// Independent upstream search benchmark: prepared cosine data, masks and exact truth.
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/Heap.h>
#if FANN_navix
#include <faiss/IndexHNSW.h>
#else
#include <faiss/IndexACORN.h>
#endif

namespace filtered_bench {
using Clock = std::chrono::steady_clock;

// Reject malformed inputs before building or reporting measurements.
static void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Read a positive integer with complete-string validation.
static int32_t positive(const std::string& value) {
    size_t used = 0;
    const int64_t parsed = std::stoll(value, &used);
    require(used == value.size() && parsed > 0 && parsed <= INT32_MAX, "invalid positive integer: " + value);
    return static_cast<int32_t>(parsed);
}

// Read benchmark configuration without inheriting Knowhere's global state.
static std::string env(const char* key, const char* fallback) {
    const char* value = std::getenv(key);
    return value == nullptr ? fallback : value;
}

struct Config {
    int32_t topk = positive(env("FANN_TOPK", "100"));
    int32_t recall_nq = positive(env("FANN_RECALL_NQ", "100"));
    int32_t batch = positive(env("FANN_NQ", "10"));
    int32_t concurrency = positive(env("FANN_CONCURRENCY", "12"));
    int32_t seconds = positive(env("FANN_SECONDS", "60"));
    int32_t build_threads = positive(env("FANN_BUILD_THREADS", "16"));
    int32_t m = positive(env("FANN_M", "32"));
    int32_t efc = positive(env("FANN_EFC", "200"));
    int32_t gamma = positive(env("FANN_GAMMA", "10"));
};

// Own an mmap rather than an additional full in-memory copy of 6M vectors.
class Matrix {
 public:
    explicit Matrix(const std::string& path) {
        const int32_t fd = open(path.c_str(), O_RDONLY);
        require(fd >= 0, "cannot open matrix: " + path);
        struct stat info {};
        const int32_t status = fstat(fd, &info);
        bytes_ = static_cast<size_t>(info.st_size);
        mapping_ = status == 0 ? mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
        close(fd);
        require(mapping_ != MAP_FAILED, "cannot map matrix: " + path);
        require(bytes_ >= 16, "matrix header missing: " + path);
        const auto* shape = static_cast<const uint64_t*>(mapping_);
        rows = shape[0];
        dimension = shape[1];
        require(rows > 0 && rows <= INT32_MAX && dimension > 0 && dimension <= INT32_MAX,
                "unsupported matrix dimensions: " + path);
        require(bytes_ == 16 + rows * dimension * sizeof(float), "matrix size mismatch: " + path);
    }
    ~Matrix() { munmap(mapping_, bytes_); }
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
    // Return a checked-by-caller row in the prepared matrix.
    const float* row(size_t id) const {
        return reinterpret_cast<const float*>(static_cast<const char*>(mapping_) + 16) + id * dimension;
    }
    uint64_t rows = 0;
    uint64_t dimension = 0;
 private:
    void* mapping_ = MAP_FAILED;
    size_t bytes_ = 0;
};

// Read masks and ground truth strictly; truncated files must never yield measurements.
template <typename T>
static std::vector<T> read_array(const std::string& path, size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.good() && input.tellg() == static_cast<std::streamoff>(count * sizeof(T)),
            "file size mismatch: " + path);
    std::vector<T> values(count);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    require(input.good(), "cannot read file: " + path);
    return values;
}

struct Scene {
    std::string name;
    std::vector<char> mask;
    std::vector<int64_t> truth;
};

// Supply the same eligibility predicate to ACORN's initial result insertion.
class MaskSelector : public faiss::IDSelector {
 public:
    explicit MaskSelector(const std::vector<char>& mask) : mask_(mask) {}
    bool is_member(faiss::idx_t id) const override {
        return id >= 0 && static_cast<size_t>(id) < mask_.size() && mask_[id] != 0;
    }
 private:
    const std::vector<char>& mask_;
};

#if FANN_navix
using UpstreamIndex = faiss::IndexHNSWFlat;
#else
using UpstreamIndex = faiss::IndexACORNFlat;
#endif

// Build without business metadata. Zero-valued ACORN metadata only satisfies its ABI.
static std::unique_ptr<UpstreamIndex> build_index(const Matrix& train, const Config& config,
                                                std::vector<int32_t>& metadata) {
    omp_set_num_threads(config.build_threads);
#if FANN_navix
    (void)metadata;
    auto index = std::make_unique<UpstreamIndex>(train.dimension, config.m, faiss::METRIC_L2);
    index->hnsw.efConstruction = config.efc;
#else
    metadata.resize(train.rows, 0);
    const int32_t beta = config.gamma == 1 ? 2 * config.m : config.m;
    auto index = std::make_unique<UpstreamIndex>(train.dimension, config.m, config.gamma, metadata, beta);
    index->acorn.efConstruction = std::max(config.efc, config.m * config.gamma);
#endif
    index->add(train.rows, train.row(0));
    return index;
}

// Every worker owns its visited table and result buffers; no global Faiss search statistics are updated.
class Searcher {
 public:
    Searcher(const UpstreamIndex& index, const Config& config, Scene& scene)
        : index_(index), scene_(scene), visited_(index.ntotal), selector_(scene.mask),
          ids(config.topk), distances(config.topk) {}
    // Reproduce upstream single-query traversal and heap handling with worker-local state.
    void search(const float* query) {
#if FANN_navix
        faiss::HNSWStats stats;
        index_.navix_single_search(query, ids.size(), distances.data(), ids.data(),
                                  scene_.mask.data(), visited_, stats);
#else
        std::unique_ptr<faiss::DistanceComputer> computer(index_.storage->get_distance_computer());
        computer->set_query(query);
        faiss::maxheap_heapify(ids.size(), distances.data(), ids.data());
        faiss::SearchParametersACORN params;
        params.efSearch = index_.acorn.efSearch;
        params.sel = &selector_;
        index_.acorn.hybrid_search(*computer, ids.size(), ids.data(), distances.data(),
                                  visited_, scene_.mask.data(), &params);
        faiss::maxheap_reorder(ids.size(), distances.data(), ids.data());
#endif
    }
 private:
    const UpstreamIndex& index_;
    Scene& scene_;
    faiss::VisitedTable visited_;
    MaskSelector selector_;
 public:
    std::vector<faiss::idx_t> ids;
    std::vector<float> distances;
};

// Count set intersection, rejecting duplicate or ineligible results independently of recall.
static size_t count_hits(const Searcher& searcher, const Scene& scene, size_t query) {
    const auto first = scene.truth.begin() + query * searcher.ids.size();
    const std::unordered_set<int64_t> expected(first, first + searcher.ids.size());
    std::unordered_set<int64_t> seen;
    size_t hits = 0;
    for (const int64_t id : searcher.ids) {
        if (id >= 0) {
            require(static_cast<size_t>(id) < scene.mask.size() && scene.mask[id], "ineligible search result");
            require(seen.insert(id).second, "duplicate search result");
            hits += expected.count(id);
        }
    }
    return hits;
}

// Measure true filtered recall separately from the timed throughput loop.
static double measure_recall(Searcher& searcher, const Matrix& queries, const Config& config, Scene& scene) {
    size_t hits = 0;
    omp_set_num_threads(1);
    for (int32_t query = 0; query < config.recall_nq; ++query) {
        searcher.search(queries.row(query));
        hits += count_hits(searcher, scene, query);
    }
    return static_cast<double>(hits) / (static_cast<int64_t>(config.recall_nq) * config.topk);
}

struct Timing {
    std::atomic<int32_t> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> failed{false};
    std::atomic<uint64_t> count{0};
    Clock::time_point deadline;
    std::mutex error_mutex;
    std::exception_ptr error;
};

struct WorkerInput {
    Searcher& searcher;
    const Matrix& queries;
    const Config& config;
    size_t offset = 0;
};

// Warm up outside the clock, then repeat the same nq batch as Knowhere's current harness.
static void run_worker(const WorkerInput& input, Timing& timing) {
    try {
        omp_set_num_threads(1);
        for (int32_t i = 0; i < input.config.batch; ++i) {
            input.searcher.search(input.queries.row(input.offset + i));
        }
        timing.ready.fetch_add(1);
        while (!timing.start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        uint64_t count = 0;
        while (!timing.failed.load() && Clock::now() < timing.deadline) {
            for (int32_t i = 0; i < input.config.batch; ++i) {
                input.searcher.search(input.queries.row(input.offset + i));
            }
            count += input.config.batch;
        }
        timing.count.fetch_add(count);
    } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(timing.error_mutex);
        std::cerr << "ERROR worker offset=" << input.offset << " batch=" << input.config.batch
                  << " action=abort benchmark cause=" << error.what() << std::endl;
        timing.error = std::current_exception();
        timing.failed.store(true);
    }
}

// Start all warmed-up workers together and include completion of their final batches in elapsed time.
static double measure_qps(std::vector<std::unique_ptr<Searcher>>& searchers, const Matrix& queries,
                          const Config& config) {
    Timing timing;
    std::vector<std::thread> workers;
    for (int32_t worker = 0; worker < config.concurrency; ++worker) {
        const size_t offset = (static_cast<size_t>(worker) * config.batch) % (queries.rows - config.batch + 1);
        const WorkerInput input{*searchers[worker], queries, config, offset};
        workers.emplace_back([input, &timing]() { run_worker(input, timing); });
    }
    while (timing.ready.load() < config.concurrency && !timing.failed.load()) {
        std::this_thread::yield();
    }
    const auto started = Clock::now();
    timing.deadline = started + std::chrono::seconds(config.seconds);
    timing.start.store(true, std::memory_order_release);
    for (auto& worker : workers) {
        worker.join();
    }
    if (timing.error) {
        std::rethrow_exception(timing.error);
    }
    return timing.count.load() / std::chrono::duration<double>(Clock::now() - started).count();
}

// Sweep ef on one built index so build time is never charged to QPS.
static void run_scene(UpstreamIndex& index, const Matrix& queries, Scene& scene, std::ostream& output) {
    const Config config;
    std::vector<std::unique_ptr<Searcher>> searchers;
    for (int32_t i = 0; i < config.concurrency; ++i) {
        searchers.push_back(std::make_unique<Searcher>(index, config, scene));
    }
    std::istringstream efs(env("FANN_EFS", "100,200,400,800,1600,3200"));
    std::string token;
    while (std::getline(efs, token, ',')) {
        const int32_t ef = positive(token);
        require(ef >= config.topk, "efSearch must be >= topk");
#if FANN_navix
        index.hnsw.efSearch = ef;
#else
        index.acorn.efSearch = ef;
#endif
        const double recall = measure_recall(*searchers[0], queries, config, scene);
        const double qps = measure_qps(searchers, queries, config);
        const size_t accepted = std::count(scene.mask.begin(), scene.mask.end(), 1);
        output << scene.name << ',' << accepted << ',' << ef << ',' << recall << ',' << qps << '\n' << std::flush;
        std::cout << "RESULT scene=" << scene.name << " ef=" << ef << " recall=" << recall
                  << " qps=" << qps << std::endl;
    }
}

// Validate prepared inputs, build once, then measure every scene in the manifest.
static void benchmark(const std::string& directory, const std::string& result) {
    const Config config;
    const Matrix train(directory + "/train.f32");
    const Matrix queries(directory + "/test.f32");
    require(train.dimension == queries.dimension, "train/query dimension mismatch");
    require(queries.rows >= static_cast<uint64_t>(std::max(config.recall_nq, config.batch)), "too few queries");
    require(train.rows >= static_cast<uint64_t>(config.topk), "too few base vectors");
    std::ifstream names(directory + "/scenes.txt");
    require(names.good(), "cannot open scene manifest");
    std::ofstream output(result);
    require(output.good(), "cannot open result: " + result);
    output << "scene,accepted,ef,recall,qps\n";
    std::vector<int32_t> metadata;
    const auto started = Clock::now();
    auto index = build_index(train, config, metadata);
    std::cout << "BUILD seconds=" << std::chrono::duration<double>(Clock::now() - started).count()
              << " rows=" << train.rows << " dimension=" << train.dimension << std::endl;
    std::string name;
    size_t scenes = 0;
    while (names >> name) {
        Scene scene{name, read_array<char>(directory + "/" + name + ".mask", train.rows),
                    read_array<int64_t>(directory + "/" + name + ".gt", config.recall_nq * config.topk)};
        require(std::count(scene.mask.begin(), scene.mask.end(), 1) >= config.topk, "too few eligible vectors");
        run_scene(*index, queries, scene, output);
        ++scenes;
    }
    require(scenes > 0 && output.good(), "no scenes or result write failed");
}
}  // namespace filtered_bench

// Report failures to the runner; never turn a failed benchmark into a successful CSV.
int main(int argc, char** argv) {
    int32_t status = 0;
    try {
        filtered_bench::require(argc == 3, "usage: filtered_benchmark PREPARED_DIRECTORY RESULT.csv");
        filtered_bench::benchmark(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << "ERROR benchmark action=abort cause=" << error.what() << std::endl;
        status = 1;
    }
    return status;
}
