# Faiss-Navix

Implementation of Navix predicate agnostic filtered vector search algorithm (adaptive-local) in Faiss.

## What is Navix?

A Native Vector Index Design for Graph DBMSs With Robust and Fast Predicate-Agnostic Search Performance.

- A novel Prefiltering-based predicate agnostic filtered vector search algorithm (adaptive-global) that is fast and robust acorss various selectivities and correlation scenarios. This algorithm works directly on top of HNSW thus making it easily integrable into most existing systems.
- Disk-Based HNSW Index backed by Buffer Manager.
- Zero Copy Fast Distance Computations through the buffer manager.
- Easy to use as implemented in an embedded database.

Navix is originally implemented on top of Kuzu, an embedded graph database system. Here is the repo: https://github.com/gaurav8297/kuzu

Our Paper for more info and benchmarks against SOTA baselines: https://cs.uwaterloo.ca/~ssalihog/papers/navix-tr.pdf

## Build From Source

The library is mostly implemented in C++, the only dependency is a [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) implementation. It compiles with cmake. 

Simple build instructions for Linux:
```shell
> cmake -B build -FAISS_OPT_LEVEL=avx512 -DCMAKE_BUILD_TYPE=Release -DFAISS_USE_LTO=ON -DBUILD_TESTING=OFF  .
> make -C build -j faiss_avx512 install
```

See Building from source section in [INSTALL.md](INSTALL.md) for more details.

## How it works?

Initialize the index
```c++
int M = 32;
int dimension = 1024;
auto index = faiss::IndexHNSWFlat(dimension, M, faiss::METRIC_L2);
index.efConstruction = 200
```

Construct the index
```c++
float* data = ...; // Your data points
int num_data_points = ...; // Number of data points
index.add(num_data_points, data);
```

Search the index
```c++
// Single query search

int k = 100; // Number of nearest neighbors to search
uint8_t* filter_mask = new uint8_t[num_data_points]; // Filter mask for predicate
for (int i = 0; i < num_data_points; ++i) {
    filter_mask[i] = ...; // Set filter mask based on your predicate (1 for include, 0 for exclude)
}

float* query = ...; // Your query point
auto labels = new faiss::idx_t[k];
auto distances = new float[k];
index.efSearch = 200
faiss::VisitedTable visited(num_data_points);
faiss::HNSWStats stats;

// Hybrid search using Navix
index.navix_single_search(query, k, distances, labels, reinterpret_cast<char*>(filter_mask), visited, stats);

// Regular search without Navix
index.single_search(query, k, distances, labels, visited, stats);

// ====================================== OR ==========================================
// Multiple queries and parallel search

int num_queries = ...; // Number of queries
float* queries = ...; // Your query points
labels = new faiss::idx_t[num_queries * k];
distances = new float[num_queries * k];
uint8_t* filter_masks = new uint8_t[num_data_points * num_queries]; // Filter mask for multiple queries
for (int i = 0; i < num_queries; ++i) {
    for (int j = 0; j < num_data_points; ++j) {
        filter_mask[i * num_data_points + j] = ...; // Set filter mask for each query
    }
}

// Hybrid search for multiple queries
index.navix_search(num_queries, queries, k, distances, labels, reinterpret_cast<char*>(filter_masks));

// Regular search for multiple queries
index.search(num_queries, queries, k, distances, labels);
```
