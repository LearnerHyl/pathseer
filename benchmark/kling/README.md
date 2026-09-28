# Kling 独立测试

测试代码、构建依赖和结果全部在当前仓库内。Knowhere 仅提供测试口径参考，运行不链接 Knowhere，也不需要它的源码目录。

## 一键运行

线上在仓库根目录直接运行，无需参数：

```bash
./run_kling_benchmark.sh
```

无参数入口固定读取 `/media/nvme1n1/huayelin/knowhere/datasets/kling_1b` 的前六个训练分片和
`test.parquet`，top-100、100 条召回查询、每批 10 条、12 并发、每配置 60 秒，建图和真值各 16 线程。
NaviX 固定 M=32、efConstruction=200，efSearch 扫 100/200/400/800/1600/3200，
保留比例为 5%/10%/15%/20%/30%，编译并行度 4。无参数运行会覆盖同名 FANN 环境变量，避免旧设置影响结果。
数据路径来自已有线上记录，本机没有该全量数据目录；仍需在线上完成全量验证。

也可以显式运行功能检查或指定其他数据：

```bash
./run_kling_benchmark.sh --smoke
./run_kling_benchmark.sh --dataset /path/to/kling-6shards.hdf5
./run_kling_benchmark.sh --parquet-dir /path/to/kling_1b
```

`--smoke` 是 4000 条、2048 维合成数据的功能检查，不能作为 Kling 的 QPS 结论。
无参数使用上述线上路径，也可显式指定数据。HDF5 包含 `train`、`test` 两个 FP32 矩阵；Parquet 入口读取
`train-00-of-1000.parquet` 至 `train-05-of-1000.parquet` 以及 `test.parquet` 的 `emb` 列。
仓库内 `prepare_kling_parquet_dataset.py` 原样取自 Knowhere 的数据准备脚本。

首次入口会在 `build/kling/venv` 创建独立 Python 环境并安装固定版本 numpy/h5py/pyarrow。
系统需提供 CMake >=3.24、支持 C++17 的编译器、OpenMP、OpenBLAS，以及 ACORN 构建需要的 zlib 开发文件。
可用 `CXX` 指定编译器，`FANN_BOOTSTRAP_PYTHON` 指定 Python（推荐 3.11），
`FANN_PYPI_INDEX_URL` 指定包源。构建默认 4 个任务，不安装系统库。
已有 Python 依赖时也可以 `python benchmark/kling/run.py --build-only`，此入口不自动安装包。

## 测试口径

参考 Knowhere `benchmark/curator/run_pag_kling_2048.sh` 及 `MeasureQps`：

- 归一化 FP32；原生 L2 搜索单位向量，排序等价于 cosine。拒绝零向量和非有限值。
- top-100；100 条查询计算 recall，分母是 `recall_nq * topk`。真值是对合格向量的完整 FP32 扫描，未命中的结果不补算为命中。
- 默认保留 5%、10%、15%、20%、30%，谓词为 `id % 100 < percentage`。
- 12 个客户端，每批 10 条查询，计时 60 秒；每个客户端循环自己的固定查询段，匹配当前 Knowhere 热查询口径。
- 每个客户端顺序调用底层单查询接口，内部 OpenMP/BLAS 线程数为 1；建图默认 16 线程。
- 先预热，再统一开始计时；最终批次完成时间计入分母。QPS 是向量查询数/秒。
- `efSearch` 默认扫 100、200、400、800、1600、3200；每个图只构建一次，然后遍历所有场景。
- 结果检查越界、重复和不合格 ID；少于 k 个结果按真实 recall 体现。
- 汇总仅选取实测 recall 达到 0.90/0.95/0.98/0.99 的点，不对 QPS 插值，不把“返回满 100 个”作为召回率。

默认 M=32、efConstruction=200，是独立候选的起始参数，不宣称与 PAG 的量化、图度数或索引大小一致。
本测试使用 FP32；比较 PAG INT16/INT8 结果时应同时报告编码和内存差别。尚未实现索引持久化，重新启动会重新建图；
归一化数据和真值按输入文件状态、场景参数及准备脚本散列缓存。请勿在正在读取时修改数据。
6M × 2048 FP32 向量约 46 GiB，另需图、建图临时空间和准备数据的磁盘空间；本机 smoke 不代表全量可运行。

## 调参和任意 bitset

```bash
FANN_M=64 FANN_EFC=400 FANN_EFS=200,400,800,1600,3200,6400 \
  ./run_kling_benchmark.sh --dataset /path/to/kling.hdf5
./run_kling_benchmark.sh --dataset /path/to/kling.hdf5 --mask-mode random
./run_kling_benchmark.sh --dataset /path/to/kling.hdf5 --percentages '' \
  --bitset actual=/path/to/excluded.bits
```

自定义 bitset 恰好 `ceil(N/8)` 字节，低位优先，**1 表示排除**，与 Knowhere 一致。
测试准备阶段转为上游需要的 N 字节 mask，**1 表示保留**。转换和真值计算均不计入搜索 QPS。
当前一个场景对所有查询使用同一份 mask，与参考基准一致；不同查询不同 mask 的业务负载尚未覆盖。
随机模式固定种子 20260928；相关/反相关场景可以传入外部生成的 bitset，索引不接收业务标签。

环境变量：`FANN_TOPK`、`FANN_RECALL_NQ`、`FANN_NQ`、`FANN_CONCURRENCY`、`FANN_SECONDS`、
`FANN_BUILD_THREADS`、`FANN_M`、`FANN_EFC`、`FANN_EFS`、`FANN_GT_THREADS`。
`--smoke` 固定小规模计时设置；真实参数以各结果目录的 `*-settings.json` 为准。
`--work-dir` 改变缓存/构建位置，`--result-dir` 必须是尚不存在的目录。

## 源码适配细节

### NaviX

检查版本：`192fafbff7ead780185891e6056bbf67cbb19606`，Faiss 1.10.0 分支。
`faiss/IndexHNSW.cpp::navix_single_search` 接收 mask 和调用方提供的 `VisitedTable`、`HNSWStats`；
核心策略位于 `faiss/impl/HNSW.cpp::navix_hybrid_search`。构图为 `IndexHNSWFlat`。
适配器直接调用该入口，每个客户端保留自己的 visited/result 缓冲区，不更新全局统计量。
README 中直接访问 index.efSearch/efConstruction 的示例不符实际接口，实际成员位于 index.hnsw。

### ACORN

检查版本：`c259f11cdbec880671658eaa0d8747e2cc9de79d`，Faiss 1.7.3 分支。
实际 `IndexACORNFlat` 构造函数比 README 多一个 metadata 参数；适配传入全零数组，未传入任何业务属性。
默认顺序为 gamma=10、gamma=1，对照可用 `--gammas 4,10,20,1`；gamma=1 的 M_beta=2M，其余 M_beta=M。
efConstruction 取 max(配置值, M×gamma)。
原始批量 search 会合并全局 acorn_stats，适配直接调用相同的 `acorn.hybrid_search` 并复用线程私有 visited。
使用官方可选 IDSelector 提供相同的 mask，以约束初始种子加入结果的分支；不改上游遍历代码。
编译时使用上游声明的 nlohmann/json v3.10.4，固定提交并记录来源。

两个仓库分别编译各自版本的 Faiss，不在同一进程混合符号。CMake 使用 Release、`-O3 -march=native`，
并统一禁用 GPU/Python 扩展。当前源码修改可以直接测试，结果记录 HEAD 和 tracked diff，不会重置工作区。
不同 Faiss 版本及底层实现仍可能影响性能，因此这里测得的是独立实现效果，不是纯算法消融。

## 输出与验证

`benchmark-results/independent-*` 保存构建/运行日志、CSV、summary.md、数据 manifest、CPU、参数和版本信息。
任何编译或搜索错误均返回非零状态；完成标记在全部测量成功后才打印。

```bash
build/kling/venv/bin/python -m unittest discover -s benchmark/kling -p 'test_*.py'
```

MCI 源码入口不可访问，按用户要求跳过。
