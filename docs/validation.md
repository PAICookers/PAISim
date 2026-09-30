# 验证范围

## 自动化验证矩阵

| 范围 | 关键检查 |
| --- | --- |
| 离线数值 | full/half/fold、dense/CSC、整数位宽、MAC/maxpool/direct-add、LUT、溢出、输入环 |
| 可变对象 | 神经元参数/电位、在线 raw、Weights、Lut 写回，revision 和缓存失效，公开数组只读 |
| 帧事务 | 半包跨 feed、完整包原子提交、错误无半写、多播 WORK 原子失败、TEST/INIT |
| 在线核 | 低层对象可观察；Simulator 配置路径整体拒绝在线核，SYNC/UPDATE 不执行数值计算 |
| 板卡 DATA | 预设四条 X/Y 边、自定义显式边、缺边对角拒绝、所有跨片 XY 负例、CPU 端口 |
| 板卡 signal | 9→1→9、receive 门控、跨线程物理传播、relay/completion、busy/done |
| 线程 | 多线程同时 runnable、独立 tick/完成、稳定轮转、统一提交、对象插入顺序 |
| 重放/产物 | manifest v3、自包含板卡/入口/thread、Protobuf 与 JSON 夹具、哈希与路径边界 |

高位 `lcn`/`target_lcn`（大于 7）以及同时启用 `max_pooling` 与 `add_potential` 的
硬件行为当前未由实测或芯片设计资料确认。模拟器会明确报告“硬件行为未定义”，不把
该诊断解释成芯片实际行为；需要向芯片设计人员确认或补充硬件测试。

主门命令：

```bash
uv run pytest
uv run ruff format --check src scripts tests
uv run ruff check src scripts tests
uv build
```

外部 DVS 测试需要仓库外数据；缺少数据时跳过，不冒充通过。Protobuf
测试随基础依赖运行。构建后在全新临时环境安装 wheel，检查导入、
`py.typed` 和最小对象 API。

## 性能门

固定负载为 4 核×128 神经元×32 步，固定 BLAS 线程、一次预热、九轮独立会话。旧基线中位数
为 126.57 ms，最大允许值为 139.22 ms；输出 SHA-256 必须保持
`c79e0c0c43d1ba97531b3c30894f58d47a06df6d651e5aab7a69fa8b17a594f3`。

优化只针对 profiler 证明的热点。当前实现缓存不可变输出路由和帧路由，调度 runnable 集合
避免无关扫描；不引入线程池或额外 profiling 依赖。最终计时、cProfile 和 tracemalloc 结果
记录在 `tasks/todo.md` 的 Review 中。

这些结果证明确定性软件行为，不证明物理 lane 带宽、板级时序、RTL 等价或硅片周期精度。
