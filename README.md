<div align="center">

# PAISim

</div>

<p align="center">
    <a href="https://github.com/PAICookers/PAISim/blob/main/pyproject.toml">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/paisim">
    </a>
    <a href="https://pypi.org/project/paisim">
        <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/paisim?color=pink">
    </a>
    <a href="https://www.codefactor.io/repository/github/PAICookers/PAISim">
        <img alt="CodeFactor Grade" src="https://img.shields.io/codefactor/grade/github/PAICookers/PAISim?color=orange">
    </a>
    <a href="https://results.pre-commit.ci/latest/github/PAICookers/PAISim/main">
        <img alt="pre-commit.ci status" src="https://results.pre-commit.ci/badge/github/PAICookers/PAISim/main.svg">
    </a>
    <a href="https://codecov.io/gh/PAICookers/PAISim" >
        <img src="https://codecov.io/gh/PAICookers/PAISim/graph/badge.svg?token=978U1BIZRE"/>
    </a>
    <a href="https://github.com/PAICookers/PAISim/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/PAICookers/PAISim">
    </a>
</p>

PAISim 是面向 PAICORE 2.5 的确定性、帧驱动功能仿真器。它在 Python 中执行原始帧，
维护板卡拓扑、路由、核状态和线程调度，并返回仿真产生的输出。它适合检查编译器生成的
帧、运行时协议和板卡级功能行为。

PAISim 关注功能语义，不模拟真实硬件的物理周期、lane 带宽、MUX/DEMUX、仲裁、流水延迟
或墙钟时间。应用层输入输出编码、TensorCodec、准确率比较和真实板卡验证属于外围工具。

当前运行路径面向离线核。在线核对象可用于较低层观察，但 `Simulator` 在在线核配置阶段
会拒绝完整在线执行；它不支持在线核的 SYNC/UPDATE 数值计算。仿真结果可以证明帧级状态
转换和输出队列行为，不能单独证明编译器成功、RTL 等价、应用精度或真实板卡性能。

详细的仿真操作和结果查看方式见 [用户指南](docs/user-guide.md)。

## 使用前提

- Python 3.11 或更高版本
- `uv`
- PAICORE 编译器
- Protobuf 编译产物读取是核心运行路径，基础安装会包含 Protobuf 运行时

开发环境可以直接同步：

```bash
uv sync --dev
```

运行时依赖 NumPy、`paicorelib>=2.0.0,<3` 和 Protobuf。只使用原始帧时，Protobuf
仍随基础安装提供，因为编译产物是 PAISim 的主要输入边界。

## 最短示例

仿真器接收已编码的 `np.uint64` 帧。`feed()` 只排队输入，`run()` 才推进仿真，
`drain_output()` 返回并清空当前输出队列：

```python
import numpy as np

from paisim import ChipCoord, Port, Simulator, SingleBoard

board = SingleBoard()
cpu = Port(ChipCoord(0, 0))
sim = Simulator(board)

words = np.asarray(encoded_words, dtype=np.uint64)
sim.feed(words, ingress=cpu)
sim.run()
outputs = sim.drain_output()
print(outputs)
```

`encoded_words` 必须由编译器或帧编码器提供。PAISim 不会从 Python 数值、张量或应用样本
自动生成协议帧。

如果输入来自 Protobuf 编译产物：

```python
import numpy as np

from paisim import ChipCoord, Port, Simulator, SingleBoard

cpu = Port(ChipCoord(0, 0))
sim = Simulator.from_artifact("config.pb", SingleBoard())
sim.feed(np.asarray(input_frames, dtype=np.uint64), ingress=cpu)
sim.run()
outputs = sim.drain_output()
```

上述接口给出原始 `np.uint64` 输出。需要记录来源、方向、端口、线程和 tick 时，使用
`Recorder`；需要在终端查看格式化输出时，使用 `TerminalPrinter`。两者的完整用法见
[用户指南](docs/user-guide.md) 和 [观测契约](docs/observability.md)。

## 仓库结构

```text
src/paisim/
├── frames.py       PAICORE 帧编码、路由位和输入地址
├── topology/       芯片坐标、板卡声明、芯片阵列和路由
├── engine/         帧解码、数值 kernel、SRAM 和 core 状态
├── artifact/       Protobuf/JSON artifact 读取与输入输出映射
├── runtime/        Simulator、scheduler、观测、错误和 replay
└── _generated/     固定的 Protobuf Python 绑定

tests/
├── artifact/       编译产物、映射和 fixtures
├── engine/         core 解码、数值和存储行为
├── protocol/       paisim.frames 的帧编码和输入地址测试
├── runtime/        Simulator、调度、replay 和观测
└── topology/       板卡、芯片和路由
```

## 开发

在仓库根目录运行 `uv` 命令。源码位于 `src/paisim`，新增功能应放入对应职责的模块；
公共入口保持在 `paisim.__init__`，内部实现从具体子模块导入。

提交代码前运行完整检查：

```bash
uv run pytest -q
uv run ruff format --check src scripts tests
uv run ruff check src scripts tests
uv run mypy --strict src/paisim
uv build
```

常用的局部检查：

```bash
uv run pytest -q tests/runtime
uv run pytest -q tests/artifact
uv run pytest -q tests/protocol
```

## 文档入口

- [用户指南](docs/user-guide.md)：从输入帧到输出结果的完整操作示例
- [运行契约](docs/runtime.md)：帧事务、线程、输出队列和板级路由
- [观测契约](docs/observability.md)：`TraceFilter`、`SimEvent`、`Recorder` 和终端打印
- [架构说明](docs/architecture.md)：对象层次、状态所有权和功能仿真边界
- [验证范围](docs/validation.md)：测试矩阵、验收命令和证据边界
- [资料索引](docs/references.md)：硬件、编译器、运行时和仿真器资料的使用边界
- [工程规则](AGENTS.md)：开发、测试、文档和共享索引规则
