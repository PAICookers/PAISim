# 观测契约

状态：当前实现契约。

PAISim 是确定性、帧驱动的硬件功能仿真器。库代码不自动打印、不配置全局日志。用户可以
通过 `Recorder` 有界保存结构化事件，或直接使用回调和 `snapshot()`。

## 追踪选择

```python
from paisim import Simulator, TraceFilter, TraceLevel

events = []
sim = Simulator(
    board,
    events.append,
    trace_filter=TraceFilter(level=TraceLevel.TRAFFIC),
)
```

`TraceLevel` 按包含关系递增：`OUTPUT` 记录所有内部到外部（I2E）数据、读回 payload
和 COMPLETE；`PROGRESS` 再加 `step`；`TRAFFIC` 再加 E2I/I2I 的 `frame`、`payload`、
`signal`；`DEBUG` 再加核级事件（如 `core_step`）。`TraceFilter.outputs()` 等价于
`TraceFilter(level=TraceLevel.OUTPUT)`。

过滤器还可以按 `directions`、`ports`、`cores`、`threads`、`ticks` 和 `kinds` 筛选。
所有字段按 AND 组合；`None` 表示不限制，空集合表示不接收匹配事件。路由事件在 source
或 destination 命中所选核时保留，`core_step` 通过 `core` 字段命中。

## 事件

`SimEvent` 是不可变值对象，字段为 `sequence`、`kind`、`direction`、`source`、
`destination`、`port`、`word`、`raw_word`、`thread`、`tick`、`core` 和预留的 `route`。
`destination` 始终是物理 `CoreAddr`；I2E 事件通过 `port` 标识声明的外部端口。事件类型为：

- `frame`：完整协议帧完成路由、写入或读回投递；
- `payload`：payload 被接受或读回数据词被投递；
- `signal`：全局控制信号沿物理连接传播；
- `core_step`：计算核完成一次状态提交及其输出交付；
- `step`：线程完成一次调度步；
- `complete`：COMPLETE 词进入声明的输出端队列。

方向值为 `PortDirection.E2I`、`I2I`、`I2E`。事件按仿真器单实例的递增序号产生。
普通线程步中，`core_step`（如果有）先于 `step` 回调。最终线程步中，调度器会先交付
计算输出并回调 `complete`，然后 `advance()` 再回调该线程的 `step`；因此最终步的回调
顺序是 `core_step`、`complete`、`step`，而不是 `step`、`complete`。用户应按 `kind`、
`direction` 和 `sequence` 处理事件，不要依赖一个未筛选回调序列中的固定位置。输出队列
本身仍保证普通计算输出先于 COMPLETE。异常继续通过 `SimulationFailure` 报告，不转换成
普通事件。

## Recorder 与终端

最基础的逐步输出查看方式是把 Recorder 的过滤器同时交给模拟器，并连接终端打印器：

```python
from paisim import PortDirection, Recorder, Simulator, TerminalPrinter, TraceFilter

recorder = Recorder(
    trace_filter=TraceFilter.outputs(),
    printer=TerminalPrinter(format="hex"),
)
sim = Simulator(board, on_event=recorder, trace_filter=recorder.trace_filter)
sim.feed(words, ingress=cpu)
sim.run()
```

终端只显示 I2E 输出。`TerminalPrinter` 支持 `hex`、`decimal` 和固定 64 位、每 8 位分组
的 `bin`。Recorder 默认保留最近 10,000 条记录，超限丢弃最旧记录并累计 `dropped_count`；
可通过 `max_records` 修改，`records` 返回当前快照，`clear()` 清空记录和计数。

需要按线程步观察边界时，选择 `PROGRESS` 并使用 `advance(1)`：

```python
from paisim import PortDirection, Recorder, Simulator, TerminalPrinter, TraceFilter, TraceLevel

recorder = Recorder(
    trace_filter=TraceFilter(level=TraceLevel.PROGRESS),
    printer=TerminalPrinter(format="hex"),
)
sim = Simulator(board, on_event=recorder, trace_filter=recorder.trace_filter)
sim.feed(words, ingress=cpu)
while sim.advance(1):
    new_records = recorder.records
    recorder.clear()
    step = next((item for item in new_records if item.kind == "step"), None)
    if step is not None:
        outputs = [
            item for item in new_records if item.direction is PortDirection.I2E
        ]
        print(f"thread={step.thread} tick={step.tick} outputs={outputs}")
```

若只需程序化取得输出，直接读取 `recorder.records` 中 `direction is PortDirection.I2E`
的记录；若需要原始输出数组，继续使用 `sim.drain_output()`。

记录可导出为单个 JSONL 或小端 uint64 文件：

```python
recorder.export("events.jsonl", format="jsonl")
recorder.export("outputs.bin", format="raw-u64-le")
```

`raw-u64-le` 只写 I2E 记录的 `word`；v1 不提供文件轮转。

## 回调错误

回调在对应状态提交后调用。回调抛出的异常包装为 `SimulationFailure`，诊断码为
`observer_error`；已经提交的核状态、路由状态和输出不会回滚，诊断中的 `effect` 记录
已提交的状态前缀。`Recorder` 也遵循这一语义。
