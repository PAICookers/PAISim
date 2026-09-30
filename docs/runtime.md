# PAISim 运行行为

## 公共 API

| API | 行为 |
| --- | --- |
| `Simulator(board, on_event=None)` | 校验 `TargetBoard` 声明并固定板卡快照，创建独立 `ChipArray`；不隐式发送 INIT |
| `feed(words, *, ingress)` | 校验并复制一维整数词，按 `Port` 排队；不推进仿真 |
| `advance(max_events=1)` | 推进最多 N 个输入事务或线程步，返回实际数量 |
| `run(max_events=None)` | 运行至空闲，或在事件预算耗尽时保留状态 |
| `drain_output()` | 返回并清空原始 uint64 输出词 |
| `snapshot(selection=None)` | 返回所选核和线程的稳定只读副本 |
| `finish_input()` | 检查是否仍有未完成配置包 |
| `array.chip(coord)` / `chip.core(coord)` | 按物理层次取得对象 |
| `OfflineCore.neurons[index]` | 按需取得可变离线神经元视图 |

`SingleBoard()` 与 `Array2x2Board()` 是不可修改的预设，声明了芯片、四条 Mesh 边
（单片无边）及各芯片 `(0,0)` CPU 入口/出口。自定义子类通过 `declaration` 属性返回
`BoardDecl(chips, links, ingress, outputs)`；`CustomBoard` 提供可编辑的声明集合。
声明对象只承载输入数据；`snapshot_board()` 在创建仿真器时统一复制并校验，
运行态以不可变 `BoardSnapshot` 保存。
自定义坐标为非负二维整数，边只允许已声明芯片间的单位距离 X/Y 邻接，不自动补边。
自定义 CPU 端口需明确列出 `Port(chip, core)`，`feed` 的入口必须在声明集合内；
更改板卡对象只影响随后创建的仿真器，运行态仍在 `sim.array` 中。

观测事件、追踪筛选、Recorder、终端打印和回调错误边界见
[observability.md](observability.md)。使用 `TraceFilter.outputs()` 和 `Recorder` 可以只查看
板卡向 CPU/外部端口的输出；`Recorder.records` 保存有界的结构化结果，`drain_output()`
仍提供原始 `uint64` 输出队列的主动读取接口。

## 帧事务

CONFIG header 0..3 分别写配置寄存器、LUT、神经元/权重 SRAM 和 Input SRAM。包头记录
后续 payload 数量；跨 `feed` 按入口保存半包。完整包先校验所有目标和配置内容，失败不创建
核、不留下半写。运行中有 runnable 线程时拒绝重配置。

TEST header 4..7 且 `package_type=1`，请求只有头，count 是读回词数。离线和在线计算核都
支持读回；空核没有 SRAM。WORK header 8..11 写目标核输入环；多播先验证全部目标再提交。
SYNC、INIT、COMPLETE、UPDATE 分别使用 header 12..15；外部 COMPLETE 注入拒绝。

每个入口独立保存包状态。输入非法值、容量超限和完整包失败不会截断成合法值。响应保留原始
协议词；调用方应及时 `drain_output()`，输出队列本身不设历史保留策略。一个线程的最后一步
会先交付计算输出，再追加 COMPLETE，因此 COMPLETE 是该线程输出队列中的最后一词。
事件回调的先后不同于输出队列的先后：最终步中 `complete` 回调发生在 `step` 回调之前，
即通常为 `core_step`、`complete`、`step`。调用方应读取事件的 `kind` 和 `sequence`，不要
根据回调位置推断输出队列顺序。

## 线程语义

SYNC payload 是非负增量步数；`SYNC(0)` 校验并激活控制树，不推进时间步，随后立即
产生 COMPLETE。控制 root 通过 `global_send` /
`global_receive` 建立 `SignalTree`；local 位决定计算或 completion 成员。每个线程有独立
tick、预算和完成状态。调度器在输入事件之间按升序 `ThreadId` 稳定轮转所有 runnable 线程。

线程一步执行 `read -> compute -> commit -> deliver`。新输出不会被同一线程的其他成员在
该步读取。预算归零后置 `done=True`、`busy=False`，并从 root 返回一条 COMPLETE。INIT
清该线程输入和动态状态并重置 tick。不同线程无需共享芯片级 tick 或同步屏障。

当前 `Simulator` 的普通配置和线程执行路径面向离线核。在线核相关对象可以在较低层被
观察，但普通配置路径会在在线核配置阶段报告 `online core execution is unsupported`；
因此在线核不能通过当前 `Simulator` 完成完整配置、SYNC 数值执行或 UPDATE 数值执行。

## 板级路由

预设 `array2x2` 只有四条 X/Y 邻接。DATA 可在路由的 X/Y 阶段跨已声明的边；对角芯片路由必须先后经过
两条边。任何需要在 `XY+` 或 `XY-` 阶段离开芯片的路由返回不支持，不自动拆成 X 和 Y。

跨片控制信号通过 `SignalLink` 做 `9 -> 1 -> 9`。对端九个边界核均是物理候选，反向
`global_receive` 位决定是否接入本核控制路径。仿真器不依据 `ThreadId` 拦截物理传播；
编译器负责保证生成配置的逻辑正确性。

## 对象修改

`OfflineNeuron.voltage` 和 `set_parameter()`、`Lut.__setitem__`、`Weights.__setitem__`
是受控写入口。成功写入会递增所属核 revision；离线解码和输出路由缓存
随即失效。公开数组视图只读，快照数组也只读且与后续执行隔离。

## 重放 manifest v3

旧 manifest 不兼容。顶层字段必须恰好是 `version`、`target`、`board`、`events`，
其中 version=3 只标识文件结构、target=`paicore-2.5` 标识硬件类型。board 可为
`"single"`、`"array2x2"`，或含 `chips`、`links`、`ingress`、`outputs` 的对象。
芯片坐标为 `[x,y]`，边为两组芯片坐标，端口为 `[chip_x,chip_y,core_x,core_y]`。
`board_document(sim.board)` 可导出运行态快照的等效声明，不序列化用户 Python 子类。
每个事件必须记录 `path`、`format`、`sha256`、`ingressChip`、
`ingressCore`、`threadId`，可选 `start`/`stop`。

文件仅支持一维整数 NPY 或显式小端 `raw-u64-le`。路径必须位于 manifest 目录内，读取受
字节和事件预算限制。回放不补 CONFIG、INIT、SYNC，也不推断张量或线程。

## 明确不支持

- 旧 API、旧内部类型、旧 manifest 和兼容适配层。
- 在线核 SYNC/UPDATE 数值计算。
- 跨芯片 `XY+`/`XY-`、3D 板级拓扑和从帧自动推断板卡。
- 自定义拓扑只有功能仿真语义，不宣称存在对应的实物板卡。
- lane、MUX/DEMUX、仲裁、带宽、物理周期和 OS/Python 线程池。
