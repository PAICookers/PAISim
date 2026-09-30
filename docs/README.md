# PAISim 文档

状态：共享工程文档。内容只描述可复现的 PAISim 行为、设计边界和验证方法。

## 入口

- [用户指南](user-guide.md)：环境准备、仿真操作、结果获取和终端输出。
- [架构](architecture.md)：对象层次、路由、线程和硬件保真边界。
- [运行契约](runtime.md)：原始帧 API、线程行为和输出语义。
- [观测](observability.md)：`TraceFilter`、`SimEvent` 和回调错误边界。
- [验证](validation.md)：测试矩阵、证据等级和验收命令。
- [资料索引](references.md)：公开来源和引用规则。
- [决策记录](decisions/)：影响公共行为的长期决策。

## 共享规则

共享文档使用仓库相对链接，不包含本机路径、用户名、私有数据集路径、临时日志或
未脱敏的机器证据。详细 profiling、调试转储和历史发布材料存放在未跟踪的
`docs-local/`，不作为公共 API 或功能证明。
