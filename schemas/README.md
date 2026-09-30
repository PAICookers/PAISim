# Protobuf compiler artifact schema

Source: PAIBox `paibox/backendv2/schemas/compile_artifacts.proto` at the pinned
compiler revision. Protobuf is the only formal artifact input. The runtime schema
version is 1.

Generation tools: protoc 27.1. From the PAISim root:

```sh
protoc -I schemas --python_out=src/paisim/_generated --pyi_out=src/paisim/_generated schemas/compile_artifacts.proto
```

The generated directory is excluded from Ruff formatting and linting.

The Protobuf bindings include generated `.pyi` declarations. Generated code is
excluded from strict static analysis because generator output does not satisfy
strict typing; the handwritten container and tensor boundaries are checked.

No schema or Python source is loaded from a customer's artifact directory.

The base project dependency supplies the Protobuf runtime used by both binary
and JSON artifact readers. Schemas do not encode a full stimulus stream,
initial state, or tensor preprocessing policy.
