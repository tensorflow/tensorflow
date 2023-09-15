# Experimental backend/runtime for XLA:GPU

This is an experimental backend for XLA:GPU that uses StreamExecutor, IREE VM
and (optional) IREE CUDA HAL to execute programs compiled by XLA:GPU compiler.

We do not build it by default, and do not build at all in open source. This is
a prototype with a goal to build a runtime that will allow us to use better
integration with latest CUDA APIs, primarily relying on explicit CUDA graph
construction instead of graph captures.

## Coding Style

Few recommendations on how to mix `xla`, `absl` and `iree` in a single code
base and keep some sanity.

### xla::Status vs absl::Status vs iree::Status

In XLA `xla::Status` is an alias to `absl::Status`. IREE has its own
`iree::Status`.

Warning: NEVER bring `iree::Status` into the scope with using declarations.

Warning: Try not to bring any `iree` types into the scope with using
declarations.

### Implementing IREE VM custom modules

Tip: When implementing APIs for exporting as IREE VM custom modules, it's
required to return `iree:Status` or `iree::StatusOr` result. In all other APIs
use `xla::Status` or `xla::StatusOr`.

Tip: All of reference-counted types (inherit from `iree::vm::RefObject`) should
be declared in the `xla::gpu::vm` namespace.

Tip: All of IREE VM custom modules and corresponding helper functions also
should be implemented in the `xla::gpu::vm` namespace.

### Passing IREE structs to functions

IREE runtime is written in C, and most of the types are opaque structs passed
around as pointers. As large number of IREEs APIs require non-const pointers,
pass them as non-const pointers in XLA APIs.

Tip: `iree_vm_list_t* list` is ok! No need to add `const`.
