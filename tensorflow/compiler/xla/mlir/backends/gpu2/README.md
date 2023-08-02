# Experimental backend/runtime for XLA:GPU

This is an experimental backend for XLA:GPU that uses StreamExecutor, IREE VM
and (optional) IREE CUDA HAL to execute programs compiled by XLA:GPU compiler.

This is an experimental compiler backend for lowering `lmhlo` and `lmhlo_gpu`
dialects to `iree_input` and `xla_gpu` dialects implementing "XLA:GPU runtime".

We do not build it by default, and do not build at all in open source. This is
a prototype with a goal to build a runtime that will allow us to use better
integration with latest CUDA APIs, primarily relying on explicit CUDA graph
construction instead of graph captures.
