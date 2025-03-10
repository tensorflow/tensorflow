# PJRT Examples

## Example: JAX CUDA plugin

1. PJRT C API implementation through wrapper ([pjrt\_c\_api\_gpu.h](https://github.com/openxla/xla/blob/c23fbd601a017be25726fd6d624b22daa6a8a4e5/xla/pjrt/c/pjrt_c_api_gpu.h)).
1. Set up the entry point for the package ([setup.py](https://github.com/google/jax/blob/main/jax_plugins/cuda/setup.py)).
1. Implement an initialize() method ([\_\_init\_\_.py](https://github.com/google/jax/blob/a10854786b6d1bc92a65dd314916b151640789af/plugins/cuda/__init__.py#L31-L51)).
1. Can be tested with any jax tests for CUDA.


## Frameworks Implementations

Some references for using PJRT on the framework side, to interface with PJRT
devices:

- JAX
  + [jax-ml/jax](https://github.com/jax-ml/jax/blob/main/jax/_src/compiler.py#L248)
    interacts with PJRT APIs via the `xla_client` APIs
- GoMLX
  + [gomlx/gopjrt](https://github.com/gomlx/gopjrt)
  + [gomlx/gomlx > backends/xla](https://github.com/gomlx/gomlx/tree/main/backends/xla/xla.go)
- ZML
  + PJRT API wrapper [pjrt.zig](https://github.com/zml/zml/blob/master/pjrt/pjrt.zig)
  + Load PJRT Plugin [context.zig](https://github.com/zml/zml/blob/master/zml/context.zig#L30-L34)
  + Interacting with PJRT Buffers [buffer.zig](https://github.com/zml/zml/blob/master/zml/buffer.zig#L36)
  + Execute a module via PJRT [module.zig](https://github.com/zml/zml/blob/master/zml/module.zig#L863-L886)

## Hardware Implementations

- Full integration plugins (PJRT+MLIR+XLA):
  + [XLA CPU Plugin](https://github.com/openxla/xla/tree/main/xla/pjrt/cpu/cpu_client.cc)
  + [XLA GPU Plugin](https://github.com/openxla/xla/tree/main/xla/pjrt/gpu/se_gpu_pjrt_client.cc)
  + [Intel XLA Plugin](https://github.com/intel/intel-extension-for-openxla)
- Light integration plugins (PJRT+MLIR):
  + StableHLO Reference Interpreter plugin
    (MLIR-based, C++ plugin, to be linked after devlabs)
  + [Tenstorrent-XLA plugin](https://github.com/tenstorrent/tt-xla/blob/main/src/common/pjrt_implementation/api_bindings.cc)
    (MLIR-based, C plugin)
