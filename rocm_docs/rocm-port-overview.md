# TensorFlow ROCm port high-level design document

## Introduction

This document serves as the overall document to explain what was changed to allow TensorFlow 1.14.0 running on ROCm platform.

In this port efforts were made to try ensure logic for existing CUDA / NVPTX path stay as-is. Places where platform neutrality were broken are marked as **XXX**.

---

## Make system

- **configure**:
  - set default value of `TF_ENABLE_XLA` to *1*
  - added `TF_NEED_ROCM`, default value set to *1*
  - added `ROCM_TOOLKIT_PATH`, default value set to */opt/rocm*
- **third_party/gpus**:
  - added **rocm_configure.bzl** for ROCm platform
  - added **rocm/** directory for custom bazel functions for ROCm platform
  - added **crosstool/CROSSTOOL_hipcc.tpl** to add a new crosstool toolchain be used by **rocm_configure.bzl**
  - added **crosstool/clang/bin/crosstool_wrapper_driver_rocm.tpl** as the wrapper for compiler and linker
- **tensorflow/workspace.bzl**:
  - adopted `rocm_configrue()` to be ROCm-aware
  - changed how Eigen is fetched to cope with HIP on ROCm platform
  - removed some dead links
- **tensorflow/tensorflow.bzl**:
  - renamed `tf_cuda_library()` to `tf_gpu_library`
  - renamed `cuda_py_tests()` to `gpu_py_tests()`
  - renamed `tf_cuda_test_tags()` to `tf_gpu_tests_tags()`
- **BUILD** files within TensorFlow directories
  - adopted naming changes introduced in **tensorflow/tensorflow.bzl**
  - added logic to load ROCm-specific functions such as `if_rocm()` or `if_rocm_is_configured()`

---

## StreamExecutor

An ROCm backend is added to implement StreamExecutor interface. Existing CUDA backend is completely retained.

- added **tensorflow/stream_executor/rocm** to contain ROCm implementation for StreamExecutor interface
- integrated with HIP runtime APIs for stream, memory management, and copy
- integrated with MIOpen
- integrated with rocBLAS for certain GEMM operations
- integrated with rocRAND for RNG operations. Thoughly practically speaking it doesn't seem to be used by any TensorFlow operators.
- intergated with rocFFT for FFT operations.

---

## Common Runtime

**XXX** Changes under **tensorflow/core/commmon_runtime/gpu** directory are largely ROCm-specific and drops CUDA platform due to its current design.

- **XXX** **gpu_device.cc**
  - force use ROCm platform
  - rename `EigenCudaStreamDevice` to `EigenROCmStreamDevice`
  - removed CUDA compute capabilities and use AMDGPU ISA version
- **XXX** **gpu_init.cc**, **pool_allocator.h**, **process_state.cc**, **process_state.h**
  - force use ROCM instead of CUDA
- **XXX** removed dependency to CUPTI
- **gpu_util.h**, **gpu_util.cc**
  - added `DnnScratchAllocator` as a replacement for `CudnnScratchAllocator`

---

## GPU kernel implementation

- renamed the following files under **tensorflow/core/kernels** as they can be shared between CUDA and ROCm platform
  - **cuda_device_array.h** to **gpu_device_array.h**
  - **cuda_device_array_gpu.h** to **gpu_device_array_gpu.h**
  - **cudnn_pooling_gpu.cc** to **dnn_pooling_gpu.cc**
  - **util/cuda_kernel_helper.h** to **util/gpu_kernel_helper.h**
- introduced `TENSORFLOW_USE_ROCM` macro and use it in conjunction with `GOOGLE_CUDA`
  - on CUDA platform: `GOOGLE_CUDA` would be enabled
  - on ROCm platform: `TENSORFLOW_USE_ROCM` would be enabled
  - in certain kernels these macros are used to distinguish different logic be used on CUDA or ROCm platform
- introduced `EIGEN_USE_HIP` in kernels where Eigen is used
  - guarded with `TENSORFLOW_USE_ROCM` macro so they won't be enabled on CUDA platform
- replaced CUDA `<<< >>>` kernel launch syntax with `GpuLaunchKernel` macro
- added `GpuLaunchKernel` in **util/gpu_kernel_helper.h**
- renamed `CudaLaunchConfig` to `GpuLaunchConfig`
- renamed macros with perfix CUDA/Cuda to GPU/Gpu shall they are usable on ROCm platform
- Eigen is downloaded and [patched](https://github.com/ROCmSoftwarePlatform/tensorflow/blob/rocm-v1/third_party/eigen3/eigen.patch) to support HIP and ROCm platform
- [List of supported operators](https://github.com/ROCmSoftwarePlatform/tensorflow/blob/rocm-v1/rocm_docs/core_kernels.md)


---

## XLA

XLA support for LLVM AMDGPU backend is still highly experimental at this point. MIOpen and rocBLAS kernels would be invoked as libcalls via StreamExecutor interface.

- **tensorflow/compiler/jit/xla_gpu_device.cc**
  - **XXX**: disable registering XLA devices for CUDA
- **tensorflow/compiler/xla/service/computation_placer.cc**
  - register for ROCm platform
- **tensorflow/compiler/xla/service/generic_transfer_manager.cc**
  - register for ROCm platform
- added the following files for AMDGPU backend
  - **tensorflow/compiler/xla/service/gpu/amdgpu_compiler.cc**
  - **tensorflow/compiler/xla/service/gpu/amdgpu_compiler.h**
  - **tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/amdgpu_backend_lib.cc**
  - **tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/amdgpu_backend_lib.h**
- renamed the following files for NVPTX backend
  - **tensorflow/compiler/xla/service/gpu/gpu_compiler.cc** to **tensorflow/compiler/xla/service/gpu/nvptx_compiler.cc**
  - **tensorflow/compiler/xla/service/gpu/gpu_compiler.h** to **tensorflow/compiler/xla/service/gpu/nvptx_compiler.h**
  - **tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h** to **tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/nvptx_backend_lib.h**
  - **tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc** to **tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/nvptx_backend_lib.hcc**
- **tensorflow/compiler/xla/service/gpu/BUILD**
  - modified rule `gpu_compiler()` to cope with file name changes and added depedency to *ROCm Device Libs*
- **tensorflow/compiler/xla/service/gpu/convolution_thunk.cc**
  - **XXX**: removed autotuning logic for *CuDNN*
- **tensorflow/compiler/xla/service/gpu/elemenmtal_ir_emitter.cc**
  - **XXX**: replaced logic dependent to *Libdevice* to *ROCm Device Libs*
- **tensorflow/compiler/xla/service/gpu/gpu_executable.cc**, **tensorflow/compiler/xla/service/gpu/gpu_executable.h**, **tensorflow/compiler/xla/service/gpu/kernel_thunk.cc**
  - renamed *ptx* to *text*
- **tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.cc**
  - **XXX**: changed logic to use `llvm::ConstantExpr::getAddrSpaceCast()` due to address space differences in AMDGPU
- **tensorflow/compiler/xla/service/gpu/ir_emission_utils.cc**, **tensorflow/compiler/xla/service/gpu/ir_emission_utils.h**
  - **XXX**: replaced logic dependent to *Libdevice* to *ROCm Device Libs*
- **tensorflow/compiler/xla/service/gpu/ir_emitter.cc**
  - **XXX**: modified logic to cope with AMDGPU memory address spaces
- **tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.cc**
  - **XXX**: disabled NVPTX-specific logic
  - **XXX**: replaced logic dependent to *Libdevice* to *ROCm Device Libs*
- **tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.cc**
  - **XXX**: replaced logic dependent to *Libdevice* to *ROCm Device Libs*
- **tensorflow/compiler/xla/service/platform_util.cc**
  - added logic to check ROCm platform ISA version

---

## Fusion Support

This release introduces support for automatically "fusing" certain sequences of ops/nodes in the Tensorflow graph, into a single op/node. This is only done for ops/nodes that have been placed on the GPU partition by Tensorflow. This idea here is to improve performance because the optimized GPU kernel implementation for the single fused node will perform better than the plural GPU kernel implementations (one for each node/op) of the individual node/op sequence.

The ROCm Fusion feature is disabled by default, and can be enabled by the setting the env var `TF_ROCM_FUSION_ENABLE` to `1`.  The current release supports the fusion of following op/node sequences

1. Convolution --> Bias --> Activation (forward and inference)
2. BatchNorm --> Activation (forward, backward, and inferece)
3. Add + Relu
4. AddN + ReluGrad

By default you will only see a single message durung runtime that indicates that ROCm Fusion is turned ON

```
2018-11-14 23:03:09.721057: I tensorflow/core/graph/gpu_fusion_pass.cc:434] ROCm Fusion is enabled.
```

Setting the env var `TF_CPP_MIN_VLOG_LEVEL` to `2` will enable the display of verbose ROCm fusion details.


For the Convolution+Bias+Activation and BatchNorm+Activation fusion nodes, it is possible that the runtime is unable to create a custom kernel implementation for the fused node, and errors out. Should this happen, you will need to disable the fusion

When ROCm Fusion is enabled, the following env-vars can be used to disable individual fusions

- set `TF_ROCM_FUSION_DISABLE_CBA` to `1` to disable to Convolution+Bias+Activation fusion (forward and inference)
- set `TF_ROCM_FUSION_DISABLE_BNA` to `1` to disable to BatchNorm+Activation fusions (forward, backward and inference)
- set `TF_ROCM_FUSION_DISABLE_ADDRELU` to `1` to disable to Add+Relu fusion
- set `TF_ROCM_FUSION_DISABLE_ADDNRELUGRAD` to `1` to disable to AddN+ReluGrad fusion

---

## Verbs Support

This release enables the [community-contributed Verbs module](../tensorflow/contrib/verbs/README.md) for ROCm platforms.  The Verbs module provides a new distributed TensorFlow server protocol for RDMA transfers of Tensors over high-speed infiniband interconnects.  When building TensorFlow from source, you enable the Verbs module by

- adding `--config=verbs` to your bazel build command, and
- you must have the OFED headers and libraries installed, e.g., verbs.h, libibverbs.so.

To use the Verbs module, you specify the new server protocol when constructing your `tf.train.Server` instances.  For example

```python
# In task 0:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0, protocol='grpc+verbs')
```

Additional details for running distributed TensorFlow applications can be found online:

- <https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md>
- <https://www.tensorflow.org/guide/distribute_strategy>
