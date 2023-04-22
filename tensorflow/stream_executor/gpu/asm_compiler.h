/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/stream_executor/gpu/gpu_asm_opts.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#if GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#endif  // GOOGLE_CUDA

namespace stream_executor {
namespace gpu {
class GpuContext;
}

// Compiles the given PTX string using ptxas and returns the resulting machine
// code (i.e. a cubin) as a byte array. The generated cubin matches the compute
// capabilities of the device associated with 'device_ordinal'.
//
// 'options' is used to query for the CUDA location in case it is
// customized in a passed flag, and for controlling ptxas optimizations.
port::StatusOr<std::vector<uint8>> CompileGpuAsm(int device_ordinal,
                                                 const char* ptx_contents,
                                                 GpuAsmOpts options);

// Compiles the given PTX string using ptxas and returns the resulting machine
// code (i.e. a cubin) as a byte array. The generated cubin matches the compute
// capabilities provided by 'cc_major' and 'cc_minor'.
//
// 'options' is used to query for the CUDA location in case it is
// customized in a passed flag, and for controlling ptxas optimizations.
port::StatusOr<std::vector<uint8>> CompileGpuAsm(int cc_major, int cc_minor,
                                                 const char* ptx_contents,
                                                 GpuAsmOpts options);

// Same as CompileGpuAsm, but caches the result, and returns unowned view of
// the compiled binary.
//
// A copy of the string provided in ptx will be made.
port::StatusOr<absl::Span<const uint8>> CompileGpuAsmOrGetCached(
    int device_ordinal, const char* ptx, GpuAsmOpts compilation_options);

struct CubinOrPTXImage {
  std::string profile;
  std::vector<uint8> bytes;
};

// Bundles the GPU machine code (cubins) and PTX if requested and returns the
// resulting binary (i.e. a fatbin) as a byte array.
port::StatusOr<std::vector<uint8>> BundleGpuAsm(
    std::vector<CubinOrPTXImage> images, GpuAsmOpts options);

struct HsacoImage {
  std::string gfx_arch;
  std::vector<uint8> bytes;
};

// Bundles the GPU machine code (HSA Code Object) and returns the resulting
// binary (i.e. a fatbin) as a byte array.
port::StatusOr<std::vector<uint8>> BundleGpuAsm(
    std::vector<HsacoImage> images, const std::string rocm_root_dir);

// Links multiple relocatable GPU images (e.g. results of ptxas -c) into a
// single image.
port::StatusOr<std::vector<uint8>> LinkGpuAsm(
    gpu::GpuContext* context, std::vector<CubinOrPTXImage> images);

#if GOOGLE_CUDA
// Maintains a cache of pointers to loaded kernels
template <typename... Args>
port::StatusOr<std::shared_ptr<TypedKernel<Args...>>> LoadKernelOrGetPtr(
    StreamExecutor* executor, absl::string_view kernel_name,
    absl::string_view ptx, absl::Span<const uint8> cubin_data) {
  using KernelPtrCacheKey =
      std::tuple<CUcontext, absl::string_view, absl::string_view>;

  static tensorflow::mutex kernel_ptr_cache_mutex(
      tensorflow::LINKER_INITIALIZED);
  static auto& kernel_ptr_cache TF_GUARDED_BY(kernel_ptr_cache_mutex) =
      *new absl::flat_hash_map<KernelPtrCacheKey,
                               std::shared_ptr<TypedKernel<Args...>>>();
  CUcontext current_context = cuda::CurrentContextOrDie();
  KernelPtrCacheKey kernel_ptr_cache_key{current_context, kernel_name, ptx};
  tensorflow::mutex_lock lock(kernel_ptr_cache_mutex);

  auto it = kernel_ptr_cache.find(kernel_ptr_cache_key);
  if (it == kernel_ptr_cache.end()) {
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<TypedKernel<Args...>> loaded,
        executor->CreateTypedKernel<Args...>(kernel_name, ptx, cubin_data));
    it =
        kernel_ptr_cache.emplace(kernel_ptr_cache_key, std::move(loaded)).first;
  }

  CHECK(it != kernel_ptr_cache.end());
  return it->second;
}
#endif  // GOOGLE_CUDA

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_
