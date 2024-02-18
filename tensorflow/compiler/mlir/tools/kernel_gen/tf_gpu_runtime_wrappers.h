/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_GPU_RUNTIME_WRAPPERS_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_GPU_RUNTIME_WRAPPERS_H_

#include "absl/container/flat_hash_map.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"  // from @llvm-project
#include "tensorflow/core/framework/resource_base.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/hash.h"
#include "tsl/platform/thread_annotations.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif
#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#endif

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

class GPURuntimeCache : public tensorflow::ResourceBase {
 public:
#if GOOGLE_CUDA
  using GPUModule = CUmodule;
  using GPUFunction = CUfunction;
#endif
#if TENSORFLOW_USE_ROCM
  using GPUModule = hipModule_t;
  using GPUFunction = hipFunction_t;
#endif

  ~GPURuntimeCache() override;
  static constexpr const char* kDefaultResourceName = "mlir-gpu-runtime-cache";
  static tensorflow::Status Create(GPURuntimeCache** dst);
  std::string DebugString() const override;

  // Assumes that no two modules are loaded from the same memory location over
  // the lifetime of this cache. This allows to use the pointer as a key. All
  // modules are unloaded on destruction of this cache.
  GPUModule LookupOrLoadModule(void* data);

  GPUFunction LookupOrGetFunction(GPUModule module, const char* kernel_name);

 private:
  struct FunctionKey {
    GPUModule module;
    const char* kernel_name;

    friend bool operator==(const FunctionKey& lhs, const FunctionKey& rhs) {
      return lhs.module == rhs.module && lhs.kernel_name == rhs.kernel_name;
    }

    struct Hash {
      size_t operator()(const FunctionKey& key) const {
        return tsl::Hash64Combine(tsl::hash<GPUModule>()(key.module),
                                  tsl::Hash64(key.kernel_name));
      }
    };
  };

  tensorflow::mutex mu_;
  absl::flat_hash_map<void*, GPUModule> gpu_module_by_data_ptr_
      TF_GUARDED_BY(mu_);
  absl::flat_hash_map<FunctionKey, GPUFunction, FunctionKey::Hash>
      gpu_function_by_module_and_name_ TF_GUARDED_BY(mu_);
};

// Implements a C wrapper around the TensorFlow runtime and CUDA (or ROCm)
// library that allows launching a kernel on the current device and stream from
// a binary blob for the module and function name.
extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_tf_launch_kernel(
    void* ctx, void* module_blob, char* kernel_name, intptr_t gridX,
    intptr_t gridY, intptr_t gridZ, intptr_t blockX, intptr_t blockY,
    intptr_t blockZ, void** params);

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_GPU_RUNTIME_WRAPPERS_H_
