/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_GEMM_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_GEMM_H_

#include "llvm/ADT/DenseMap.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/runtime/logical_result.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"

namespace xla {
namespace gpu {

class JitRtGemmConfigCache;

class JitRtGemmConfigCache {
 public:
  const GemmConfig* Get(int64_t uid);
  const GemmConfig* Set(int64_t uid, GemmConfig config);

 private:
  mutable absl::Mutex mutex_;

  llvm::SmallDenseMap<int64_t, GemmConfig> configs_ ABSL_GUARDED_BY(mutex_);
};

// Registers XLA Gpu runtime Gemm# custom calls.
void RegisterGemmCustomCalls(runtime::DirectCustomCallRegistry& registry);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_GEMM_H_
