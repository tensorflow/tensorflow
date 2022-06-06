// Copyright 2022 The TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_JITRT_CUSTOM_CALLS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_JITRT_CUSTOM_CALLS_H_

#include <cstdint>

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tfrt/jitrt/custom_call.h"  // from @tf_runtime
#include "tfrt/support/type_id.h"  // from @tf_runtime

namespace xla {
namespace gpu {
class JitRtKernelsCache;
class JitRtGemmConfigCache;
}  // namespace gpu
}  // namespace xla

// Declare explicit dense type ids for all types passed to the custom calls
// as a user data to generate template specializations for fast id lookup.
TFRT_DECLARE_EXPLICIT_DENSE_TYPE_ID(tfrt::jitrt::CustomCall,
                                    xla::gpu::JitRtKernelsCache);
TFRT_DECLARE_EXPLICIT_DENSE_TYPE_ID(tfrt::jitrt::CustomCall,
                                    xla::gpu::JitRtGemmConfigCache);
TFRT_DECLARE_EXPLICIT_DENSE_TYPE_ID(tfrt::jitrt::CustomCall,
                                    const xla::ServiceExecutableRunOptions);
TFRT_DECLARE_EXPLICIT_DENSE_TYPE_ID(tfrt::jitrt::CustomCall,
                                    const xla::DebugOptions);

namespace xla {
namespace gpu {

class JitRtKernelsCache {
 public:
  JitRtKernelsCache() = default;

  ::stream_executor::KernelBase* Get(
      ::stream_executor::StreamExecutor* executor, const char* data);

  ::stream_executor::KernelBase* Set(
      ::stream_executor::StreamExecutor* executor, const char* data,
      std::unique_ptr<::stream_executor::KernelBase> kernel);

 private:
  mutable absl::Mutex mutex_;

  using Key = std::pair<::stream_executor::StreamExecutor*, const char*>;
  llvm::SmallDenseMap<Key, std::unique_ptr<::stream_executor::KernelBase>>
      kernels_cache_ ABSL_GUARDED_BY(mutex_);
};

class JitRtGemmConfigCache {
 public:
  const GemmConfig* Get(int64_t uid);
  const GemmConfig* Set(int64_t uid, GemmConfig config);

 private:
  mutable absl::Mutex mutex_;

  llvm::SmallDenseMap<int64_t, GemmConfig> configs_ ABSL_GUARDED_BY(mutex_);
};

llvm::orc::SymbolMap JitRtCustomCallsSymbolMap(
    llvm::orc::MangleAndInterner mangle);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_JITRT_CUSTOM_CALLS_H_
