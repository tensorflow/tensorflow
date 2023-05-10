/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_CONCURRENT_REGION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_CONCURRENT_REGION_H_

#include <vector>

#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"

namespace xla {
namespace gpu {

// Registers XLA Gpu runtime kernel launch custom calls.
void RegisterConcurrentRegionCustomCalls(
    runtime::DirectCustomCallRegistry& registry);

// The state to keep track of the information regarding concurrent regions
// between custom calls.
class ConcurrentRegionStatus {
 public:
  explicit ConcurrentRegionStatus(
      const ServiceExecutableRunOptions* run_options,
      int num_borrowed_streams = 10);

  absl::Status StartConcurrentRegion();
  void EndConcurrentRegion();

  // Get a stream on which the concurrent-executable kernel runs. It returns a
  // different stream each time to avoid building dependencies in the CUDA
  // graph.
  se::Stream* GetNextStream();

  bool is_in_concurrent_region();

 private:
  int num_borrowed_streams_;
  std::vector<StreamPool::Ptr> borrowed_streams_;

  bool is_in_concurrent_region_;
  int32_t stream_index_;
  const ServiceExecutableRunOptions* run_options_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_CONCURRENT_REGION_H_
