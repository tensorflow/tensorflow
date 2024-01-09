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

#ifndef XLA_SERVICE_GPU_RUNTIME_NORM_H_
#define XLA_SERVICE_GPU_RUNTIME_NORM_H_

#include <memory>
#include <utility>

#include "absl/container/node_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/service/gpu/gpu_norm_runner.h"

namespace xla {
namespace gpu {

// Registers XLA GPU runtime norm custom calls.
void RegisterNormCustomCalls(runtime::DirectCustomCallRegistry& registry);

// Register type names for norm attributes defined by MHLO dialect.
void RegisterNormTypeIdNames(runtime::TypeIDNameRegistry& registry);

void PopulateNormAlgorithmConfigAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding);

// State of the norm runners between invocations.
struct NormRunnerState {
  explicit NormRunnerState(GpuNormConfig config)
      : config(std::move(config)), runner(this->config) {}
  GpuNormConfig config;
  NormRunner runner;
};

class StreamExecutorNormRunners : public runtime::StateVector<NormRunnerState> {
};

// XLA executable keeps a mapping from stream executors to norm runners.
class NormRunnerStates {
 public:
  StreamExecutorNormRunners* operator()(se::StreamExecutor* executor);

 private:
  mutable absl::Mutex mutex_;
  absl::node_hash_map<se::StreamExecutor*, StreamExecutorNormRunners> runners_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_NORM_H_
