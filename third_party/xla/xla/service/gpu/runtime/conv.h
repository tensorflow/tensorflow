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

#ifndef XLA_SERVICE_GPU_RUNTIME_CONV_H_
#define XLA_SERVICE_GPU_RUNTIME_CONV_H_

#include <memory>
#include <utility>

#include "absl/container/node_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/service/gpu/gpu_conv_runner.h"

namespace xla {
namespace gpu {

// Registers XLA Gpu runtime Conv custom calls.
void RegisterConvCustomCalls(runtime::DirectCustomCallRegistry& registry);

// Register type names for convoluttion attributes defined by MHLO dialect.
void RegisterConvTypeIdNames(runtime::TypeIDNameRegistry& registry);

// Add attributes encoding for convoluttion attributes defined by MHLO dialect.
void PopulateConvAttrEncoding(runtime::CustomCallAttrEncodingSet& encoding);

//===----------------------------------------------------------------------===//
// Cache conv runners between invocations of convolution custom calls.
//===----------------------------------------------------------------------===//

struct ConvRunner {
  explicit ConvRunner(GpuConvConfig config)
      : config(std::move(config)), runner(this->config) {}
  GpuConvConfig config;
  GenericConvRunner runner;
};

class StreamExecutorConvRunners : public runtime::StateVector<ConvRunner> {};

// Xla executable keeps a mapping from stream executors to convolution runners.
class ConvRunners {
 public:
  StreamExecutorConvRunners* operator()(se::StreamExecutor* executor);

 private:
  mutable absl::Mutex mutex_;
  absl::node_hash_map<se::StreamExecutor*, StreamExecutorConvRunners> runners_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_CONV_H_
