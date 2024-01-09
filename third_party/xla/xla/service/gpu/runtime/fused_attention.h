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

#ifndef XLA_SERVICE_GPU_RUNTIME_FUSED_ATTENTION_H_
#define XLA_SERVICE_GPU_RUNTIME_FUSED_ATTENTION_H_

#include <memory>
#include <utility>

#include "absl/container/node_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/service/gpu/gpu_fused_mha_runner.h"

namespace xla {
namespace gpu {

// Registers XLA Gpu runtime fused attention custom calls.
void RegisterFusedAttentionCustomCalls(
    runtime::DirectCustomCallRegistry& registry);

// Register type names for fused attention attributes defined by MHLO dialect.
void RegisterFusedAttentionTypeIdNames(runtime::TypeIDNameRegistry& registry);

// Add attributes encoding for fused attention attributes defined by LMHLO
// dialect.
void PopulateFusedAttentionForwardDAGSignatureAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding);

// Registers XLA Gpu runtime fused attention backward custom calls.
void RegisterFusedAttentionBackwardCustomCalls(
    runtime::DirectCustomCallRegistry& registry);

// Add attributes encoding for fused attention backward attributes defined by
// LMHLO dialect.
void PopulateFusedAttentionBackwardDAGSignatureAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding);

void PopulateFusedAttentionAlgorithmConfigAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding);

//===----------------------------------------------------------------------===//
// Cache fused dot attention runners between invocations of fused dot attention
// custom calls.
//===----------------------------------------------------------------------===//
struct FusedAttentionRunner {
  explicit FusedAttentionRunner(GpufMHAConfig config)
      : config(std::move(config)), runner(this->config) {}
  GpufMHAConfig config;
  FusedMultiHeadedAttentionRunner runner;
};

struct FusedAttentionBackwardRunner {
  explicit FusedAttentionBackwardRunner(GpufMHABackwardConfig config)
      : config(std::move(config)), runner(this->config) {}
  GpufMHABackwardConfig config;
  FusedMultiHeadedAttentionBackwardRunner runner;
};

class StreamExecutorFusedAttentionRunners
    : public runtime::StateVector<FusedAttentionRunner> {};

class StreamExecutorFusedAttentionBackwardRunners
    : public runtime::StateVector<FusedAttentionBackwardRunner> {};

// Xla executable keeps a mapping from stream executors to fused attention
// runners.
class FusedAttentionRunners {
 public:
  StreamExecutorFusedAttentionRunners* operator()(se::StreamExecutor* executor);

 private:
  mutable absl::Mutex mutex_;
  absl::node_hash_map<se::StreamExecutor*, StreamExecutorFusedAttentionRunners>
      runners_ ABSL_GUARDED_BY(mutex_);
};

// Xla executable keeps a mapping from stream executors to fused attention
// backward runners.
class FusedAttentionBackwardRunners {
 public:
  StreamExecutorFusedAttentionBackwardRunners* operator()(
      se::StreamExecutor* executor);

 private:
  mutable absl::Mutex mutex_;
  absl::node_hash_map<se::StreamExecutor*,
                      StreamExecutorFusedAttentionBackwardRunners>
      runners_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_FUSED_ATTENTION_H_
