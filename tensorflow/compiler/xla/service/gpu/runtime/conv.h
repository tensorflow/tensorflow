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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_CONV_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_CONV_H_

#include <utility>

#include "absl/container/node_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/custom_call_registry.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {
namespace gpu {

using llvm::ArrayRef;

struct ConvDimensionNumbers {
  int64_t input_batch_dim;
  int64_t input_feature_dim;
  ArrayRef<int64_t> input_spatial_dims;

  int64_t kernel_in_feature_dim;
  int64_t kernel_out_feature_dim;
  ArrayRef<int64_t> kernel_spatial_dims;

  int64_t output_batch_dim;
  int64_t output_feature_dim;
  ArrayRef<int64_t> output_spatial_dims;
};

struct ConvBackendConfig {
  int64_t algorithm;
  bool tensor_ops_enabled;
  bool is_cudnn_frontend;
  ArrayRef<int64_t> knob_ids;
  ArrayRef<int64_t> knob_values;
  ArrayRef<int64_t> operand_0_layout;
  ArrayRef<int64_t> operand_1_layout;
  ArrayRef<int64_t> result_layout;
  int64_t workspace_size;
};

// Registers XLA Gpu runtime Conv custom calls.
void RegisterConvCustomCalls(runtime::DirectCustomCallRegistry& registry);

// Add conv arguments and attributes encoding for custom HLO enums and
// structs, so that we can pass them to custom calls.
void PopulateConvAttrEncoding(runtime::CustomCallAttrEncodingSet& encoding);

// Cache conv runners between invocations of convolution custom calls.
class ConvRunnerCache {
 public:
  using Key = std::pair<::stream_executor::Stream*, int64_t>;

  struct Entry {
    MaybeFusedConvRunner* runner;
    GpuConvConfig* config;
  };

  // Returns cached conv runner and the gpu config it was constructed from for
  // the given id, or creates a new one using user-provided config construction
  // function.
  absl::StatusOr<Entry> GetOrCreate(
      Key key, absl::FunctionRef<absl::StatusOr<GpuConvConfig>()> config);

 private:
  mutable absl::Mutex mutex_;

  absl::node_hash_map<Key, std::pair<MaybeFusedConvRunner, GpuConvConfig>>
      runners_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

namespace xla {
namespace runtime {

using llvm::ArrayRef;

XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(stream_executor::dnn::ActivationMode);

XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(
    xla::gpu::ConvDimensionNumbers,
    // --- input dimensions
    AggregateMember<int64_t>("input_batch_dim"),
    AggregateMember<int64_t>("input_feature_dim"),
    AggregateMember<ArrayRef<int64_t>>("input_spatial_dims"),
    // --- kernel dimensions
    AggregateMember<int64_t>("kernel_in_feature_dim"),
    AggregateMember<int64_t>("kernel_out_feature_dim"),
    AggregateMember<ArrayRef<int64_t>>("kernel_spatial_dims"),
    // --- output dimensions
    AggregateMember<int64_t>("output_batch_dim"),
    AggregateMember<int64_t>("output_feature_dim"),
    AggregateMember<ArrayRef<int64_t>>("output_spatial_dims"));

XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(
    xla::gpu::ConvBackendConfig,  //
    AggregateMember<int64_t>("algorithm"),
    AggregateMember<bool>("tensor_ops_enabled"),
    AggregateMember<bool>("is_cudnn_frontend"),
    AggregateMember<ArrayRef<int64_t>>("knob_ids"),
    AggregateMember<ArrayRef<int64_t>>("knob_values"),
    AggregateMember<ArrayRef<int64_t>>("operand_0_layout"),
    AggregateMember<ArrayRef<int64_t>>("operand_1_layout"),
    AggregateMember<ArrayRef<int64_t>>("result_layout"),
    AggregateMember<int64_t>("workspace_size"));

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME_CONV_H_
