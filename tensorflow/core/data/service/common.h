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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_COMMON_H_
#define TENSORFLOW_CORE_DATA_SERVICE_COMMON_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

// Increment this when making backwards-incompatible changes to communication
// between tf.data clients and servers.
constexpr int kDataServiceVersion = 9;

// If the user starts a colocated tf.data worker on each TF host, the worker
// will be applied a "COLOCATED" tag. This is used to avoid reading from tf.data
// workers on other TF hosts when the host runs a local tf.data service worker.
constexpr absl::string_view kColocatedWorkerTag = "COLOCATED";

// Container to hold the result of a `GetNext` call.
struct GetNextResult final {
  explicit GetNextResult() = default;
  GetNextResult(const GetNextResult&) = delete;
  GetNextResult& operator=(const GetNextResult&) = delete;
  GetNextResult(GetNextResult&&) = default;
  GetNextResult& operator=(GetNextResult&&) = delete;

  static GetNextResult EndOfSequence() {
    GetNextResult result;
    result.end_of_sequence = true;
    return result;
  }

  std::vector<Tensor> tensors;
  bool end_of_sequence = false;
};

// Returns true if `processing_mode` specifies no sharding policy.
bool IsNoShard(const ProcessingModeDef& processing_mode);

// Returns true if `processing_mode` is dynamic sharding.
bool IsDynamicShard(const ProcessingModeDef& processing_mode);

// Returns true if `processing_mode` is static sharding.
bool IsStaticShard(const ProcessingModeDef& processing_mode);

// Returns an internal error if `processing_mode` is invalid.
absl::Status ValidateProcessingMode(const ProcessingModeDef& processing_mode);

// Converts tf.data service `sharding_policy` to `AutoShardPolicy`. Returns an
// internal error if `sharding_policy` is not supported.
absl::StatusOr<AutoShardPolicy> ToAutoShardPolicy(
    ProcessingModeDef::ShardingPolicy sharding_policy);

// Parses a string representing a `TargetWorkers` (case-insensitive).
// Returns InvalidArgument if the string is not recognized.
absl::StatusOr<TargetWorkers> ParseTargetWorkers(absl::string_view s);

// Converts a `TargetWorkers` enum to string.
std::string TargetWorkersToString(TargetWorkers target_workers);

// Parses a string representing a `DeploymentMode` (case-insensitive).
// Returns InvalidArgument if the string is not recognized.
absl::StatusOr<DeploymentMode> ParseDeploymentMode(absl::string_view s);

// Returns true if `status` is a retriable error that indicates preemption.
bool IsPreemptedError(const absl::Status& status);

// Base class for data service clients. Data service clients are
// threadsafe.
class DataServiceClientBase {
 public:
  DataServiceClientBase(const std::string& address, const std::string& protocol)
      : address_(address), protocol_(protocol) {}

  virtual ~DataServiceClientBase() = default;
  // Not copyable or movable.
  DataServiceClientBase(const DataServiceClientBase&) = delete;
  DataServiceClientBase& operator=(const DataServiceClientBase&) = delete;

  // Initializes the client. Calling `Initialize()` is not required since the
  // first RPC will perform any necessary initialization. However, it can be
  // useful to call `Initialize()` proactively so that any errors that happen
  // during initialization can be surfaced earlier.
  virtual absl::Status Initialize() { return EnsureInitialized(); }

 protected:
  // Initializes the client if it isn't already initialized.
  virtual absl::Status EnsureInitialized() = 0;

  const std::string address_;
  const std::string protocol_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_COMMON_H_
