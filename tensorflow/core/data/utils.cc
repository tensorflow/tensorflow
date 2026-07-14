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
#include "tensorflow/core/data/utils.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tf_data_file_logger_options.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

void AddLatencySample(int64_t microseconds) {
  metrics::RecordTFDataGetNextDuration(microseconds);
}

void IncrementThroughput(int64_t bytes) {
  metrics::RecordTFDataBytesFetched(bytes);
}

std::string TranslateFileName(const std::string& fname) { return fname; }

std::string DefaultDataTransferProtocol() { return "grpc"; }

std::string LocalityOptimizedPath(const std::string& path) { return path; }

absl::StatusOr<bool> DisableCompressionAtRuntime(
    const std::string& data_transfer_protocol, DeploymentMode deployment_mode,
    DataServiceMetadata::Compression compression) {
  return false;
}

void LogFilenames(const LogFilenamesOptions& options) {}

}  // namespace data
}  // namespace tensorflow
