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
#ifndef TENSORFLOW_CORE_DATA_UTILS_H_
#define TENSORFLOW_CORE_DATA_UTILS_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

// Records latency of fetching data from tf.data iterator.
void AddLatencySample(int64_t microseconds);

// Records bytes produced by a tf.data iterator.
void IncrementThroughput(int64_t bytes);

// Returns a modified file name that can be used to do implementation specific
// file name manipulation/optimization.
std::string TranslateFileName(const std::string& fname);

// Returns the data transfer protocol to use if one is not specified by the
// user.
std::string DefaultDataTransferProtocol();

// Returns a path pointing to the same file as `path` with a potential locality
// optimization.
std::string LocalityOptimizedPath(const std::string& path);

// Returns `true` if tf.data service compression should be disabled at runtime
// based on (1) the inputs or (2) the properties of the calling trainer.
absl::StatusOr<bool> DisableCompressionAtRuntime(
    const std::string& data_transfer_protocol, DeploymentMode deployment_mode,
    DataServiceMetadata::Compression compression);

// Log filenames into TfDataLogger. Uses the same  TfDataFileLoggerClient at
// every call. Thread safe.
// TODO (shushanik) Implement streamz error reporting in case the logging is not
// successful
void LogFilenames(const std::vector<std::string>& files);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_UTILS_H_
