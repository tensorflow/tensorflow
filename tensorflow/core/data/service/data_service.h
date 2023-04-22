/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_DATA_SERVICE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_DATA_SERVICE_H_

#include <string>

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {

// Increment this when making backwards-incompatible changes to communication
// between tf.data servers.
constexpr int kDataServiceVersion = 3;

// Modes for how a tf.data service job should process a dataset.
enum class ProcessingMode : int64 {
  UNSET = 0,
  // Each tf.data worker processes an entire epoch. If a dataset contains 2
  // elements and there are 3 workers, the job will produce 6 elements.
  PARALLEL_EPOCHS = 1,
  // Processing of a single epoch is distributed across all tf.data workers.
  DISTRIBUTED_EPOCH = 2,
};

// Parses a string representing a processing mode and stores the result in
// `mode`. Returns an InvalidArgument status if the string is not recognized.
Status ParseProcessingMode(const std::string& s, ProcessingMode& mode);

// Converts a processing mode to its corresponding string.
std::string ProcessingModeToString(ProcessingMode mode);

// Specifies which tf.data service workers to read from.
enum class TargetWorkers : int64 {
  UNSET = 0,
  // tf.data service runtime decides which workers to read from.
  AUTO = 1,
  // Reads from any available worker.
  ANY = 2,
  // Only reads from local workers. If no local worker is found, it is an error.
  LOCAL = 3,
};

// Parses a string representing a `TargetWorkers` (case-insensitive).
// Returns InvalidArgument if the string is not recognized.
StatusOr<TargetWorkers> ParseTargetWorkers(absl::string_view s);

// Converts a `TargetWorkers` enum to string.
std::string TargetWorkersToString(TargetWorkers target_workers);

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
  Status Initialize() { return EnsureInitialized(); }

 protected:
  // Initializes the client if it isn't already initialized.
  virtual Status EnsureInitialized() = 0;

  const std::string address_;
  const std::string protocol_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DATA_SERVICE_H_
