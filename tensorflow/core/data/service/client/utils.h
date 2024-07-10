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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_CLIENT_UTILS_H_
#define TENSORFLOW_CORE_DATA_SERVICE_CLIENT_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

// Gets the `DataServiceMetadata` for `dataset_id`.
absl::StatusOr<DataServiceMetadata> GetDataServiceMetadata(
    const std::string& dataset_id, const std::string& address,
    const std::string& protocol);

// Gets the `DisableCompressAtRuntimeResponse.compression_disabled_at_runtime`
// for the given dataset.
absl::StatusOr<bool> CompressionDisabledAtRuntime(
    const std::string& dataset_id, const std::string& address,
    const std::string& protocol, bool disable_compression_at_runtime);

// Gets the `DataServiceConfig` for the data service running at `address`.
absl::StatusOr<DataServiceConfig> GetDataServiceConfig(
    const std::string& address, const std::string& protocol);

// Gets the compression from `metadata`. If `metadata` specifies no valid
// compression, returns an internal error.
absl::StatusOr<DataServiceMetadata::Compression> GetValidatedCompression(
    const std::string& dataset_id, const DataServiceMetadata& metadata);

// Estimates the cardinality of a data service dataset.
int64_t EstimateCardinality(const ProcessingModeDef& processing_mode,
                            const DataServiceMetadata& metadata,
                            bool is_coordinated_read);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_CLIENT_UTILS_H_
