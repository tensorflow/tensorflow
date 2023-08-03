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
#include "tensorflow/core/data/service/client/utils.h"

#include <cstdint>
#include <string>

#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {
// Same timeout used by the RegisterDatasetOp.
constexpr absl::Duration kGetMetadataRetryTimeout = absl::Hours(1);
}  // namespace

StatusOr<DataServiceMetadata> GetDataServiceMetadata(
    const std::string& dataset_id, const std::string& address,
    const std::string& protocol) {
  DataServiceDispatcherClient client(address, protocol);
  DataServiceMetadata metadata;
  absl::Time deadline =
      absl::FromUnixMicros(EnvTime::NowMicros()) + kGetMetadataRetryTimeout;

  Status status = grpc_util::Retry(
      [&]() { return client.GetDataServiceMetadata(dataset_id, metadata); },
      absl::Substitute("Get data service metadata for dataset $0, "
                       "with dispatcher at $1.",
                       dataset_id, address),
      absl::ToUnixMicros(deadline));
  if (errors::IsNotFound(status)) {
    return errors::NotFound(
        "Dataset id ", dataset_id,
        " not found. It must be registered with `register_dataset` before "
        "calling `from_dataset_id`.");
  }
  TF_RETURN_IF_ERROR(status);
  return metadata;
}

StatusOr<DataServiceConfig> GetDataServiceConfig(const std::string& address,
                                                 const std::string& protocol) {
  DataServiceDispatcherClient client(address, protocol);
  DataServiceConfig config;
  absl::Time deadline =
      absl::FromUnixMicros(EnvTime::NowMicros()) + kGetMetadataRetryTimeout;

  TF_RETURN_IF_ERROR(grpc_util::Retry(
      [&]() { return client.GetDataServiceConfig(config); },
      absl::Substitute("Get data service config with dispatcher at $0.",
                       address),
      absl::ToUnixMicros(deadline)));
  return config;
}

StatusOr<DataServiceMetadata::Compression> GetValidatedCompression(
    const std::string& dataset_id, const DataServiceMetadata& metadata) {
  if (metadata.compression() == DataServiceMetadata::COMPRESSION_UNSPECIFIED) {
    return errors::Internal(absl::Substitute(
        "Got invalid compression $0 for dataset $1. A proper compression "
        "should be registered in `register_dataset`.",
        DataServiceMetadata::Compression_Name(metadata.compression()),
        dataset_id));
  }
  return metadata.compression();
}

int64_t EstimateCardinality(const ProcessingModeDef& processing_mode,
                            const DataServiceMetadata& metadata,
                            bool is_coordinated_read) {
  if (is_coordinated_read) {
    // Coordinated reads require the dataset to be infinite.
    return kInfiniteCardinality;
  }

  if (metadata.cardinality() == 0) {
    return 0;
  }

  if (metadata.cardinality() == kInfiniteCardinality) {
    // Sharding may cause an infinite dataset to be empty. For example, in
    // `range(10).batch(10, drop_remainder=True).repeat()`, inserting `shard`
    // before `batch` will cause the dataset to be empty.
    // This case is rare, and there is significant performance hit for dynamic
    // sharding if it reports unknown cardinality, so it is reasonable to
    // report infinite cardinality. For DATA sharding, it is ok to report
    // infinite cardinality since it inserts `shard` after `repeat`.
    if (processing_mode.sharding_policy() == ProcessingModeDef::OFF ||
        processing_mode.sharding_policy() == ProcessingModeDef::DYNAMIC ||
        processing_mode.sharding_policy() == ProcessingModeDef::DATA) {
      return kInfiniteCardinality;
    }
  }
  return kUnknownCardinality;
}
}  // namespace data
}  // namespace tensorflow
