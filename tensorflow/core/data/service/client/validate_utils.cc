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
#include "tensorflow/core/data/service/client/validate_utils.h"

#include "absl/status/status.h"
#include "tensorflow/core/data/service/client/common.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/worker_impl.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {
namespace {

// Validates local worker related parameters.
absl::Status ValidateLocalWorkers(
    const DataServiceParams& data_service_params) {
  if (data_service_params.target_workers != TARGET_WORKERS_LOCAL) {
    return absl::OkStatus();
  }
  if (LocalWorkers::Empty()) {
    if (IsStaticShard(data_service_params.processing_mode)) {
      return errors::InvalidArgument(
          "Static sharding policy <",
          ProcessingModeDef::ShardingPolicy_Name(
              data_service_params.processing_mode.sharding_policy()),
          "> requires local tf.data workers, but no local worker is found. "
          "You need to run local tf.data service workers in your training "
          "workers. Static sharding also requires a fixed worker pool and "
          "a list of worker addresses in the DispatcherConfig. See the "
          "\"Processing Modes\" section in the module doc for details.");
    }
    return errors::InvalidArgument(
        "Local reads require local tf.data workers, but no local worker "
        "is found. You need to run local tf.data service workers in your "
        "training workers.");
  }
  if (data_service_params.num_consumers.has_value()) {
    return errors::InvalidArgument(
        "Coordinated reads require non-local workers, but `target_workers` "
        "is \"LOCAL\".");
  }
  return absl::OkStatus();
}

// Validates cross-trainer cache related parameters.
absl::Status ValidateCrossTrainerCache(
    const DataServiceParams& data_service_params) {
  if (!data_service_params.cross_trainer_cache_options.has_value()) {
    return absl::OkStatus();
  }
  if (data_service_params.job_name.empty()) {
    return errors::InvalidArgument(
        "Cross-trainer caching requires named jobs. Got empty `job_name`.");
  }
  if (data_service_params.metadata.cardinality() >= 0) {
    return errors::InvalidArgument(
        "Cross-trainer caching requires the input dataset to be infinite. "
        "Got input with cardinality ",
        data_service_params.metadata.cardinality());
  }
  if (data_service_params.repetition > 1) {
    return errors::InvalidArgument(
        "Cross-trainer caching requires infinite datasets and disallows "
        "multiple repetitions of the same dataset. Got repetition ",
        data_service_params.repetition);
  }
  if (data_service_params.num_consumers.has_value()) {
    return errors::InvalidArgument(
        "Cross-trainer caching does not support coordinated reads. "
        "Got number of coordinated consumers: ",
        data_service_params.num_consumers.value());
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status ValidateDataServiceParams(
    const DataServiceParams& data_service_params) {
  TF_RETURN_IF_ERROR(ValidateLocalWorkers(data_service_params));
  TF_RETURN_IF_ERROR(ValidateCrossTrainerCache(data_service_params));
  return absl::OkStatus();
}

}  // namespace data
}  // namespace tensorflow
