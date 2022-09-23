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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_CLIENT_COMMON_H_
#define TENSORFLOW_CORE_DATA_SERVICE_CLIENT_COMMON_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

// tf.data service parameters.
struct DataServiceParams final {
  std::string dataset_id;
  ProcessingModeDef processing_mode;
  std::string address;
  std::string protocol;
  std::string data_transfer_protocol;
  std::string job_name;
  int64_t repetition = 0;
  std::optional<int64_t> num_consumers;
  std::optional<int64_t> consumer_index;
  int64_t max_outstanding_requests = 0;
  absl::Duration task_refresh_interval;
  TargetWorkers target_workers = TargetWorkers::TARGET_WORKERS_UNSPECIFIED;
  DataServiceMetadata metadata;
  std::optional<CrossTrainerCacheOptions> cross_trainer_cache_options;
};

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

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_CLIENT_COMMON_H_
