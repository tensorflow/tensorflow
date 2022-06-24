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
#include <string>
#include <vector>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

// tf.data service parameters.
struct DataServiceParams final {
  int64_t dataset_id = 0;
  ProcessingModeDef processing_mode;
  std::string address;
  std::string protocol;
  std::string data_transfer_protocol;
};

// Container to hold the result of a `GetNext` call.
struct GetNextResult final {
  explicit GetNextResult() = default;
  GetNextResult(const GetNextResult&) = delete;
  GetNextResult& operator=(const GetNextResult&) = delete;

  std::vector<Tensor> tensors;
  bool end_of_sequence;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_CLIENT_COMMON_H_
