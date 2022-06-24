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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_CLIENT_DATA_SERVICE_CLIENT_H_
#define TENSORFLOW_CORE_DATA_SERVICE_CLIENT_DATA_SERVICE_CLIENT_H_

#include "tensorflow/core/data/service/client/common.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace data {

// Interface for reading data from tf.data service. This class is thread-safe.
// It is intended to be used by `data_service_dataset_op.cc` to read data from
// tf.data service.
//
// TODO(b/230344331): Support the first-come-first-served client.
// TODO(b/230344331): Support the coordinated read client.
class DataServiceClient {
 public:
  explicit DataServiceClient(const DataServiceParams& params);
  virtual ~DataServiceClient() = default;

  // Reads the next element from tf.data servers. Blocks if the next element is
  // not ready.
  virtual StatusOr<GetNextResult> GetNext();

 private:
  const DataServiceParams params_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_CLIENT_DATA_SERVICE_CLIENT_H_
