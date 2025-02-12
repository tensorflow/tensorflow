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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_VALIDATE_UTILS_H_
#define TENSORFLOW_CORE_DATA_SERVICE_VALIDATE_UTILS_H_

#include <string>

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

// Verifies the datasets with the same ID have the same metadata. If the
// metadata differs, returns an invalid argument error.
absl::Status ValidateMatchingDataset(const std::string& dataset_id,
                                     const DataServiceMetadata& metadata1,
                                     const DataServiceMetadata& metadata2);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_VALIDATE_UTILS_H_
