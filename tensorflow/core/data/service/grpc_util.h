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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_GRPC_UTIL_H_
#define TENSORFLOW_CORE_DATA_SERVICE_GRPC_UTIL_H_

#include "grpcpp/grpcpp.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace data {
namespace grpc_util {

// Wraps a grpc::Status in a tensorflow::Status with the given message.
Status WrapError(const std::string& message, const grpc::Status& status);

}  // namespace grpc_util
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_GRPC_UTIL_H_
