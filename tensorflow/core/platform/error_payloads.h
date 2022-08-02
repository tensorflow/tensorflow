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

#ifndef TENSORFLOW_CORE_PLATFORM_ERROR_PAYLOADS_H_
#define TENSORFLOW_CORE_PLATFORM_ERROR_PAYLOADS_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/core_platform_payloads.pb.h"
// This file contains macros and payload keys for the error counter in
// EagerClient.

namespace tensorflow {

// Proto: tensorflow::core::platform::ErrorSourceProto
// Location: tensorflow/core/protobuf/core_platform_payloads.proto
// Usage: Payload key for recording the error raised source. Payload value is
// retrieved to update counter in
// tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.cc.
constexpr char kErrorSource[] =
    "type.googleapis.com/tensorflow.core.platform.ErrorSourceProto";

// Set payload when status is not ok and ErrorSource payload hasn't been set.
// The code below will be used at every place where we would like to catch
// the error for the error counter in EagerClient.

void OkOrSetErrorCounterPayload(
    const tensorflow::core::platform::ErrorSourceProto::ErrorSource&
        error_source,
    tensorflow::Status& status);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_ERROR_PAYLOADS_H_
