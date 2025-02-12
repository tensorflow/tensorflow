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

#include "tensorflow/core/platform/error_payloads.h"

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorflow/core/protobuf/core_platform_payloads.pb.h"

namespace tsl {

using ::tensorflow::core::platform::ErrorSourceProto;

void OkOrSetErrorCounterPayload(
    const ErrorSourceProto::ErrorSource& error_source, absl::Status& status) {
  if (!status.ok() &&
      !status.GetPayload(tensorflow::kErrorSource).has_value()) {
    ErrorSourceProto error_source_proto;
    error_source_proto.set_error_source(error_source);
    status.SetPayload(tensorflow::kErrorSource,
                      absl::Cord(error_source_proto.SerializeAsString()));
  }
}

}  // namespace tsl
