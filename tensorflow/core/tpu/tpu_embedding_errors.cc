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

#include "tensorflow/core/tpu/tpu_embedding_errors.h"

#include <string>

#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"

namespace tensorflow::tpu {

absl::Status AppendTpuEmbeddingErrorPayload(absl::Status obj) {
  if (obj.ok()) {
    return absl::OkStatus();
  } else {
    const std::string error_message =
        absl::StrCat(kTpuEmbeddingErrorMessage, ". ", obj.message());
    absl::Status status(obj.code(), error_message);
    TPUEmbeddingError error_payload;
    status.SetPayload(kTpuEmbeddingErrorUrl,
                      absl::Cord(error_payload.SerializeAsString()));
    return status;
  }
}

bool HasTpuEmbeddingErrorPayload(const absl::Status& status) {
  return status.GetPayload(kTpuEmbeddingErrorUrl).has_value();
}

bool HasTpuEmbeddingErrorMessage(const absl::Status& status) {
  return absl::StrContains(status.message(), kTpuEmbeddingErrorMessage);
}

}  // namespace tensorflow::tpu
