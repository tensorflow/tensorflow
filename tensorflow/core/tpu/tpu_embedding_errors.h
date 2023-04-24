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

#ifndef TENSORFLOW_CORE_TPU_TPU_EMBEDDING_ERRORS_H_
#define TENSORFLOW_CORE_TPU_TPU_EMBEDDING_ERRORS_H_

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"

namespace tensorflow::tpu {

// The payload URL for TPU embedding initialization permanent errors.
constexpr absl::string_view kTpuEmbeddingErrorUrl =
    "type.googleapis.com/tensorflow.tpu.TPUEmbeddingError";

constexpr absl::string_view kTpuEmbeddingErrorMessage =
    "TPUEmbedding permanent error";

// Appends a payload of type tensorflow::tpu::kTpuEmbeddingErrorUrl to the
// tensorflow::Status obj if the status is NOT OK. Returns the
// tensorflow::Status obj unchanged if the status is OK.
Status AppendTpuEmbeddingErrorPayload(Status obj);

// Appends a payload of type tensorflow::tpu::kTpuEmbeddingErrorUrl to the
// tensorflow::Status obj if the status is NOT OK. Returns obj.value() if the
// status is OK.
template <typename T>
StatusOr<T> AppendTpuEmbeddingErrorPayload(StatusOr<T> obj) {
  if (obj.ok()) {
    return std::move(obj.value());
  } else {
    const std::string error_message =
        absl::StrCat(kTpuEmbeddingErrorMessage, ". ", obj.status().message());
    Status status(obj.status().code(), error_message);
    TPUEmbeddingError error_payload;
    status.SetPayload(kTpuEmbeddingErrorUrl,
                      absl::Cord(error_payload.SerializeAsString()));
    return status;
  }
}

// Returns true if the tensorflow::Status obj has a payload of type
// tensorflow::tpu::kTpuEmbeddingErrorUrl.
bool HasTpuEmbeddingErrorPayload(const Status& status);

// Returns true if the tensorflow::Status obj error message contains
// tensorflow::tpu::kTpuEmbeddingErrorMessage as a substring.
bool HasTpuEmbeddingErrorMessage(const Status& status);

}  // namespace tensorflow::tpu

#endif  // TENSORFLOW_CORE_TPU_TPU_EMBEDDING_ERRORS_H_
