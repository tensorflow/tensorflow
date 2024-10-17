// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/lrt/core/experimental/litert_model_serialize.h"

#include <cstddef>
#include <string>

#include "absl/strings/str_format.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/lite_rt_model_init.h"

//
// METADATA Strategy
//

// [EXPERIMENTAL]
LrtStatus LiteRtModelAddByteCodeMetadata(LrtModel model,
                                         const char* soc_manufacturer,
                                         const char* soc_model,
                                         const void* byte_code,
                                         size_t byte_code_size) {
  // Register custom code shared by all NPU dispatch ops.
  LRT_RETURN_STATUS_IF_NOT_OK(
      RegisterCustomOpCode(model, kLiteRtDispatchOpCustomCode));

  // Add the build tag to the model.
  const std::string m_buffer =
      absl::StrFormat(kLiteRtBuildTagTpl, soc_manufacturer, soc_model,
                      kLiteRtMetadataSerializationStrategy);
  LRT_RETURN_STATUS_IF_NOT_OK(AppendMetadata(
      model, m_buffer.data(), m_buffer.size(), kLiteRtBuildTagKey));

  // Add the raw byte code.
  LRT_RETURN_STATUS_IF_NOT_OK(AppendMetadata(model, byte_code, byte_code_size,
                                             kLiteRtMetadataByteCodeKey));

  return kLrtStatusOk;
}

//
// APPEND Strategy
//

// [EXPERIMENTAL]
LrtStatus LiteRtModelPrepareForByteCodeAppend(LrtModel model,
                                              const char* soc_manufacturer,
                                              const char* soc_model) {
  // Register custom code shared by all NPU dispatch ops.
  LRT_RETURN_STATUS_IF_NOT_OK(
      RegisterCustomOpCode(model, kLiteRtDispatchOpCustomCode));

  // Add the build tag to the model.
  const std::string m_buffer =
      absl::StrFormat(kLiteRtBuildTagTpl, soc_manufacturer, soc_model,
                      kLiteRtAppendSerializationStrategy);
  LRT_RETURN_STATUS_IF_NOT_OK(AppendMetadata(
      model, m_buffer.data(), m_buffer.size(), kLiteRtBuildTagKey));

  // Add the byte code placeholder.
  LRT_RETURN_STATUS_IF_NOT_OK(AppendMetadata(
      model, kLiteRtAppendedByteCodePlaceholder,
      sizeof(kLiteRtAppendedByteCodePlaceholder), kLiteRtMetadataByteCodeKey));

  return kLrtStatusOk;
}
