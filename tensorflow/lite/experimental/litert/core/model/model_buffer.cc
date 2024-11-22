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

#include "tensorflow/lite/experimental/litert/core/model/model_buffer.h"

#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/byte_code_util.h"
#include "tensorflow/lite/experimental/litert/core/filesystem.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_load.h"
#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"

namespace litert {
namespace internal {

Expected<OwningBufferRef<uint8_t>> GetModelBufWithByteCode(
    absl::string_view tfl_file, absl::string_view npu_file) {
  auto model = LoadModelFromFile(tfl_file);
  if (!model) {
    return model.Error();
  }

  auto npu_file_buf = LoadBinaryFile(npu_file);
  if (!npu_file_buf) {
    return npu_file_buf.Error();
  }

  LiteRtModelT& internal_model = *model->Get();
  LITERT_EXPECT_OK(internal_model.PushMetadata(kByteCodeMetadataKey,
                                               MakeByteCodePlaceholder()));

  for (auto& subgraph : internal_model.subgraphs) {
    for (auto& op : subgraph.ops) {
      if (op->op_code != kLiteRtOpCodeTflCustom) {
        continue;
      }
      auto exec_info =
          MakeExecInfo(op->custom_options.StrView(), kByteCodeMetadataKey);
      if (!exec_info) {
        return exec_info.Error();
      }
      op->custom_options = std::move(*exec_info);
    }
  }

  internal_model.custom_op_code = kLiteRtDispatchOpCustomCode;

  auto serialized = SerializeModel(std::move(*model));
  if (!serialized) {
    return serialized;
  }

  LITERT_EXPECT_OK(
      FinishByteCodePlaceholders(*serialized, npu_file_buf->Size()));

  OwningBufferRef<uint8_t> with_append(serialized->Size() +
                                       npu_file_buf->Size());

  uint8_t* write = with_append.Data();
  std::memcpy(write, serialized->Data(), serialized->Size());
  write += serialized->Size();
  std::memcpy(write, npu_file_buf->Data(), npu_file_buf->Size());

  return with_append;
}

}  // namespace internal
}  // namespace litert
