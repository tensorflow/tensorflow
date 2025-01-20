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
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/byte_code_util.h"
#include "tensorflow/lite/experimental/litert/core/filesystem.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_load.h"
#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"

namespace litert {
namespace internal {

Expected<OwningBufferRef<uint8_t>> GetModelBufWithByteCode(
    LiteRtModelT&& model, BufferRef<uint8_t> npu_byte_code) {
  LITERT_EXPECT_OK(model.PushMetadata(
      kByteCodeMetadataKey, npu_byte_code.Data(), npu_byte_code.Size()));

  for (auto* subgraph : model.Subgraphs()) {
    for (auto* op : subgraph->Ops()) {
      if (op->OpCode() != kLiteRtOpCodeTflCustom) {
        continue;
      }
      auto exec_info =
          MakeExecInfo(op->CustomOptions().StrView(), kByteCodeMetadataKey);
      if (!exec_info) {
        return exec_info.Error();
      }
      op->SetCustomOptions(std::move(*exec_info));
    }
  }

  auto build_stamp = MakeBuildStamp("", "", Serialization::kAppend);
  LITERT_EXPECT_OK(model.PushMetadata(kLiteRtBuildStampKey, *build_stamp));

  return SerializeModel(std::move(model));
}

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

  return GetModelBufWithByteCode(std::move(**model), std::move(*npu_file_buf));
}

}  // namespace internal
}  // namespace litert
