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
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/filesystem.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_load.h"
#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"

namespace litert {
namespace internal {

Expected<OwningBufferRef<uint8_t>> GetModelBufWithByteCode(
    LiteRtModelT&& model, BufferRef<uint8_t> npu_byte_code) {
  if (model.NumSubgraphs() != 1) {
    return Error(kLiteRtStatusErrorUnsupported);
  }
  LiteRtOpT* custom_op = nullptr;
  auto* subgraph = model.Subgraphs().front();
  int num_custom_ops = 0;

  for (auto op : subgraph->Ops()) {
    if (op->OpCode() == kLiteRtOpCodeTflCustom) {
      custom_op = op;
      num_custom_ops++;
    }
  }
  if (custom_op == nullptr || num_custom_ops != 1) {
    return Error(kLiteRtStatusErrorUnsupported);
  }

  OwningBufferRef<uint8_t> byte_code(npu_byte_code.Data(),
                                     npu_byte_code.Size());
  const auto buf_id =
      model.Buffers()->RegisterOwnedBuffer(std::move(byte_code));

  model.AttachAssetToOp(custom_op, buf_id, "");

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
