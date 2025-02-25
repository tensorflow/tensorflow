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
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
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
    LiteRtModelT&& model,
    const absl::flat_hash_map<std::string, OwningBufferRef<uint8_t>>&
        custom_code_to_npu_bytecode) {
  for (const auto& subgraph : model.Subgraphs()) {
    for (auto op : subgraph->Ops()) {
      if (op->OpCode() == kLiteRtOpCodeTflCustom) {
        auto custom_code = GetCustomOpCode(model, *op);
        if (!custom_code) {
          continue;
        }

        auto iter = custom_code_to_npu_bytecode.find(*custom_code);
        if (iter == custom_code_to_npu_bytecode.end()) {
          return Error(kLiteRtStatusErrorUnsupported,
                       absl::StrFormat("Unexpected custom code: %s",
                                       custom_code->c_str()));
        }

        LiteRtOpT* custom_op = op;
        OwningBufferRef<uint8_t> byte_code(iter->second);
        const auto buf_id =
            model.Buffers()->RegisterOwnedBuffer(std::move(byte_code));
        model.AttachAssetToOp(custom_op, buf_id, "");
      }
    }
  }

  return SerializeModel(std::move(model));
}

Expected<OwningBufferRef<uint8_t>> GetModelBufWithByteCode(
    absl::string_view tfl_file,
    const absl::flat_hash_map<std::string, std::string>&
        custom_code_to_npu_file) {
  auto model = LoadModelFromFile(tfl_file);
  if (!model) {
    return model.Error();
  }

  absl::flat_hash_map<std::string, OwningBufferRef<uint8_t>>
      custom_code_to_npu_bytecode;
  for (auto& iter : custom_code_to_npu_file) {
    auto npu_file_buf = LoadBinaryFile(iter.second);
    if (!npu_file_buf) {
      return npu_file_buf.Error();
    }
    custom_code_to_npu_bytecode[iter.first] = std::move(*npu_file_buf);
  }

  return GetModelBufWithByteCode(std::move(**model),
                                 custom_code_to_npu_bytecode);
}

Expected<OwningBufferRef<uint8_t>> GetModelBufWithByteCode(
    LiteRtModelT&& model, BufferRef<uint8_t> npu_byte_code) {
  absl::flat_hash_map<std::string, OwningBufferRef<uint8_t>>
      custom_code_to_npu_bytecode;
  for (const auto& subgraph : model.Subgraphs()) {
    for (auto op : subgraph->Ops()) {
      if (op->OpCode() == kLiteRtOpCodeTflCustom) {
        auto custom_code = GetCustomOpCode(model, *op);
        if (!custom_code) {
          continue;
        }
        OwningBufferRef<uint8_t> byte_code(npu_byte_code.Data(),
                                           npu_byte_code.Size());
        custom_code_to_npu_bytecode[*custom_code] = std::move(byte_code);
      }
    }
  }

  return GetModelBufWithByteCode(std::move(model), custom_code_to_npu_bytecode);
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
