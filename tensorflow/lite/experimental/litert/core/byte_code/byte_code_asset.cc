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

#include "tensorflow/lite/experimental/litert/core/byte_code/byte_code_asset.h"

#include <cstddef>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/byte_code/schema.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"

namespace litert::internal {

namespace {

static constexpr absl::string_view kMetadataKeySuffix = "NPU_BYTE_CODE";

}  // namespace

std::string SharedByteCodeAsset::MakeMetadataKey() const {
  return absl::StrFormat("%s:%s", BackendId(), kMetadataKeySuffix);
}

SharedByteCodeAsset::SharedByteCodeAsset(std::string backend_id,
                                         std::string entry_point,
                                         SharedByteCode shared_bytecode,
                                         size_t shared_bytecode_size)
    : ByteCodeAsset(std::move(backend_id)),
      entry_point_(std::move(entry_point)),
      shared_bytecode_(std::move(shared_bytecode)),
      shared_bytecode_size_(shared_bytecode_size) {}

void SharedByteCodeAsset::AddCaller(LiteRtOp op, std::string entry_point) {
  ops_.push_back({op, std::move(entry_point)});
}

LiteRtStatus SharedByteCodeAsset::Serialize(LiteRtModelT& model) const {
  const auto metadata_key = MakeMetadataKey();

  for (const auto& [op, entry_point] : ops_) {
    auto exec_info =
        SerializeExecInfo({BackendId(), entry_point, metadata_key});
    if (!exec_info) {
      return exec_info.Error().Status();
    }
    op->SetCustomOptions(std::move(*exec_info));
  }

  model.PushMetadata(metadata_key, ByteCode().Span());

  return kLiteRtStatusOk;
}

LiteRtStatus UniqueByteCodeAsset::Serialize(LiteRtModelT& model) const {
  const auto metadata_key = MakeMetadataKey();
  auto exec_info = SerializeExecInfo({BackendId(), entry_point_, metadata_key});
  if (!exec_info) {
    return exec_info.Error().Status();
  }
  callers_->SetCustomOptions(std::move(*exec_info));

  LITERT_RETURN_STATUS_IF_NOT_OK(
      model.PushMetadata(metadata_key, ByteCode().Span()));

  return kLiteRtStatusOk;
}

std::string UniqueByteCodeAsset::MakeMetadataKey() const {
  return absl::StrFormat("%s:%s:%s", BackendId(), entry_point_,
                         kMetadataKeySuffix);
}

UniqueByteCodeAsset::UniqueByteCodeAsset(std::string backend_id, LiteRtOp op,
                                         std::string entry_point,
                                         OwnedByteCode owned_bytecode,
                                         size_t owned_bytecode_size)
    : ByteCodeAsset(std::move(backend_id)),
      callers_(op),
      entry_point_(std::move(entry_point)),
      owned_bytecode_(std::move(owned_bytecode)),
      owned_bytecode_size_(owned_bytecode_size) {}

}  // namespace litert::internal
