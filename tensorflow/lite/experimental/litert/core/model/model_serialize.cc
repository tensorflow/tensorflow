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

#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_util.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::internal {

namespace {

class ModelRepacker {
 public:
  static LiteRtStatus Repack(LiteRtModel model);

 private:
  static void BuildOpCodeMap(LiteRtModel model,
                             absl::flat_hash_map<LiteRtOpCode, uint32_t>& map);

  explicit ModelRepacker(LiteRtModel model) : model_(model) {
    BuildOpCodeMap(model_, op_code_map_);
  }

  LiteRtStatus SerializeTensor(LiteRtTensor tensor, tflite::TensorT& target);

  LiteRtStatus SerializeOp(
      LiteRtOp op, tflite::OperatorT& target,
      const absl::flat_hash_map<LiteRtTensor, int32_t>& tensor_map);

  LiteRtStatus SerializeSubgraph(LiteRtSubgraph subgraph,
                                 tflite::SubGraphT& target);

  uint32_t SubmitBuffer(std::unique_ptr<tflite::BufferT> buffer) {
    OldFb().buffers.push_back(std::move(buffer));
    return OldFb().buffers.size() - 1;
  }

  tflite::ModelT& OldFb() { return *model_->flatbuffer_model; }

  LiteRtModel model_;
  absl::flat_hash_map<LiteRtOpCode, uint32_t> op_code_map_;
};

void ModelRepacker::BuildOpCodeMap(
    LiteRtModel model, absl::flat_hash_map<LiteRtOpCode, uint32_t>& map) {
  // Add the user set custom code to the flatbuffers known codes.
  auto& custom_code = model->flatbuffer_model->operator_codes.emplace_back(
      std::make_unique<tflite::OperatorCodeT>());
  custom_code->builtin_code = tflite::BuiltinOperator_CUSTOM;
  custom_code->custom_code = model->custom_op_code;
  custom_code->version = 1;

  auto& codes = model->flatbuffer_model->operator_codes;

  for (int i = 0; i < codes.size(); ++i) {
    const auto tfl_code = codes[i]->builtin_code;
    map.insert({static_cast<LiteRtOpCode>(tfl_code), i});
  }
}

LiteRtStatus ModelRepacker::SerializeTensor(LiteRtTensor tensor,
                                            tflite::TensorT& target) {
  target.has_rank = true;
  const auto& type = tensor->type_detail.ranked_tensor_type;
  // TODO: b/365299994 - Map litert element types to flatbuffer elements types.
  target.type = tflite::TensorType_FLOAT32;

  for (int i = 0; i < type.layout.rank; ++i) {
    target.shape.push_back(type.layout.dimensions[i]);
  }

  // TFL tensors don't support strides yet.
  ABSL_DCHECK(type.layout.strides == nullptr);

  ABSL_DCHECK(tensor->weights.fb_buffer != nullptr)
      << "Submitting a null buffer";
  target.buffer = SubmitBuffer(std::move(tensor->weights.fb_buffer));

  target.name = tensor->name;

  return kLiteRtStatusOk;
}

LiteRtStatus ModelRepacker::SerializeOp(
    LiteRtOp op, tflite::OperatorT& target,
    const absl::flat_hash_map<LiteRtTensor, int32_t>& tensor_map) {
  target.opcode_index = op_code_map_.at(op->op_code);

  for (auto in : op->inputs) {
    target.inputs.push_back(tensor_map.at(in));
  }

  for (auto out : op->outputs) {
    target.outputs.push_back(tensor_map.at(out));
  }

  // TODO: b/365299994 - Support options in serialize.
  LITERT_RETURN_STATUS_IF_NOT_OK(
      SetDefaultOptions(target.builtin_options, op->op_code));

  if (op->custom_options.Size() != 0) {
    target.custom_options = op->custom_options.ToVec();
    target.custom_options_format = tflite::CustomOptionsFormat_FLEXBUFFERS;
  }
  // TODO: b/365299994 - Support exotic op fields in serialize.

  return kLiteRtStatusOk;
}

LiteRtStatus ModelRepacker::SerializeSubgraph(LiteRtSubgraph subgraph,
                                              tflite::SubGraphT& target) {
  absl::flat_hash_map<LiteRtTensor, int32_t> tensor_map;

  for (auto tensor : subgraph->tensors) {
    tensor_map.insert({tensor, tensor_map.size()});
    target.tensors.push_back(std::make_unique<tflite::TensorT>());
    LITERT_RETURN_STATUS_IF_NOT_OK(
        SerializeTensor(tensor, *target.tensors.back()));
  }

  for (auto op : subgraph->ops) {
    target.operators.push_back(std::make_unique<tflite::OperatorT>());
    LITERT_RETURN_STATUS_IF_NOT_OK(
        SerializeOp(op, *target.operators.back(), tensor_map));
  }

  for (auto in : subgraph->inputs) {
    target.inputs.push_back(tensor_map.at(in));
  }
  for (auto out : subgraph->outputs) {
    target.outputs.push_back(tensor_map.at(out));
  }

  return kLiteRtStatusOk;
}

LiteRtStatus ModelRepacker::Repack(LiteRtModel model) {
  ModelRepacker repacker(model);

  auto& target = repacker.OldFb();

  std::vector<std::pair<std::string, std::unique_ptr<tflite::BufferT>>>
      metadata;
  for (auto& flatbuffer_metadata : target.metadata) {
    const auto metadata_buffer_ind = flatbuffer_metadata->buffer;
    metadata.push_back({flatbuffer_metadata->name,
                        std::move(target.buffers[metadata_buffer_ind])});
  }

  target.subgraphs.clear();
  target.buffers.clear();
  target.metadata.clear();
  target.metadata_buffer.clear();

  target.buffers.push_back(std::make_unique<tflite::BufferT>());

  for (auto& subgraph : model->subgraphs) {
    target.subgraphs.push_back(std::make_unique<tflite::SubGraphT>());
    LITERT_RETURN_STATUS_IF_NOT_OK(
        repacker.SerializeSubgraph(&subgraph, *target.subgraphs.back()));
  }

  for (auto& [name, buf] : metadata) {
    const auto new_ind = target.buffers.size();
    auto new_metadata = std::make_unique<tflite::MetadataT>();
    new_metadata->name = name;
    new_metadata->buffer = new_ind;
    target.metadata.emplace_back(std::move(new_metadata));
    target.metadata_buffer.push_back(new_ind);
    target.buffers.emplace_back(std::move(buf));
  }

  return kLiteRtStatusOk;
}

}  // namespace

Expected<OwningBufferRef<uint8_t>> SerializeModel(Model&& model) {
  LITERT_EXPECT_OK(ModelRepacker::Repack(model.Get()));

  flatbuffers::FlatBufferBuilder b;
  auto model_offset =
      tflite::Model::Pack(b, model.Get()->flatbuffer_model.get());
  tflite::FinishModelBuffer(b, model_offset);

  OwningBufferRef<uint8_t> buffer;
  auto [new_buf, new_size, new_offset] = buffer.GetWeak();
  new_buf = b.ReleaseRaw(new_size, new_offset);

  if (!VerifyFlatbuffer(buffer.Span())) {
    return Unexpected(kLiteRtStatusErrorInvalidFlatbuffer);
  }

  return std::move(buffer);
}

}  // namespace litert::internal

LiteRtStatus LiteRtSerializeModel(LiteRtModel model, uint8_t** buf,
                                  size_t* size, size_t* offset) {
  auto serialized =
      SerializeModel(::litert::Model::CreateFromOwnedHandle(model));
  if (!serialized) {
    return serialized.Error().Status();
  }
  std::tie(*buf, *size, *offset) = serialized->Release();
  return kLiteRtStatusOk;
}
