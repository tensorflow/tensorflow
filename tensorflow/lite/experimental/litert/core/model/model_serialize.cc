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
#include <functional>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/litert_to_flatbuffer.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using OpCodeMap = absl::flat_hash_map<LiteRtOpCode, uint32_t>;
using TensorMap = absl::flat_hash_map<LiteRtTensor, int32_t>;

TflOpCodePtr MakeCustomOpCode(absl::string_view custom_code_name) {
  auto custom_code = std::make_unique<TflOpCode>();
  custom_code->builtin_code = ::tflite::BuiltinOperator_CUSTOM;
  custom_code->custom_code.assign(custom_code_name.begin(),
                                  custom_code_name.end());
  custom_code->version = 1;
  return custom_code;
}

OpCodeMap BuildOpCodeMap(const std::vector<TflOpCodePtr>& op_codes) {
  OpCodeMap map;
  for (auto i = 0; i < op_codes.size(); ++i) {
    const auto tfl_code = op_codes[i]->builtin_code;
    map.insert({static_cast<LiteRtOpCode>(tfl_code), i});
  }
  return map;
}

void SetOptions(const LiteRtOpT& litert_op, TflOp& tfl_op) {
  tfl_op.builtin_options = litert_op.option;

  if (litert_op.custom_options.Size() != 0) {
    tfl_op.custom_options = litert_op.custom_options.ToVec();
    tfl_op.custom_options_format = tflite::CustomOptionsFormat_FLEXBUFFERS;
  }
}

class ModelRepacker {
 public:
  static LiteRtStatus Repack(LiteRtModelT& model);

 private:
  explicit ModelRepacker(LiteRtModelT::Ref model) : model_(model) {
    model_.get().flatbuffer_model->operator_codes.emplace_back(
        MakeCustomOpCode(model_.get().custom_op_code));
    op_code_map_ =
        BuildOpCodeMap(model_.get().flatbuffer_model->operator_codes);
  }

  LiteRtStatus SerializeTensor(LiteRtTensorT& tensor, TflTensor& target);

  LiteRtStatus SerializeOp(LiteRtOpT& op, TflOp& target,
                           const TensorMap& tensor_map);

  LiteRtStatus SerializeSubgraph(LiteRtSubgraphT& subgraph,
                                 TflSubgraph& target);

  uint32_t SubmitBuffer(TflBufferPtr buffer) {
    OldFb().buffers.push_back(std::move(buffer));
    return OldFb().buffers.size() - 1;
  }

  TflModel& OldFb() { return *model_.get().flatbuffer_model; }

  LiteRtModelT::Ref model_;
  OpCodeMap op_code_map_;
};

LiteRtStatus ModelRepacker::SerializeTensor(LiteRtTensorT& tensor,
                                            TflTensor& target) {
  auto tfl_tensor_type = MapTensorType({tensor.type_id, tensor.type_detail});
  if (!tfl_tensor_type) {
    return tfl_tensor_type.Error().Status();
  }
  auto [tfl_elem_type, tfl_shape] = *tfl_tensor_type;

  target.type = tfl_elem_type;
  target.shape.assign(tfl_shape.shape.begin(), tfl_shape.shape.end());
  target.has_rank = tfl_shape.has_rank;
  target.shape_signature.assign(tfl_shape.shape_signature.begin(),
                                tfl_shape.shape_signature.end());

  auto tfl_quantization =
      MapQuantization(std::make_pair(tensor.q_type_id, tensor.q_type_detail));
  if (!tfl_quantization) {
    return tfl_quantization.Error().Status();
  }
  target.quantization = std::move(*tfl_quantization);

  ABSL_DCHECK(tensor.weights.fb_buffer != nullptr)
      << "Submitting a null buffer";
  target.buffer = SubmitBuffer(std::move(tensor.weights.fb_buffer));

  target.name = tensor.name;

  return kLiteRtStatusOk;
}

LiteRtStatus ModelRepacker::SerializeOp(LiteRtOpT& op, TflOp& target,
                                        const TensorMap& tensor_map) {
  target.opcode_index = op_code_map_.at(op.op_code);

  for (auto in : op.inputs) {
    target.inputs.push_back(tensor_map.at(in));
  }

  for (auto out : op.outputs) {
    target.outputs.push_back(tensor_map.at(out));
  }

  SetOptions(op, target);

  // TODO: b/365299994 - Support exotic op fields in serialize.

  return kLiteRtStatusOk;
}

LiteRtStatus ModelRepacker::SerializeSubgraph(LiteRtSubgraphT& subgraph,
                                              TflSubgraph& target) {
  TensorMap tensor_map;

  for (auto tensor : subgraph.tensors) {
    tensor_map.insert({tensor, tensor_map.size()});
    target.tensors.push_back(std::make_unique<TflTensor>());
    LITERT_RETURN_STATUS_IF_NOT_OK(
        SerializeTensor(*tensor, *target.tensors.back()));
  }

  for (auto op : subgraph.ops) {
    target.operators.push_back(std::make_unique<TflOp>());
    LITERT_RETURN_STATUS_IF_NOT_OK(
        SerializeOp(*op, *target.operators.back(), tensor_map));
  }

  for (auto in : subgraph.inputs) {
    target.inputs.push_back(tensor_map.at(in));
  }
  for (auto out : subgraph.outputs) {
    target.outputs.push_back(tensor_map.at(out));
  }

  return kLiteRtStatusOk;
}

LiteRtStatus ModelRepacker::Repack(LiteRtModelT& model) {
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

  target.buffers.push_back(std::make_unique<TflBuffer>());

  for (auto& subgraph : model.subgraphs) {
    target.subgraphs.push_back(std::make_unique<TflSubgraph>());
    LITERT_RETURN_STATUS_IF_NOT_OK(
        repacker.SerializeSubgraph(subgraph, *target.subgraphs.back()));
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
  LITERT_EXPECT_OK(ModelRepacker::Repack(*model.Get()));

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
