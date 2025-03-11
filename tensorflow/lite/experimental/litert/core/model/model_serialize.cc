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

#include <sys/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// schema/mutable/schema_generated.h and schema/schema_generated.h (included
// through flatbuffer_tools.h via model.h) have the same #ifdef, thus this line
// need to be put at the top to ensure we get the "mutable" version.
#if 1
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"
#endif

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/build_stamp.h"
#include "tensorflow/lite/experimental/litert/core/dispatch_op_schema.h"
#include "tensorflow/lite/experimental/litert/core/insert_order_map.h"
#include "tensorflow/lite/experimental/litert/core/model/litert_to_flatbuffer.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/schema/mutable/schema_generated.h"

namespace litert::internal {
namespace {

using TensorMap = absl::flat_hash_map<LiteRtTensor, int32_t>;

// This is expected to be used to serialize the dispatch op custom code.
TflOpCodePtr MakeCustomOpCode(std::string custom_code_name) {
  auto custom_code = std::make_unique<TflOpCode>();
  custom_code->builtin_code = ::tflite::BuiltinOperator_CUSTOM;
  custom_code->custom_code = std::move(custom_code_name);
  custom_code->version = 1;
  return custom_code;
}

// Utility for accessing flatbuffer state and other relevant state.
class SerializationContext {
 public:
  // Subgraph and op index pair.
  using TflOpInd = std::pair<size_t, size_t>;
  using TflOpAssetMap =
      absl::flat_hash_map<TflOpInd, LiteRtModelT::OpAssetReference>;
  using TflBufferInd = uint32_t;
  using TflOffsetTensorMap =
      absl::flat_hash_map<TflBufferInd, LiteRtModelT::BufferId>;
  using TflBufferIdMap =
      absl::flat_hash_map<LiteRtModelT::BufferId, TflBufferInd>;

  explicit SerializationContext(uint32_t dispatch_op_code_ind,
                                LiteRtModelT& litert_model,
                                size_t bytecode_alignment)
      : tfl_model_(std::make_unique<TflModel>()),
        dispatch_op_code_ind_(dispatch_op_code_ind),
        litert_model_(litert_model),
        bytecode_alignment_(bytecode_alignment) {
    // Tfl expects empty buffer 0.
    tfl_model_->buffers.push_back(std::make_unique<TflBuffer>());
  }

  TflModel& Model() { return *tfl_model_.get(); }

  TflModelPtr Release() && { return std::move(tfl_model_); }

  LiteRtModelT& LitertModel() { return litert_model_; }

  size_t BytecodeAlignment() const { return bytecode_alignment_; }

  LiteRtStatus HandleTensorBuffer(TflTensor& tfl_tensor,
                                  const LiteRtTensorT& litert_tensor) {
    const auto litert_buf_id = litert_tensor.Weights().GetBufferId();
    auto* buffer_manager = litert_tensor.Weights().GetBufferManager();

    auto litert_buf_ctx = buffer_manager->GetContext(litert_buf_id);
    if (!litert_buf_ctx) {
      LITERT_LOG(LITERT_ERROR, "Failed to get buffer context");
      return litert_buf_ctx.Error().Status();
    }

    auto litert_buf = buffer_manager->GetBuffer(litert_buf_id);
    if (!litert_buf) {
      LITERT_LOG(LITERT_ERROR, "Failed to get buffer");
      return litert_buf.Error().Status();
    }

    TflBufferInd tfl_buffer_ind;
    if (buffer_id_map_.contains(litert_buf_id)) {
      tfl_buffer_ind = buffer_id_map_.at(litert_buf_id);
    } else {
      auto& tfl_buffer =
          tfl_model_->buffers.emplace_back(std::make_unique<TflBuffer>());
      tfl_buffer_ind = tfl_model_->buffers.size() - 1;

      if (litert_buf_ctx->get().should_append) {
        tfl_buffer->offset = 1;
        tfl_buffer->size = 1;
        offset_tensor_map_.emplace(tfl_buffer_ind, litert_buf_id);
      } else {
        tfl_buffer->data.assign(litert_buf->Data(),
                                litert_buf->Data() + litert_buf->Size());
      }
      buffer_id_map_[litert_buf_id] = tfl_buffer_ind;
    }

    tfl_tensor.buffer = tfl_buffer_ind;

    return kLiteRtStatusOk;
  }

  // Add to tfl model metadata.
  void PushMetadata(std::string key, BufferRef<uint8_t> data) {
    auto& tfl_buffer =
        tfl_model_->buffers.emplace_back(std::make_unique<TflBuffer>());
    const auto tfl_buffer_ind = tfl_model_->buffers.size() - 1;
    tfl_buffer->data.assign(data.Data(), data.Data() + data.Size());
    tfl_model_->metadata_buffer.push_back(tfl_buffer_ind);
    auto tfl_metadata = std::make_unique<TflMetadata>();
    tfl_metadata->name = key;
    tfl_metadata->buffer = tfl_buffer_ind;
    tfl_model_->metadata.push_back(std::move(tfl_metadata));
  }

  // Keep track of the given ops index as having a particular asset.
  // These will be used to update the ops with the correct offset and size
  // after the model is fully packed.
  void AttachAssetToOp(size_t subgraph_ind, size_t op_ind,
                       LiteRtModelT::OpAssetReference asset) {
    TflOpInd tfl_op_ind = {subgraph_ind, op_ind};
    op_asset_map_.emplace(tfl_op_ind, asset);
  }

  const TflOpAssetMap& OpAssetMap() const { return op_asset_map_; }

  const TflOffsetTensorMap& OffsetTensorMap() const {
    return offset_tensor_map_;
  }

  // Get the index in the tfl op codes for the dispatch custom code.
  // This should be the only new custom code added after loading the initial
  // tfl.
  uint32_t DispatchOpCodeInd() const { return dispatch_op_code_ind_; }

 private:
  TflModelPtr tfl_model_;
  uint32_t dispatch_op_code_ind_;
  LiteRtModelT& litert_model_;

  TflOpAssetMap op_asset_map_;
  TflOffsetTensorMap offset_tensor_map_;
  TflBufferIdMap buffer_id_map_;
  size_t bytecode_alignment_ = 0;
};

void SetOptions(const LiteRtOpT& litert_op, TflOp& tfl_op) {
  tfl_op.builtin_options = detail::GetTflOptions(litert_op);
  if (litert_op.CustomOptions().Size() != 0) {
    tfl_op.custom_options = litert_op.CustomOptions().ToVec();
    tfl_op.custom_options_format = tflite::CustomOptionsFormat_FLEXBUFFERS;
  }
}

LiteRtStatus PackOp(SerializationContext& builder, LiteRtOpT& litert_op,
                    TflOp& tfl_op, const TensorMap& tensor_map) {
  // Get index of the op code in the tfl model.
  auto tfl_op_code_ind = detail::GetTflOpCodeInd(litert_op);
  const bool is_dispatch_op = tfl_op_code_ind == detail::kDispatchOpCodeTflInd;

  if (is_dispatch_op) {
    tfl_op_code_ind = builder.DispatchOpCodeInd();
  }

  tfl_op.opcode_index = tfl_op_code_ind;

  // Look up the tensor indices in the tfl model.
  for (auto* in : litert_op.Inputs()) {
    tfl_op.inputs.push_back(tensor_map.at(in));
  }
  for (auto* out : litert_op.Outputs()) {
    tfl_op.outputs.push_back(tensor_map.at(out));
  }

  // Set generic options.
  tfl_op.builtin_options = detail::GetTflOptions(litert_op);

  return kLiteRtStatusOk;
}

LiteRtStatus PackTensor(SerializationContext& builder,
                        LiteRtTensorT& litert_tensor, TflTensor& tfl_tensor) {
  auto tfl_tensor_type = MapTensorType(litert_tensor.Type());
  if (!tfl_tensor_type) {
    return tfl_tensor_type.Error().Status();
  }
  auto [tfl_elem_type, tfl_shape] = *tfl_tensor_type;

  tfl_tensor.type = tfl_elem_type;
  tfl_tensor.shape.assign(tfl_shape.shape.begin(), tfl_shape.shape.end());
  tfl_tensor.has_rank = tfl_shape.has_rank;
  tfl_tensor.shape_signature.assign(tfl_shape.shape_signature.begin(),
                                    tfl_shape.shape_signature.end());

  auto tfl_quantization = MapQuantization(litert_tensor.Qparams());
  if (!tfl_quantization) {
    return tfl_quantization.Error().Status();
  }
  tfl_tensor.quantization = std::move(*tfl_quantization);

  LITERT_RETURN_IF_ERROR(builder.HandleTensorBuffer(tfl_tensor, litert_tensor));

  tfl_tensor.name = std::string(litert_tensor.Name());

  return kLiteRtStatusOk;
}

LiteRtStatus PackSubgraph(SerializationContext& builder,
                          LiteRtSubgraphT& litert_subgraph,
                          TflSubgraph& tfl_subgraph, TensorMap& tensor_map,
                          size_t subgraph_ind) {
  for (auto* tensor : litert_subgraph.Tensors()) {
    tfl_subgraph.tensors.push_back(std::make_unique<TflTensor>());
    tensor_map.insert({tensor, tfl_subgraph.tensors.size() - 1});
    LITERT_RETURN_IF_ERROR(
        PackTensor(builder, *tensor, *tfl_subgraph.tensors.back()));
  }

  for (auto i = 0; i < litert_subgraph.Ops().size(); ++i) {
    auto* op = litert_subgraph.Ops().at(i);

    tfl_subgraph.operators.push_back(std::make_unique<TflOp>());
    auto& tfl_op = *tfl_subgraph.operators.back();
    LITERT_RETURN_IF_ERROR(PackOp(builder, *op, tfl_op, tensor_map));

    // Set custom options.
    if (auto op_asset = builder.LitertModel().FindOpAsset(op)) {
      // This mechanism is currently only used for dispatch ops to store
      // location of bytecode. Here we update the name and placeholder values
      // for offset and size. These will be updated when the model is fully
      // packed.
      auto dispatch_opts = MakeDispatchOpOptions({
          1,
          1,
          std::string(op_asset->second),
      });
      tfl_op.custom_options = dispatch_opts.ToVec();

      // Save the "location" of the op and its asset.
      builder.AttachAssetToOp(subgraph_ind, i, *op_asset);

    } else if (op->CustomOptions().Size() != 0) {
      tfl_op.custom_options = op->CustomOptions().ToVec();
    }

    tfl_op.custom_options_format = tflite::CustomOptionsFormat_FLEXBUFFERS;
  }

  for (auto* in : litert_subgraph.Inputs()) {
    tfl_subgraph.inputs.push_back(tensor_map.at(in));
  }

  for (auto* out : litert_subgraph.Outputs()) {
    tfl_subgraph.outputs.push_back(tensor_map.at(out));
  }

  return kLiteRtStatusOk;
}

Expected<TflModelPtr> PackAsTflite(SerializationContext& builder) {
  auto& litert_model = builder.LitertModel();

  // Pack litert subgraphs into tfl subgraphs and save the mapping of
  // tensors.
  TensorMap tensor_map;
  for (auto i = 0; i < litert_model.Subgraphs().size(); ++i) {
    auto& litert_subgraph = litert_model.Subgraph(i);
    auto& tfl_subgraph = *builder.Model().subgraphs.emplace_back(
        std::make_unique<TflSubgraph>());
    LITERT_RETURN_IF_ERROR(
        PackSubgraph(builder, litert_subgraph, tfl_subgraph, tensor_map, i));
  }

  // Serialize the signatures using saved tensor mapping.
  for (auto* litert_signature : litert_model.Signatures()) {
    auto* litert_subgraph = &litert_signature->GetSubgraph();

    auto& tfl_signature = *builder.Model().signature_defs.emplace_back(
        std::make_unique<TflSignature>());
    tfl_signature.signature_key = std::string(litert_signature->Key());

    auto begin = litert_model.Subgraphs().cbegin();
    auto end = litert_model.Subgraphs().cend();
    const auto litert_subgraph_ind =
        std::find(begin, end, litert_subgraph) - begin;
    tfl_signature.subgraph_index = litert_subgraph_ind;

    auto input_ind = 0;
    for (const auto& litert_name : litert_signature->InputNames()) {
      auto& tfl_input = *tfl_signature.inputs.emplace_back(
          std::make_unique<::tflite::TensorMapT>());
      tfl_input.name = litert_name;
      tfl_input.tensor_index =
          tensor_map.find(litert_subgraph->Inputs().at(input_ind))->second;
      ++input_ind;
    }

    auto output_ind = 0;
    for (const auto& litert_name : litert_signature->OutputNames()) {
      auto& tfl_output = *tfl_signature.outputs.emplace_back(
          std::make_unique<::tflite::TensorMapT>());
      tfl_output.name = litert_name;
      tfl_output.tensor_index =
          tensor_map.find(litert_subgraph->Outputs().at(output_ind))->second;
      ++output_ind;
    }
  }

  // Serialize metadata.
  for (auto it = litert_model.MetadataBegin(); it != litert_model.MetadataEnd();
       ++it) {
    const auto& [key, buf_id] = *it;
    auto buf = litert_model.Buffers()->GetBuffer(buf_id);
    if (!buf) {
      LITERT_LOG(LITERT_ERROR, "Failed to find metadata buffer");
      return buf.Error();
    }
    builder.PushMetadata(key, *buf);
  }

  builder.Model().version = 3;

  return std::move(builder).Release();
}

// Appends external buffers to the back of the serialized tflite model. Updates
// the ops that references them with the correct offset and size in-place.
Expected<OwningBufferRef<uint8_t>> SerializeWithAppendedBuffers(
    SerializationContext& builder, OwningBufferRef<uint8_t> serialized_tfl,
    LiteRtModelT& litert_model) {
  if (builder.OpAssetMap().empty() && builder.OffsetTensorMap().empty()) {
    return serialized_tfl;
  }

  const auto align = builder.BytecodeAlignment();
  // Pad the original model to the next multiple of the alignment.
  auto align_offset = [align](size_t& cur_offset) {
    cur_offset = (cur_offset + align - 1) & ~(align - 1);
  };

  size_t cur_offset = serialized_tfl.Size();
  align_offset(cur_offset);

  // Calculate the offset and size of each op asset.
  InsertOrderMap<LiteRtModelT::BufferId, std::pair<size_t, size_t>>
      asset_buffer_offsets;
  for (auto it = builder.OpAssetMap().cbegin();
       it != builder.OpAssetMap().cend(); ++it) {
    const auto& [buf_id, name] = it->second;
    auto asset_buf = litert_model.Buffers()->GetBuffer(buf_id);
    if (!asset_buf) {
      return asset_buf.Error();
    }
    if (asset_buffer_offsets.Contains(buf_id)) {
      continue;
    }
    asset_buffer_offsets.InsertOrAssign(buf_id,
                                        {cur_offset, asset_buf->Size()});
    cur_offset += asset_buf->Size();
    align_offset(cur_offset);
  }

  // Calculate the offset and size of each offset tensor.
  InsertOrderMap<SerializationContext::TflBufferInd, std::pair<size_t, size_t>>
      offset_tensor_offsets;
  for (auto it = builder.OffsetTensorMap().cbegin();
       it != builder.OffsetTensorMap().cend(); ++it) {
    const auto& [tfl_buffer_ind, litert_buf_id] = *it;
    auto litert_buf = litert_model.Buffers()->GetBuffer(litert_buf_id);
    if (!litert_buf) {
      LITERT_LOG(LITERT_ERROR, "Failed to find offset tensor buffer");
      return litert_buf.Error();
    }
    if (offset_tensor_offsets.Contains(tfl_buffer_ind)) {
      continue;
    }
    offset_tensor_offsets.InsertOrAssign(tfl_buffer_ind,
                                         {cur_offset, litert_buf->Size()});
    cur_offset += litert_buf->Size();
  }

  // Read serialized tflite in packed form.
  auto* tfl_model = tflite::GetMutableModel(serialized_tfl.Data());

  // Find the ops that have external buffers and mark them with the future size
  // and offset.
  for (auto sg_ind = 0; sg_ind < tfl_model->mutable_subgraphs()->size();
       ++sg_ind) {
    auto* sg = tfl_model->mutable_subgraphs()->GetMutableObject(sg_ind);

    for (auto op_ind = 0; op_ind < sg->mutable_operators()->size(); ++op_ind) {
      SerializationContext::TflOpInd ind = {sg_ind, op_ind};

      auto asset_buffer = builder.OpAssetMap().find(ind);
      if (asset_buffer == builder.OpAssetMap().end()) {
        // No external buffer for this op.
        continue;
      }

      auto* op = sg->mutable_operators()->GetMutableObject(op_ind);

      // The id of the buffer in the litert model.
      const auto buf_id = asset_buffer->second.first;

      // The real offset and size of the buffer in the serialized tflite model.
      const auto offset_and_size = asset_buffer_offsets.Find(buf_id);
      if (!offset_and_size) {
        LITERT_LOG(LITERT_ERROR, "Failed to find offset and size for buffer");
        return Error(kLiteRtStatusErrorInvalidFlatbuffer);
      }
      const auto [offset, size] = offset_and_size->get().second;

      // The custom options should have already been set with the name and
      // placeholder values for size and offset.
      MutableBufferRef<uint8_t> old_raw_opts(
          op->mutable_custom_options()->data(),
          op->mutable_custom_options()->size());

      // Update with real size and offset.
      DispatchOpOptions dispach_opts(GetDispatchOpOptions(old_raw_opts));
      dispach_opts.bytecode_offset = offset;
      dispach_opts.bytecode_size = size;

      if (!UpdateDispatchOpOptionsInPlace(dispach_opts, old_raw_opts)) {
        LITERT_LOG(LITERT_ERROR, "Failed to update dispatch op options");
        return Error(kLiteRtStatusErrorInvalidFlatbuffer);
      }
    }
  }

  // Find the buffers that are offset buffers and mark them with the future
  // size and offset.
  for (auto i = 0; i < tfl_model->mutable_buffers()->size(); ++i) {
    auto* tfl_buffer = tfl_model->mutable_buffers()->GetMutableObject(i);
    auto offset_size = offset_tensor_offsets.Find(i);
    if (!offset_size) {
      // Not offset buffer.
      continue;
    }
    const auto [offset, size] = offset_size->get().second;
    const auto offset_ok = tfl_buffer->mutate_offset(offset);
    const auto size_ok = tfl_buffer->mutate_size(size);
    if (!offset_ok || !size_ok) {
      LITERT_LOG(LITERT_ERROR, "Failed to update offset and size for buffer");
      return Error(kLiteRtStatusErrorInvalidFlatbuffer);
    }
  }

  // Allocate buffer enough for original model and appendd buffers and copy.
  OwningBufferRef<uint8_t> final_model(cur_offset);

  // Copy serialized tflite model.
  uint8_t* const start = final_model.Data();
  std::memcpy(start, serialized_tfl.Data(), serialized_tfl.Size());

  // Copy asset buffers (aligned).
  for (auto it = asset_buffer_offsets.Begin(); it != asset_buffer_offsets.End();
       ++it) {
    const auto buf_id = it->first;

    auto asset_buf = litert_model.Buffers()->GetBuffer(buf_id);
    if (!asset_buf) {
      LITERT_LOG(LITERT_ERROR, "Failed to find asset buffer");
      return asset_buf.Error();
    }
    uint8_t* const offset = start + it->second.first;
    std::memcpy(offset, asset_buf->Data(), asset_buf->Size());
  }

  // Copy offset tensor buffers.
  for (auto it = offset_tensor_offsets.Begin();
       it != offset_tensor_offsets.End(); ++it) {
    const auto buf_id = it->first;

    auto offset_buf = litert_model.Buffers()->GetBuffer(buf_id);
    if (!offset_buf) {
      LITERT_LOG(LITERT_ERROR, "Failed to find offset tensor buffer");
      return offset_buf.Error();
    }

    uint8_t* const offset = start + it->second.first;
    std::memcpy(offset, offset_buf->Data(), offset_buf->Size());
  }

  return final_model;
}

}  // namespace

Expected<OwningBufferRef<uint8_t>> SerializeModel(LiteRtModelT&& model,
                                                  size_t bytecode_alignment) {
  // Pass the op code list through that was saved during loading. Add one more
  // op code for the dispatch ops
  auto tfl_op_codes = detail::TakeTflOpCodes(model);
  tfl_op_codes.push_back(
      MakeCustomOpCode(std::string(kLiteRtDispatchOpCustomCode)));

  SerializationContext builder(tfl_op_codes.size() - 1, model,
                               bytecode_alignment);
  builder.Model().operator_codes = std::move(tfl_op_codes);

  auto tfl_model = PackAsTflite(builder);
  if (!tfl_model) {
    LITERT_LOG(LITERT_ERROR, "Failed to pack as tflite");
    return tfl_model.Error();
  }

  auto serialized_tfl = SerializeFlatbuffer(**tfl_model);
  auto serialized_with_buffers =
      SerializeWithAppendedBuffers(builder, std::move(serialized_tfl), model);
  if (!serialized_with_buffers) {
    LITERT_LOG(LITERT_ERROR, "Failed to serialize with appended buffers");
    return serialized_with_buffers.Error();
  }

  if (!VerifyFlatbuffer(serialized_with_buffers->Span())) {
    LITERT_LOG(LITERT_ERROR, "Failed to verify flatbuffer");
    return Error(kLiteRtStatusErrorInvalidFlatbuffer);
  }

  return serialized_with_buffers;
}

}  // namespace litert::internal
