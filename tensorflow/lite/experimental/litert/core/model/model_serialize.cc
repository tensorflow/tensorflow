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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/byte_code_util.h"
#include "tensorflow/lite/experimental/litert/core/dispatch_op_schema.h"
#include "tensorflow/lite/experimental/litert/core/model/litert_to_flatbuffer.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/schema/mutable/schema_generated.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::internal {
namespace {

using TensorMap = absl::flat_hash_map<LiteRtTensor, int32_t>;

// Pop npu related stuff from model if it exists and requires a post process
// step (i.e. appending byte code to tflite).
std::optional<OwningBufferRef<uint8_t>> PopByteCodeIfNeedsPostProcess(
    LiteRtModelT& model) {
  auto build_stamp_buf = model.FindMetadata(kLiteRtBuildStampKey);
  if (!build_stamp_buf) {
    return std::nullopt;
  }

  auto build_stamp = ParseBuildStamp(*build_stamp_buf);
  if (!build_stamp) {
    LITERT_LOG(LITERT_WARNING,
               "Model has a build stamp but it couldn't be parsed");
    return std::nullopt;
  }

  // Only appending needs separate strategy.
  if (std::get<2>(*build_stamp) != kAppend) {
    return std::nullopt;
  }

  // Pop the actual byte and and replace it with a placeholder value
  // which will be
  auto byte_code = model.PopMetadata(kByteCodeMetadataKey);
  if (!byte_code) {
    LITERT_LOG(LITERT_WARNING, "Model has npu build stamp but no byte code");
    return std::nullopt;
  }
  model.PushMetadata(kByteCodeMetadataKey, MakeByteCodePlaceholder());

  return *byte_code;
}

Expected<OwningBufferRef<uint8_t>> AppendByteCode(
    OwningBufferRef<uint8_t> flatbuffer,
    OwningBufferRef<uint8_t> npu_byte_code) {
  LITERT_RETURN_IF_ERROR(
      FinishByteCodePlaceholders(flatbuffer, npu_byte_code.Size()));

  const auto res_size = flatbuffer.Size() + npu_byte_code.Size();
  OwningBufferRef<uint8_t> res(res_size);

  uint8_t* it = res.Data();
  std::memcpy(it, flatbuffer.Data(), flatbuffer.Size());
  it += flatbuffer.Size();
  std::memcpy(it, npu_byte_code.Data(), npu_byte_code.Size());

  return res;
}

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
  struct TflExternalBufferRef {
    size_t subgraph_ind;
    size_t op_ind;
    LiteRtModelT::ExternalBufferId buf_id;
  };

  explicit SerializationContext(uint32_t dispatch_op_code_ind,
                                LiteRtModelT& litert_model)
      : tfl_model_(std::make_unique<TflModel>()),
        dispatch_op_code_ind_(dispatch_op_code_ind),
        litert_model_(litert_model) {
    // Tfl expects empty buffer 0.
    tfl_model_->buffers.push_back(std::make_unique<TflBuffer>());
  }

  TflModel& Model() { return *tfl_model_.get(); }

  TflModelPtr Release() && { return std::move(tfl_model_); }

  LiteRtModelT& LitertModel() { return litert_model_; }

  // Move given buffer into tfl model and get its index.
  uint32_t SubmitBuffer(TflBufferPtr tfl_buffer) {
    tfl_model_->buffers.push_back(std::move(tfl_buffer));
    return tfl_model_->buffers.size() - 1;
  }

  // Add to tfl model metadata.
  void PushMetadata(std::string key, BufferRef<uint8_t> data) {
    auto tfl_buffer = std::make_unique<TflBuffer>();
    tfl_buffer->data.assign(data.Data(), data.Data() + data.Size());
    auto tfl_buffer_ind = SubmitBuffer(std::move(tfl_buffer));
    tfl_model_->metadata_buffer.push_back(tfl_buffer_ind);
    auto tfl_metadata = std::make_unique<TflMetadata>();
    tfl_metadata->name = key;
    tfl_metadata->buffer = tfl_buffer_ind;
    tfl_model_->metadata.push_back(std::move(tfl_metadata));
  }

  // Keep track of the given ops index as having a particular external buffer.
  // These will be used to update the ops with the correct offset and size
  // after the model is fully packed. Call this in "sorted" op order.
  void MarkOpWithExternalBuffer(size_t subgraph_ind, size_t op_ind,
                                LiteRtModelT::ExternalBufferId buf_id) {
    external_buffer_refs_.push_back(
        TflExternalBufferRef{subgraph_ind, op_ind, buf_id});
  }

  const std::vector<TflExternalBufferRef>& ExternalBufferRefs() {
    return external_buffer_refs_;
  }

  // Get the index in the tfl op codes for the dispatch custom code.
  // This should be the only new custom code added after loading the initial
  // tfl.
  uint32_t DispatchOpCodeInd() const { return dispatch_op_code_ind_; }

 private:
  TflModelPtr tfl_model_;
  uint32_t dispatch_op_code_ind_;
  LiteRtModelT& litert_model_;
  // Expected to be sorted at all times, with respect to subgraph/op indices.
  std::vector<TflExternalBufferRef> external_buffer_refs_;
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

  tfl_tensor.buffer =
      builder.SubmitBuffer(detail::TakeTflBuffer(litert_tensor.Weights()));
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
    LITERT_RETURN_STATUS_IF_NOT_OK(
        PackTensor(builder, *tensor, *tfl_subgraph.tensors.back()));
  }

  for (auto i = 0; i < litert_subgraph.Ops().size(); ++i) {
    auto* op = litert_subgraph.Ops().at(i);

    tfl_subgraph.operators.push_back(std::make_unique<TflOp>());
    auto& tfl_op = *tfl_subgraph.operators.back();
    LITERT_RETURN_STATUS_IF_NOT_OK(PackOp(builder, *op, tfl_op, tensor_map));

    // Set custom options.
    if (auto external_buffer = builder.LitertModel().FindExternalBuffer(op)) {
      // This mechanism is currently only used for dispatch ops to store
      // location of bytecode. Here we update the name and placeholder values
      // for offset and size. These will be updated when the model is fully
      // packed.
      auto dispatch_opts = MakeDispatchOpOptions({
          0,
          0,
          std::string(external_buffer->second),
      });
      tfl_op.custom_options = dispatch_opts.ToVec();

      // Save the "location" of the external buffer in the model.
      builder.MarkOpWithExternalBuffer(subgraph_ind, i, external_buffer->first);

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
    builder.PushMetadata(it->first, it->second);
  }

  builder.Model().version = 3;

  return std::move(builder).Release();
}

// Appends external buffers to the back of the serialized tflite model. Updates
// the ops that references them with the correct offset and size in-place.
Expected<OwningBufferRef<uint8_t>> SerializeWithAppendedExternalBuffers(
    SerializationContext& builder, OwningBufferRef<uint8_t> serialized_tfl,
    LiteRtModelT& litert_model) {
  // The final offset/size of the external buffers in the serialized tflite
  // model.
  std::vector<std::pair<size_t, size_t>> external_buffer_offsets;
  size_t cur_offset = serialized_tfl.Size();
  for (auto i = 0; i < litert_model.NumExternalBuffers(); ++i) {
    auto external_buf = litert_model.GetExternalBuffer(i);
    if (!external_buf) {
      return external_buf.Error();
    }
    external_buffer_offsets.push_back({cur_offset, external_buf->Size()});
    cur_offset += external_buf->Size();
  }

  // Read serialized tflite in packed form.
  auto* tfl_model = tflite::GetMutableModel(serialized_tfl.Data());

  // Get "sorted" locations of ops that use external buffers.
  auto external_buf_it = builder.ExternalBufferRefs().cbegin();

  // Iterate in the same order that ops were packed, upating with the correct
  // offset and size.
  for (auto sg_ind = 0; sg_ind < tfl_model->mutable_subgraphs()->size();
       ++sg_ind) {
    auto* sg = tfl_model->mutable_subgraphs()->GetMutableObject(sg_ind);

    if (external_buf_it == builder.ExternalBufferRefs().cend()) {
      break;
    }

    if (external_buf_it->subgraph_ind != sg_ind) {
      ++external_buf_it;
      continue;
    }

    for (auto op_ind = 0; op_ind < sg->mutable_operators()->size(); ++op_ind) {
      auto* op = sg->mutable_operators()->GetMutableObject(op_ind);

      if (external_buf_it == builder.ExternalBufferRefs().cend()) {
        break;
      }

      if (external_buf_it->op_ind != op_ind) {
        ++external_buf_it;
        continue;
      }

      // If we are here than this op has an external buffer. Update with
      // the appropriate offset and size.
      const auto buf_id = external_buf_it->buf_id;
      const auto [real_offset, real_size] = external_buffer_offsets.at(buf_id);

      MutableBufferRef<uint8_t> old_raw_opts(
          op->mutable_custom_options()->data(),
          op->mutable_custom_options()->size());
      DispatchOpOptions dispach_opts(GetDispatchOpOptions(old_raw_opts));
      dispach_opts.bytecode_offset = real_offset;
      dispach_opts.bytecode_size = real_size;

      if (!UpdateDispatchOpOptionsInPlace(dispach_opts, old_raw_opts)) {
        return Error(kLiteRtStatusErrorInvalidFlatbuffer);
      }

      ++external_buf_it;
    }
  }

  // Allocate buffer enough for original model and appendd buffers and copy.
  OwningBufferRef<uint8_t> final_model(cur_offset);

  uint8_t* start = final_model.Data();
  std::memcpy(start, serialized_tfl.Data(), serialized_tfl.Size());
  start += serialized_tfl.Size();
  for (auto i = 0; i < litert_model.NumExternalBuffers(); ++i) {
    auto external_buf = litert_model.GetExternalBuffer(i);
    if (!external_buf) {
      return external_buf.Error();
    }
    std::memcpy(start, external_buf->Data(), external_buf->Size());
    start += external_buf->Size();
  }

  return final_model;
}

}  // namespace

Expected<OwningBufferRef<uint8_t>> SerializeModel(LiteRtModelT&& model) {
  // Check if the model has fresh npu stuff. It it does, pop it off
  // for post processing after packing to tflite model.
  auto maybe_byte_code = PopByteCodeIfNeedsPostProcess(model);

  // Pass the op code list through that was saved during loading. Add one more
  // op code for the dispatch ops
  auto tfl_op_codes = detail::TakeTflOpCodes(model);
  tfl_op_codes.push_back(
      MakeCustomOpCode(std::string(kLiteRtDispatchOpCustomCode)));

  SerializationContext builder(tfl_op_codes.size() - 1, model);
  builder.Model().operator_codes = std::move(tfl_op_codes);

  auto tfl_model = PackAsTflite(builder);
  if (!tfl_model) {
    return tfl_model.Error();
  }

  auto serialized_tfl = SerializeFlatbuffer(**tfl_model);
  if (!VerifyFlatbuffer(serialized_tfl.Span())) {
    return Error(kLiteRtStatusErrorInvalidFlatbuffer);
  }

  if (model.NumExternalBuffers() != 0) {
    return SerializeWithAppendedExternalBuffers(
        builder, std::move(serialized_tfl), model);
  }
  if (maybe_byte_code) {
    return AppendByteCode(std::move(serialized_tfl), *maybe_byte_code);
  }
  return serialized_tfl;
}

}  // namespace litert::internal

LiteRtStatus LiteRtSerializeModel(LiteRtModel model, uint8_t** buf,
                                  size_t* size, size_t* offset,
                                  bool destroy_model) {
  auto serialized = litert::internal::SerializeModel(std::move(*model));
  if (destroy_model) {
    delete model;
  }
  if (!serialized) {
    return serialized.Error().Status();
  }
  std::tie(*buf, *size, *offset) = serialized->Release();
  return kLiteRtStatusOk;
}
