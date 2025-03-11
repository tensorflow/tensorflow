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

#include "tensorflow/lite/experimental/litert/core/model/model_load.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/model/buffer_manager.h"
#include "tensorflow/lite/experimental/litert/core/model/flatbuffer_to_litert.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_graph.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::internal {
namespace {

// Provides a view of model-level resources when constructing litert graph.
class FlatbufferContext {
 public:
  using LiteRtBufferId = uint32_t;
  using TflBufferInd = uint32_t;
  using BufferIdMap = absl::flat_hash_map<TflBufferInd, LiteRtBufferId>;

  FlatbufferContext(const FlatbufferWrapper& tfl_flatbuffer,
                    BufferManager* buffer_manager)
      : tfl_flatbuffer_(tfl_flatbuffer), buffer_manager_(buffer_manager) {}

  void SetOpCode(LiteRtOpT& litert_op, uint32_t ind) {
    const auto builtin_code =
        PackedModel()->operator_codes()->Get(ind)->builtin_code();
    litert_op.SetOpCode(static_cast<LiteRtOpCode>(builtin_code));
    detail::SetTflOpCodeInd(litert_op, ind);
  }

  // Get the buffer at the given index in the tflite model.
  Expected<const TflPackedBuffer*> GetTflBuffer(uint32_t ind) const {
    const auto* packed_model = tfl_flatbuffer_.PackedModel();
    if (ind >= packed_model->buffers()->size()) {
      LITERT_LOG(LITERT_ERROR, "Buffer index out of range");
      return Error(kLiteRtStatusErrorInvalidArgument);
    }
    return packed_model->buffers()->Get(ind);
  }

  BufferManager* GetBufferManager() { return buffer_manager_; }

  const uint8_t* AllocBase() const { return tfl_flatbuffer_.AllocBase(); }

  const TflPackedModel* PackedModel() const {
    return tfl_flatbuffer_.PackedModel();
  }

  BufferIdMap& RegisteredTflBufferIds() { return registered_tfl_buffer_ids_; }

 private:
  const FlatbufferWrapper& tfl_flatbuffer_;
  BufferManager* buffer_manager_;
  BufferIdMap registered_tfl_buffer_ids_;
};

LiteRtStatus UnpackOp(FlatbufferContext& context, LiteRtSubgraphT& parent,
                      const TflPackedOp& tfl_op, LiteRtOpT& litert_op) {
  // I/O TENSORS

  if (tfl_op.intermediates() && tfl_op.intermediates()->size() != 0) {
    // TODO: b/365299994 - Support intermediates.
    LITERT_LOG(LITERT_ERROR, "Intermediate tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (tfl_op.mutating_variable_inputs() &&
      tfl_op.mutating_variable_inputs()->size() != 0) {
    // TODO: b/365299994 - Support mutating variable inputs.
    LITERT_LOG(LITERT_ERROR, "Mutating variable inputs not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  const auto num_inputs = tfl_op.inputs()->size();
  for (auto i = 0; i < num_inputs; ++i) {
    const auto input_ind = tfl_op.inputs()->Get(i);
    // Skipping optional input tensor.
    if (input_ind == -1) {
      continue;
    }
    AttachInput(&parent.Tensor(input_ind), litert_op);
  }

  const auto num_outputs = tfl_op.outputs()->size();
  for (auto i = 0; i < num_outputs; ++i) {
    const auto output_ind = tfl_op.outputs()->Get(i);
    AttachOutput(&parent.Tensor(output_ind), litert_op);
  }

  // OPTIONS

  if (tfl_op.large_custom_options_size() != 0) {
    // TODO: b/365299994 - Support large custom options.
    LITERT_LOG(LITERT_ERROR, "Large custom options not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  const auto* custom_opts = tfl_op.custom_options();
  if (custom_opts) {
    litert_op.SetCustomOptions(custom_opts->data(), custom_opts->size());
  }

  // TODO figure out how to parse builtins with the packed flatbuffer api.
  TflOpPtr tfl_op_ptr(tfl_op.UnPack());
  detail::SetTflOptions(litert_op, std::move(tfl_op_ptr->builtin_options));
  detail::SetTflOptions2(litert_op, std::move(tfl_op_ptr->builtin_options_2));

  // OP CODE

  context.SetOpCode(litert_op, tfl_op.opcode_index());

  return kLiteRtStatusOk;
}

struct TflBufferContext {
  BufferRef<uint8_t> buffer;
  // Is buffer appended to the flatbuffer?
  bool is_external;
};

Expected<TflBufferContext> ReadBuffer(FlatbufferContext& context,
                                      uint32_t buffer_ind) {
  auto buffer = context.GetTflBuffer(buffer_ind);
  if (!buffer) {
    return buffer.Error();
  }

  const auto& tfl_buffer = **buffer;

  if (tfl_buffer.offset() != 0) {
    // Data is appended to the end of the flatbuffer.

    const auto* alloc_base = context.AllocBase();
    const auto offset = tfl_buffer.offset();
    const auto size = tfl_buffer.size();

    return TflBufferContext{BufferRef<uint8_t>(alloc_base + offset, size),
                            true};
  } else if (tfl_buffer.data()) {
    // Data is in the flatbuffer.

    const auto* start = tfl_buffer.data()->data();
    const auto size = tfl_buffer.data()->size();

    return TflBufferContext{BufferRef<uint8_t>(start, size), false};
  } else {
    return TflBufferContext{};
  }
}

LiteRtStatus UnpackTensor(FlatbufferContext& context,
                          const TflPackedTensor& tfl_tensor,
                          LiteRtTensorT& litert_tensor) {
  const auto buffer_ind = tfl_tensor.buffer();
  if (buffer_ind != 0) {
    auto buffer = ReadBuffer(context, buffer_ind);
    if (!buffer) {
      return buffer.Error().Status();
    }

    auto it = context.RegisteredTflBufferIds().find(buffer_ind);
    if (it != context.RegisteredTflBufferIds().end()) {
      litert_tensor.Weights().SetBufferId(it->second);
    } else {
      BufferContext lrt_buf_ctx;
      lrt_buf_ctx.should_append = buffer->is_external;
      SetWeightsFromUnownedBuffer(litert_tensor.Weights(), buffer->buffer,
                                  lrt_buf_ctx);
      context.RegisteredTflBufferIds()[buffer_ind] =
          litert_tensor.Weights().GetBufferId();
    }
  }

  // TENSOR TYPE

  TflTensorType tfl_tensor_type(tfl_tensor.type(), TflShapeInfo(tfl_tensor));
  auto tensor_type = MapTensorType(tfl_tensor_type);
  if (!tensor_type) {
    return tensor_type.Error().Status();
  }

  litert_tensor.SetType(std::move(*tensor_type));

  // QUANTIZATION

  if (tfl_tensor.quantization()) {
    TflQuantizationPtr tfl_quantization(tfl_tensor.quantization()->UnPack());
    auto quantization = MapQuantization(tfl_quantization.get(), litert_tensor);
    if (!quantization) {
      return quantization.Error().Status();
    }
    litert_tensor.SetQarams(std::move(*quantization));
  }

  // MISC

  if (tfl_tensor.name()) {
    litert_tensor.SetName(tfl_tensor.name()->str());
  }

  if (tfl_tensor.is_variable()) {
    // TODO: b/365299994 - Support variable tensors.
    LITERT_LOG(LITERT_ERROR, "Variable tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (tfl_tensor.variant_tensors() &&
      tfl_tensor.variant_tensors()->size() != 0) {
    // TODO: b/365299994 - Support variant tensors.
    LITERT_LOG(LITERT_ERROR, "Variant tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (tfl_tensor.sparsity() != nullptr) {
    // TODO: b/365299994 - Support sparsity tensors.
    LITERT_LOG(LITERT_ERROR, "Sparsity tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus UnpackSubgraph(FlatbufferContext& context,
                            const TflPackedSubgraph& tfl_subgraph,
                            LiteRtSubgraphT& litert_subgraph) {
  // Unpack tensors.
  const auto num_tensors = tfl_subgraph.tensors()->size();
  for (auto i = 0; i < num_tensors; ++i) {
    const auto* tfl_tensor = tfl_subgraph.tensors()->Get(i);
    LITERT_RETURN_IF_ERROR(
        UnpackTensor(context, *tfl_tensor, litert_subgraph.EmplaceTensor()));
  }

  // Unpack ops, pass litert_subgraph so they can look up the new litert
  // tensors.
  const auto num_ops = tfl_subgraph.operators()->size();
  for (auto i = 0; i < num_ops; ++i) {
    const auto* tfl_op = tfl_subgraph.operators()->Get(i);
    LITERT_RETURN_IF_ERROR(UnpackOp(context, litert_subgraph, *tfl_op,
                                    litert_subgraph.EmplaceOp()));
  }

  // Update subgraph I/O.
  const auto num_inputs = tfl_subgraph.inputs()->size();
  for (auto i = 0; i < num_inputs; ++i) {
    const auto tfl_input_ind = tfl_subgraph.inputs()->Get(i);
    litert_subgraph.Inputs().push_back(&litert_subgraph.Tensor(tfl_input_ind));
  }
  const auto num_outputs = tfl_subgraph.outputs()->size();
  for (auto i = 0; i < num_outputs; ++i) {
    const auto tfl_output_ind = tfl_subgraph.outputs()->Get(i);
    litert_subgraph.Outputs().push_back(
        &litert_subgraph.Tensor(tfl_output_ind));
  }

  return kLiteRtStatusOk;
}

LiteRtStatus UnpackSignatures(std::vector<TflSignaturePtr>& tfl_signatures,
                              LiteRtModelT& parent) {
  for (auto& tfl_signature : tfl_signatures) {
    if (tfl_signature->subgraph_index >= parent.Subgraphs().size()) {
      LITERT_LOG(LITERT_ERROR,
                 "Signature does not refer to a valid subgraph index.");
      return kLiteRtStatusErrorInvalidArgument;
    }

    auto* litert_subgraph =
        parent.Subgraphs().at(tfl_signature->subgraph_index);

    auto& tfl_inputs = tfl_signature->inputs;
    auto& tfl_outputs = tfl_signature->outputs;

    // Tflite signatures map a tensor index to a name. The input & output
    // indexes of signatures and subgraph are not matched, but the nubmer of
    // inputs and outputs should be the same.
    if (tfl_inputs.size() != litert_subgraph->Inputs().size() ||
        tfl_outputs.size() != litert_subgraph->Outputs().size()) {
      LITERT_LOG(LITERT_ERROR,
                 "Signature has incorrect number of input/outputs");
      return kLiteRtStatusErrorInvalidFlatbuffer;
    }

    // The tensor names may not be matched between signature and subgraph.
    // Update the tensor names with the signature names since the signature
    // names are used for LiteRT APIs.
    for (auto i = 0; i < tfl_inputs.size(); ++i) {
      const auto& tfl_input = tfl_inputs.at(i);
      auto* index_litert_input =
          litert_subgraph->Tensors().at(tfl_input->tensor_index);
      index_litert_input->SetName(tfl_input->name);
    }
    for (auto i = 0; i < tfl_outputs.size(); ++i) {
      const auto& tfl_output = tfl_outputs.at(i);
      auto* index_litert_output =
          litert_subgraph->Tensors().at(tfl_output->tensor_index);
      index_litert_output->SetName(tfl_output->name);
    }

    // Keep signature input/output names in the same order as the subgraph.
    std::vector<std::string> input_names;
    input_names.reserve(tfl_inputs.size());
    for (auto& tensor : litert_subgraph->Inputs()) {
      input_names.push_back(std::string(tensor->Name()));
    }
    std::vector<std::string> output_names;
    output_names.reserve(tfl_outputs.size());
    for (auto& tensor : litert_subgraph->Outputs()) {
      output_names.push_back(std::string(tensor->Name()));
    }

    parent.EmplaceSignature(litert_subgraph, std::move(input_names),
                            std::move(output_names),
                            tfl_signature->signature_key);
  }

  if (tfl_signatures.empty()) {
    parent.EmplaceSignature(MakeDefaultSignature(parent.MainSubgraph()));
  }

  return kLiteRtStatusOk;
}

Expected<LiteRtModelT::Ptr> UnpackModel(FlatbufferWrapper&& flatbuffer) {
  auto litert_model = std::make_unique<LiteRtModelT>(std::move(flatbuffer));

  FlatbufferContext context(detail::GetTflFlatbuffer(*litert_model),
                            litert_model->Buffers());
  const auto* packed_model = context.PackedModel();

  if (packed_model->subgraphs()) {
    const auto num_subgraphs = packed_model->subgraphs()->size();
    for (auto i = 0; i < num_subgraphs; ++i) {
      const auto* tfl_subgraph = packed_model->subgraphs()->Get(i);
      LITERT_RETURN_IF_ERROR(UnpackSubgraph(context, *tfl_subgraph,
                                            litert_model->EmplaceSubgraph()));
    }
  }

  // TODO Figure out how to load signatures in packed flatbuffer.
  if (packed_model->signature_defs()) {
    std::vector<TflSignaturePtr> tfl_signatures;
    for (auto i = 0; i < packed_model->signature_defs()->size(); ++i) {
      const auto* tfl_signature = packed_model->signature_defs()->Get(i);
      tfl_signatures.push_back(TflSignaturePtr(tfl_signature->UnPack()));
    }
    LITERT_RETURN_IF_ERROR(UnpackSignatures(tfl_signatures, *litert_model));
  } else {
    litert_model->EmplaceSignature(
        MakeDefaultSignature(litert_model->MainSubgraph()));
  }

  if (packed_model->metadata()) {
    const auto num_metadata = packed_model->metadata()->size();
    for (auto i = 0; i < num_metadata; ++i) {
      const auto* tfl_metadata = packed_model->metadata()->Get(i);
      auto name = tfl_metadata->name()->str();
      const auto buf_id = tfl_metadata->buffer();
      auto buf = ReadBuffer(context, buf_id);
      if (!buf) {
        return buf.Error();
      }

      litert_model->PushMetadata(name, buf->buffer.Data(), buf->buffer.Size());
    }
  }

  if (packed_model->operator_codes()) {
    const auto num_operator_codes = packed_model->operator_codes()->size();
    std::vector<TflOpCodePtr> tfl_op_codes(num_operator_codes);
    for (auto i = 0; i < num_operator_codes; ++i) {
      const auto* tfl_op_code = packed_model->operator_codes()->Get(i);
      TflOpCodePtr tfl_op_code_ptr(tfl_op_code->UnPack());
      tfl_op_codes[i] = std::move(tfl_op_code_ptr);
    }
    detail::SetTflOpCodes(*litert_model, std::move(tfl_op_codes));
  }

  return litert_model;
}

}  // namespace

Expected<LiteRtModelT::Ptr> LoadModelFromBuffer(BufferRef<uint8_t> buffer) {
  auto flatbuffer = FlatbufferWrapper::CreateFromBuffer(buffer);
  if (!flatbuffer) {
    return flatbuffer.Error();
  }
  return UnpackModel(std::move(**flatbuffer));
}

Expected<LiteRtModelT::Ptr> LoadModelFromFile(absl::string_view filename) {
  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(filename);
  if (!flatbuffer) {
    return flatbuffer.Error();
  }
  return UnpackModel(std::move(**flatbuffer));
}

}  // namespace litert::internal
