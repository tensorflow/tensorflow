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
#include <utility>
#include <vector>

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
  FlatbufferContext(TflModel& tfl_model, BufferManager* buffer_manager,
                    const uint8_t* alloc_base)
      : tfl_model_(tfl_model),
        buffer_manager_(buffer_manager),
        alloc_base_(alloc_base) {}

  void SetOpCode(LiteRtOpT& litert_op, uint32_t ind) {
    auto tfl_op_code = GetTflOpCode(tfl_model_, ind);
    litert_op.SetOpCode(static_cast<LiteRtOpCode>(*tfl_op_code));
    detail::SetTflOpCodeInd(litert_op, ind);
  }

  // Get the buffer at the given index in the tflite model. This must be a
  // copied currently since we are using the object based flatbuffer API.
  // TODO switch to use the lower level flatbuffer api to avoid the copy.
  Expected<const tflite::BufferT*> GetTflBuffer(uint32_t ind) const {
    return GetBuffer(tfl_model_, ind);
  }

  BufferManager* GetBufferManager() { return buffer_manager_; }

 private:
  TflModel& tfl_model_;
  BufferManager* buffer_manager_;
  // NOLINTNEXTLINE
  const uint8_t* alloc_base_;
};

LiteRtStatus UnpackOp(FlatbufferContext& context, LiteRtSubgraphT& parent,
                      TflOpPtr tfl_op, LiteRtOpT& litert_op) {
  // I/O TENSORS

  if (!tfl_op->intermediates.empty()) {
    // TODO: b/365299994 - Support intermediates.
    LITERT_LOG(LITERT_ERROR, "Intermediate tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  for (auto m_input : tfl_op->mutating_variable_inputs) {
    if (m_input) {
      // TODO: b/365299994 - Support mutating variable inputs.
      LITERT_LOG(LITERT_ERROR, "Mutating variable inputs not yet supported.");
      return kLiteRtStatusErrorUnsupported;
    }
  }

  for (auto input_ind : tfl_op->inputs) {
    // Skipping optional input tensor.
    if (input_ind == -1) {
      continue;
    }
    AttachInput(&parent.Tensor(input_ind), litert_op);
  }

  for (auto output_ind : tfl_op->outputs) {
    AttachOutput(&parent.Tensor(output_ind), litert_op);
  }

  // OPTIONS

  if (tfl_op->large_custom_options_size != 0) {
    // TODO: b/365299994 - Support large custom options.
    LITERT_LOG(LITERT_ERROR, "Large custom options not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  const auto& tfl_custom_opts = tfl_op->custom_options;
  litert_op.SetCustomOptions(tfl_custom_opts.data(), tfl_custom_opts.size());
  detail::SetTflOptions(litert_op, std::move(tfl_op->builtin_options));

  // OP CODE

  context.SetOpCode(litert_op, tfl_op->opcode_index);

  return kLiteRtStatusOk;
}

LiteRtStatus UnpackTensor(FlatbufferContext& context, TflTensorPtr tfl_tensor,
                          LiteRtTensorT& litert_tensor) {
  // WEIGHTS

  litert_tensor.Weights().SetBufferManager(context.GetBufferManager());

  const auto buffer_ind = tfl_tensor->buffer;
  if (buffer_ind != 0) {
    auto buffer = context.GetTflBuffer(buffer_ind);
    if (!buffer) {
      return buffer.Error().Status();
    }
    const auto& tfl_buffer = **buffer;

    if (tfl_buffer.offset != 0) {
      // TODO: b/365299994 - Support buffer with offset.
      LITERT_LOG(LITERT_ERROR, "Buffers with offset not yet supported.");
      return kLiteRtStatusErrorUnsupported;
    }

    // TODO we can switch to lower level flatbuffer api here and just pass the
    // view through w/ no copy.
    OwningBufferRef<uint8_t> weights_buffer(tfl_buffer.data.data(),
                                            tfl_buffer.data.size());
    SetWeightsFromOwnedBuffer(litert_tensor.Weights(),
                              std::move(weights_buffer));
  }

  // TENSOR TYPE

  TflTensorType tfl_tensor_type(tfl_tensor->type, TflShapeInfo(*tfl_tensor));
  auto tensor_type = MapTensorType(tfl_tensor_type);
  if (!tensor_type) {
    return tensor_type.Error().Status();
  }

  litert_tensor.SetType(std::move(*tensor_type));

  // QUANTIZATION

  auto quantization =
      MapQuantization(tfl_tensor->quantization.get(), litert_tensor);
  if (!quantization) {
    return quantization.Error().Status();
  }

  litert_tensor.SetQarams(std::move(*quantization));

  // MISC

  litert_tensor.SetName(tfl_tensor->name);

  if (tfl_tensor->is_variable) {
    // TODO: b/365299994 - Support variable tensors.
    LITERT_LOG(LITERT_ERROR, "Variable tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (!tfl_tensor->variant_tensors.empty()) {
    // TODO: b/365299994 - Support variant tensors.
    LITERT_LOG(LITERT_ERROR, "Variant tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (tfl_tensor->sparsity) {
    // TODO: b/365299994 - Support sparsity tensors.
    LITERT_LOG(LITERT_ERROR, "Sparsity tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus UnpackSubgraph(FlatbufferContext& context,
                            TflSubgraphPtr tfl_subgraph,
                            LiteRtSubgraphT& litert_subgraph) {
  // Unpack tensors.
  for (auto& tfl_tensor : tfl_subgraph->tensors) {
    LITERT_RETURN_IF_ERROR(UnpackTensor(context, std::move(tfl_tensor),
                                        litert_subgraph.EmplaceTensor()));
  }

  // Unpack ops, pass litert_subgraph so they can look up the new litert
  // tensors.
  for (auto& tfl_op : tfl_subgraph->operators) {
    LITERT_RETURN_IF_ERROR(UnpackOp(context, litert_subgraph, std::move(tfl_op),
                                    litert_subgraph.EmplaceOp()));
  }

  // Update subgraph I/O.
  for (auto tfl_input_ind : tfl_subgraph->inputs) {
    litert_subgraph.Inputs().push_back(&litert_subgraph.Tensor(tfl_input_ind));
  }
  for (auto tfl_output_ind : tfl_subgraph->outputs) {
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

    auto get_name = [](const auto& tfl_tensor) { return tfl_tensor->name; };

    std::vector<std::string> input_names(tfl_inputs.size());
    std::transform(tfl_inputs.cbegin(), tfl_inputs.cend(), input_names.begin(),
                   get_name);

    std::vector<std::string> output_names(tfl_outputs.size());
    std::transform(tfl_outputs.cbegin(), tfl_outputs.cend(),
                   output_names.begin(), get_name);

    parent.EmplaceSignature(litert_subgraph, std::move(input_names),
                            std::move(output_names),
                            tfl_signature->signature_key);
  }

  if (tfl_signatures.empty()) {
    parent.EmplaceSignature(MakeDefaultSignature(parent.MainSubgraph()));
  }

  return kLiteRtStatusOk;
}

LiteRtStatus UnpackMetadata(FlatbufferContext& context,
                            std::vector<TflMetadataPtr>& tfl_metadata,
                            LiteRtModelT& parent) {
  for (auto& tfl_m_data : tfl_metadata) {
    auto tfl_buffer = context.GetTflBuffer(tfl_m_data->buffer);
    if (!tfl_buffer) {
      return tfl_buffer.Error().Status();
    }
    const auto& tfl_buf = **tfl_buffer;
    // TODO switch to lower level flatbuffer api here and just pass the view
    // through w/ no copy.
    parent.PushMetadata(tfl_m_data->name, tfl_buf.data.data(),
                        tfl_buf.data.size());
  }

  return kLiteRtStatusOk;
}

Expected<LiteRtModelT::Ptr> UnpackModel(TflModelPtr tfl_model,
                                        const uint8_t* alloc_base) {
  auto litert_model = std::make_unique<LiteRtModelT>();
  FlatbufferContext context(*tfl_model, litert_model->Buffers(), alloc_base);

  for (auto& tfl_subgraph : tfl_model->subgraphs) {
    LITERT_RETURN_IF_ERROR(UnpackSubgraph(context, std::move(tfl_subgraph),
                                          litert_model->EmplaceSubgraph()));
  }

  LITERT_RETURN_IF_ERROR(
      UnpackSignatures(tfl_model->signature_defs, *litert_model));
  LITERT_RETURN_IF_ERROR(
      UnpackMetadata(context, tfl_model->metadata, *litert_model));
  detail::SetTflOpCodes(*litert_model, std::move(tfl_model->operator_codes));

  return litert_model;
}

}  // namespace

Expected<LiteRtModelT::Ptr> LoadModelFromBuffer(BufferRef<uint8_t> buffer) {
  auto flatbuffer = FlatbufferWrapper::CreateFromBuffer(buffer);
  if (!flatbuffer) {
    return flatbuffer.Error();
  }
  auto litert_model = UnpackModel(flatbuffer->get()->Unpack(), buffer.Data());
  if (litert_model) {
    // Save the original FB pointer to use it later on CompiledModel.
    detail::SetTflInitFlatbuffer(**litert_model, buffer);
  }
  return litert_model;
}

Expected<LiteRtModelT::Ptr> LoadModelFromFile(absl::string_view filename) {
  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(filename);
  if (!flatbuffer) {
    return flatbuffer.Error();
  }
  return UnpackModel(flatbuffer->get()->Unpack(),
                     flatbuffer->get()->Buf().Data());
}

}  // namespace litert::internal
