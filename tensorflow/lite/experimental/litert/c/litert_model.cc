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

#include "tensorflow/lite/experimental/litert/c/litert_model.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_load.h"
#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Model
//

LiteRtStatus LiteRtCreateModelFromFile(const char* filename,
                                       LiteRtModel* model) {
  if (!filename || !model) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto new_model = litert::internal::LoadModelFromFile(filename);
  if (!new_model) {
    return new_model.Error().Status();
  }
  *model = new_model->release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCreateModelFromBuffer(const void* buffer_addr,
                                         size_t buffer_size,
                                         LiteRtModel* model) {
  if (!buffer_addr || !buffer_size || !model) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  auto new_model = litert::internal::LoadModelFromBuffer(
      litert::BufferRef<uint8_t>(buffer_addr, buffer_size));
  if (!new_model) {
    return new_model.Error().Status();
  }
  *model = new_model->release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumModelSubgraphs(LiteRtModel model,
                                        LiteRtParamIndex* num_subgraphs) {
  if (model == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_subgraphs = model->Subgraphs().size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetModelSubgraph(LiteRtModel model,
                                    LiteRtParamIndex subgraph_index,
                                    LiteRtSubgraph* subgraph) {
  if (model == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (subgraph_index >= model->Subgraphs().size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *subgraph = &model->Subgraph(subgraph_index);
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMainModelSubgraphIndex(
    LiteRtModel model, LiteRtParamIndex* main_subgraph_index) {
  if (!model || !main_subgraph_index) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *main_subgraph_index = LiteRtModelT::kMainSubgraphIndex;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetModelMetadata(LiteRtModel model, const char* metadata_key,
                                    const void** metadata_buffer,
                                    size_t* metadata_buffer_size) {
  if (!model || !metadata_key || !metadata_buffer || !metadata_buffer_size) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  auto m_buf = model->FindMetadata(metadata_key);
  if (!m_buf) {
    return m_buf.Error().Status();
  }
  *metadata_buffer = m_buf->Data();
  *metadata_buffer_size = m_buf->Size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumModelSignatures(LiteRtModel model,
                                         LiteRtParamIndex* num_signatures) {
  if (!model || !num_signatures) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_signatures = model->Signatures().size();
  return kLiteRtStatusOk;
}

// Get the signature at the given index in the model
LiteRtStatus LiteRtGetModelSignature(LiteRtModel model,
                                     LiteRtParamIndex signature_index,
                                     LiteRtSignature* signature) {
  if (!model || !signature) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (signature_index >= model->Signatures().size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *signature = model->Signatures().at(signature_index);
  return kLiteRtStatusOk;
}

void LiteRtDestroyModel(LiteRtModel model) { delete model; }

LiteRtStatus LiteRtSerializeModel(LiteRtModel model, uint8_t** buf,
                                  size_t* size, size_t* offset,
                                  bool destroy_model,
                                  LiteRtModelSerializationOptions options) {
  auto serialized = litert::internal::SerializeModel(
      std::move(*model), options.bytecode_alignment);
  // Even if we fail to serialize, we still need to destroy the model if
  // requested. This is because the model may have been partially serialized
  // and we don't want to leak memory. Also if ownership of the model is
  // transferred to the caller, we need to ensure that the model is destroyed
  // when the caller is done with it.
  if (destroy_model) {
    delete model;
  }
  if (!serialized) {
    return serialized.Error().Status();
  }
  std::tie(*buf, *size, *offset) = serialized->Release();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtPushOp(LiteRtOpList op_list, LiteRtOp op,
                          LiteRtParamIndex index) {
  if (!op_list || !op) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  op_list->Push(op, index);
  return kLiteRtStatusOk;
}

//
// Signature
//

LiteRtStatus LiteRtGetDefaultSignatureKey(const char** signature_key) {
  if (!signature_key) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *signature_key = LiteRtSignatureT::kDefaultSignatureKey.data();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSignatureKey(LiteRtSignature signature,
                                   const char** signature_key) {
  if (!signature || !signature_key) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *signature_key = signature->Key().data();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSignatureSubgraph(LiteRtSignature signature,
                                        LiteRtSubgraph* subgraph) {
  if (signature == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *subgraph = &signature->GetSubgraph();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumSignatureInputs(LiteRtSignature signature,
                                         LiteRtParamIndex* num_inputs) {
  if (!signature || !num_inputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_inputs = signature->InputNames().size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSignatureInputName(LiteRtSignature signature,
                                         LiteRtParamIndex input_idx,
                                         const char** input_name) {
  if (!signature || !input_name) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (input_idx >= signature->InputNames().size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *input_name = signature->InputNames().at(input_idx).data();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumSignatureOutputs(LiteRtSignature signature,
                                          LiteRtParamIndex* num_outputs) {
  if (!signature || !num_outputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_outputs = signature->OutputNames().size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSignatureOutputName(LiteRtSignature signature,
                                          LiteRtParamIndex output_idx,
                                          const char** output_name) {
  if (!signature || !output_name) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (output_idx >= signature->OutputNames().size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *output_name = signature->OutputNames().at(output_idx).data();
  return kLiteRtStatusOk;
}

//
// Subgraph
//

LiteRtStatus LiteRtGetNumSubgraphInputs(LiteRtSubgraph subgraph,
                                        LiteRtParamIndex* num_inputs) {
  if (!subgraph || !num_inputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_inputs = subgraph->Inputs().size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSubgraphInput(LiteRtSubgraph subgraph,
                                    LiteRtParamIndex input_index,
                                    LiteRtTensor* input) {
  if (!subgraph || !input) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (input_index < 0 || input_index >= subgraph->Inputs().size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *input = subgraph->Inputs()[input_index];
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumSubgraphOutputs(LiteRtSubgraph subgraph,
                                         LiteRtParamIndex* num_outputs) {
  if (!subgraph || !num_outputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_outputs = subgraph->Outputs().size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSubgraphOutput(LiteRtSubgraph subgraph,
                                     LiteRtParamIndex output_index,
                                     LiteRtTensor* output) {
  if (!subgraph || !output) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (output_index < 0 || output_index >= subgraph->Outputs().size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *output = subgraph->Outputs()[output_index];
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumSubgraphOps(LiteRtSubgraph subgraph,
                                     LiteRtParamIndex* num_ops) {
  if (!subgraph || !num_ops) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_ops = subgraph->Ops().size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSubgraphOp(LiteRtSubgraph subgraph,
                                 LiteRtParamIndex op_index, LiteRtOp* op) {
  if (!subgraph || !op) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (op_index < 0 || op_index >= subgraph->Ops().size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *op = subgraph->Ops()[op_index];
  return kLiteRtStatusOk;
}

//
// Op
//

LiteRtStatus LiteRtGetOpCode(LiteRtOp op, LiteRtOpCode* code) {
  if (!op || !code) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *code = op->OpCode();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumOpInputs(LiteRtOp op, LiteRtParamIndex* num_inputs) {
  if (!op || !num_inputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_inputs = op->Inputs().size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetOpInput(LiteRtOp op, LiteRtParamIndex input_index,
                              LiteRtTensor* input) {
  if (!op || !input) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (input_index < 0 || input_index >= op->Inputs().size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *input = op->Inputs()[input_index];
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumOpOutputs(LiteRtOp op, LiteRtParamIndex* num_outputs) {
  if (!op || !num_outputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_outputs = op->Outputs().size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetOpOutput(LiteRtOp op, LiteRtParamIndex output_index,
                               LiteRtTensor* output) {
  if (!op || !output) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (output_index < 0 || output_index >= op->Outputs().size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *output = op->Outputs()[output_index];
  return kLiteRtStatusOk;
}

//
// Weights
//

LiteRtStatus LiteRtGetWeightsBytes(LiteRtWeights weights, const void** addr,
                                   size_t* size) {
  if (!weights || !addr || !size) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *addr = weights->Buffer().Data();
  *size = weights->Buffer().Size();
  return kLiteRtStatusOk;
}

//
// Tensor
//

LiteRtStatus LiteRtGetTensorWeights(LiteRtTensor tensor,
                                    LiteRtWeights* weights) {
  if (!tensor || !weights) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *weights = &tensor->Weights();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumTensorUses(LiteRtTensor tensor,
                                    LiteRtParamIndex* num_uses) {
  if (!tensor || !num_uses) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_uses = tensor->Users().size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorUse(LiteRtTensor tensor, LiteRtParamIndex use_index,
                                LiteRtOp* user,
                                LiteRtParamIndex* user_arg_index) {
  if (!tensor || !user || !user_arg_index) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (use_index < 0 || use_index >= tensor->Users().size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *user = tensor->Users()[use_index];
  *user_arg_index = tensor->UserArgInds()[use_index];
  return kLiteRtStatusOk;
}

// Null if subgraph input or constant.
LiteRtStatus LiteRtGetTensorDefiningOp(LiteRtTensor tensor,
                                       bool* has_defining_op,
                                       LiteRtTensorDefiningOp* defining_op) {
  if (!tensor || !has_defining_op || !defining_op) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (tensor->DefiningOp() != nullptr) {
    *has_defining_op = true;
    defining_op->op = tensor->DefiningOp();
    defining_op->op_output_index = tensor->DefiningOpOutInd();
  } else {
    *has_defining_op = false;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorTypeId(LiteRtTensor tensor,
                                   LiteRtTensorTypeId* type_id) {
  if (!tensor || !type_id) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *type_id = tensor->Type().first;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetUnrankedTensorType(
    LiteRtTensor tensor, LiteRtUnrankedTensorType* unranked_tensor_type) {
  if (!tensor || !unranked_tensor_type) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (tensor->Type().first != kLiteRtUnrankedTensorType) {
    return kLiteRtStatusErrorInvalidIrType;
  }
  *unranked_tensor_type = tensor->Type().second.unranked_tensor_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetRankedTensorType(
    LiteRtTensor tensor, LiteRtRankedTensorType* ranked_tensor_type) {
  if (!tensor || !ranked_tensor_type) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (tensor->Type().first != kLiteRtRankedTensorType) {
    return kLiteRtStatusErrorInvalidIrType;
  }
  *ranked_tensor_type = tensor->Type().second.ranked_tensor_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorName(LiteRtTensor tensor, const char** name) {
  if (!tensor || !name) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *name = tensor->Name().data();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetQuantizationTypeId(LiteRtTensor tensor,
                                         LiteRtQuantizationTypeId* q_type_id) {
  if (!tensor || !q_type_id) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *q_type_id = tensor->Qparams().first;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetPerTensorQuantization(
    LiteRtTensor tensor, LiteRtQuantizationPerTensor* per_tensor_quantization) {
  if (!tensor || !per_tensor_quantization) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (tensor->Qparams().first != kLiteRtQuantizationPerTensor) {
    return kLiteRtStatusErrorInvalidIrType;
  }
  auto& per_tensor = tensor->Qparams().second.per_tensor;
  per_tensor_quantization->scale = per_tensor.scale;
  per_tensor_quantization->zero_point = per_tensor.zero_point;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetPerChannelQuantization(
    LiteRtTensor tensor,
    LiteRtQuantizationPerChannel* per_channel_quantization) {
  if (!tensor || !per_channel_quantization) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (tensor->Qparams().first != kLiteRtQuantizationPerChannel) {
    return kLiteRtStatusErrorInvalidIrType;
  }
  auto& per_channel = tensor->Qparams().second.per_channel;
  per_channel_quantization->scales = per_channel.scales;
  per_channel_quantization->zero_points = per_channel.zero_points;
  per_channel_quantization->num_channels = per_channel.num_channels;
  per_channel_quantization->quantized_dimension =
      per_channel.quantized_dimension;
  return kLiteRtStatusOk;
}

#ifdef __cplusplus
}  // extern "C"
#endif
