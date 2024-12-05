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

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_load.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char* LiteRtDefaultSignatureKey = LITERT_DEFAULT_SIGNATURE_KEY;

//
// Model
//

LiteRtStatus LiteRtLoadModelFromFile(const char* filename, LiteRtModel* model) {
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

LiteRtStatus LiteRtLoadModelFromBuffer(const void* buffer_addr,
                                       size_t buffer_size, LiteRtModel* model) {
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
  if (!model || !num_subgraphs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_subgraphs = model->subgraphs.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetModelSubgraph(LiteRtModel model,
                                    LiteRtParamIndex subgraph_index,
                                    LiteRtSubgraph* subgraph) {
  if (!model) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (subgraph_index >= model->subgraphs.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *subgraph = model->subgraphs.data() + subgraph_index;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetMainModelSubgraphIndex(
    LiteRtModel model, LiteRtParamIndex* main_subgraph_index) {
  if (!model || !main_subgraph_index) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *main_subgraph_index = model->MainSubgraphIndex();
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
  *num_signatures = model->signatures.size();
  return kLiteRtStatusOk;
}

// Get the signature at the given index in the model
LiteRtStatus LiteRtGetModelSignature(LiteRtModel model,
                                     LiteRtParamIndex signature_index,
                                     LiteRtSignature* signature) {
  if (!model || !signature) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (signature_index >= model->signatures.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *signature = model->signatures[signature_index].get();
  return kLiteRtStatusOk;
}

void LiteRtModelDestroy(LiteRtModel model) { delete model; }

LiteRtStatus LiteRtPushOp(LiteRtOpList op_list, LiteRtOp op) {
  if (!op_list || !op) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  op_list->Push(op);
  return kLiteRtStatusOk;
}

//
// Signature
//

LiteRtStatus LiteRtGetDefaultSignatureKey(const char** signature_key) {
  if (!signature_key) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *signature_key = LiteRtDefaultSignatureKey;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSignatureKey(LiteRtSignature signature,
                                   const char** signature_key) {
  if (!signature || !signature_key) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *signature_key = signature->key.data();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSignatureSubgraphIndex(LiteRtSignature signature,
                                             LiteRtParamIndex* subgraph_index) {
  *subgraph_index = signature->subgraph_index;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumSignatureInputs(LiteRtSignature signature,
                                         LiteRtParamIndex* num_inputs) {
  if (!signature || !num_inputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_inputs = signature->input_names.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSignatureInputName(LiteRtSignature signature,
                                         LiteRtParamIndex input_idx,
                                         const char** input_name) {
  if (!signature || !input_name) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (input_idx >= signature->input_names.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *input_name = signature->input_names[input_idx].data();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumSignatureOutputs(LiteRtSignature signature,
                                          LiteRtParamIndex* num_outputs) {
  if (!signature || !num_outputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_outputs = signature->output_names.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSignatureOutputName(LiteRtSignature signature,
                                          LiteRtParamIndex output_idx,
                                          const char** output_name) {
  if (!signature || !output_name) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (output_idx >= signature->output_names.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *output_name = signature->output_names[output_idx].data();
  return kLiteRtStatusOk;
}

//
// Subgraph
//

LiteRtStatus LiteRtGetSubgraphInputs(LiteRtSubgraph subgraph,
                                     LiteRtParamIndex* num_inputs,
                                     LiteRtTensorArray* inputs) {
  if (!subgraph || !num_inputs || !inputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_inputs = subgraph->inputs.size();
  *inputs = subgraph->inputs.data();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSubgraphOutputs(LiteRtSubgraph subgraph,
                                      LiteRtParamIndex* num_outputs,
                                      LiteRtTensorArray* outputs) {
  if (!subgraph || !num_outputs || !outputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_outputs = subgraph->outputs.size();
  *outputs = subgraph->outputs.data();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetSubgraphOps(LiteRtSubgraph subgraph,
                                  LiteRtParamIndex* num_ops,
                                  LiteRtOpArray* ops) {
  if (!subgraph || !num_ops || !ops) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_ops = subgraph->ops.size();
  *ops = subgraph->ops.data();
  return kLiteRtStatusOk;
}

//
// Op
//

LiteRtStatus LiteRtGetOpOutputs(LiteRtOp op, LiteRtParamIndex* num_outputs,
                                LiteRtTensorArray* outputs) {
  if (!op || !num_outputs || !outputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_outputs = op->outputs.size();
  *outputs = op->outputs.data();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetOpInputs(LiteRtOp op, LiteRtParamIndex* num_inputs,
                               LiteRtTensorArray* inputs) {
  if (!op || !num_inputs || !inputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_inputs = op->inputs.size();
  *inputs = op->inputs.data();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetOpCode(LiteRtOp op, LiteRtOpCode* code) {
  if (!op || !code) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *code = op->op_code;
  return kLiteRtStatusOk;
}

//
// Tensor
//

LiteRtStatus LiteRtGetWeightsBytes(LiteRtWeights weights, const void** addr,
                                   size_t* size) {
  if (!weights || !addr || !size) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (weights->fb_buffer == nullptr) {
    *addr = nullptr;
    *size = 0;
  } else {
    *addr = weights->fb_buffer->data.data();
    *size = weights->fb_buffer->data.size();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorWeights(LiteRtTensor tensor,
                                    LiteRtWeights* weights) {
  if (!tensor || !weights) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *weights = &tensor->weights;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorUses(LiteRtTensor tensor,
                                 LiteRtParamIndex* num_uses,
                                 LiteRtOpArray* use_users,
                                 LiteRtParamIndex** use_user_arg_inds) {
  if (!tensor || !num_uses || !use_users || !use_user_arg_inds) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_uses = tensor->users.size();
  *use_users = tensor->users.data();
  *use_user_arg_inds = tensor->user_arg_inds.data();
  return kLiteRtStatusOk;
}

// Null if subgraph input or constant.
LiteRtStatus LiteRtGetTensorDefiningOp(LiteRtTensor tensor,
                                       bool* has_defining_op,
                                       LiteRtTensorDefiningOp* defining_op) {
  if (!tensor || !has_defining_op || !defining_op) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (tensor->defining_op != nullptr) {
    *has_defining_op = true;
    defining_op->op = tensor->defining_op;
    defining_op->op_output_index = tensor->defining_op_out_ind;
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
  *type_id = tensor->type_id;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetUnrankedTensorType(
    LiteRtTensor tensor, LiteRtUnrankedTensorType* unranked_tensor_type) {
  if (!tensor || !unranked_tensor_type) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (tensor->type_id != kLiteRtUnrankedTensorType) {
    return kLiteRtStatusErrorInvalidIrType;
  }
  *unranked_tensor_type = tensor->type_detail.unranked_tensor_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetRankedTensorType(
    LiteRtTensor tensor, LiteRtRankedTensorType* ranked_tensor_type) {
  if (!tensor || !ranked_tensor_type) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (tensor->type_id != kLiteRtRankedTensorType) {
    return kLiteRtStatusErrorInvalidIrType;
  }
  *ranked_tensor_type = tensor->type_detail.ranked_tensor_type;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetTensorName(LiteRtTensor tensor, const char** name) {
  if (!tensor || !name) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *name = tensor->name.data();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetQuantizationTypeId(LiteRtTensor tensor,
                                         LiteRtQuantizationTypeId* q_type_id) {
  if (!tensor || !q_type_id) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *q_type_id = tensor->q_type_id;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetPerTensorQuantization(
    LiteRtTensor tensor, LiteRtQuantizationPerTensor* per_tensor_quantization) {
  if (!tensor || !per_tensor_quantization) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (tensor->q_type_id != kLiteRtQuantizationPerTensor) {
    return kLiteRtStatusErrorInvalidIrType;
  }
  *per_tensor_quantization = tensor->q_type_detail.per_tensor;
  return kLiteRtStatusOk;
}
