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

#include "tensorflow/lite/experimental/lrt/core/model.h"

#include <cstddef>
#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/c/litert_op_code.h"
#include "tensorflow/lite/experimental/lrt/cc/litert_support.h"
#include "tensorflow/lite/experimental/lrt/core/util/buffer_ref.h"
#include "tensorflow/lite/schema/schema_generated.h"

using ::litert::MutableBufferRef;

//
// Model
//

LiteRtStatus GetModelNumSubgraphs(LiteRtModel model,
                                  LiteRtParamIndex* num_subgraphs) {
  *num_subgraphs = model->subgraphs.size();
  return kLiteRtStatusOk;
}

LiteRtStatus GetModelSubgraph(LiteRtModel model,
                              LiteRtParamIndex subgraph_index,
                              LiteRtSubgraph* subgraph) {
  if (subgraph_index >= model->subgraphs.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  *subgraph = model->subgraphs.data() + subgraph_index;
  return kLiteRtStatusOk;
}

LiteRtStatus GetModelMainSubgraph(LiteRtModel model,
                                  LiteRtParamIndex* main_subgraph_index) {
  // TODO replace this with signature.
  *main_subgraph_index = 0;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtModelGetMetadata(LiteRtModel model, const char* metadata_key,
                                    const void** metadata_buffer,
                                    size_t* metadata_buffer_size) {
  LITERT_ASSIGN_OR_RETURN_STATUS(auto m_buf, model->FindMetadata(metadata_key));
  *metadata_buffer = m_buf.Data();
  *metadata_buffer_size = m_buf.Size();
  return kLiteRtStatusOk;
}

void ModelDestroy(LiteRtModel model) {
  if (model != nullptr) {
    delete model;
  }
}

LiteRtStatus PushOp(LiteRtOpList op_list, LiteRtOp op) {
  op_list->Push(op);
  return kLiteRtStatusOk;
}

LiteRtResult<MutableBufferRef<uint8_t>> LiteRtModelT::FindMetadata(
    const absl::string_view key) const {
  using ResT = LiteRtResult<MutableBufferRef<uint8_t>>;

  tflite::MetadataT* fb_metadata = nullptr;
  for (auto& m : flatbuffer_model->metadata) {
    if (m->name == key) {
      fb_metadata = m.get();
      break;
    }
  }
  if (fb_metadata == nullptr) {
    return ResT::FromStatus(kLiteRtStatusErrorNotFound);
  }

  const uint32_t m_buffer_idx = fb_metadata->buffer;
  if (m_buffer_idx >= flatbuffer_model->buffers.size()) {
    return ResT::FromStatus(kLiteRtStatusErrorIndexOOB);
  }
  tflite::BufferT* m_buffer = flatbuffer_model->buffers.at(m_buffer_idx).get();

  return ResT::FromValue(
      MutableBufferRef(m_buffer->data.data(), m_buffer->data.size()));
}

//
// Subgraph
//

LiteRtStatus GetSubgraphInputs(LiteRtSubgraph subgraph,
                               LiteRtParamIndex* num_inputs,
                               LiteRtTensorArray* inputs) {
  *num_inputs = subgraph->inputs.size();
  *inputs = subgraph->inputs.data();
  return kLiteRtStatusOk;
}

LiteRtStatus GetSubgraphOutputs(LiteRtSubgraph subgraph,
                                LiteRtParamIndex* num_outputs,
                                LiteRtTensorArray* outputs) {
  *num_outputs = subgraph->outputs.size();
  *outputs = subgraph->outputs.data();
  return kLiteRtStatusOk;
}

LiteRtStatus GetSubgraphOps(LiteRtSubgraph subgraph, LiteRtParamIndex* num_ops,
                            LiteRtOpArray* ops) {
  *num_ops = subgraph->ops.size();
  *ops = subgraph->ops.data();
  return kLiteRtStatusOk;
}

//
// Op
//

LiteRtStatus GetOpOutputs(LiteRtOp op, LiteRtParamIndex* num_outputs,
                          LiteRtTensorArray* outputs) {
  *num_outputs = op->outputs.size();
  *outputs = op->outputs.data();
  return kLiteRtStatusOk;
}

LiteRtStatus GetOpInputs(LiteRtOp op, LiteRtParamIndex* num_inputs,
                         LiteRtTensorArray* inputs) {
  *num_inputs = op->inputs.size();
  *inputs = op->inputs.data();
  return kLiteRtStatusOk;
}

LiteRtStatus GetOpCode(LiteRtOp op, LiteRtOpCode* code) {
  *code = op->op_code;
  return kLiteRtStatusOk;
}

//
// Tensor
//

LiteRtStatus GetWeightsInfo(LiteRtWeights weights, size_t* size,
                            const void** addr) {
  if (weights->fb_buffer == nullptr) {
    *size = 0;
    *addr = nullptr;
  } else {
    *size = weights->fb_buffer->data.size();
    *addr = weights->fb_buffer->data.data();
  }
  return kLiteRtStatusOk;
}

LiteRtStatus GetTensorWeights(LiteRtTensor tensor, LiteRtWeights* weights) {
  *weights = &tensor->weights;
  return kLiteRtStatusOk;
}

LiteRtStatus GetTensorUses(LiteRtTensor tensor, LiteRtParamIndex* num_uses,
                           LiteRtOpArray* use_users,
                           LiteRtParamIndex** use_user_arg_inds) {
  *num_uses = tensor->users.size();
  *use_users = tensor->users.data();
  *use_user_arg_inds = tensor->user_arg_inds.data();
  return kLiteRtStatusOk;
}

// Null if subgraph input or constant.
LiteRtStatus GetTensorDefiningOp(
    LiteRtTensor tensor, LiteRtOp* maybe_defining_op,
    LiteRtParamIndex* maybe_defining_op_output_ind) {
  if (tensor->defining_op != nullptr) {
    *maybe_defining_op = tensor->defining_op;
    *maybe_defining_op_output_ind = tensor->defining_op_out_ind;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus GetTensorTypeId(LiteRtTensor tensor, LiteRtTensorTypeId* type_id) {
  *type_id = tensor->type_id;
  return kLiteRtStatusOk;
}

LiteRtStatus GetUrankedTensorType(
    LiteRtTensor tensor, LiteRtUnrankedTensorType* unranked_tensor_type) {
  if (tensor->type_id != kLiteRtUnrankedTensorType) {
    return kLiteRtStatusErrorInvalidIrType;
  }
  *unranked_tensor_type = tensor->type_detail.unranked_tensor_type;
  return kLiteRtStatusOk;
}

LiteRtStatus GetRankedTensorType(LiteRtTensor tensor,
                                 LiteRtRankedTensorType* ranked_tensor_type) {
  if (tensor->type_id != kLiteRtRankedTensorType) {
    return kLiteRtStatusErrorInvalidIrType;
  }
  *ranked_tensor_type = tensor->type_detail.ranked_tensor_type;
  return kLiteRtStatusOk;
}
