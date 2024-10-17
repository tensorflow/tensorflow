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
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/schema/schema_generated.h"

//
// Model
//

LrtStatus GetModelNumSubgraphs(LrtModel model,
                               lrt_param_index_t* num_subgraphs) {
  *num_subgraphs = model->subgraphs.size();
  return kLrtStatusOk;
}

LrtStatus GetModelSubgraph(LrtModel model, lrt_param_index_t subgraph_index,
                           LrtSubgraph* subgraph) {
  if (subgraph_index >= model->subgraphs.size()) {
    return kLrtStatusErrorIndexOOB;
  }
  *subgraph = model->subgraphs.data() + subgraph_index;
  return kLrtStatusOk;
}

LrtStatus GetModelMainSubgraph(LrtModel model,
                               lrt_param_index_t* main_subgraph_index) {
  // TODO replace this with signature.
  *main_subgraph_index = 0;
  return kLrtStatusOk;
}

LrtStatus LiteRtModelGetMetadata(LrtModel model, const char* metadata_key,
                                 const void** metadata_buffer,
                                 size_t* metadata_buffer_size) {
  LRT_ASSIGN_OR_RETURN_STATUS(auto m_buffer_view,
                              model->FindMetadata(metadata_key));
  *metadata_buffer = m_buffer_view.data();
  *metadata_buffer_size = m_buffer_view.size();
  return kLrtStatusOk;
}

void ModelDestroy(LrtModel model) {
  if (model != nullptr) {
    delete model;
  }
}

LrtStatus PushOp(LrtOpList op_list, LrtOp op) {
  op_list->Push(op);
  return kLrtStatusOk;
}

LrtResult<FbBufferT> LrtModelT::FindMetadata(
    const absl::string_view key) const {
  using ResT = LrtResult<FbBufferT>;

  tflite::MetadataT* fb_metadata = nullptr;
  for (auto& m : flatbuffer_model->metadata) {
    if (m->name == key) {
      fb_metadata = m.get();
      break;
    }
  }
  if (fb_metadata == nullptr) {
    return ResT::FromStatus(kLrtStatusErrorNotFound);
  }

  const uint32_t m_buffer_idx = fb_metadata->buffer;
  if (m_buffer_idx >= flatbuffer_model->buffers.size()) {
    return ResT::FromStatus(kLrtStatusErrorIndexOOB);
  }
  tflite::BufferT* m_buffer = flatbuffer_model->buffers.at(m_buffer_idx).get();

  return ResT::FromValue(
      absl::MakeSpan(m_buffer->data.data(), m_buffer->data.size()));
}

//
// Subgraph
//

LrtStatus GetSubgraphInputs(LrtSubgraph subgraph, lrt_param_index_t* num_inputs,
                            LrtTensorArray* inputs) {
  *num_inputs = subgraph->inputs.size();
  *inputs = subgraph->inputs.data();
  return kLrtStatusOk;
}

LrtStatus GetSubgraphOutputs(LrtSubgraph subgraph,
                             lrt_param_index_t* num_outputs,
                             LrtTensorArray* outputs) {
  *num_outputs = subgraph->outputs.size();
  *outputs = subgraph->outputs.data();
  return kLrtStatusOk;
}

LrtStatus GetSubgraphOps(LrtSubgraph subgraph, lrt_param_index_t* num_ops,
                         LrtOpArray* ops) {
  *num_ops = subgraph->ops.size();
  *ops = subgraph->ops.data();
  return kLrtStatusOk;
}

//
// Op
//

LrtStatus GetOpOutputs(LrtOp op, lrt_param_index_t* num_outputs,
                       LrtTensorArray* outputs) {
  *num_outputs = op->outputs.size();
  *outputs = op->outputs.data();
  return kLrtStatusOk;
}

LrtStatus GetOpInputs(LrtOp op, lrt_param_index_t* num_inputs,
                      LrtTensorArray* inputs) {
  *num_inputs = op->inputs.size();
  *inputs = op->inputs.data();
  return kLrtStatusOk;
}

LrtStatus GetOpCode(LrtOp op, LrtOpCode* code) {
  *code = op->op_code;
  return kLrtStatusOk;
}

//
// Tensor
//

LrtStatus GetWeightsInfo(LrtWeights weights, size_t* size, const void** addr) {
  if (weights->fb_buffer == nullptr) {
    *size = 0;
    *addr = nullptr;
  } else {
    *size = weights->fb_buffer->data.size();
    *addr = weights->fb_buffer->data.data();
  }
  return kLrtStatusOk;
}

LrtStatus GetTensorWeights(LrtTensor tensor, LrtWeights* weights) {
  *weights = &tensor->weights;
  return kLrtStatusOk;
}

LrtStatus GetTensorUses(LrtTensor tensor, lrt_param_index_t* num_uses,
                        LrtOpArray* use_users,
                        lrt_param_index_t** use_user_arg_inds) {
  *num_uses = tensor->users.size();
  *use_users = tensor->users.data();
  *use_user_arg_inds = tensor->user_arg_inds.data();
  return kLrtStatusOk;
}

// Null if subgraph input or constant.
LrtStatus GetTensorDefiningOp(LrtTensor tensor, LrtOp* maybe_defining_op,
                              lrt_param_index_t* maybe_defining_op_output_ind) {
  if (tensor->defining_op != nullptr) {
    *maybe_defining_op = tensor->defining_op;
    *maybe_defining_op_output_ind = tensor->defining_op_out_ind;
  }
  return kLrtStatusOk;
}

LrtStatus GetTensorTypeId(LrtTensor tensor, LrtTensorTypeId* type_id) {
  *type_id = tensor->type_id;
  return kLrtStatusOk;
}

LrtStatus GetUrankedTensorType(LrtTensor tensor,
                               LrtUnrankedTensorType* unranked_tensor_type) {
  if (tensor->type_id != kLrtUnrankedTensorType) {
    return kLrtStatusErrorInvalidIrType;
  }
  *unranked_tensor_type = tensor->type_detail.unranked_tensor_type;
  return kLrtStatusOk;
}

LrtStatus GetRankedTensorType(LrtTensor tensor,
                              LrtRankedTensorType* ranked_tensor_type) {
  if (tensor->type_id != kLrtRankedTensorType) {
    return kLrtStatusErrorInvalidIrType;
  }
  *ranked_tensor_type = tensor->type_detail.ranked_tensor_type;
  return kLrtStatusOk;
}
