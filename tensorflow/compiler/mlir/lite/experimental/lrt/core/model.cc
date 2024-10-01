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

#include "tensorflow/compiler/mlir/lite/experimental/lrt/core/model.h"

#include <cstddef>

#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_op_code.h"

//
// Model
//

LrtStatus GetModelNumSubgraphs(LrtModel model,
                               lrt_param_index_t* num_subgraphs) {
  *num_subgraphs = model->subgraphs.size();
  return StatusOk();
}

LrtStatus GetModelSubgraph(LrtModel model, lrt_param_index_t subgraph_index,
                           LrtSubgraph* subgraph) {
  if (subgraph_index >= model->subgraphs.size()) {
    return StatusCreate(kLrtParamIndexOOB);
  }
  *subgraph = model->subgraphs.data() + subgraph_index;
  return StatusOk();
}

LrtStatus GetModelMainSubgraph(LrtModel model,
                               lrt_param_index_t* main_subgraph_index) {
  // TODO replace this with signature.
  *main_subgraph_index = 0;
  return StatusOk();
}

void ModelDestroy(LrtModel model) { delete model; }

LrtStatus PushOp(LrtOpList op_list, LrtOp op) {
  op_list->ops.push_back(op);
  return StatusOk();
}

//
// Subgraph
//

LrtStatus GetSubgraphInputs(LrtSubgraph subgraph, lrt_param_index_t* num_inputs,
                            LrtTensorArray* inputs) {
  *num_inputs = subgraph->inputs.size();
  *inputs = subgraph->inputs.data();
  return StatusOk();
}

LrtStatus GetSubgraphOutputs(LrtSubgraph subgraph,
                             lrt_param_index_t* num_outputs,
                             LrtTensorArray* outputs) {
  *num_outputs = subgraph->outputs.size();
  *outputs = subgraph->outputs.data();
  return StatusOk();
}

LrtStatus GetSubgraphOps(LrtSubgraph subgraph, lrt_param_index_t* num_ops,
                         LrtOpArray* ops) {
  *num_ops = subgraph->ops.size();
  *ops = subgraph->ops.data();
  return StatusOk();
}

//
// Op
//

LrtStatus GetOpOutputs(LrtOp op, lrt_param_index_t* num_outputs,
                       LrtTensorArray* outputs) {
  *num_outputs = op->outputs.size();
  *outputs = op->outputs.data();
  return StatusOk();
}

LrtStatus GetOpInputs(LrtOp op, lrt_param_index_t* num_inputs,
                      LrtTensorArray* inputs) {
  *num_inputs = op->inputs.size();
  *inputs = op->inputs.data();
  return StatusOk();
}

LrtStatus GetOpCode(LrtOp op, LrtOpCode* code) {
  *code = op->op_code;
  return StatusOk();
}

//
// Tensor
//

LrtStatus GetBufferInfo(LrtBuffer buffer, size_t* size, const void** addr) {
  if (buffer->fb_buffer == nullptr) {
    *size = 0;
    *addr = nullptr;
  } else {
    *size = buffer->fb_buffer->data.size();
    *addr = buffer->fb_buffer->data.data();
  }
  return StatusOk();
}

LrtStatus GetTensorBuffer(LrtTensor tensor, LrtBuffer* buffer) {
  *buffer = &tensor->buffer;
  return StatusOk();
}

LrtStatus GetTensorUses(LrtTensor tensor, lrt_param_index_t* num_uses,
                        LrtOpArray* use_users,
                        lrt_param_index_t** use_user_arg_inds) {
  *num_uses = tensor->users.size();
  *use_users = tensor->users.data();
  *use_user_arg_inds = tensor->user_arg_inds.data();
  return StatusOk();
}

// Null if subgraph input or constant.
LrtStatus GetTensorDefiningOp(LrtTensor tensor, LrtOp* maybe_defining_op,
                              lrt_param_index_t* maybe_defining_op_output_ind) {
  if (tensor->defining_op != nullptr) {
    *maybe_defining_op = tensor->defining_op;
    *maybe_defining_op_output_ind = tensor->defining_op_out_ind;
  }
  return StatusOk();
}

LrtStatus GetTensorTypeId(LrtTensor tensor, LrtTensorTypeId* type_id) {
  *type_id = tensor->type_id;
  return StatusOk();
}

LrtStatus GetUrankedTensorType(LrtTensor tensor,
                               LrtUnrankedTensorType* unranked_tensor_type) {
  if (tensor->type_id != kLrtUnrankedTensorType) {
    return StatusCreate(kLrtStatusBadTensorType);
  }
  *unranked_tensor_type = tensor->type_detail.unranked_tensor_type;
  return StatusOk();
}

LrtStatus GetRankedTensorType(LrtTensor tensor,
                              LrtRankedTensorType* ranked_tensor_type) {
  if (tensor->type_id != kLrtRankedTensorType) {
    return StatusCreate(kLrtStatusBadTensorType);
  }
  *ranked_tensor_type = tensor->type_detail.ranked_tensor_type;
  return StatusOk();
}
