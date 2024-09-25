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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_C_LITE_RT_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_C_LITE_RT_MODEL_H_

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/lite/core/c/c_api_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITE_RT_DEFINE_HANDLE(LrtBuffer);

LITE_RT_DEFINE_HANDLE(LrtTensor);
LITE_RT_DEFINE_HANDLE_ARRAY(LrtTensor);

LITE_RT_DEFINE_HANDLE(LrtOp);
LITE_RT_DEFINE_HANDLE_ARRAY(LrtOp);

LITE_RT_DEFINE_HANDLE(LrtSubgraph);
LITE_RT_DEFINE_HANDLE_ARRAY(LrtSubgraph);

LITE_RT_DEFINE_HANDLE(LrtModel);

// Append only list of ops.
LITE_RT_DEFINE_HANDLE(LrtOpList);

// For indexing into lrt collections or counting lrt things.
typedef uint64_t lrt_param_index_t;

//
// Tensors
//

typedef enum {
  kLrtElementTypeNone = kTfLiteNoType,
  kLrtElementTypeBool = kTfLiteBool,
  kLrtElementTypeInt4 = kTfLiteInt4,
  kLrtElementTypeInt8 = kTfLiteInt8,
  kLrtElementTypeInt16 = kTfLiteInt16,
  kLrtElementTypeInt32 = kTfLiteInt32,
  kLrtElementTypeInt64 = kTfLiteInt64,
  kLrtElementTypeUInt8 = kTfLiteUInt8,
  kLrtElementTypeUInt16 = kTfLiteUInt16,
  kLrtElementTypeUInt32 = kTfLiteUInt32,
  kLrtElementTypeUInt64 = kTfLiteUInt64,
  kLrtElementTypeFloat16 = kTfLiteFloat16,
  kLrtElementTypeBFloat16 = kTfLiteBFloat16,
  kLrtElementTypeFloat32 = kTfLiteFloat32,
  kLrtElementTypeFloat64 = kTfLiteFloat64,
  kLrtElementTypeComplex64 = kTfLiteComplex64,
  kLrtElementTypeComplex128 = kTfLiteComplex128,
  kLrtElementTypeTfResource = kTfLiteResource,
  kLrtElementTypeTfString = kTfLiteString,
  kLrtElementTypeTfVariant = kTfLiteVariant,
} LrtElementType;

typedef struct {
  uint32_t rank;
  // TODO: b/365299994 - Decide on canonical type(s) for indices({s}32/64). Also
  // representation of dynamic dim.
  const int32_t* dimensions;
} LrtLayout;

// Tensor whose rank is dynamic.
typedef struct {
  LrtElementType element_type;
} LrtUnrankedTensorType;

// Tensor whose rank is static but dimenions may be dynamic.
typedef struct {
  LrtElementType element_type;
  LrtLayout layout;
} LrtRankedTensorType;

typedef enum {
  kLrtRankedTensorType = 0,
  kLrtUnrankedTensorType = 1,
  // TODO: b/365299994 - q types.
} LrtTensorTypeId;

// Get type identifier from tensor.
LrtStatus GetTensorTypeId(LrtTensor tensor, LrtTensorTypeId* type_id);

// Get unranked tensor type info, return bad status if not unranked.
LrtStatus GetUrankedTensorType(LrtTensor tensor,
                               LrtUnrankedTensorType* unranked_tensor_type);

// Get ranked tensor type info, return bad status if not ranked.
LrtStatus GetRankedTensorType(LrtTensor tensor,
                              LrtRankedTensorType* ranked_tensor_type);

// Get opaque array from given buffer.
LrtStatus GetBufferInfo(LrtBuffer buffer, size_t* size, const void** addr);

// Get buffer associated with given tensor. All tensors have a buffer,
// null buffers have size = 0;
LrtStatus GetTensorBuffer(LrtTensor tensor, LrtBuffer* buffer);

// Get all the ops that reference given tensor, and at what operand index.
LrtStatus GetTensorUses(LrtTensor tensor, lrt_param_index_t* num_uses,
                        LrtOpArray* users, lrt_param_index_t** user_arg_inds);

// Get the op that defines this tensor and the corresponding output index. If
// tensor is a subgraph input, defining op will be null.
LrtStatus GetTensorDefiningOp(LrtTensor tensor, LrtOp* maybe_defining_op,
                              lrt_param_index_t* maybe_defining_op_output_ind);

//
// Op
//

// Get output tensors of given op.
LrtStatus GetOpOutputs(LrtOp op, lrt_param_index_t* num_outputs,
                       LrtTensorArray* output);

// Get input tensors of given op.
LrtStatus GetOpInputs(LrtOp op, lrt_param_index_t* num_inputs,
                      LrtTensorArray* inputs);

// Get code corresponding to operation type for given op.
LrtStatus GetOpCode(LrtOp op, LrtOpCode* code);

//
// Subgraph
//

// Get input tensors for given subgraph.
LrtStatus GetSubgraphInputs(LrtSubgraph subgraph, lrt_param_index_t* num_inputs,
                            LrtTensorArray* inputs);

// Get output tensors for given subgraph.
LrtStatus GetSubgraphOutputs(LrtSubgraph subgraph,
                             lrt_param_index_t* num_outputs,
                             LrtTensorArray* outputs);

// Get all ops in given subgraph in a topological order.
LrtStatus GetSubgraphOps(LrtSubgraph subgraph, lrt_param_index_t* num_ops,
                         LrtOpArray* ops);

//
// Model
//

// Get number of subgraphs in model.
LrtStatus GetModelNumSubgraphs(LrtModel model,
                               lrt_param_index_t* num_subgraphs);

// Get subgraph at given index in model.
LrtStatus GetModelSubgraph(LrtModel model, lrt_param_index_t subgraph_index,
                           LrtSubgraph* subgraph);

// Get the index of the entry subgraph.
// TODO: b/365299994 - Figure out signatures.
LrtStatus GetModelMainSubgraph(LrtModel model,
                               lrt_param_index_t* main_subgraph_index);

LrtStatus PushOp(LrtOpList op_list, LrtOp op);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_C_LITE_RT_MODEL_H_
