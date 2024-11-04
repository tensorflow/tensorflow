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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_MODEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_MODEL_H_

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LiteRtWeights);

LITERT_DEFINE_HANDLE(LiteRtTensor);
LITERT_DEFINE_HANDLE_ARRAY(LiteRtTensor);

LITERT_DEFINE_HANDLE(LiteRtOp);
LITERT_DEFINE_HANDLE_ARRAY(LiteRtOp);

LITERT_DEFINE_HANDLE(LiteRtSubgraph);
LITERT_DEFINE_HANDLE_ARRAY(LiteRtSubgraph);

LITERT_DEFINE_HANDLE(LiteRtModel);

// Append only list of ops.
LITERT_DEFINE_HANDLE(LiteRtOpList);

// For indexing into litert collections or counting litert things.
typedef uint64_t LiteRtParamIndex;

//
// Tensors
//

typedef enum {
  kLiteRtElementTypeNone = kTfLiteNoType,
  kLiteRtElementTypeBool = kTfLiteBool,
  kLiteRtElementTypeInt4 = kTfLiteInt4,
  kLiteRtElementTypeInt8 = kTfLiteInt8,
  kLiteRtElementTypeInt16 = kTfLiteInt16,
  kLiteRtElementTypeInt32 = kTfLiteInt32,
  kLiteRtElementTypeInt64 = kTfLiteInt64,
  kLiteRtElementTypeUInt8 = kTfLiteUInt8,
  kLiteRtElementTypeUInt16 = kTfLiteUInt16,
  kLiteRtElementTypeUInt32 = kTfLiteUInt32,
  kLiteRtElementTypeUInt64 = kTfLiteUInt64,
  kLiteRtElementTypeFloat16 = kTfLiteFloat16,
  kLiteRtElementTypeBFloat16 = kTfLiteBFloat16,
  kLiteRtElementTypeFloat32 = kTfLiteFloat32,
  kLiteRtElementTypeFloat64 = kTfLiteFloat64,
  kLiteRtElementTypeComplex64 = kTfLiteComplex64,
  kLiteRtElementTypeComplex128 = kTfLiteComplex128,
  kLiteRtElementTypeTfResource = kTfLiteResource,
  kLiteRtElementTypeTfString = kTfLiteString,
  kLiteRtElementTypeTfVariant = kTfLiteVariant,
} LiteRtElementType;

typedef struct {
  uint32_t rank;
  // TODO: b/365299994 - Decide on canonical type(s) for indices({s}32/64). Also
  // representation of dynamic dim.
  const int32_t* dimensions;
  // Strides for a nomimal NWHC layout. NULL if unused.
  const uint32_t* strides;
} LiteRtLayout;

// Tensor whose rank is dynamic.
typedef struct {
  LiteRtElementType element_type;
} LiteRtUnrankedTensorType;

// Tensor whose rank is static but dimenions may be dynamic.
typedef struct {
  LiteRtElementType element_type;
  LiteRtLayout layout;
} LiteRtRankedTensorType;

typedef enum {
  kLiteRtRankedTensorType = 0,
  kLiteRtUnrankedTensorType = 1,
  // TODO: b/365299994 - q types.
} LiteRtTensorTypeId;

// Get type identifier from tensor.
LiteRtStatus GetTensorTypeId(LiteRtTensor tensor, LiteRtTensorTypeId* type_id);

// Get unranked tensor type info, return bad status if not unranked.
LiteRtStatus GetUrankedTensorType(
    LiteRtTensor tensor, LiteRtUnrankedTensorType* unranked_tensor_type);

// Get ranked tensor type info, return bad status if not ranked.
LiteRtStatus GetRankedTensorType(LiteRtTensor tensor,
                                 LiteRtRankedTensorType* ranked_tensor_type);

// Get opaque array from given tensor weights.
LiteRtStatus GetWeightsInfo(LiteRtWeights weights, size_t* size,
                            const void** addr);

// Get static weights associated with a given tensor. All tensors have weights,
// null weights have size = 0;
LiteRtStatus GetTensorWeights(LiteRtTensor tensor, LiteRtWeights* weights);

// Get all the ops that reference given tensor, and at what operand index.
LiteRtStatus GetTensorUses(LiteRtTensor tensor, LiteRtParamIndex* num_uses,
                           LiteRtOpArray* users,
                           LiteRtParamIndex** user_arg_inds);

// Get the op that defines this tensor and the corresponding output index. If
// tensor is a subgraph input, defining op will be null.
LiteRtStatus GetTensorDefiningOp(
    LiteRtTensor tensor, LiteRtOp* maybe_defining_op,
    LiteRtParamIndex* maybe_defining_op_output_ind);

//
// Op
//

// Get output tensors of given op.
LiteRtStatus GetOpOutputs(LiteRtOp op, LiteRtParamIndex* num_outputs,
                          LiteRtTensorArray* output);

// Get input tensors of given op.
LiteRtStatus GetOpInputs(LiteRtOp op, LiteRtParamIndex* num_inputs,
                         LiteRtTensorArray* inputs);

// Get code corresponding to operation type for given op.
LiteRtStatus GetOpCode(LiteRtOp op, LiteRtOpCode* code);

//
// Subgraph
//

// Get input tensors for given subgraph.
LiteRtStatus GetSubgraphInputs(LiteRtSubgraph subgraph,
                               LiteRtParamIndex* num_inputs,
                               LiteRtTensorArray* inputs);

// Get output tensors for given subgraph.
LiteRtStatus GetSubgraphOutputs(LiteRtSubgraph subgraph,
                                LiteRtParamIndex* num_outputs,
                                LiteRtTensorArray* outputs);

// Get all ops in given subgraph in a topological order.
LiteRtStatus GetSubgraphOps(LiteRtSubgraph subgraph, LiteRtParamIndex* num_ops,
                            LiteRtOpArray* ops);

//
// Model
//

// Get the metadata buffer associated with given key if it exists.
LiteRtStatus LiteRtModelGetMetadata(LiteRtModel model, const char* metadata_key,
                                    const void** metadata_buffer,
                                    size_t* metadata_buffer_size);

// Get number of subgraphs in model.
LiteRtStatus GetModelNumSubgraphs(LiteRtModel model,
                                  LiteRtParamIndex* num_subgraphs);

// Get subgraph at given index in model.
LiteRtStatus GetModelSubgraph(LiteRtModel model,
                              LiteRtParamIndex subgraph_index,
                              LiteRtSubgraph* subgraph);

// Get the index of the entry subgraph.
// TODO: b/365299994 - Figure out signatures.
LiteRtStatus GetModelMainSubgraph(LiteRtModel model,
                                  LiteRtParamIndex* main_subgraph_index);

//
// Utility Types
//

LiteRtStatus PushOp(LiteRtOpList op_list, LiteRtOp op);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_MODEL_H_
