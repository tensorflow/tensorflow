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

#include <stdbool.h>  // NOLINT: To use bool type in C
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_layout.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//
// Handles + Common
//

// Constant data behind a tensor stored in the model.
LITERT_DEFINE_HANDLE(LiteRtWeights);

// Values/edges of the models graph.
LITERT_DEFINE_HANDLE(LiteRtTensor);

// Operations/nodes of the models graph.
LITERT_DEFINE_HANDLE(LiteRtOp);

// Fundamental block of program, i.e. a function body.
LITERT_DEFINE_HANDLE(LiteRtSubgraph);

// Signature of the model.
LITERT_DEFINE_HANDLE(LiteRtSignature);

// A collection of subgraph + metadata + signature.
LITERT_DEFINE_HANDLE(LiteRtModel);

// Append only list of ops.
LITERT_DEFINE_HANDLE(LiteRtOpList);

// For indexing into litert collections or counting litert things.
typedef size_t LiteRtParamIndex;

//
// LiteRtTensor + Types
//

// Get the string name associated with this tensor. This is an optional
// attribute and if not set will return a zero-length string.
LiteRtStatus LiteRtGetTensorName(LiteRtTensor tensor, const char** name);

// TENSOR TYPES

// Primitive types for elements in a tensor.
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

// Tensor whose rank is dynamic.
typedef struct {
  // The primitive element type of the constituent data.
  LiteRtElementType element_type;
} LiteRtUnrankedTensorType;

// Tensor whose rank is static but dimenions may be dynamic.
typedef struct {
  // The primitive element type of the constituent data.
  LiteRtElementType element_type;

  // Shape information.
  LiteRtLayout layout;
} LiteRtRankedTensorType;

// The identifier for tensor type union.
typedef enum {
  // Type with fix ranked and possibly dynamic dimensions.
  kLiteRtRankedTensorType = 0,

  // Type with dynamic rank.
  kLiteRtUnrankedTensorType = 1,
} LiteRtTensorTypeId;

// Get type identifier from tensor.
LiteRtStatus LiteRtGetTensorTypeId(LiteRtTensor tensor,
                                   LiteRtTensorTypeId* type_id);

// Get unranked tensor type info, return bad status if not unranked.
LiteRtStatus LiteRtGetUnrankedTensorType(
    LiteRtTensor tensor, LiteRtUnrankedTensorType* unranked_tensor_type);

// Get ranked tensor type info, return bad status if not ranked.
LiteRtStatus LiteRtGetRankedTensorType(
    LiteRtTensor tensor, LiteRtRankedTensorType* ranked_tensor_type);

// QUANTIZATION

// Schema for tensors quantized with one set of q-params.
typedef struct {
  // Scaling factor.
  float scale;

  // The value that float:0 maps to in q-space.
  int64_t zero_point;
} LiteRtQuantizationPerTensor;

// Schema for tensors quantized with one set of q-params per channel.
typedef struct {
  int32_t quantized_dimension;
  uint64_t num_channels;
  float* scales;
  int64_t* zero_points;
} LiteRtQuantizationPerChannel;

// The identifier for quantization scheme type union.
typedef enum {
  // Tag for tensors without quantization.
  kLiteRtQuantizationNone = 0,

  // Basic quantization, one set of q-params per tensor.
  kLiteRtQuantizationPerTensor = 1,

  // [NOT IMPLEMENTED YET] Q-params for each element accross a single dimension.
  kLiteRtQuantizationPerChannel = 2,

  // [NOT IMPLEMENTED YET] Q-params accross blocks of fixed size (e.g. 2048).
  kLiteRtQuantizationBlockWise = 3,
} LiteRtQuantizationTypeId;

// Get the identifier for the type of quantization for a given tensor.
LiteRtStatus LiteRtGetQuantizationTypeId(LiteRtTensor tensor,
                                         LiteRtQuantizationTypeId* q_type_id);

// Get the per-tensor quantization information for a given tensor if it has it.
LiteRtStatus LiteRtGetPerTensorQuantization(
    LiteRtTensor tensor, LiteRtQuantizationPerTensor* per_tensor_quantization);

// Get the per-channel quantization information for a given tensor if it has it.
LiteRtStatus LiteRtGetPerChannelQuantization(
    LiteRtTensor tensor,
    LiteRtQuantizationPerChannel* per_channel_quantization);

// EDGES

// Information about the about that defines a tensor.
typedef struct LiteRtTensorDefiningOp {
  // The defining op itself.
  LiteRtOp op;

  // The op output index that defines the specific tensor.
  LiteRtParamIndex op_output_index;
} LiteRtTensorDefiningOp;

// Information about a reference to a tensor in the graph.
typedef struct LiteRtTensorUserOp {
  // The referring op itself.
  LiteRtOp op;

  // Index of which operand the op refers to a specific tensor on.
  LiteRtParamIndex op_input_index;
} LiteRtTensorUserOp;

// Get all the ops that reference given tensor, and at what operand index.
LiteRtStatus LiteRtGetNumTensorUses(LiteRtTensor tensor,
                                    LiteRtParamIndex* num_uses);
LiteRtStatus LiteRtGetTensorUse(LiteRtTensor tensor, LiteRtParamIndex use_index,
                                LiteRtOp* user,
                                LiteRtParamIndex* user_arg_index);

// Get the op that defines this tensor and the corresponding output index. If
// tensor is a subgraph input, has_defining_op will be false.
LiteRtStatus LiteRtGetTensorDefiningOp(LiteRtTensor tensor,
                                       bool* has_defining_op,
                                       LiteRtTensorDefiningOp* defining_op);

// WEIGHTS (constant data)

// Get static weights associated with a given tensor. All tensors have weights,
// null weights have size = 0;
LiteRtStatus LiteRtGetTensorWeights(LiteRtTensor tensor,
                                    LiteRtWeights* weights);

//
// LiteRtWeights
//

// Get opaque array from given tensor weights.
LiteRtStatus LiteRtGetWeightsBytes(LiteRtWeights weights, const void** addr,
                                   size_t* size);

//
// LiteRtOp
//

// Get code corresponding to operation type for given op.
LiteRtStatus LiteRtGetOpCode(LiteRtOp op, LiteRtOpCode* code);

// Get input tensors of given op.
LiteRtStatus LiteRtGetNumOpInputs(LiteRtOp op, LiteRtParamIndex* num_inputs);
LiteRtStatus LiteRtGetOpInput(LiteRtOp op, LiteRtParamIndex input_index,
                              LiteRtTensor* input);

// Get output tensors of given op.
LiteRtStatus LiteRtGetNumOpOutputs(LiteRtOp op, LiteRtParamIndex* num_outputs);
LiteRtStatus LiteRtGetOpOutput(LiteRtOp op, LiteRtParamIndex output_index,
                               LiteRtTensor* output);

//
// LiteRtSubgraph
//

// Get input tensors for given subgraph.
LiteRtStatus LiteRtGetNumSubgraphInputs(LiteRtSubgraph subgraph,
                                        LiteRtParamIndex* num_inputs);
LiteRtStatus LiteRtGetSubgraphInput(LiteRtSubgraph subgraph,
                                    LiteRtParamIndex input_index,
                                    LiteRtTensor* input);

// Get output tensors for given subgraph.
LiteRtStatus LiteRtGetNumSubgraphOutputs(LiteRtSubgraph subgraph,
                                         LiteRtParamIndex* num_outputs);
LiteRtStatus LiteRtGetSubgraphOutput(LiteRtSubgraph subgraph,
                                     LiteRtParamIndex output_index,
                                     LiteRtTensor* output);

// Get all ops in given subgraph in a topological order.
LiteRtStatus LiteRtGetNumSubgraphOps(LiteRtSubgraph subgraph,
                                     LiteRtParamIndex* num_ops);
LiteRtStatus LiteRtGetSubgraphOp(LiteRtSubgraph subgraph,
                                 LiteRtParamIndex op_index, LiteRtOp* op);

//
// LiteRtSignature
//

// Default signature key. This is the key that is used if the model does not
// define any signatures.
LiteRtStatus LiteRtGetDefaultSignatureKey(const char** signature_key);

// Get the signature key string defined in the model.
LiteRtStatus LiteRtGetSignatureKey(LiteRtSignature signature,
                                   const char** signature_key);

// Get the associated subgraph for the given signature.
LiteRtStatus LiteRtGetSignatureSubgraph(LiteRtSignature signature,
                                        LiteRtSubgraph* subgraph);

// Get the number of inputs for the given signature.
LiteRtStatus LiteRtGetNumSignatureInputs(LiteRtSignature signature,
                                         LiteRtParamIndex* num_inputs);

// Get the name of the i-th of input tensor name for the given signature.
LiteRtStatus LiteRtGetSignatureInputName(LiteRtSignature signature,
                                         LiteRtParamIndex input_idx,
                                         const char** input_name);

// Get the number of outputs for the given signature.
LiteRtStatus LiteRtGetNumSignatureOutputs(LiteRtSignature signature,
                                          LiteRtParamIndex* num_outputs);

// Get the name of the i-th of output tensor name for the given signature.
LiteRtStatus LiteRtGetSignatureOutputName(LiteRtSignature signature,
                                          LiteRtParamIndex output_idx,
                                          const char** output_name);

//
// LiteRtModel
//

LiteRtStatus LiteRtCreateModelFromFile(const char* filename,
                                       LiteRtModel* model);

LiteRtStatus LiteRtCreateModelFromBuffer(const void* buffer_addr,
                                         size_t buffer_size,
                                         LiteRtModel* model);

// Get the metadata buffer associated with given key if it exists.
LiteRtStatus LiteRtGetModelMetadata(LiteRtModel model, const char* metadata_key,
                                    const void** metadata_buffer,
                                    size_t* metadata_buffer_size);

// Get the index of the entry subgraph.
// TODO: b/365299994 - Figure out signatures.
LiteRtStatus LiteRtGetMainModelSubgraphIndex(
    LiteRtModel model, LiteRtParamIndex* main_subgraph_index);

// Get number of subgraphs in model.
LiteRtStatus LiteRtGetNumModelSubgraphs(LiteRtModel model,
                                        LiteRtParamIndex* num_subgraphs);

// Get subgraph at given index in model.
LiteRtStatus LiteRtGetModelSubgraph(LiteRtModel model,
                                    LiteRtParamIndex subgraph_index,
                                    LiteRtSubgraph* subgraph);

// Get the number of signatures defined in the model.
LiteRtStatus LiteRtGetNumModelSignatures(LiteRtModel model,
                                         LiteRtParamIndex* num_signatures);

// Get the signature at the given index in the model
LiteRtStatus LiteRtGetModelSignature(LiteRtModel model,
                                     LiteRtParamIndex signature_index,
                                     LiteRtSignature* signature);

// Destroy the given model, freeing any memory it owns.
void LiteRtDestroyModel(LiteRtModel model);

//
// Utility Types
//

// An append only list of ops.
LiteRtStatus LiteRtPushOp(LiteRtOpList op_list, LiteRtOp op);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_MODEL_H_
