/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_YNNPACK_UTILS_H_
#define TENSORFLOW_LITE_DELEGATES_YNNPACK_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"

#define TF_LITE_ENSURE_YNN_STATUS(x) \
  TF_LITE_ENSURE(context, (x) == ynn_status_success)

namespace tflite {
namespace ynnpack {

using TensorToValueIdMap = absl::flat_hash_map<int, uint32_t>;

struct NodeInfo {
  int node_index;
  int builtin_code;
  std::vector<int> inputs;
  std::vector<int> outputs;
  TfLiteFusedActivation activation;
};

// Generic helpers
ynn_type GetYnnType(TfLiteType type);
size_t YnnTypeElementCount(ynn_type type);
ynn_unary_operator GetYnnUnaryOperator(int builtin_code);
ynn_binary_operator GetYnnBinaryOperator(int builtin_code);
ynn_reduce_operator GetYnnReduceOperator(int builtin_code);
bool IsUnaryOp(int builtin_code);
bool IsBinaryOp(int builtin_code);
bool IsStablehloOp(int builtin_code);
bool IsQuantized(const TfLiteTensor& tensor);
bool IsSupportedQuantization(const TfLiteTensor& tensor,
                             bool allow_per_channel = false);
bool IsTensorSupported(const TfLiteTensor& tensor,
                       bool allow_per_channel = false);
bool QuantizationParamsEqual(const TfLiteTensor& tensor1,
                             const TfLiteTensor& tensor2);
bool IsActivationSupported(TfLiteFusedActivation activation,
                           TfLiteType output_type);
TfLiteFusedActivation GetFusedActivation(const TfLiteRegistration* registration,
                                         const TfLiteNode* node);

TfLiteStatus GetTfLiteTensorValueAsDouble(TfLiteContext* context,
                                          const TfLiteTensor& tensor, int index,
                                          double* value);

// YNNPACK graph building helpers
TfLiteStatus DefineYnnStencil(TfLiteContext* context, ynn_subgraph_t subgraph,
                              const TfLiteTensor& input_tensor,
                              uint32_t input_id, size_t filter_height,
                              size_t filter_width, size_t stride_height,
                              size_t stride_width, size_t dilation_height,
                              size_t dilation_width, TfLitePadding padding,
                              float padding_value, uint32_t* stencil_id);

TfLiteStatus DefineScalarConstant(TfLiteContext* context,
                                  ynn_subgraph_t subgraph, ynn_type type,
                                  double value, uint32_t* id_out);

TfLiteStatus DefineQuantizationParams(TfLiteContext* context,
                                      ynn_subgraph_t subgraph,
                                      const TfLiteTensor& tensor,
                                      uint32_t* scale_id, uint32_t* zp_id,
                                      int32_t zp_offset = 0);

TfLiteStatus ApplyActivation(TfLiteContext* context, ynn_subgraph_t subgraph,
                             TfLiteFusedActivation activation,
                             uint32_t input_id, uint32_t& output_id,
                             int output_tensor_index, ynn_type internal_type);

TfLiteStatus ApplyClamp(TfLiteContext* context, ynn_subgraph_t subgraph,
                        double min_val, double max_val, uint32_t input_id,
                        uint32_t& output_id, int output_tensor_index,
                        ynn_type internal_type);

TfLiteStatus DequantizeIfNeeded(TfLiteContext* context, ynn_subgraph_t subgraph,
                                TensorToValueIdMap& tensor_to_value_id,
                                int tensor_index, uint32_t val_id,
                                uint32_t* float_val_id);

TfLiteStatus Quantize(TfLiteContext* context, ynn_subgraph_t subgraph,
                      TensorToValueIdMap& tensor_to_value_id, int tensor_index,
                      uint32_t float_val_id, uint32_t quant_val_id);

uint32_t GetOrCreateValueId(TfLiteContext* context, ynn_subgraph_t subgraph,
                            TensorToValueIdMap& mapping, int tensor_index);

TfLiteStatus ImplementMutualBroadcasting(TfLiteContext* context,
                                         ynn_subgraph_t subgraph, int rank_a,
                                         int rank_b, int exclude_a,
                                         int exclude_b, uint32_t& current_a_id,
                                         uint32_t& current_b_id);

// Template helper for decomposed unary nodes
template <typename F>
TfLiteStatus DefineDecomposedUnaryNode(TfLiteContext* context,
                                       ynn_subgraph_t subgraph,
                                       TensorToValueIdMap& tensor_to_value_id,
                                       const NodeInfo& node,
                                       F&& define_float_subgraph) {
  TF_LITE_ENSURE_EQ(context, node.inputs.size(), 1);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  int input_tensor_index = node.inputs[0];
  int output_tensor_index = node.outputs[0];

  uint32_t input_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_tensor_index);
  TF_LITE_ENSURE(context, input_val_id != YNN_INVALID_VALUE_ID);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];
  bool is_output_quantized = IsQuantized(output_tensor);
  TfLiteFusedActivation activation = node.activation;

  uint32_t float_input_val_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DequantizeIfNeeded(
      context, subgraph, tensor_to_value_id, input_tensor_index, input_val_id,
      &float_input_val_id));

  ynn_type internal_type = IsQuantized(output_tensor)
                               ? ynn_type_fp32
                               : GetYnnType(output_tensor.type);
  uint32_t float_output_val_id = YNN_INVALID_VALUE_ID;

  if (!is_output_quantized && activation == kTfLiteActNone) {
    float_output_val_id = output_val_id;
  }

  TF_LITE_ENSURE_STATUS(
      define_float_subgraph(float_input_val_id, float_output_val_id));

  uint32_t active_output_val_id = float_output_val_id;

  if (activation != kTfLiteActNone) {
    uint32_t activation_output_val_id = output_val_id;
    if (is_output_quantized) {
      activation_output_val_id = YNN_INVALID_VALUE_ID;
    }
    TF_LITE_ENSURE_STATUS(ApplyActivation(
        context, subgraph, activation, float_output_val_id,
        activation_output_val_id, output_tensor_index, internal_type));
    active_output_val_id = activation_output_val_id;
  }

  if (is_output_quantized) {
    TF_LITE_ENSURE_STATUS(Quantize(context, subgraph, tensor_to_value_id,
                                   output_tensor_index, active_output_val_id,
                                   output_val_id));
  }

  return kTfLiteOk;
}

}  // namespace ynnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_YNNPACK_UTILS_H_
