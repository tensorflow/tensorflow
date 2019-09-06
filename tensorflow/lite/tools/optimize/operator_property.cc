/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/optimize/operator_property.h"

namespace tflite {
namespace optimize {
namespace operator_property {
OperatorProperty GetOperatorProperty(const BuiltinOperator& op) {
  OperatorProperty property;
  switch (op) {
    case BuiltinOperator_ADD:
      property.inputs = {{0, {}}, {1, {}}};
      property.outputs = {{0, {}}};
      property.version = 2;
      break;
    case BuiltinOperator_ARG_MAX:
      property.inputs = {{0, {}}};
      // ArgMax has no quantizable output.
      property.version = 2;
      break;
    case BuiltinOperator_AVERAGE_POOL_2D:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 2;
      break;
    case BuiltinOperator_BATCH_TO_SPACE_ND:
    case BuiltinOperator_SPACE_TO_BATCH_ND:
    case BuiltinOperator_SPACE_TO_DEPTH:
      // We skip inputs 1 and 2 since they aren't real valued (they are shapes).
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 2;
      break;
    case BuiltinOperator_SPLIT:
      property.arbitrary_outputs = true;
      // We skip input 0 since it is the split dim which is not real valued.
      property.inputs = {{1, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 2;
      break;
    case BuiltinOperator_CONCATENATION:
      property.arbitrary_inputs = true;
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 2;
      break;
    case BuiltinOperator_CONV_2D: {
      TensorProperty tensor_property;
      tensor_property.per_axis = true;
      tensor_property.per_axis_index = 0;
      tensor_property.symmetric = true;
      property.inputs = {{0, {}}, {1, tensor_property}};
      property.outputs = {{0, {}}};
      property.biases = {2};
      property.version = 3;
      break;
    }
    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      TensorProperty tensor_property;
      tensor_property.per_axis = true;
      tensor_property.per_axis_index = 3;
      tensor_property.symmetric = true;
      property.inputs = {
          {0, {}},
          {1, tensor_property},
      };
      property.outputs = {{0, {}}};
      property.biases = {2};
      property.version = 3;
      break;
    }
    case BuiltinOperator_EQUAL:
    case BuiltinOperator_NOT_EQUAL:
    case BuiltinOperator_GREATER:
    case BuiltinOperator_GREATER_EQUAL:
    case BuiltinOperator_LESS:
    case BuiltinOperator_LESS_EQUAL:
      property.inputs = {{0, {}}, {1, {}}};
      // Comparisons have no quantizable outputs.
      property.version = 2;
      break;
    case BuiltinOperator_EXPAND_DIMS:
      // We skip input 1 as it is not real valued (it's the index of axis) and
      // hence does not need to be quantized.
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.version = 1;
      break;
    case BuiltinOperator_FULLY_CONNECTED: {
      TensorProperty tensor_property;
      tensor_property.symmetric = true;
      property.inputs = {{0, {}}, {1, tensor_property}};
      property.outputs = {{0, {}}};
      property.biases = {2};
      property.version = 4;
      break;
    }
    case BuiltinOperator_GATHER:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 2;
      break;
    case BuiltinOperator_HARD_SWISH: {
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.version = 1;
      break;
    }
    case BuiltinOperator_LOG_SOFTMAX: {
      property.inputs = {{0, {}}};
      // LogSoftmax requires output with 16/256 as scale and 127 as zero point.
      TensorProperty tensor_property;
      tensor_property.restriction = true;
      tensor_property.restricted_value = {16.0 / 256.0, 127};
      property.outputs = {{0, tensor_property}};
      property.version = 2;
      break;
    }
    case BuiltinOperator_LOGISTIC: {
      property.inputs = {{0, {}}};
      // Logistic requires output with 1/256 as scale and -128 as zero point.
      TensorProperty tensor_property;
      tensor_property.restriction = true;
      tensor_property.restricted_value = {1 / 256.0, -128};
      property.outputs = {{0, tensor_property}};
      property.version = 2;
      break;
    }
    case BuiltinOperator_L2_NORMALIZATION: {
      property.inputs = {{0, {}}};
      // L2 Norm requires output with 1/128 as scale and 0 as zero point.
      TensorProperty tensor_property;
      tensor_property.restriction = true;
      tensor_property.restricted_value = {1 / 128.0, 0};
      property.outputs = {{0, tensor_property}};
      property.version = 2;
      break;
    }
    case BuiltinOperator_MAX_POOL_2D:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 2;
      break;
    case BuiltinOperator_MAXIMUM:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 2;
      break;
    case BuiltinOperator_MEAN:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.version = 2;
      break;
    case BuiltinOperator_MINIMUM:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 2;
      break;
    case BuiltinOperator_MUL:
      property.inputs = {{0, {}}, {1, {}}};
      property.outputs = {{0, {}}};
      property.version = 2;
      break;
    case BuiltinOperator_PAD:
    case BuiltinOperator_PADV2:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 2;
      break;
    case BuiltinOperator_QUANTIZE:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.version = 2;
      break;
    case BuiltinOperator_RELU6: {
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.version = 2;
      break;
    }
    case BuiltinOperator_RESHAPE:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 1;
      break;
    case BuiltinOperator_RESIZE_BILINEAR:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 2;
      break;
    case BuiltinOperator_SHAPE:
      property.inputs = {{0, {}}};
      // Shape has no quantizable output.
      property.version = 1;
      break;
    case BuiltinOperator_SLICE:
      // We skip inputs 1 and 2 since they aren't real valued (they are the
      // index and size).
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 2;
      break;
    case BuiltinOperator_SQUEEZE:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 1;
      break;
    case BuiltinOperator_SOFTMAX: {
      property.inputs = {{0, {}}};
      // Softmax requires output with 1/256 as scale and -128 as zero point.
      TensorProperty tensor_property;
      tensor_property.restriction = true;
      tensor_property.restricted_value = {1 / 256.0, -128};
      property.outputs = {{0, tensor_property}};
      property.version = 2;
      break;
    }
    case BuiltinOperator_SUB:
      property.inputs = {{0, {}}, {1, {}}};
      property.outputs = {{0, {}}};
      property.version = 2;
      break;
    case BuiltinOperator_SUM:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.version = 2;
      break;
    case BuiltinOperator_TANH: {
      property.inputs = {{0, {}}};
      // Tanh requires output with 1/128 as scale and 0 as zero point.
      TensorProperty tensor_property;
      tensor_property.restriction = true;
      tensor_property.restricted_value = {1 / 128.0, 0};
      property.outputs = {{0, tensor_property}};
      property.version = 2;
      break;
    }
    case BuiltinOperator_TRANSPOSE:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 2;
      break;
    default:
      // No quantized implementation exists for this operation.
      property.quantizable = false;
  }
  return property;
}

}  // namespace operator_property
}  // namespace optimize
}  // namespace tflite
