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

#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace operator_property {

namespace {

// The op as well as it variants.
// TODO(jianlijianli): extend it to support ops that has multiple variants.
struct OpVariant {
  BuiltinOperator op_code;
};

const OpVariant GetOperatorVariant(const ModelT* model, int subgraph_index,
                                   int op_index) {
  OpVariant op_variant;
  OperatorT* op =
      model->subgraphs.at(subgraph_index)->operators[op_index].get();
  op_variant.op_code = model->operator_codes[op->opcode_index]->builtin_code;
  return op_variant;
}
}  // namespace

OperatorProperty GetOperatorProperty(const ModelT* model, int subgraph_index,
                                     int op_index) {
  OpVariant op_variant = GetOperatorVariant(model, subgraph_index, op_index);
  BuiltinOperator op_code = op_variant.op_code;
  OperatorProperty property;
  switch (op_code) {
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
      // We skip input 0 since it is the split dim which is not real valued.
      property.inputs = {{1, {}}};
      property.arbitrary_outputs = true;
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
    case BuiltinOperator_TRANSPOSE_CONV: {
      TensorProperty tensor_property;
      tensor_property.per_axis = true;
      tensor_property.per_axis_index = 0;
      tensor_property.symmetric = true;
      property.inputs = {{1, tensor_property}, {2, {}}};
      property.outputs = {{0, {}}};
      property.version = 2;
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
    case BuiltinOperator_LSTM: {
      // TODO(jianlijianli): extend LSTM op spec to inlucde input, bias etc.
      // TODO(jianlijianli): extend this to other variants of LSTM.
      // LSTM needs 5 intermediate tensors. This agrees with the fully quantized
      // kernels in lstm_eval.cc
      static const float alpha = static_cast<float>(std::pow(2, -10));

      TensorProperty tensor_property_12;
      tensor_property_12.use_derived_scale = true;
      tensor_property_12.number_of_bits = 32;
      tensor_property_12.derived_scale = {{20}, {}, {alpha}};
      TensorProperty tensor_property_13;
      tensor_property_13.use_derived_scale = true;
      tensor_property_13.number_of_bits = 32;
      tensor_property_13.derived_scale = {{21}, {}, {alpha}};
      TensorProperty tensor_property_14;
      tensor_property_14.use_derived_scale = true;
      tensor_property_14.number_of_bits = 32;
      tensor_property_14.derived_scale = {{22}, {}, {alpha}};
      TensorProperty tensor_property_15;
      tensor_property_15.use_derived_scale = true;
      tensor_property_15.number_of_bits = 32;
      tensor_property_15.derived_scale = {{23}, {}, {alpha}};
      TensorProperty tensor_property_17;
      tensor_property_17.use_derived_scale = true;
      tensor_property_17.number_of_bits = 32;
      tensor_property_17.derived_scale = {{16}, {4}, {}};
      TensorProperty tensor_property_19;
      tensor_property_19.extend_to_power_of_two = true;
      tensor_property_19.number_of_bits = 16;
      tensor_property_19.state_tensor = true;
      tensor_property_19.symmetric = true;
      TensorProperty tensor_property_20;
      tensor_property_20.number_of_bits = 16;
      tensor_property_20.symmetric = true;

      property.inputs = {
          {0, {}},
          {1, {}},
          {2, {}},
          {3, {}},
          {4, {}},
          {5, {}},
          {6, {}},
          {7, {}},
          {8, {}},
          {9, {}},
          {10, {}},
          {11, {}},
          {16, {}},
          {19, tensor_property_19},
          {20, tensor_property_20},
          {21, tensor_property_20},
          {22, tensor_property_20},
          {23, tensor_property_20},
          {12, tensor_property_12},
          {13, tensor_property_13},
          {14, tensor_property_14},
          {15, tensor_property_15},
          {17, tensor_property_17},
      };
      property.outputs = {{0, {}}};
      property.intermediates = {
          {0, tensor_property_20},
          {1, tensor_property_20},
          {2, tensor_property_20},
          {3, tensor_property_20},
          {4, {}},
      };
      property.restrict_scale = {{18, 0}};
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
      property.arbitrary_inputs = true;
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.restrict_multiple_inputs_scale = true;
      property.version = 2;
      break;
    case BuiltinOperator_MEAN:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.version = 2;
      break;
    case BuiltinOperator_MINIMUM:
      property.arbitrary_inputs = true;
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.restrict_multiple_inputs_scale = true;
      property.version = 2;
      break;
    case BuiltinOperator_MUL:
      property.inputs = {{0, {}}, {1, {}}};
      property.outputs = {{0, {}}};
      property.version = 2;
      break;
    case BuiltinOperator_PACK:
      property.arbitrary_inputs = true;
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
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
    case BuiltinOperator_RELU:
    case BuiltinOperator_RELU6:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.version = 2;
      break;
    case BuiltinOperator_RELU_N1_TO_1:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.version = 1;
      break;
    case BuiltinOperator_RESHAPE:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 1;
      break;
    case BuiltinOperator_RESIZE_BILINEAR:
    case BuiltinOperator_RESIZE_NEAREST_NEIGHBOR:
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
    case BuiltinOperator_STRIDED_SLICE:
      property.inputs = {{0, {}}};
      property.outputs = {{0, {}}};
      property.restrict_same_input_output_scale = true;
      property.version = 2;
      break;
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
    case BuiltinOperator_UNPACK:
      property.inputs = {{0, {}}};
      property.arbitrary_outputs = true;
      property.restrict_same_input_output_scale = true;
      property.version = 1;
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
