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
TfLiteStatus GetOperatorProperty(const BuiltinOperator& op,
                                 OperatorProperty* property) {
  // Set up default values.
  property->per_axis = false;
  property->per_axis_index = 0;
  property->arbitrary_inputs = false;
  property->input_indexes = {};
  property->output_indexes = {};
  property->biases = {};
  property->restrict_same_input_output_scale = false;
  property->restriction_on_output = false;
  property->restricted_value_on_output = {0.0, 0.0};
  property->version = 0;
  switch (op) {
    case BuiltinOperator_ADD:
      property->input_indexes = {0, 1};
      property->output_indexes = {0};
      property->version = 2;
      return kTfLiteOk;
    case BuiltinOperator_ARG_MAX:
      property->input_indexes = {0};
      // ArgMax has no quantizable output.
      property->version = 2;
      return kTfLiteOk;
    case BuiltinOperator_AVERAGE_POOL_2D:
      property->input_indexes = {0};
      property->output_indexes = {0};
      property->restrict_same_input_output_scale = true;
      property->version = 2;
      return kTfLiteOk;
    case BuiltinOperator_CONCATENATION:
      property->arbitrary_inputs = true;
      property->input_indexes = {};
      property->output_indexes = {0};
      property->restrict_same_input_output_scale = true;
      property->version = 2;
      return kTfLiteOk;
    case BuiltinOperator_CONV_2D:
      property->per_axis = true;
      property->per_axis_index = 0;
      property->input_indexes = {0, 1};
      property->output_indexes = {0};
      property->biases = {2};
      property->version = 2;
      return kTfLiteOk;
    case BuiltinOperator_DEPTHWISE_CONV_2D:
      property->per_axis = true;
      property->per_axis_index = 3;
      property->input_indexes = {0, 1};
      property->output_indexes = {0};
      property->biases = {2};
      property->version = 3;
      return kTfLiteOk;
    case BuiltinOperator_FULLY_CONNECTED:
      property->input_indexes = {0, 1};
      property->output_indexes = {0};
      property->biases = {2};
      property->version = 4;
      return kTfLiteOk;
    case BuiltinOperator_MAX_POOL_2D:
      property->input_indexes = {0};
      property->output_indexes = {0};
      property->restrict_same_input_output_scale = true;
      property->version = 2;
      return kTfLiteOk;
    case BuiltinOperator_MEAN:
      property->input_indexes = {0};
      property->output_indexes = {0};
      property->version = 2;
      return kTfLiteOk;
    case BuiltinOperator_MUL:
      property->input_indexes = {0, 1};
      property->output_indexes = {0};
      property->version = 2;
      return kTfLiteOk;
    case BuiltinOperator_PAD:
      property->input_indexes = {0};
      property->output_indexes = {0};
      property->version = 2;
      return kTfLiteOk;
    case BuiltinOperator_QUANTIZE:
      property->input_indexes = {0};
      property->output_indexes = {0};
      property->version = 1;
      return kTfLiteOk;
    case BuiltinOperator_RESHAPE:
      property->input_indexes = {0};
      property->output_indexes = {0};
      property->restrict_same_input_output_scale = true;
      property->version = 1;
      return kTfLiteOk;
    case BuiltinOperator_SQUEEZE:
      property->input_indexes = {0};
      property->output_indexes = {0};
      property->restrict_same_input_output_scale = true;
      property->version = 1;
      return kTfLiteOk;
    case BuiltinOperator_SOFTMAX:
      property->input_indexes = {0};
      property->output_indexes = {0};
      // Softmax requires output with 1/256 as scale and -128 as zero point.
      property->restriction_on_output = true;
      property->restricted_value_on_output = {1 / 256.0, -128};
      property->version = 2;
      return kTfLiteOk;
    case BuiltinOperator_TANH:
      property->input_indexes = {0};
      property->output_indexes = {0};
      // Tanh requires output with 1/128 as scale and 0 as zero point.
      property->restriction_on_output = true;
      property->restricted_value_on_output = {1 / 128.0, 0};
      property->version = 2;
      return kTfLiteOk;
    default:
      return kTfLiteError;
  }
  return kTfLiteError;
}

}  // namespace operator_property
}  // namespace optimize
}  // namespace tflite
