/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_OPERATOR_PROPERTY_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_OPERATOR_PROPERTY_H_

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace operator_property {

// The scales of a certain tensor can be derived from the multiplications of all
// the scales. For example, for bias in conv, derived_scale = {{0, 1}, {}, {}}
// and for lstm gate bias, the derived scale is {{}, {0}, {2^-10}}
struct DerivedScale {
  std::vector<int> input_tensors = {};
  std::vector<int> intermediate_tensors = {};
  // This is a list of extra factors that are not associated with any other
  // tensor.
  std::vector<float> factors = {};
};

struct TensorProperty {
  // per_axis also implies symmetric currently.
  bool per_axis = false;
  // TODO(jianlijianli): remove dimension index and read it from tensor instead.
  int per_axis_index = 0;
  bool symmetric = false;

  // Constraints.
  bool restriction = false;
  // scale/zero_point hardcoded.
  std::pair<float, int> restricted_value = {0.0f, 0};

  // Use derived scale.
  bool use_derived_scale = false;
  // The derived scale.
  DerivedScale derived_scale;

  // The number of bits for this tensor. It could be 8, 16, 32 or even not power
  // of two.
  int number_of_bits = 8;

  // Extend the range to power of two.
  bool extend_to_power_of_two = false;

  // State tensor.
  bool state_tensor = false;
};

struct OperatorProperty {
  // Is a quantized operations currently supported.
  bool quantizable = true;

  // Op has arbitrary number of inputs, such as concat.
  bool arbitrary_inputs = false;
  // Op has arbitrary number of outputs, such as slice.
  bool arbitrary_outputs = false;
  // Input indexes -> input tensor property.
  // Must be topologically sorted since there are derived scales.
  std::vector<std::pair<int, TensorProperty>> inputs = {};
  // Output indexes -> output tensor property.
  std::vector<std::pair<int, TensorProperty>> outputs = {};
  // Bias indexes.
  // TODO(jianlijianli): remove this by putting biases into inputs as well since
  // we now can model "derived scale".
  std::vector<int> biases = {};

  // Intermediate indexes -> intermediate tensor property.
  std::vector<std::pair<int, TensorProperty>> intermediates = {};

  // Force output to reuse the same scale and zero point of input.
  bool restrict_same_input_output_scale = false;

  // Use same min of min and max of max for each group.
  // Incompatible with restrict_same_input_output_scale and restricted_value.
  // TODO(jianlijianli): make it compatible with other restrictions when there
  // is a use case.
  std::vector<std::vector<int>> restrict_scale = {};

  // Op version.
  int version = 1;
};

OperatorProperty GetOperatorProperty(const ModelT* model, int subgraph_index,
                                     int op_index);

}  // namespace operator_property
}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_OPERATOR_PROPERTY_H_
