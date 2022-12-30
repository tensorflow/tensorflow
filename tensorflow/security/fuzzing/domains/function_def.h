/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SECURITY_FUZZING_DOMAINS_FUNCTION_DEF_H_
#define TENSORFLOW_SECURITY_FUZZING_DOMAINS_FUNCTION_DEF_H_

#include "tensorflow/core/framework/function.h"

namespace tensorflow {
namespace fuzzing {
namespace domain {

inline auto NumberOfInputArguments() {
  return fuzztest::InRange(0, 7);
}

inline auto NumberOfOutputArguments() {
  return fuzztest::InRange(1, 7);
}

inline FunctionDef EmptyFunctionDefGenerator(int number_of_input_arguments, int number_of_output_arguments) {
  std::vector<string> in_def_vec;
  in_def_vec.reserve(number_of_input_arguments);
  for (int c = 0; c < number_of_input_arguments; ++c) {
    in_def_vec.push_back("in" + std::to_string(c) + ":float");
  }
  std::vector<FunctionDefHelper::Node> body_nodes;
  if (number_of_output_arguments > number_of_input_arguments) {
    Tensor const_value(DataTypeToEnum<float>::value, {});
    const_value.scalar<float>()() = 0;
    body_nodes.push_back({{"zero"}, "Const", {}, {{"value", const_value}, {"dtype", DT_FLOAT}}});
  }
  std::vector<string> out_def_vec;
  out_def_vec.reserve(number_of_output_arguments);
  std::vector<std::pair<string, string>> ret_def;
  ret_def.reserve(number_of_output_arguments);
  for (int c = 0; c < number_of_output_arguments; ++c) {
    auto output_id = "out" + std::to_string(c);
    out_def_vec.push_back(output_id + ":float");
    if (c < number_of_input_arguments) {
      ret_def.emplace_back(output_id, "in" + std::to_string(c));
    } else {
      ret_def.emplace_back(output_id, "zero:output");
    }
  }
  return FunctionDefHelper::Create("TestFunction", in_def_vec, out_def_vec, {}, body_nodes, ret_def);
}

inline auto FunctionDef() {
  return fuzztest::Map(EmptyFunctionDefGenerator, NumberOfInputArguments(), NumberOfOutputArguments());
}

}  // end namespace domain
}  // end namespace fuzzing
}  // end namespace tensorflow

#endif  // TENSORFLOW_SECURITY_FUZZING_DOMAINS_FUNCTION_DEF_H_
