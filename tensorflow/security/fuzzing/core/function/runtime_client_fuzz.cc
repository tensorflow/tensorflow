// Copyright 2022 Google LLC
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

#include <vector>

#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "tensorflow/core/function/runtime_client.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace fuzzing {

using namespace core::function;


FunctionDef EmptyFunctionDefGenerator(int number_of_input_arguments, int number_of_output_arguments) {
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

auto FunctionDefDomain() {
  auto number_of_input_arguments = fuzztest::InRange(0, 7);
  auto number_of_output_arguments = fuzztest::InRange(0, 7);
  return fuzztest::Map(EmptyFunctionDefGenerator, number_of_input_arguments, number_of_output_arguments);
}

void CreateFunctionFuzz(FunctionDef def) {
  auto& ctx = GlobalEagerContext();
  Runtime rt(ctx);
  TF_ASSERT_OK(rt.CreateFunction(def));
}

FUZZ_TEST(FuzzRuntimeClient, CreateFunctionFuzz).WithDomains(FunctionDefDomain());

void CreateCallFunction(int number_of_input_arguments, int number_of_output_arguments) {
  auto& ctx = GlobalEagerContext();
  Runtime rt(ctx);
  TF_ASSERT_OK(rt.CreateFunction(EmptyFunctionDefGenerator(number_of_input_arguments, number_of_output_arguments)));

  AbstractTensorPtr tensor(ctx.CreateFloatScalar(42));
  ImmediateTensorHandlePtr handle(ctx.CreateLocalHandle(tensor.get()));
  std::vector<AbstractTensorHandle*> args(number_of_input_arguments, handle.get());

  StatusOr<ReturnValues> rets = rt.CallFunction("TestFunction", args);
  TF_ASSERT_OK(rets.status());
  ASSERT_EQ(rets->size(), number_of_output_arguments);
  ASSERT_EQ(rets->at(0)->DataType(), DT_FLOAT);
}

FUZZ_TEST(FuzzRuntimeClient, CreateCallFunction).WithDomains(fuzztest::InRange(0, 7), fuzztest::InRange(0, 7));

}  // end namespace fuzzing
}  // end namespace tensorflow
