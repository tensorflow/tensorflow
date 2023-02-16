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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "tensorflow/core/function/runtime_client/runtime_client.h"

namespace tensorflow {
namespace fuzzing {

FunctionDef EmptyFunctionDefGenerator(int number_of_input_arguments,
                                      int number_of_output_arguments) {
  std::vector<string> in_def_vec;
  in_def_vec.reserve(number_of_input_arguments);
  for (int c = 0; c < number_of_input_arguments; ++c) {
    in_def_vec.push_back(absl::StrCat("in", c, ":float"));
  }
  std::vector<FunctionDefHelper::Node> body_nodes;
  if (number_of_output_arguments > number_of_input_arguments) {
    Tensor const_value(DataTypeToEnum<float>::value, {});
    const_value.scalar<float>()() = 0;
    body_nodes.push_back(
        {{"zero"}, "Const", {}, {{"value", const_value}, {"dtype", DT_FLOAT}}});
  }
  std::vector<string> out_def_vec;
  out_def_vec.reserve(number_of_output_arguments);
  std::vector<std::pair<string, string>> ret_def;
  ret_def.reserve(number_of_output_arguments);
  for (int c = 0; c < number_of_output_arguments; ++c) {
    string output_id = "out" + std::to_string(c);
    out_def_vec.push_back(output_id + ":float");
    if (c < number_of_input_arguments) {
      ret_def.emplace_back(output_id, "in" + std::to_string(c));
    } else {
      ret_def.emplace_back(output_id, "zero:output");
    }
  }
  return FunctionDefHelper::Create("TestFunction", in_def_vec, out_def_vec, {},
                                   body_nodes, ret_def);
}

fuzztest::Domain<FunctionDef> FunctionDefDomain() {
  return fuzztest::Map(EmptyFunctionDefGenerator, fuzztest::InRange(0, 7),
                       fuzztest::InRange(1, 7));
}

class FuzzRuntimeClient {
 public:
  FuzzRuntimeClient()
      : ctx_(InitLocalEagerContextPtr()), rt_(core::function::Runtime(*ctx_)) {}

  void CreateFunctionFuzz(FunctionDef def) {
    TF_CHECK_OK(rt_.CreateFunction(def));
  }

 private:
  EagerContextPtr ctx_;
  core::function::Runtime rt_;

  EagerContextPtr InitLocalEagerContextPtr() {
    SessionOptions opts;
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(
        opts, "/job:localhost/replica:0/task:0", &devices));
    return EagerContextPtr(new EagerContext(
        opts, ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
        /*async=*/false,
        /*device_mgr=*/new DynamicDeviceMgr(std::move(devices)),
        /*device_mgr_owned=*/true,
        /*rendezvous=*/nullptr,
        /*cluster_flr=*/nullptr,
        /*collective_executor_mgr=*/nullptr,
        /*run_eager_op_as_function=*/true));
  }
};

FUZZ_TEST_F(FuzzRuntimeClient, CreateFunctionFuzz)
    .WithDomains(FunctionDefDomain());

}  // end namespace fuzzing
}  // end namespace tensorflow
