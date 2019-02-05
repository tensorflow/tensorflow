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

#include "tensorflow/core/grappler/optimizers/function_api_info.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {
void SetArg(const string& name, const string& type_name,
            OpDef::ArgDef* arg_def) {
  arg_def->set_name(name);
  arg_def->set_type_attr(type_name);
}

typedef std::pair<string, string> ArgSpec;  // name, type.

void SetArgs(const std::vector<ArgSpec>& input_args_spec,
             const std::vector<ArgSpec>& output_args_spec, OpDef* sig) {
  for (const auto& arg_spec : input_args_spec)
    SetArg(arg_spec.first, arg_spec.second, sig->add_input_arg());
  for (const auto& arg_spec : output_args_spec)
    SetArg(arg_spec.first, arg_spec.second, sig->add_output_arg());
}

void PopulateFunction(const string& name, const string& api_interface_name,
                      const string& preferred_device,
                      const std::vector<ArgSpec>& input_args,
                      const std::vector<ArgSpec>& output_args,
                      const string& forward_function_name,
                      const string& backward_function_name,
                      FunctionDef* func_def) {
  OpDef* sig = func_def->mutable_signature();
  sig->set_name(name);

  SetArgs(input_args, output_args, sig);

  auto* func_attr = func_def->mutable_attr();
  if (!api_interface_name.empty())
    (*func_attr)["experimental_api_implements"].set_s(api_interface_name);
  if (!preferred_device.empty())
    (*func_attr)["experimental_api_preferred_device"].set_s(preferred_device);
  if (!forward_function_name.empty())
    (*func_attr)["forward_function_name"].set_s(forward_function_name);
  if (!backward_function_name.empty())
    (*func_attr)["backward_function_name"].set_s(backward_function_name);
}

void PopulateSampleLibrary(const bool mismatch_args,
                           FunctionDefLibrary* func_lib) {
  const std::vector<ArgSpec> func_args{{"in1", "float32"}, {"in2", "int32"}};
  const std::vector<ArgSpec> func_wrong_args{{"in1", "int32"},
                                             {"in2", "int32"}};
  const std::vector<ArgSpec> output_args{{"out", "float32"}};
  PopulateFunction("DoStuffCpu", "DoStuff", "CPU", func_args, output_args, "",
                   "", func_lib->add_function());
  PopulateFunction("DoStuffGpu", "DoStuff", "GPU",
                   mismatch_args ? func_wrong_args : func_args, output_args, "",
                   "", func_lib->add_function());
  PopulateFunction("DoThings", "DoThings", "", func_args, output_args, "", "",
                   func_lib->add_function());
  PopulateFunction("OneOff", "", "", func_args, output_args, "", "",
                   func_lib->add_function());
  PopulateFunction("AnotherOneOff", "", "", func_args, output_args, "", "",
                   func_lib->add_function());
}

void PopulateComplexLibrary(FunctionDefLibrary* func_lib) {
  const std::vector<ArgSpec> input_args{{"in1", "float32"}, {"in2", "int32"}};
  const std::vector<ArgSpec> output_args{{"out", "float32"}};
  const std::vector<ArgSpec> output_with_state{
      {"out", "float32"}, {"state1", "int32"}, {"state2", "int32"}};

  PopulateFunction("DoStuffCpu", "DoStuff", "CPU", input_args, output_args, "",
                   "DoStuffCpu_gradient", func_lib->add_function());
  PopulateFunction("DoStuffCpu_gradient", "DoStuff", "CPU", output_args,
                   input_args, "DoStuffCpu", "", func_lib->add_function());
  PopulateFunction("DoStuffGpu", "DoStuff", "GPU", input_args,
                   output_with_state, "", "DoStuffGpu_gradient",
                   func_lib->add_function());
  PopulateFunction("DoStuffGpu_gradient", "DoStuff", "GPU", output_with_state,
                   input_args, "DoStuffGpu", "", func_lib->add_function());
}

bool CheckEquivImpl(const FunctionLibraryApiInfo& lib_api_info,
                    const string& func_name,
                    const std::vector<string>& expected_other) {
  std::vector<string> other_impl;
  Status status =
      lib_api_info.GetEquivalentImplementations(func_name, &other_impl);
  EXPECT_EQ(status, Status::OK());
  const std::unordered_set<string> actual(other_impl.begin(), other_impl.end());
  const std::unordered_set<string> expected(expected_other.begin(),
                                            expected_other.end());
  return actual == expected;
}

string GetInterfaceName(const FunctionLibraryApiInfo& lib_api_info,
                        const string& func_name) {
  auto* info = lib_api_info.GetApiInfo(func_name);
  CHECK_NOTNULL(info);
  return info->interface_name();
}

string GetPreferredDevice(const FunctionLibraryApiInfo& lib_api_info,
                          const string& func_name) {
  auto* info = lib_api_info.GetApiInfo(func_name);
  CHECK_NOTNULL(info);
  return info->preferred_device();
}

TEST(FunctionApiInfoTest, ParseTags) {
  FunctionDefLibrary func_lib;
  PopulateSampleLibrary(/* mismatch_args */ false, &func_lib);
  FunctionLibraryApiInfo lib_api_info;
  TF_ASSERT_OK(lib_api_info.Init(func_lib));

  EXPECT_EQ("DoStuff", GetInterfaceName(lib_api_info, "DoStuffCpu"));
  EXPECT_EQ("DoStuff", GetInterfaceName(lib_api_info, "DoStuffGpu"));
  EXPECT_EQ("DoThings", GetInterfaceName(lib_api_info, "DoThings"));

  EXPECT_EQ("CPU", GetPreferredDevice(lib_api_info, "DoStuffCpu"));
  EXPECT_EQ("GPU", GetPreferredDevice(lib_api_info, "DoStuffGpu"));
  EXPECT_EQ("", GetPreferredDevice(lib_api_info, "DoThings"));

  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "DoStuffCpu", {"DoStuffGpu"}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "DoStuffGpu", {"DoStuffCpu"}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "Undefined", {}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "OneOff", {}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "AnotherOneOff", {}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "DoThings", {}));
}

TEST(FunctionApiInfoTest, ComplexFunctionLib) {
  FunctionDefLibrary func_lib;
  PopulateComplexLibrary(&func_lib);
  FunctionLibraryApiInfo lib_api_info;
  TF_ASSERT_OK(lib_api_info.Init(func_lib));

  EXPECT_EQ("DoStuff", GetInterfaceName(lib_api_info, "DoStuffCpu"));
  EXPECT_EQ("DoStuff", GetInterfaceName(lib_api_info, "DoStuffCpu_gradient"));
  EXPECT_EQ("DoStuff", GetInterfaceName(lib_api_info, "DoStuffGpu"));
  EXPECT_EQ("DoStuff", GetInterfaceName(lib_api_info, "DoStuffGpu_gradient"));

  EXPECT_EQ("CPU", GetPreferredDevice(lib_api_info, "DoStuffCpu"));
  EXPECT_EQ("CPU", GetPreferredDevice(lib_api_info, "DoStuffCpu_gradient"));
  EXPECT_EQ("GPU", GetPreferredDevice(lib_api_info, "DoStuffGpu"));
  EXPECT_EQ("GPU", GetPreferredDevice(lib_api_info, "DoStuffGpu_gradient"));

  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "DoStuffCpu", {"DoStuffGpu"}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "DoStuffGpu", {"DoStuffCpu"}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "DoStuffCpu_gradient",
                             {"DoStuffGpu_gradient"}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "DoStuffGpu_gradient",
                             {"DoStuffCpu_gradient"}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "Undefined", {}));
}

TEST(FunctionApiInfoTest, MismatchedArguments) {
  FunctionDefLibrary func_lib;
  PopulateSampleLibrary(/* mismatch_args */ true, &func_lib);
  FunctionLibraryApiInfo lib_api_info;
  const Status ret = lib_api_info.Init(func_lib);
  EXPECT_FALSE(ret.ok());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
