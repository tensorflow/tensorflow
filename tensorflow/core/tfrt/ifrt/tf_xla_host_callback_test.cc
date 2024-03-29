/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/tf_xla_host_callback.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/tfrt/runtime/step_id.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

absl::StatusOr<tensorflow::FunctionDef> ToFunctionDef(
    tensorflow::Scope scope, const std::string& function_name) {
  auto graph =
      std::make_unique<tensorflow::Graph>(tensorflow::OpRegistry::Global());
  TF_RETURN_IF_ERROR(scope.ToGraph(graph.get()));
  tensorflow::FunctionDef function_def;
  TF_RETURN_IF_ERROR(
      tensorflow::GraphToFunctionDef(*graph, function_name, &function_def));
  return function_def;
}

absl::StatusOr<tensorflow::FunctionDef> MakeAddOneFunctionDef(
    const std::string& function_name) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  {
    auto arg0 = tensorflow::ops::_Arg(scope.WithOpName("arg0"),
                                      tensorflow::DT_FLOAT, 0);
    auto const0_value = tensorflow::test::AsScalar<float>(1);
    auto const0 =
        tensorflow::ops::Const(scope.WithOpName("const0"),
                               tensorflow::Input::Initializer(const0_value));
    auto add0 = tensorflow::ops::Add(scope.WithOpName("add0"), arg0, const0);
    auto retval0 =
        tensorflow::ops::_Retval(scope.WithOpName("retval0"), add0, 0);
  }
  return ToFunctionDef(std::move(scope), function_name);
}

absl::StatusOr<std::vector<tensorflow::FunctionDef>>
MakeAddOneWithCallFunctionDef(const std::string& function_name) {
  std::vector<tensorflow::FunctionDef> function_defs;
  TF_ASSIGN_OR_RETURN(function_defs.emplace_back(),
                      MakeAddOneFunctionDef("add"));

  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  {
    auto arg0 = tensorflow::ops::_Arg(scope.WithOpName("arg0"),
                                      tensorflow::DT_FLOAT, 0);
    tensorflow::NameAttrList f;
    f.set_name("add");
    auto call = tensorflow::ops::StatefulPartitionedCall(
        scope.WithOpName("call"), {arg0.output}, {tensorflow::DT_FLOAT}, f);
    auto retval0 = tensorflow::ops::_Retval(scope.WithOpName("retval0"),
                                            call.output[0], 0);
  }
  TF_ASSIGN_OR_RETURN(function_defs.emplace_back(),
                      ToFunctionDef(std::move(scope), function_name));

  return function_defs;
}

absl::StatusOr<tensorflow::FunctionDef> MakeAssignVarFunctionDef(
    const std::string& function_name) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  {
    auto arg0 = tensorflow::ops::_Arg(scope.WithOpName("arg0"),
                                      tensorflow::DT_INT32, 0);
    auto var = tensorflow::ops::VarHandleOp(
        scope.WithOpName("var"), tensorflow::DT_INT32,
        tensorflow::TensorShape(),
        tensorflow::ops::VarHandleOp::Attrs().SharedName("var"));
    tensorflow::ops::AssignVariableOp assign_op(scope.WithOpName("assign"), var,
                                                arg0);
  }
  return ToFunctionDef(std::move(scope), function_name);
}

absl::StatusOr<tensorflow::FunctionDef> MakeAddVarFunctionDef(
    const std::string& function_name) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  {
    auto arg0 = tensorflow::ops::_Arg(scope.WithOpName("arg0"),
                                      tensorflow::DT_INT32, 0);
    auto var = tensorflow::ops::VarHandleOp(
        scope.WithOpName("var"), tensorflow::DT_INT32,
        tensorflow::TensorShape(),
        tensorflow::ops::VarHandleOp::Attrs().SharedName("var"));
    auto read = tensorflow::ops::ReadVariableOp(scope.WithOpName("read"), var,
                                                tensorflow::DT_INT32);
    auto add = tensorflow::ops::Add(scope.WithOpName("add"), read, arg0);
    tensorflow::ops::AssignVariableOp assign_op(scope.WithOpName("assign"), var,
                                                add);
    auto retval0 =
        tensorflow::ops::_Retval(scope.WithOpName("retval0"), add, 0);
  }
  return ToFunctionDef(std::move(scope), function_name);
}

TEST(TfXlaHostCallbackTest, Simple) {
  constexpr absl::string_view kMainProgram = "main";
  ASSERT_OK_AND_ASSIGN(auto function_defs, MakeAddOneWithCallFunctionDef(
                                               std::string(kMainProgram)));
  tensorflow::ConfigProto session_config;

  ASSERT_OK_AND_ASSIGN(auto device_mgr, CreateTfStaticDeviceMgr());
  ASSERT_OK_AND_ASSIGN(
      auto tf_host_callback,
      tensorflow::ifrt_serving::TfXlaHostCallback::CreateCallback(
          session_config, function_defs, kMainProgram, device_mgr));

  tensorflow::tfrt_stub::StepId step_id;

  // Construct the input & output literals.
  xla::Literal in = xla::LiteralUtil::CreateR0(2.5f);
  std::vector<std::unique_ptr<xla::LiteralBase>> operands;
  operands.push_back(std::make_unique<xla::Literal>(std::move(in)));

  xla::Literal out = xla::LiteralUtil::CreateR0(0.0f);
  std::vector<std::unique_ptr<xla::MutableLiteralBase>> results;
  results.push_back(std::make_unique<xla::Literal>(std::move(out)));

  ASSERT_OK(tf_host_callback->Call(step_id, operands, results));

  EXPECT_EQ(*results[0], xla::LiteralUtil::CreateR0(3.5f));
}

TEST(TfXlaHostCallbackTest, SharedState) {
  tensorflow::ConfigProto session_config;
  constexpr absl::string_view kMainProgram = "main";
  // Verify that two host callbacks can share the same TF resource (a variable
  // with the same shared name in this case).

  ASSERT_OK_AND_ASSIGN(auto state, CreateTfStaticDeviceMgr());

  // Build the first host callback that assigns the argument to a variable.
  std::unique_ptr<TfXlaHostCallback> assign_callback;
  {
    ASSERT_OK_AND_ASSIGN(auto functions, MakeAssignVarFunctionDef("main"));

    ASSERT_OK_AND_ASSIGN(assign_callback,
                         TfXlaHostCallback::CreateCallback(
                             session_config, {functions}, kMainProgram, state));
  }

  // Build the second host callback that adds the argument to the same variable
  // and returns its value.
  std::unique_ptr<TfXlaHostCallback> incr_callback;
  {
    ASSERT_OK_AND_ASSIGN(auto functions, MakeAddVarFunctionDef("main"));

    ASSERT_OK_AND_ASSIGN(incr_callback,
                         TfXlaHostCallback::CreateCallback(
                             session_config, {functions}, kMainProgram, state));
  }

  // Assign `kInit` to the variable.
  constexpr int32_t kInit = 2;
  {
    // Construct the output literals.
    std::vector<std::unique_ptr<xla::LiteralBase>> operands;
    xla::Literal in = xla::LiteralUtil::CreateR0(kInit);
    operands.push_back(std::make_unique<xla::Literal>(std::move(in)));

    std::vector<std::unique_ptr<xla::MutableLiteralBase>> results;

    tensorflow::tfrt_stub::StepId step_id;
    ASSERT_OK(assign_callback->Call(step_id, operands, results));
  }

  // Add one to the variable every iteration and check its value. Its value
  // should start from `kInit`.
  for (int i = 0; i < 3; ++i) {
    // Construct the output literals.
    std::vector<std::unique_ptr<xla::LiteralBase>> operands;
    xla::Literal in = xla::LiteralUtil::CreateR0(int32_t{1});
    operands.push_back(std::make_unique<xla::Literal>(std::move(in)));
    std::vector<std::unique_ptr<xla::MutableLiteralBase>> results;
    xla::Literal out = xla::LiteralUtil::CreateR0(int32_t{0});
    results.push_back(std::make_unique<xla::Literal>(std::move(out)));

    tensorflow::tfrt_stub::StepId step_id;
    ASSERT_OK(incr_callback->Call(step_id, operands, results));

    EXPECT_EQ(*results[0], xla::LiteralUtil::CreateR0(kInit + i + 1));
  }
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
