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

#include "tensorflow/core/tfrt/ifrt/tf_host_callback.h"

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
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_matcher.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

using ::tensorflow::test::AsTensor;
using ::tensorflow::test::TensorEq;

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

TEST(TfHostCallbackTest, Simple) {
  ASSERT_OK_AND_ASSIGN(auto function_defs,
                       MakeAddOneWithCallFunctionDef("main"));

  // Construct the input & output
  auto in = AsTensor<float>({2.5f}, tensorflow::TensorShape({1}));
  void* in_ptrs[1] = {in.data()};
  std::vector<DtypeAndShape> in_dtype_shapes;
  in_dtype_shapes.push_back({.dtype = in.dtype(), .shape = in.shape()});

  auto out = AsTensor<float>({0.0f}, tensorflow::TensorShape({1}));
  void* out_ptrs[1] = {out.data()};
  std::vector<DtypeAndShape> out_dtype_shapes;
  out_dtype_shapes.push_back({.dtype = out.dtype(), .shape = out.shape()});

  ASSERT_OK_AND_ASSIGN(auto device_mgr, CreateTfStaticDeviceMgr());
  ASSERT_OK_AND_ASSIGN(auto tf_host_callback,
                       tensorflow::ifrt_serving::TfHostCallback::Create(
                           function_defs, "main", in_dtype_shapes,
                           out_dtype_shapes, device_mgr.get()));

  ASSERT_OK(tf_host_callback->Call(in_ptrs, out_ptrs));

  EXPECT_THAT(out,
              TensorEq(AsTensor<float>({3.5f}, tensorflow::TensorShape({1}))));
}

TEST(TfHostCallbackTest, SharedState) {
  tensorflow::ConfigProto session_config;
  // Verify that two host callbacks can share the same TF resource (a variable
  // with the same shared name in this case).

  ASSERT_OK_AND_ASSIGN(auto state, CreateTfStaticDeviceMgr());

  // Build the first host callback that assigns the argument to a variable.
  std::unique_ptr<TfHostCallback> assign_callback;
  {
    ASSERT_OK_AND_ASSIGN(auto functions, MakeAssignVarFunctionDef("main"));

    std::vector<DtypeAndShape> in_dtype_shapes;
    in_dtype_shapes.push_back(
        {.dtype = DT_INT32, .shape = tensorflow::TensorShape({1})});
    std::vector<DtypeAndShape> out_dtype_shapes;

    ASSERT_OK_AND_ASSIGN(
        assign_callback,
        TfHostCallback::Create({functions}, "main", in_dtype_shapes,
                               out_dtype_shapes, state.get()));
  }

  // Build the second host callback that adds the argument to the same variable
  // and returns its value.
  std::unique_ptr<TfHostCallback> incr_callback;
  {
    ASSERT_OK_AND_ASSIGN(auto functions, MakeAddVarFunctionDef("main"));

    std::vector<DtypeAndShape> in_dtype_shapes;
    in_dtype_shapes.push_back(
        {.dtype = DT_INT32, .shape = tensorflow::TensorShape({1})});
    std::vector<DtypeAndShape> out_dtype_shapes;
    out_dtype_shapes.push_back(
        {.dtype = DT_INT32, .shape = tensorflow::TensorShape({1})});

    ASSERT_OK_AND_ASSIGN(
        incr_callback,
        TfHostCallback::Create({functions}, "main", in_dtype_shapes,
                               out_dtype_shapes, state.get()));
  }

  // Assign `kInit` to the variable.
  constexpr int32_t kInit = 2;
  {
    // Construct the output literals.
    auto in = AsTensor<int32_t>({kInit}, tensorflow::TensorShape({1}));
    void* in_ptrs[1] = {in.data()};

    void* out_ptrs[0];

    ASSERT_OK(assign_callback->Call(in_ptrs, out_ptrs));
  }

  // Add one to the variable every iteration and check its value. Its value
  // should start from `kInit`.
  for (int i = 0; i < 3; ++i) {
    // Construct the output literals.

    auto in = AsTensor<int32_t>({1}, tensorflow::TensorShape({1}));
    void* in_ptrs[1] = {in.data()};

    auto out = AsTensor<int32_t>({0}, tensorflow::TensorShape({1}));
    void* out_ptrs[1] = {out.data()};

    ASSERT_OK(incr_callback->Call(in_ptrs, out_ptrs));

    EXPECT_THAT(out, TensorEq(AsTensor<int32_t>({kInit + i + 1},
                                                tensorflow::TensorShape({1}))));
  }
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
