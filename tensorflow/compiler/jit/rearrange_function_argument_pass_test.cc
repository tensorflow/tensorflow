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

#include "tensorflow/compiler/tf2xla/rearrange_function_argument_pass.h"

#include "absl/strings/match.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/encapsulate_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class RearrangeFunctionArgumentForFunctionTest : public ::testing::Test {
 public:
  void SetUp() override {
    SessionOptions session_options;
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(
        session_options, "/job:localhost/replica:0/task:0", &devices));
    device_mgr_ = absl::make_unique<DeviceMgr>(std::move(devices));
  }

  Status RearrangeFunctionArgumentTest(
      const string &func_name, const string &new_func_name,
      const protobuf::Map<string, tensorflow::AttrValue> &attrs,
      FunctionLibraryDefinition *fld, bool *modified) {
    OptimizerOptions opts;
    pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
        device_mgr_.get(), Env::Default(), TF_GRAPH_DEF_VERSION, fld, opts,
        /*default_thread_pool=*/nullptr, /*cluster_flr=*/nullptr);
    std::map<string, absl::optional<string>> canonicalized_name_to_new_name;
    auto flr = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
    return RearrangeFunctionArgumentForFunction(
        func_name, new_func_name, attrs, fld, flr,
        &canonicalized_name_to_new_name, modified);
  }

 private:
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
};

TEST_F(RearrangeFunctionArgumentForFunctionTest, Basic) {
  FunctionDefLibrary fdl;
  {
    // Function for StatefulPartitionedCall's "f", If's
    // "then_branch"/"else_branch".
    // "arg0" (T=DT_RESOURCE), "arg1" (T=DT_BOOL)
    // "ret0" = "arg1"
    // "ret1" = "arg0"
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg0 = ops::_Arg(s.WithOpName("arg0"), DT_RESOURCE, 0);
    Output arg1 = ops::_Arg(s.WithOpName("arg1"), DT_BOOL, 1);
    auto ret0 = ops::_Retval(s.WithOpName("ret0"), arg1, 0);
    auto ret1 = ops::_Retval(s.WithOpName("ret1"), arg0, 1);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "f1", xla_fdef));
  }
  {
    // Function for While's "body".
    // "arg0" (T=DT_RESOURCE), "arg1" (T=DT_BOOL)
    // "ret0" = "arg0"
    // "ret1" = "arg1"
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg0 = ops::_Arg(s.WithOpName("arg0"), DT_RESOURCE, 0);
    Output arg1 = ops::_Arg(s.WithOpName("arg1"), DT_BOOL, 1);
    auto ret0 = ops::_Retval(s.WithOpName("ret0"), arg0, 0);
    auto ret1 = ops::_Retval(s.WithOpName("ret1"), arg1, 0);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "f2", xla_fdef));
  }
  {
    // Function for While's "cond".
    // "arg0" (T=DT_RESOURCE), "arg1" (T=DT_BOOL)
    // "ret0" = "arg1"
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg0 = ops::_Arg(s.WithOpName("arg0"), DT_RESOURCE, 0);
    Output arg1 = ops::_Arg(s.WithOpName("arg1"), DT_BOOL, 1);
    auto ret0 = ops::_Retval(s.WithOpName("ret0"), arg1, 0);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "f3", xla_fdef));
  }
  {
    // Build the XLA computation func.
    // "arg0" (T=DT_RESOURCE), "arg1" (T=DT_INT32)
    // "arg0", "arg1" -> "call" (StatefulPartitionedCall) -> "ret0", "ret1"
    // "arg0", "arg1" -> "if" (If) -> "ret2", "ret3"
    // "arg0", "arg1" -> "while" (While) -> "ret4", "ret5"
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output arg0 = ops::_Arg(s.WithOpName("arg0"), DT_RESOURCE, 0);
    Output arg1 = ops::_Arg(s.WithOpName("arg1"), DT_BOOL, 1);
    NameAttrList f;
    f.set_name("f1");
    auto call = ops::StatefulPartitionedCall(
        s.WithOpName("call"), {arg0, arg1},
        std::vector<DataType>{DT_BOOL, DT_RESOURCE}, f);
    auto ret0 = ops::_Retval(s.WithOpName("ret0"), call.output[0], 0);
    auto ret1 = ops::_Retval(s.WithOpName("ret1"), call.output[1], 1);
    auto if_op = ops::If(s.WithOpName("if"), arg1,
                         std::initializer_list<Input>{arg0, arg1},
                         {DT_BOOL, DT_RESOURCE}, f, f);
    auto ret2 = ops::_Retval(s.WithOpName("ret2"), if_op.output[0], 2);
    auto ret3 = ops::_Retval(s.WithOpName("ret3"), if_op.output[1], 3);
    NameAttrList cond_fn, body_fn;
    cond_fn.set_name("f3");
    body_fn.set_name("f2");
    auto while_op =
        ops::While(s.WithOpName("while"),
                   std::initializer_list<Input>{arg0, arg1}, cond_fn, body_fn);
    auto ret4 = ops::_Retval(s.WithOpName("ret4"), while_op.output[0], 4);
    auto ret5 = ops::_Retval(s.WithOpName("ret5"), while_op.output[1], 5);
    std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
    TF_CHECK_OK(s.ToGraph(g.get()));
    FunctionDef *xla_fdef = fdl.add_function();
    TF_CHECK_OK(GraphToFunctionDef(*g, "cluster", xla_fdef));
  }
  FunctionLibraryDefinition fld(OpRegistry::Global(), fdl);

  bool modified;
  protobuf::Map<string, tensorflow::AttrValue> attrs;
  TF_CHECK_OK(RearrangeFunctionArgumentTest("cluster", "cluster_rewritten",
                                            attrs, &fld, &modified));

  // Check function f1_rearrange_0, input types should be {DT_BOOL, DT_RESOURCE}
  // and output types should be {DT_BOOL}.
  const FunctionDef *f1_rewritten = fld.Find("f1_rearrange_0");
  CHECK_NE(f1_rewritten, nullptr);
  ASSERT_EQ(f1_rewritten->signature().input_arg_size(), 2);
  EXPECT_EQ(f1_rewritten->signature().input_arg(0).type(), DT_BOOL);
  EXPECT_EQ(f1_rewritten->signature().input_arg(1).type(), DT_RESOURCE);
  ASSERT_EQ(f1_rewritten->signature().output_arg_size(), 1);
  EXPECT_EQ(f1_rewritten->signature().output_arg(0).type(), DT_BOOL);

  // Check node "call" input and output edges.
  std::unique_ptr<FunctionBody> xla_fbody;
  TF_CHECK_OK(FunctionDefToBodyHelper(*fld.Find("cluster_rewritten"),
                                      AttrSlice(), &fld, &xla_fbody));
  auto node_name_index = xla_fbody->graph->BuildNodeNameIndex();
  const Node *call_node = node_name_index.at("call");
  ASSERT_NE(call_node, nullptr);
  const Node *input_node;
  TF_CHECK_OK(call_node->input_node(0, &input_node));
  EXPECT_EQ(input_node->name(), "arg1");
  TF_CHECK_OK(call_node->input_node(1, &input_node));
  EXPECT_EQ(input_node->name(), "arg0");
  const Node *ret0_node = xla_fbody->ret_nodes[0];
  TF_CHECK_OK(ret0_node->input_node(0, &input_node));
  EXPECT_EQ(input_node->name(), "call");
  const Node *ret1_node = xla_fbody->ret_nodes[1];
  TF_CHECK_OK(ret1_node->input_node(0, &input_node));
  EXPECT_EQ(input_node->name(), "arg0");

  // Check node "if" input and output edges.
  const Node *if_node = node_name_index.at("if");
  ASSERT_NE(if_node, nullptr);
  TF_CHECK_OK(if_node->input_node(1, &input_node));
  EXPECT_EQ(input_node->name(), "arg1");
  TF_CHECK_OK(if_node->input_node(2, &input_node));
  EXPECT_EQ(input_node->name(), "arg0");
  const Node *ret2_node = xla_fbody->ret_nodes[2];
  TF_CHECK_OK(ret2_node->input_node(0, &input_node));
  EXPECT_EQ(input_node->name(), "if");
  const Node *ret3_node = xla_fbody->ret_nodes[3];
  TF_CHECK_OK(ret3_node->input_node(0, &input_node));
  EXPECT_EQ(input_node->name(), "arg0");

  // Check node "while" input and output edges.
  const Node *while_node = node_name_index.at("while");
  ASSERT_NE(while_node, nullptr);
  TF_CHECK_OK(while_node->input_node(0, &input_node));
  EXPECT_EQ(input_node->name(), "arg1");
  TF_CHECK_OK(while_node->input_node(1, &input_node));
  EXPECT_EQ(input_node->name(), "arg0");
  const Node *ret4_node = xla_fbody->ret_nodes[4];
  TF_CHECK_OK(ret4_node->input_node(0, &input_node));
  EXPECT_EQ(input_node->name(), "arg0");
  const Node *ret5_node = xla_fbody->ret_nodes[5];
  TF_CHECK_OK(ret5_node->input_node(0, &input_node));
  EXPECT_EQ(input_node->name(), "while");
}

}  // namespace tensorflow
