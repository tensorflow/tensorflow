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

#include "tensorflow/core/grappler/optimizers/function_optimizer.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace tensorflow {
namespace grappler {

namespace {
constexpr char kDevice[] = "/job:localhost/replica:0/task:0/device:CPU:0";
}  // namespace

class FunctionOptimizerTest : public GrapplerTest {};

TEST_F(FunctionOptimizerTest, InlineFunction_SimpleFunction) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);

  // Build a graph to compute y = XTimesTwo(x)
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("y", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, kDevice)},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  const string arg0 = "Func/y/input/_0";
  const string ret0 = "Func/y/output/_1";

  const Tensor kTwo = test::AsScalar<int64_t>(2);
  GraphDef expected = test::function::GDef(
      {NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}),
       NDef(arg0, "Identity", {"x"}, {{"T", DT_FLOAT}}),
       NDef("y/two", "Const", {}, {{"dtype", DT_INT64}, {"value", kTwo}}),
       NDef("y/scale", "Cast", {"y/two"},
            {{"DstT", DT_FLOAT}, {"SrcT", DT_INT64}}),
       NDef("y/y", "Mul", {arg0, "y/scale"}, {{"T", DT_FLOAT}}),
       NDef(ret0, "Identity", {"y/y"}, {{"T", DT_FLOAT}}),
       NDef("z", "Identity", {ret0}, {{"T", DT_FLOAT}})},
      {});
  for (NodeDef& node : *expected.mutable_node()) node.set_device(kDevice);

  CompareGraphs(expected, output);

  Tensor pi = test::AsScalar<float>(3.14f);
  item.fetch = {"z"};
  item.feed.emplace_back("x", pi);
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, InlineFunction_FixedTypeFunction) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);

  // Create and instantiate a version of the XTimesTwo function that only
  // accepts floats a inputs.
  const Tensor kTwo = test::AsScalar<float>(2.0f);
  FunctionDef x_times_two = FunctionDefHelper::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: float"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_FLOAT}}},
          // "enter" node is used to verify that InlineFunction would update the
          // frame name accordingly.
          {{"enter"},
           "Enter",
           {"x"},
           {{"T", DT_FLOAT}, {"frame_name", "frame"}}},
          {{"y"}, "Mul", {"x", "two"}, {{"T", DT_FLOAT}}},
      });

  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("y", "XTimesTwo", {"x"}, {}, kDevice),
       NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, kDevice)},
      // FunctionLib
      {
          x_times_two,
      });

  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // Calls to XTimesTwo were removed from the graph.
  for (const NodeDef& node : output.node()) {
    EXPECT_NE(node.op(), "XTimesTwo");
  }
  // And the function itself was removed from the library.
  EXPECT_EQ(output.library().function_size(), 0);

  Tensor pi = test::AsScalar<float>(3.14f);
  item.fetch = {"z"};
  item.feed.emplace_back("x", pi);
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, InlineFunction_FunctionWithOutputMapping) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);

  FunctionDef func = FunctionDefHelper::Create(
      // Name
      "Exp_func",
      // Args
      {"in: float"},
      // Return values
      {"out: float"},
      // Attr def
      {},
      // Nodes
      {{{"Linear_func"}, "Identity", {"in"}, {{"T", DT_FLOAT}}},
       {{"Exp"}, "Exp", {"Linear_func:output:0"}, {{"T", DT_FLOAT}}}},
      // Mapping
      {{"out", "Exp:y:0"}});

  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("y", "Exp_func", {"x"}, {}, kDevice),
       NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, kDevice)},
      // FunctionLib
      {
          func,
      });

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Function call was removed from the graph.
  for (const NodeDef& node : output.node()) {
    EXPECT_NE(node.op(), "Exp_func");
  }
  // And the function itself was removed from the library.
  EXPECT_EQ(output.library().function_size(), 0);

  Tensor pi = test::AsScalar<float>(3.14f);
  item.fetch = {"z"};
  item.feed.emplace_back("x", pi);
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, InlineFunction_FunctionWithInputForwarding) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);

  FunctionDef func = FunctionDefHelper::Create(
      // Name
      "ForwardInputs",
      // Args
      {"in0: float", "in1: float", "arg2: float", "arg3: int32", "arg4: float"},
      // Return values
      {"out0: float", "arg2: float", "arg3: int32"},
      // Attr def
      {},
      // Nodes
      {},
      // Mapping
      {{"out0", "in0"}, {"arg2", "arg2"}, {"arg3", "arg3"}});

  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("x0", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("x1", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("x2", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("x3", "Placeholder", {}, {{"dtype", DT_INT32}}, kDevice),
       NDef("x4", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("y", "ForwardInputs", {"x0", "x1", "x2", "x3", "x4"}, {}, kDevice),
       NDef("z0", "Identity", {"y:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("z1", "Identity", {"y:1"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("z2", "Identity", {"y:2"}, {{"T", DT_INT32}}, kDevice)},
      // FunctionLib
      {
          func,
      });

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Function call was removed from the graph.
  for (const NodeDef& node : output.node()) {
    EXPECT_NE(node.op(), "ForwardInputs");
  }
  // And the function itself was removed from the library.
  EXPECT_EQ(output.library().function_size(), 0);

  item.fetch = {"z0", "z1", "z2"};
  item.feed.emplace_back("x0", test::AsScalar<float>(3.14f));
  item.feed.emplace_back("x1", test::AsScalar<float>(2.7f));
  item.feed.emplace_back("x2", test::AsScalar<float>(1.0f));
  item.feed.emplace_back("x4", test::AsScalar<float>(-1.0f));
  item.feed.emplace_back("x3", test::AsScalar<int>(1234));
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  test::ExpectTensorEqual<float>(tensors_expected[1], tensors[1]);
  test::ExpectTensorEqual<int>(tensors_expected[2], tensors[2]);
}

TEST_F(FunctionOptimizerTest, InlineFunction_FunctionWithoutInput) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);

  const Tensor kTwo = test::AsScalar<int64_t>(2);
  FunctionDef func = FunctionDefHelper::Define(
      // Name
      "GenerateTwo",
      // Args
      {},
      // Return value
      {"o: T"},
      // Attr def
      {"T: {float, double}"},
      // Nodes
      {{{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
       {{"o"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}}});

  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("y", "GenerateTwo", {}, {{"T", DT_FLOAT}}, kDevice),
       NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, kDevice)},
      // FunctionLib
      {
          func,
      });

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Function call was removed from the graph.
  for (const NodeDef& node : output.node()) {
    EXPECT_NE(node.op(), "GenerateTwo");
  }
  // And the function itself was removed from the library.
  EXPECT_EQ(output.library().function_size(), 0);

  item.fetch = {"z"};
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, InlineFunction_FunctionWithNestedFunctionCall) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);

  // Define square via function library:
  //   MySquare(x) = MyMul(x, x)

  FunctionDef mul_func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"output"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "output:z:0"}});

  FunctionDef square_func = FunctionDefHelper::Create(
      "MySquare", {"x:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"output"}, "MyMul", {"x", "x"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "output:z:0"}});

  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("square", "MySquare", {"a"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("outputs", "Identity", {"square:0"}, {{"T", DT_FLOAT}}, kDevice)},
      // FunctionLib
      {mul_func, square_func});

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Function calls were removed from the graph.
  for (const NodeDef& node : output.node()) {
    EXPECT_NE(node.op(), "MySquare");
    EXPECT_NE(node.op(), "MyMul");
  }
  // And functions were removed from the library.
  EXPECT_EQ(output.library().function_size(), 0);

  item.fetch = {"outputs"};
  item.feed.emplace_back("a", test::AsScalar<float>(2.0f));
  auto tensors_expected = EvaluateFetchNodes(item);

  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);

  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, InlineSymbolicGradient_TestFunc) {
  FunctionOptimizer optimizer(RewriterConfig::ON, true);

  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  FunctionDef func = FunctionDefHelper::Define(
      "TestFunc", {"x:float", "y:float"}, {"l:float"}, {},
      {
          {{"z"}, "Add", {"x", "y"}, {{"T", DT_FLOAT}}},
          FunctionDefHelper::Const("zero", 0),
          FunctionDefHelper::Const("one", 1),
          {{"r"}, "Rank", {"z"}, {{"T", DT_FLOAT}}},
          {{"indices"}, "Range", {"zero", "r", "one"}},
          {{"l"}, "Sum", {"z", "indices"}, {{"T", DT_FLOAT}}},
      });

  auto x = ops::Const(scope, 1.0f);
  auto y = ops::Const(scope, 2.0f);
  auto dl = ops::Const(scope, 3.0f);

  NameAttrList fn;
  fn.set_name("TestFunc");
  (*fn.mutable_attr())["T"].set_type(DT_FLOAT);
  auto g0 = ops::SymbolicGradient(scope, std::initializer_list<Input>{x, y, dl},
                                  {DT_FLOAT, DT_FLOAT}, fn);
  auto out1 = ops::Identity(scope.WithOpName("out1"), g0.output[0]);
  auto out2 = ops::Identity(scope.WithOpName("out2"), g0.output[1]);

  GrapplerItem item;
  TF_EXPECT_OK(scope.ToGraphDef(&item.graph));
  *item.graph.mutable_library()->add_function() = func;

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // SymbolicGradient calls were removed from the graph.
  for (const NodeDef& node : output.node()) {
    EXPECT_NE(node.op(), "SymbolicGradient");
  }
  // And functions were removed from the library.
  EXPECT_EQ(output.library().function_size(), 0);

  std::vector<Tensor> expected =
      EvaluateNodes(item.graph, {"out1", "out2"}, {});
  std::vector<Tensor> optimized = EvaluateNodes(output, {"out1", "out2"}, {});
  test::ExpectTensorEqual<float>(expected[0], optimized[0]);
  test::ExpectTensorEqual<float>(expected[1], optimized[1]);
}

TEST_F(FunctionOptimizerTest, InlineSymbolicGradient_IdentityFunc) {
  FunctionOptimizer optimizer(RewriterConfig::ON, true);

  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  FunctionDef func = FunctionDefHelper::Create(
      // Name
      "Identity_func",
      // Args
      {"in: float"},
      // Return values
      {"out: float"},
      // Attr def
      {},
      // Nodes
      {{{"Identity"}, "Identity", {"in"}, {{"T", DT_FLOAT}}}},
      // Mapping
      {{"out", "Identity:output:0"}});

  auto x = ops::Const(scope, 1.0f, {3, 5, 7});
  auto z = ops::Const(scope, 3.0f, {3, 5, 7});

  NameAttrList fn;
  fn.set_name("Identity_func");
  auto g0 = ops::SymbolicGradient(scope, std::initializer_list<Input>{x, z},
                                  {DT_FLOAT}, fn);
  auto out = ops::Identity(scope.WithOpName("out"), g0.output[0]);

  GrapplerItem item;
  TF_EXPECT_OK(scope.ToGraphDef(&item.graph));
  *item.graph.mutable_library()->add_function() = func;

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // SymbolicGradient calls were removed from the graph.
  for (const NodeDef& node : output.node()) {
    EXPECT_NE(node.op(), "SymbolicGradient");
  }
  // And functions were removed from the library.
  EXPECT_EQ(output.library().function_size(), 0);

  std::vector<Tensor> expected = EvaluateNodes(item.graph, {"out"}, {});
  std::vector<Tensor> optimized = EvaluateNodes(output, {"out"}, {});
  test::ExpectTensorEqual<float>(expected[0], optimized[0]);
}

TEST_F(FunctionOptimizerTest, InlineSymbolicGradientNoInlineFunc) {
  FunctionOptimizer optimizer(RewriterConfig::ON, true);

  FunctionDef func = FunctionDefHelper::Define(
      "TestFunc", {"x:float", "y:float"}, {"l:float"}, {},
      {
          {{"z"}, "Add", {"x", "y"}, {{"T", DT_FLOAT}}},
          FunctionDefHelper::Const("zero", 0),
          FunctionDefHelper::Const("one", 1),
          {{"r"}, "Rank", {"z"}, {{"T", DT_FLOAT}}},
          {{"indices"}, "Range", {"zero", "r", "one"}},
          {{"l"}, "Sum", {"z", "indices"}, {{"T", DT_FLOAT}}},
      });
  (*func.mutable_attr())["_noinline"].set_b(true);

  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(scope, 1.0f);
  auto y = ops::Const(scope, 2.0f);
  auto dl = ops::Const(scope, 3.0f);

  NameAttrList fn;
  fn.set_name("TestFunc");
  (*fn.mutable_attr())["T"].set_type(DT_FLOAT);
  auto g0 = ops::SymbolicGradient(scope, std::initializer_list<Input>{x, y, dl},
                                  {DT_FLOAT, DT_FLOAT}, fn);
  auto out1 = ops::Identity(scope.WithOpName("out1"), g0.output[0]);
  auto out2 = ops::Identity(scope.WithOpName("out2"), g0.output[1]);

  GrapplerItem item;
  TF_EXPECT_OK(scope.ToGraphDef(&item.graph));
  *item.graph.mutable_library()->add_function() = func;

  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  // The optimizer should succeed but the graphs should be the same.
  TF_EXPECT_OK(status);
  CompareGraphs(item.graph, output);
}

TEST_F(FunctionOptimizerTest, InlineIndirectFunctionSimpleFunction) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  FunctionOptimizer optimizer(RewriterConfig::AGGRESSIVE, true);

  FunctionDef mul_func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "mul:z:0"}});

  // Build a graph to compute c = MyMul(a, b)
  GrapplerItem item;
  item.fetch = {"d"};
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("c", "PartitionedCall", {"a", "b"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("MyMul", {{"T", DT_FLOAT}})}},
            kDevice),
       NDef("d", "Identity", {"c"}, {{"T", DT_FLOAT}}, kDevice)},
      {mul_func} /* Function library */);

  Tensor pi = test::AsScalar<float>(3.14f);
  item.feed.emplace_back("a", pi);
  item.feed.emplace_back("b", pi);

  const string input_x = "Func/c/input/_0";
  const string input_y = "Func/c/input/_1";
  const string output_z = "Func/c/output/_2";

  // If device set is empty, inlined function body must not be placed.
  {
    GraphDef optimized_graph;
    TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));

    GraphDef expected = test::function::GDef(
        {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
         NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),

         // Function body nodes copy only job/task/replica parts of device
         // assignment, and function input nodes must copy full device
         // assignment from input arguments. Optimized graph is not fully
         // placed.
         NDef(input_x, "Identity", {"a"}, {{"T", DT_FLOAT}}, kDevice),
         NDef(input_y, "Identity", {"b"}, {{"T", DT_FLOAT}}, kDevice),
         // NOTE(ezhulenev): Currently multi-device function inlining placer
         // strategy will override all empty devices with function call device.
         NDef("c/mul", "Mul", {input_x, input_y}, {{"T", DT_FLOAT}}, kDevice),
         NDef(output_z, "Identity", {"c/mul"}, {{"T", DT_FLOAT}}),

         NDef("d", "Identity", {output_z}, {{"T", DT_FLOAT}}, kDevice)},
        // Function library.
        {mul_func});

    CompareGraphs(expected, optimized_graph);

    GrapplerItem optimized = item.WithGraph(std::move(optimized_graph));
    auto tensors_expected = EvaluateFetchNodes(item);
    auto tensors = EvaluateFetchNodes(optimized);
    ASSERT_EQ(tensors_expected.size(), 1);
    ASSERT_EQ(tensors.size(), tensors_expected.size());
    test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  }

  // If device set is not empty, inlined function body must be placed.
  {
    GraphDef optimized_graph;
    TF_EXPECT_OK(item.AddDevice(kDevice));
    TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));

    GraphDef expected = test::function::GDef(
        {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
         NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),

         NDef(input_x, "Identity", {"a"}, {{"T", DT_FLOAT}}, kDevice),
         NDef(input_y, "Identity", {"b"}, {{"T", DT_FLOAT}}, kDevice),
         NDef("c/mul", "Mul", {input_x, input_y}, {{"T", DT_FLOAT}}, kDevice),
         NDef(output_z, "Identity", {"c/mul"}, {{"T", DT_FLOAT}}, kDevice),

         NDef("d", "Identity", {output_z}, {{"T", DT_FLOAT}}, kDevice)},
        // Function library.
        {mul_func});

    CompareGraphs(expected, optimized_graph);

    GrapplerItem optimized = item.WithGraph(std::move(optimized_graph));
    auto tensors_expected = EvaluateFetchNodes(item);
    auto tensors = EvaluateFetchNodes(optimized);
    ASSERT_EQ(tensors_expected.size(), 1);
    ASSERT_EQ(tensors.size(), tensors_expected.size());
    test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  }
}

TEST_F(FunctionOptimizerTest, InlineIndirectFunctionWithControlDependencies) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  FunctionOptimizer optimizer(RewriterConfig::ON, true);

  const Tensor kOne = test::AsScalar<float>(1.0);
  const Tensor kTwo = test::AsScalar<float>(2.0);
  const TensorShape scalar = TensorShape({});

  // Compute `x*y` and add `1.0` to the variable.
  FunctionDef mul_func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T", "v: resource"}, {"z:T"}, {"T: {float, double}"},
      {{{"one"}, "Const", {}, {{"value", kOne}, {"dtype", DT_FLOAT}}},
       {{"add"},
        "AssignAddVariableOp",
        {"v", "one:output:0"},
        {{"dtype", DT_FLOAT}}},
       {{"mul"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "mul:z:0"}},
      /* Control output to ensure that side effects will be executed. */
      {{"size_effects", "add"}});

  // Build a graph to compute:
  //   a = Placeholder
  //   b = Placeholder
  //   v = VarHandleOp(init = a)
  //   f1 = MyMul(a, b, v)
  //   f2 = MyMul(f1, f1, v)
  //   return [f2, v]
  GrapplerItem item;
  TF_EXPECT_OK(item.AddDevice(kDevice));  // device for placing inlined function
  item.fetch = {"out_1", "out_2"};
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),

       // Initialize variable with one of the placeholders.
       NDef("v", "VarHandleOp", {}, {{"dtype", DT_FLOAT}, {"shape", scalar}}),
       NDef("init_v", "AssignVariableOp", {"v", "a"}, {{"dtype", DT_FLOAT}},
            kDevice),

       // Call function first time.
       NDef("f1", "PartitionedCall", {"a", "b", "v", "^init_v"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT, DT_RESOURCE}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("MyMul", {{"T", DT_FLOAT}})}},
            kDevice),

       // Call function second time.
       NDef("f2", "PartitionedCall", {"f1", "f1", "v", "^f1"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT, DT_RESOURCE}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("MyMul", {{"T", DT_FLOAT}})}},
            kDevice),

       // Return result of multiplication and a current value of the variable.
       NDef("out_1", "Identity", {"f2"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("out_2", "ReadVariableOp", {"v", "^f1", "^f2"},
            {{"dtype", DT_FLOAT}}, kDevice)},

      // Function library.
      {mul_func});

  GraphDef optimized_graph;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));

  GraphDef expected = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),

       // Initialize variable with one of the placeholders.
       NDef("v", "VarHandleOp", {}, {{"dtype", DT_FLOAT}, {"shape", scalar}},
            kDevice),
       NDef("init_v", "AssignVariableOp", {"v", "a"}, {{"dtype", DT_FLOAT}},
            kDevice),

       // Function body of a first function call inlined into the graph.
       NDef("Func/f1/input_control_node/_0", "NoOp", {"^init_v"}, {}, kDevice),

       NDef("Func/f1/input/_1", "Identity",  // input: 'x'
            {"a", "^Func/f1/input_control_node/_0"}, {{"T", DT_FLOAT}},
            kDevice),
       NDef("Func/f1/input/_2", "Identity",  // input: 'y'
            {"b", "^Func/f1/input_control_node/_0"}, {{"T", DT_FLOAT}},
            kDevice),
       NDef("Func/f1/input/_3", "Identity",  // input: 'v'
            {"v", "^Func/f1/input_control_node/_0"}, {{"T", DT_RESOURCE}},
            kDevice),

       NDef("f1/one", "Const", {"^Func/f1/input_control_node/_0"},
            {{"dtype", DT_FLOAT}, {"value", kOne}}, kDevice),
       NDef("f1/mul", "Mul", {"Func/f1/input/_1", "Func/f1/input/_2"},
            {{"T", DT_FLOAT}}, kDevice),
       NDef("f1/add", "AssignAddVariableOp", {"Func/f1/input/_3", "f1/one"},
            {{"dtype", DT_FLOAT}}, kDevice),

       NDef("Func/f1/output/_4", "Identity", {"f1/mul"}, {{"T", DT_FLOAT}},
            kDevice),
       NDef("Func/f1/output_control_node/_5", "NoOp", {"^f1/add"}, {}, kDevice),

       // Function body of a second function call also inlined into the graph,
       // and input nodes read from the output nodes of the first function call.
       NDef("Func/f2/input_control_node/_6", "NoOp",
            {"^Func/f1/output_control_node/_5"}, {}, kDevice),

       NDef("Func/f2/input/_7", "Identity",  // input: 'x'
            {"Func/f1/output/_4", "^Func/f2/input_control_node/_6"},
            {{"T", DT_FLOAT}}, kDevice),
       NDef("Func/f2/input/_8", "Identity",  // input: 'y'
            {"Func/f1/output/_4", "^Func/f2/input_control_node/_6"},
            {{"T", DT_FLOAT}}, kDevice),
       NDef("Func/f2/input/_9", "Identity",  // input: 'v'
            {"v", "^Func/f2/input_control_node/_6"}, {{"T", DT_RESOURCE}},
            kDevice),

       NDef("f2/one", "Const", {"^Func/f2/input_control_node/_6"},
            {{"dtype", DT_FLOAT}, {"value", kOne}}, kDevice),
       NDef("f2/add", "AssignAddVariableOp", {"Func/f2/input/_9", "f2/one"},
            {{"dtype", DT_FLOAT}}, kDevice),
       NDef("f2/mul", "Mul", {"Func/f2/input/_7", "Func/f2/input/_8"},
            {{"T", DT_FLOAT}}, kDevice),

       NDef("Func/f2/output/_10", "Identity", {"f2/mul"}, {{"T", DT_FLOAT}},
            kDevice),
       NDef("Func/f2/output_control_node/_11", "NoOp", {"^f2/add"}, {},
            kDevice),

       // Return values read from inlined output nodes.
       NDef("out_1", "Identity", {"Func/f2/output/_10"}, {{"T", DT_FLOAT}},
            kDevice),
       NDef("out_2", "ReadVariableOp",
            {"v", "^Func/f1/output_control_node/_5",
             "^Func/f2/output_control_node/_11"},
            {{"dtype", DT_FLOAT}}, kDevice)},

      // Function library.
      {mul_func});

  CompareGraphs(expected, optimized_graph);

  item.feed.emplace_back("a", kOne);
  item.feed.emplace_back("b", kTwo);

  auto tensors_expected = EvaluateFetchNodes(item);
  ASSERT_EQ(tensors_expected.size(), 2);
  EXPECT_EQ(tensors_expected[0].flat<float>()(0), 4.0);  // mul
  EXPECT_EQ(tensors_expected[1].flat<float>()(0), 3.0);  // read variable

  GrapplerItem optimized = item.WithGraph(std::move(optimized_graph));
  auto tensors = EvaluateFetchNodes(optimized);
  ASSERT_EQ(tensors.size(), 2);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  test::ExpectTensorEqual<float>(tensors_expected[1], tensors[1]);
}

TEST_F(FunctionOptimizerTest, InlineIndirectFunctionWithDevicePlacement) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  FunctionOptimizer optimizer(RewriterConfig::AGGRESSIVE, true);

  FunctionDef mul_func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "mul:z:0"}});
  // Add device placement spec to the function body node.
  (*mul_func.mutable_node_def())[0].set_device("/device:CPU:1");

  // We need fully defined device names to run the placer for inlined function.
  const string cpu0 = "/job:work/replica:1/task:1/device:CPU:0";
  const string cpu1 = "/job:work/replica:1/task:1/device:CPU:1";

  // Build a graph to compute c = MyMul(a, b)
  GrapplerItem item;
  item.fetch = {"d"};
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, cpu0),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, cpu1),
       NDef("c", "PartitionedCall", {"a", "b"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("MyMul", {{"T", DT_FLOAT}})}},
            cpu0),
       NDef("d", "Identity", {"c"}, {{"T", DT_FLOAT}}, cpu0)},
      // Function library.
      {mul_func});
  ASSERT_TRUE(item.InferDevicesFromGraph().ok());

  GraphDef optimized_graph;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));

  const string input_x = "Func/c/input/_0";
  const string input_y = "Func/c/input/_1";
  const string output_z = "Func/c/output/_2";

  GraphDef expected = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, cpu0),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, cpu1),

       // Function must be inlined and `mul` node placed on a requested device,
       // and input `Identity` nodes must be colocated with their source nodes.
       NDef(input_x, "Identity", {"a"}, {{"T", DT_FLOAT}}, cpu0),
       NDef(input_y, "Identity", {"b"}, {{"T", DT_FLOAT}}, cpu1),
       NDef("c/mul", "Mul", {input_x, input_y}, {{"T", DT_FLOAT}}, cpu1),
       NDef(output_z, "Identity", {"c/mul"}, {{"T", DT_FLOAT}}, cpu1),

       NDef("d", "Identity", {output_z}, {{"T", DT_FLOAT}}, cpu0)},
      // Function library.
      {mul_func});

  CompareGraphs(expected, optimized_graph);
}

TEST_F(FunctionOptimizerTest,
       InlineMultipleIndirectFunctionWithDevicePlacement) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  FunctionOptimizer optimizer(RewriterConfig::AGGRESSIVE, true);

  FunctionDef mul_func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "mul:z:0"}});
  // Add device placement spec to the function body node.
  (*mul_func.mutable_node_def())[0].set_device("/device:CPU:1");

  // We need fully defined device names to run the placer for inlined function.
  const string cpu0 = "/job:work/replica:1/task:1/device:CPU:0";
  const string cpu1 = "/job:work/replica:1/task:1/device:CPU:1";

  // Build a graph to compute c = MyMul(a, b)
  GrapplerItem item;
  item.fetch = {"e"};
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, cpu0),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, cpu1),
       NDef("c", "PartitionedCall", {"a", "b"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("MyMul", {{"T", DT_FLOAT}})}},
            cpu0),
       NDef("d", "PartitionedCall", {"a", "c"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("MyMul", {{"T", DT_FLOAT}})}},
            cpu0),
       NDef("e", "Identity", {"d"}, {{"T", DT_FLOAT}}, cpu0)},
      // Function library.
      {mul_func});
  ASSERT_TRUE(item.InferDevicesFromGraph().ok());

  GraphDef optimized_graph;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));

  const string input_c_x = "Func/c/input/_0";
  const string input_c_y = "Func/c/input/_1";
  const string output_c_z = "Func/c/output/_2";
  const string input_d_x = "Func/d/input/_3";
  const string input_d_y = "Func/d/input/_4";
  const string output_d_z = "Func/d/output/_5";

  GraphDef expected = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, cpu0),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, cpu1),

       // Function must be inlined and `mul` node placed on a requested device,
       // and input/output `Identity` nodes must be colocated with their
       // source nodes.
       NDef(input_c_x, "Identity", {"a"}, {{"T", DT_FLOAT}}, cpu0),
       NDef(input_c_y, "Identity", {"b"}, {{"T", DT_FLOAT}}, cpu1),
       NDef("c/mul", "Mul", {input_c_x, input_c_y}, {{"T", DT_FLOAT}}, cpu1),
       NDef(output_c_z, "Identity", {"c/mul"}, {{"T", DT_FLOAT}}, cpu1),

       // Function must be inlined and `mul` node placed on a requested device,
       // and input/output `Identity` nodes must be colocated with their
       // source nodes.
       NDef(input_d_x, "Identity", {"a"}, {{"T", DT_FLOAT}}, cpu0),
       NDef(input_d_y, "Identity", {output_c_z}, {{"T", DT_FLOAT}}, cpu1),
       NDef("d/mul", "Mul", {input_d_x, input_d_y}, {{"T", DT_FLOAT}}, cpu1),
       NDef(output_d_z, "Identity", {"d/mul"}, {{"T", DT_FLOAT}}, cpu1),

       NDef("e", "Identity", {output_d_z}, {{"T", DT_FLOAT}}, cpu0)},
      // Function library.
      {mul_func});

  CompareGraphs(expected, optimized_graph);
}

TEST_F(FunctionOptimizerTest,
       InlineIndirectFunctionWithControlDependencyAndNoSideEffects) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  FunctionOptimizer optimizer(RewriterConfig::AGGRESSIVE, true);

  const Tensor kOne = test::AsScalar<float>(1.0);
  const Tensor kTwo = test::AsScalar<float>(2.0);
  const TensorShape scalar = TensorShape({});

  // MyMul doesn't have any side-effectful nodes in the function body, but the
  // optimized graph has a control dependency edge `f1->f2`.
  FunctionDef mul_func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "mul:z:0"}});

  // Build a graph to compute:
  //   a = Placeholder
  //   b = Placeholder
  //   f1 = MyMul(a, b)
  //   f2 = MyMul(a, b, ^f1)  <-- control dependency on inlined function!
  //   return f2
  GrapplerItem item;
  TF_EXPECT_OK(item.AddDevice(kDevice));  // device for placing inlined function
  item.fetch = {"out"};
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),

       NDef("c", "NoOp", {}, {}, kDevice),

       // Call function first time.
       NDef("f1", "PartitionedCall", {"a", "b", "^c"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("MyMul", {{"T", DT_FLOAT}})}},
            kDevice),

       // Call function second time.
       NDef("f2", "PartitionedCall", {"f1", "f1", "^f1"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("MyMul", {{"T", DT_FLOAT}})}},
            kDevice),

       // Return result of f2.
       NDef("out", "Identity", {"f2"}, {{"T", DT_FLOAT}}, kDevice)},

      // Function library.
      {mul_func});

  GraphDef optimized_graph;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));

  GraphDef expected = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),

       NDef("c", "NoOp", {}, {}, kDevice),

       // Function body of a first function call inlined into the graph.
       NDef("Func/f1/input_control_node/_0", "NoOp", {"^c"}, {}, kDevice),

       NDef("Func/f1/input/_1", "Identity",  // input: 'x'
            {"a", "^Func/f1/input_control_node/_0"}, {{"T", DT_FLOAT}},
            kDevice),
       NDef("Func/f1/input/_2", "Identity",  // input: 'y'
            {"b", "^Func/f1/input_control_node/_0"}, {{"T", DT_FLOAT}},
            kDevice),

       NDef("f1/mul", "Mul", {"Func/f1/input/_1", "Func/f1/input/_2"},
            {{"T", DT_FLOAT}}, kDevice),

       NDef("Func/f1/output/_3", "Identity", {"f1/mul"}, {{"T", DT_FLOAT}},
            kDevice),
       // Control input from `input_control_node` node is added to ensure
       // correct frame execution.
       NDef("Func/f1/output_control_node/_4", "NoOp",
            {"^Func/f1/input_control_node/_0"}, {}, kDevice),

       // Function body of a second function call also inlined into the graph,
       // and input nodes read directly from the output nodes of the first
       // function call, and control dependency edge removed.
       NDef("Func/f2/input_control_node/_5", "NoOp",
            {"^Func/f1/output_control_node/_4"}, {}, kDevice),

       NDef("Func/f2/input/_6", "Identity",
            {"Func/f1/output/_3", "^Func/f2/input_control_node/_5"},
            {{"T", DT_FLOAT}}, kDevice),
       NDef("Func/f2/input/_7", "Identity",
            {"Func/f1/output/_3", "^Func/f2/input_control_node/_5"},
            {{"T", DT_FLOAT}}, kDevice),

       NDef("f2/mul", "Mul", {"Func/f2/input/_6", "Func/f2/input/_7"},
            {{"T", DT_FLOAT}}, kDevice),
       NDef("Func/f2/output/_8", "Identity", {"f2/mul"}, {{"T", DT_FLOAT}},
            kDevice),

       // Return directly from output node of f2.
       NDef("out", "Identity", {"Func/f2/output/_8"}, {{"T", DT_FLOAT}},
            kDevice)},

      // Function library.
      {mul_func});

  CompareGraphs(expected, optimized_graph);

  item.feed.emplace_back("a", kOne);
  item.feed.emplace_back("b", kTwo);

  auto tensors_expected = EvaluateFetchNodes(item);
  ASSERT_EQ(tensors_expected.size(), 1);

  GrapplerItem optimized = item.WithGraph(std::move(optimized_graph));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, InlineIndirectFunctionDoNotInlineDeadOutputs) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  FunctionOptimizer optimizer(RewriterConfig::AGGRESSIVE, true);

  // Function output can be dead.
  FunctionDef dead_outputs = FunctionDefHelper::Create(
      "DeadOutputs", {"x:T", "cond:bool"}, {"z:T"}, {"T: {float, double}"},
      {
          {{"switch"}, "Switch", {"x", "cond"}, {{"T", "$T"}}},
          {{"if_false"}, "Identity", {"switch:output_false:0"}, {{"T", "$T"}}},
          {{"if_true"}, "Identity", {"switch:output_true:0"}, {{"T", "$T"}}},
      },
      /* Mapping between function returns and function node outputs. */
      {{"z", "if_false:output:0"}});

  // Simple proxy functions that calls DeadOutputs from the function body.
  FunctionDef proxy_func = FunctionDefHelper::Create(
      "Proxy", {"x:T", "cond:bool"}, {"z:T"}, {"T: {float, double}"},
      {{{"dead"}, "DeadOutputs", {"x", "cond"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "dead:z:0"}});

  // Build a graph to compute:
  //   a: float
  //   b: bool
  //   fn0 = DeadOutputs(x, b)
  //   fn1 = Proxy(x, b)
  //   out0 = Identity(fn0)
  //   out1 = Identity(fn1)
  //   return [out0, out1]
  //
  GrapplerItem item;
  item.fetch = {"out0", "out1"};
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_BOOL}}, kDevice),

       NDef("fn0", "PartitionedCall", {"a", "b"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_BOOL}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("DeadOutputs", {{"T", DT_FLOAT}})}},
            kDevice),

       NDef("fn1", "PartitionedCall", {"a", "b"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_BOOL}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("Proxy", {{"T", DT_FLOAT}})}},
            kDevice),

       NDef("out0", "Identity", {"fn0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("out1", "Identity", {"fn1"}, {{"T", DT_FLOAT}}, kDevice)},
      // Function library.
      {dead_outputs, proxy_func});

  GraphDef optimized_graph;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));

  GraphDef expected = item.graph;
  CompareGraphs(expected, optimized_graph);

  const Tensor one = test::AsScalar<float>(1.0);
  item.feed.emplace_back("a", one);
  item.feed.emplace_back("b", test::AsScalar<bool>(false));

  auto tensors = EvaluateFetchNodes(item);
  ASSERT_EQ(tensors.size(), 2);
  test::ExpectTensorEqual<float>(tensors[0], one);
  test::ExpectTensorEqual<float>(tensors[1], one);
}

TEST_F(FunctionOptimizerTest, InlineIndirectFunctionWithMergedDeadTensors) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  FunctionOptimizer optimizer(RewriterConfig::AGGRESSIVE, true);

  // Function output can't be dead because it goes through the Merge node.
  FunctionDef no_dead_outputs = FunctionDefHelper::Create(
      "NoDeadOutputs", {"x:T", "cond:bool"}, {"z:T"}, {"T: {float, double}"},
      {
          {{"switch"}, "Switch", {"x", "cond"}, {{"T", "$T"}}},
          {{"if_false"}, "Identity", {"switch:output_false:0"}, {{"T", "$T"}}},
          {{"if_true"}, "Identity", {"switch:output_true:0"}, {{"T", "$T"}}},
          {{"merge"},
           "Merge",
           {"if_false:output:0", "if_true:output:0"},
           {{"T", "$T"}, {"N", 2}}},
      },
      /* Mapping between function returns and function node outputs. */
      {{"z", "merge:output:0"}});

  // Build a graph to compute:
  //   a: float
  //   b: bool
  //   d = DeadOutputs(x, b)
  //   out = Identity(d)
  //   return out
  //
  GrapplerItem item;
  TF_EXPECT_OK(item.AddDevice(kDevice));  // device for placing inlined function
  item.fetch = {"out"};
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_BOOL}}, kDevice),

       NDef("fn", "PartitionedCall", {"a", "b"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_BOOL}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("NoDeadOutputs", {{"T", DT_FLOAT}})}},
            kDevice),

       NDef("out", "Identity", {"fn"}, {{"T", DT_FLOAT}}, kDevice)},
      // Function library.
      {no_dead_outputs});

  GraphDef optimized_graph;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));

  GraphDef expected = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_BOOL}}, kDevice),

       // Function body of a first function call inlined into the graph.
       NDef("Func/fn/input/_0", "Identity", {"a"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("Func/fn/input/_1", "Identity", {"b"}, {{"T", DT_BOOL}}, kDevice),

       NDef("fn/switch", "Switch", {"Func/fn/input/_0", "Func/fn/input/_1"},
            {{"T", DT_FLOAT}}, kDevice),
       NDef("fn/if_false", "Identity", {"fn/switch"}, {{"T", DT_FLOAT}},
            kDevice),
       NDef("fn/if_true", "Identity", {"fn/switch:1"}, {{"T", DT_FLOAT}},
            kDevice),
       NDef("fn/merge", "Merge", {"fn/if_false", "fn/if_true"},
            {{"T", DT_FLOAT}, {"N", 2}}, kDevice),

       NDef("Func/fn/output/_2", "Identity", {"fn/merge"}, {{"T", DT_FLOAT}},
            kDevice),

       // Return directly from inlined function output node.
       NDef("out", "Identity", {"Func/fn/output/_2"}, {{"T", DT_FLOAT}},
            kDevice)},

      // Function library.
      {no_dead_outputs});

  CompareGraphs(expected, optimized_graph);

  const Tensor one = test::AsScalar<float>(1.0);
  item.feed.emplace_back("a", one);
  item.feed.emplace_back("b", test::AsScalar<bool>(false));

  auto tensors_expected = EvaluateFetchNodes(item);
  ASSERT_EQ(tensors_expected.size(), 1);

  GrapplerItem optimized = item.WithGraph(std::move(optimized_graph));
  auto tensors = EvaluateFetchNodes(optimized);
  ASSERT_EQ(tensors.size(), 1);

  test::ExpectTensorEqual<float>(tensors[0], tensors_expected[0]);
}

TEST_F(FunctionOptimizerTest, InlineIndirectFunctionWithNestedFunctionCall) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  FunctionOptimizer optimizer(RewriterConfig::AGGRESSIVE, true);

  FunctionDef mul_func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "mul:z:0"}});

  // `Square` implemented in terms of PartitionedCall to `MyMul`.
  FunctionDef square_func = FunctionDefHelper::Create(
      "MySquare", {"x:T"}, {"output:T"}, {"T: {float, double}"},
      {{{"square"},
        "PartitionedCall",
        {"x", "x"},
        {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
         {"Tout", DataTypeSlice{DT_FLOAT}},
         {"f", FDH::FunctionRef("MyMul", {{"T", DT_FLOAT}})}}}},
      /* Mapping between function returns and function node outputs. */
      {{"output", "square:output:0"}});

  // Build a graph to compute:
  //   b = Square(a)
  //   c = Identity(b)
  //   return c
  GrapplerItem item;
  TF_EXPECT_OK(item.AddDevice(kDevice));  // device for placing inlined function
  item.fetch = {"c"};
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "PartitionedCall", {"a"},
            {{"Tin", DataTypeSlice{DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("MySquare", {{"T", DT_FLOAT}})}},
            kDevice),
       NDef("c", "Identity", {"b"}, {{"T", DT_FLOAT}}, kDevice)},
      /* Function library */
      {mul_func, square_func});

  GraphDef optimized_graph;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));

  GraphDef expected = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),

       // Inlined inputs of `b` node.
       NDef("Func/b/input/_0", "Identity", {"a"}, {{"T", DT_FLOAT}}, kDevice),

       // Inlined inputs of `square` node inside inlined `MySquare` function.
       NDef("Func/b/square/input/_2", "Identity", {"Func/b/input/_0"},
            {{"T", DT_FLOAT}}, kDevice),
       NDef("Func/b/square/input/_3", "Identity", {"Func/b/input/_0"},
            {{"T", DT_FLOAT}}, kDevice),

       // Inlined mul node from the `MyMul` function.
       NDef("b/square/mul", "Mul",
            {"Func/b/square/input/_2", "Func/b/square/input/_3"},
            {{"T", DT_FLOAT}}, kDevice),

       NDef("Func/b/square/output/_4", "Identity", {"b/square/mul"},
            {{"T", DT_FLOAT}}, kDevice),
       NDef("Func/b/output/_1", "Identity", {"Func/b/square/output/_4"},
            {{"T", DT_FLOAT}}, kDevice),

       NDef("c", "Identity", {"Func/b/output/_1"}, {{"T", DT_FLOAT}}, kDevice)},
      // Function library.
      {mul_func});

  CompareGraphs(expected, optimized_graph);

  Tensor three = test::AsScalar<float>(3.0f);
  item.feed.emplace_back("a", three);

  GrapplerItem optimized = item.WithGraph(std::move(optimized_graph));
  auto tensors_expected = EvaluateFetchNodes(item);
  auto tensors = EvaluateFetchNodes(optimized);
  ASSERT_EQ(tensors_expected.size(), 1);
  ASSERT_EQ(tensors.size(), tensors_expected.size());
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

GrapplerItem ConditionalAdd() {
  // Returns the conditional (is_add) ? a + b : a * b;
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  FunctionDef add_func = FDH::Create(
      "MyAdd", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"add"}, "Add", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "add:z:0"}});

  FunctionDef mul_func = FDH::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "mul:z:0"}});

  // Compute: return cond ? a + b : a * b
  FunctionDef add_or_mul_func = FDH::Create(
      "AddOrMul", {"cond:bool", "x:float", "y:float"}, {"z:float"}, {},
      {
          {{"if_node"},
           "If",
           {"cond", "x", "y"},
           {
               {"Tcond", DT_BOOL},
               {"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
               {"Tout", DataTypeSlice{DT_FLOAT}},
               {"then_branch", FDH::FunctionRef("MyAdd", {{"T", DT_FLOAT}})},
               {"else_branch", FDH::FunctionRef("MyMul", {{"T", DT_FLOAT}})},
               {"_lower_using_switch_merge", true},
           }},
      },
      /* Mapping between function returns and function node outputs. */
      {{"z", "if_node:output:0"}}, {{"side_effect", "if_node"}});

  // Build a computation graph for:
  //   is_add: bool
  //   a: float
  //   b: float
  //   c = AddOrMul(is_add, a, b)  # is_add ? a + b : a * b
  //   d = Identity(c)
  //   return d

  // c = MyMul(a, b)
  GrapplerItem item;
  item.fetch = {"d"};
  item.graph = test::function::GDef(
      {NDef("is_add", "Placeholder", {}, {{"dtype", DT_BOOL}}, kDevice),
       NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),

       NDef("c", "PartitionedCall", {"is_add", "a", "b"},
            {{"Tin", DataTypeSlice{DT_BOOL, DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("AddOrMul")}},
            kDevice),

       NDef("d", "Identity", {"c", "^c"}, {{"T", DT_FLOAT}}, kDevice)},
      // Function library.
      {add_or_mul_func, add_func, mul_func});
  return item;
}

TEST_F(FunctionOptimizerTest, InlineIndirectFunctionWithFunctionalControlFlow) {
  FunctionOptimizer optimizer(RewriterConfig::AGGRESSIVE, true);

  // item.fetch['d'] == (is_add) ? a + b : a * b
  GrapplerItem item = ConditionalAdd();
  GraphDef optimized_graph;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));

  const auto count_nodes_with_op = [&](const string& op) {
    return absl::c_count_if(optimized_graph.node(), [&](const NodeDef& node) {
      return node.op() == op;
    });
  };

  // All `PartitionedCall` nodes in the optimized graph must be inlined, and
  // `If` node must be lowered to `Switch` and `Merge` nodes.
  EXPECT_EQ(count_nodes_with_op("PartitionedCall"), 0);
  EXPECT_EQ(count_nodes_with_op("If"), 0);
  EXPECT_EQ(count_nodes_with_op("Switch"), 3);
  EXPECT_EQ(count_nodes_with_op("Merge"), 2);

  GrapplerItem optimized = item.WithGraph(std::move(optimized_graph));

  Tensor one = test::AsScalar<float>(1.0);
  Tensor two = test::AsScalar<float>(2.0);
  Tensor three = test::AsScalar<float>(3.0);

  const auto feed_args = [&](bool is_add) {
    std::vector<std::pair<string, Tensor>> feed;
    feed.emplace_back("a", one);
    feed.emplace_back("b", two);
    feed.emplace_back("is_add", test::AsScalar<bool>(is_add));
    return feed;
  };

  {  // Check 'is_add == true': a + b
    item.feed = feed_args(true);
    optimized.feed = feed_args(true);

    auto tensors_expected = EvaluateFetchNodes(item);
    ASSERT_EQ(tensors_expected.size(), 1);
    test::ExpectTensorEqual<float>(tensors_expected[0], three);

    auto tensors = EvaluateFetchNodes(optimized);
    ASSERT_EQ(tensors.size(), tensors_expected.size());
    test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  }

  {  // Check 'is_add == false': a * b
    item.feed = feed_args(false);
    optimized.feed = feed_args(false);

    auto tensors_expected = EvaluateFetchNodes(item);
    ASSERT_EQ(tensors_expected.size(), 1);
    test::ExpectTensorEqual<float>(tensors_expected[0], two);

    auto tensors = EvaluateFetchNodes(optimized);
    ASSERT_EQ(tensors.size(), tensors_expected.size());
    test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  }
}

TEST_F(FunctionOptimizerTest, InlineIndirectFunctionDontLowerControlFlow) {
  FunctionOptimizer optimizer(RewriterConfig::AGGRESSIVE,
                              /*lower_control_flow=*/false);

  // item.fetch['d'] == (is_add) ? a + b : a * b
  GrapplerItem item = ConditionalAdd();
  GraphDef optimized_graph;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));

  const auto count_nodes_with_op = [&](const string& op) {
    return absl::c_count_if(optimized_graph.node(), [&](const NodeDef& node) {
      return node.op() == op;
    });
  };

  // All `PartitionedCall` nodes in the optimized graph must be inlined, and
  // `If` node must be lowered to `Switch` and `Merge` nodes.
  EXPECT_EQ(count_nodes_with_op("PartitionedCall"), 0);
  EXPECT_EQ(count_nodes_with_op("If"), 1);
  EXPECT_EQ(count_nodes_with_op("Switch"), 0);
  EXPECT_EQ(count_nodes_with_op("Merge"), 0);

  GrapplerItem optimized = item.WithGraph(std::move(optimized_graph));

  Tensor one = test::AsScalar<float>(1.0);
  Tensor two = test::AsScalar<float>(2.0);
  Tensor three = test::AsScalar<float>(3.0);

  const auto feed_args = [&](bool is_add) {
    std::vector<std::pair<string, Tensor>> feed;
    feed.emplace_back("a", one);
    feed.emplace_back("b", two);
    feed.emplace_back("is_add", test::AsScalar<bool>(is_add));
    return feed;
  };

  {  // Check 'is_add == true': a + b
    item.feed = feed_args(true);
    optimized.feed = feed_args(true);

    auto tensors_expected = EvaluateFetchNodes(item);
    ASSERT_EQ(tensors_expected.size(), 1);
    test::ExpectTensorEqual<float>(tensors_expected[0], three);

    auto tensors = EvaluateFetchNodes(optimized);
    ASSERT_EQ(tensors.size(), tensors_expected.size());
    test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  }

  {  // Check 'is_add == false': a * b
    item.feed = feed_args(false);
    optimized.feed = feed_args(false);

    auto tensors_expected = EvaluateFetchNodes(item);
    ASSERT_EQ(tensors_expected.size(), 1);
    test::ExpectTensorEqual<float>(tensors_expected[0], two);

    auto tensors = EvaluateFetchNodes(optimized);
    ASSERT_EQ(tensors.size(), tensors_expected.size());
    test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  }
}

TEST_F(FunctionOptimizerTest, SpecializeFunctionXTimesTwo) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);

  // Mark XTimesTwo as noinline.
  FunctionDef x_times_two = test::function::XTimesTwo();
  (*x_times_two.mutable_attr())["_noinline"].set_b(true);
  std::vector<FunctionDef> function_library = {x_times_two};

  // Build a graph to compute y = XTimesTwo(x).
  GrapplerItem item;
  item.id = "tf_graph";
  item.graph = test::function::GDef(
      {NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("y", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, kDevice)},
      function_library);

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Make sure that specialized function was added to the library and original
  // function was removed.
  EXPECT_EQ(1, output.library().function_size());
  EXPECT_EQ("XTimesTwo_specialized_for_y_at_tf_graph",
            output.library().function(0).signature().name());

  // And 'y' node is calling specialized function.
  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "y" && ++count) {
      EXPECT_EQ("XTimesTwo_specialized_for_y_at_tf_graph", node.op());
    }
  }
  EXPECT_EQ(1, count);

  // And that graph evaluation yields the same result.
  Tensor pi = test::AsScalar<float>(3.14f);
  item.fetch = {"z"};
  item.feed.emplace_back("x", pi);

  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, SpecializeIndirectFunctionXTimesTwo) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);

  // Mark XTimesTwo as noinline.
  FunctionDef x_times_two = test::function::XTimesTwo();
  (*x_times_two.mutable_attr())["_noinline"].set_b(true);
  std::vector<FunctionDef> function_library = {x_times_two};

  // Tensorflow graph:
  //   y = PartitionedCall[f=XTimesTwo, Tin=[DT_FLOAT], Tout=[DT_FLOAT]](x)
  GrapplerItem item;
  item.id = "tf_graph";
  item.graph = test::function::GDef(
      {NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("y", "PartitionedCall", {"x"},
            {{"Tin", DataTypeSlice{DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("XTimesTwo", {{"T", DT_FLOAT}})}},
            kDevice),
       NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, kDevice)},
      function_library);

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Make sure that specialized function was added to the library and original
  // function was removed.
  EXPECT_EQ(1, output.library().function_size());
  EXPECT_EQ("XTimesTwo_specialized_for_y_at_tf_graph",
            output.library().function(0).signature().name());

  // And 'y' node is calling specialized function.
  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "y" && ++count) {
      EXPECT_EQ("PartitionedCall", node.op());
      auto& func = AttrSlice(node).Find("f")->func();
      // Function calls into the specialized function.
      EXPECT_EQ("XTimesTwo_specialized_for_y_at_tf_graph", func.name());
      // And input/output types stay the same.
      auto& tin = AttrSlice(node).Find("Tin")->list();
      auto& tout = AttrSlice(node).Find("Tout")->list();
      ASSERT_EQ(1, tin.type_size());
      ASSERT_EQ(1, tout.type_size());
      EXPECT_EQ(DT_FLOAT, tin.type(0));
      EXPECT_EQ(DT_FLOAT, tout.type(0));
    }
  }
  EXPECT_EQ(1, count);

  // And that graph evaluation yields the same result.
  Tensor pi = test::AsScalar<float>(3.14f);
  item.fetch = {"z"};
  item.feed.emplace_back("x", pi);

  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, SpecializeFunctionPushDownConstInput) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);

  FunctionDef mul_func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"output"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "output:z:0"}});

  // Mark MyMul as noinline.
  (*mul_func.mutable_attr())["_noinline"].set_b(true);
  std::vector<FunctionDef> function_library = {mul_func};

  // Build a graph to compute y = MyMul(x, 2.0).
  const Tensor kTwo = test::AsScalar<float>(2.0);

  GrapplerItem item;
  item.id = "tf_graph";
  item.graph = test::function::GDef(
      {NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("init", "NoOp", {}, {}, kDevice),
       NDef("two", "Const", {"^init", "^x"},
            {{"dtype", DT_FLOAT}, {"value", kTwo}}, kDevice),
       NDef("y", "MyMul", {"x", "two"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, kDevice)},
      function_library);

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Make sure that specialized function was added to the library and original
  // function was removed.
  ASSERT_EQ(1, output.library().function_size());

  const FunctionDef& specialized = output.library().function(0);
  EXPECT_EQ("MyMul_specialized_for_y_at_tf_graph",
            specialized.signature().name());
  EXPECT_EQ(1, specialized.signature().input_arg_size());

  // And 'y' node has control dependencies of a pushed down const node.
  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "y" && ++count) {
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("^init", node.input(1));
    }
  }
  EXPECT_EQ(1, count);

  // And that graph evaluation yields the same result.
  Tensor pi = test::AsScalar<float>(3.14f);
  item.fetch = {"z"};
  item.feed.emplace_back("x", pi);

  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, SpecializeIndirectFunctionPushDownConstInput) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);

  FunctionDef mul_func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"output"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "output:z:0"}});

  // Mark MyMul as noinline.
  (*mul_func.mutable_attr())["_noinline"].set_b(true);
  std::vector<FunctionDef> function_library = {mul_func};

  const Tensor kTwo = test::AsScalar<float>(2.0);

  // Tensorflow graph:
  //   y = PartitionedCall[Tin=[DT_FLOAT], Tout=[DT_FLOAT], f=MyMul](x, two)
  GrapplerItem item;
  item.id = "tf_graph";
  item.graph = test::function::GDef(
      {NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("init", "NoOp", {}, {}, kDevice),
       NDef("two", "Const", {"^init", "^x"},
            {{"dtype", DT_FLOAT}, {"value", kTwo}}, kDevice),
       NDef("y", "PartitionedCall", {"x", "two"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"f", FDH::FunctionRef("MyMul", {{"T", DT_FLOAT}})}},
            kDevice),
       NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, kDevice)},
      function_library);

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Make sure that specialized function was added to the library and original
  // function was removed.
  ASSERT_EQ(1, output.library().function_size());

  const FunctionDef& specialized = output.library().function(0);
  EXPECT_EQ("MyMul_specialized_for_y_at_tf_graph",
            specialized.signature().name());
  EXPECT_EQ(1, specialized.signature().input_arg_size());

  // And 'y' node has control dependencies of a pushed down const node.
  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "y" && ++count) {
      EXPECT_EQ("PartitionedCall", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("^init", node.input(1));
      // Function calls into the specialized function.
      auto& func = AttrSlice(node).Find("f")->func();
      EXPECT_EQ("MyMul_specialized_for_y_at_tf_graph", func.name());
      // And input/output type lists were updated.
      auto& tin = AttrSlice(node).Find("Tin")->list();
      auto& tout = AttrSlice(node).Find("Tout")->list();
      ASSERT_EQ(1, tin.type_size());
      ASSERT_EQ(1, tout.type_size());
      EXPECT_EQ(DT_FLOAT, tin.type(0));
      EXPECT_EQ(DT_FLOAT, tout.type(0));
    }
  }
  ASSERT_EQ(1, count);

  // And that graph evaluation yields the same result.
  Tensor pi = test::AsScalar<float>(3.14f);
  item.fetch = {"z"};
  item.feed.emplace_back("x", pi);

  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, SpecializeFunction_OncePerUniqueContext) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);

  // Mark MyMul as noinline.
  FunctionDef mul_func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, int32}"},
      {{{"output"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "output:z:0"}});
  (*mul_func.mutable_attr())["_noinline"].set_b(true);
  std::vector<FunctionDef> function_library = {mul_func};

  const Tensor kTwo = test::AsScalar<float>(2.0);
  const Tensor kThree = test::AsScalar<float>(3.0);

  GrapplerItem item;
  item.id = "tf_graph";
  item.graph = test::function::GDef(
      {NDef("init", "NoOp", {}, {}, kDevice),

       // Float placeholders.
       NDef("xf", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("yf", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),

       // Int32 placeholders.
       NDef("xi", "Placeholder", {}, {{"dtype", DT_INT32}}, kDevice),
       NDef("yi", "Placeholder", {}, {{"dtype", DT_INT32}}, kDevice),

       // Consts. Control inputs has to be attached to specialized func calls.
       NDef("two", "Const", {"^init", "^xf"},
            {{"dtype", DT_FLOAT}, {"value", kTwo}}, kDevice),
       NDef("three", "Const", {"^init", "^xf"},
            {{"dtype", DT_FLOAT}, {"value", kThree}}, kDevice),

       // Specialization #1: DT_FLOAT type parameter.
       NDef("mul_1", "MyMul", {"xf", "yf"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("mul_2", "MyMul", {"yf", "xf"}, {{"T", DT_FLOAT}}, kDevice),

       // Specialization #2: DT_INT32 type parameter.
       NDef("mul_3", "MyMul", {"xi", "yi"}, {{"T", DT_INT32}}, kDevice),

       // Specialization #3: DT_FLOAT type parameter + const input kTwo.
       NDef("mul_4", "MyMul", {"xf", "two"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("mul_5", "MyMul", {"yf", "two"}, {{"T", DT_FLOAT}}, kDevice),

       // Specialization #4: DT_FLOAT type parameter + const input kThree.
       NDef("mul_6", "MyMul", {"three", "xf"}, {{"T", DT_FLOAT}}, kDevice)},
      function_library);

  // Specify fetch nodes before optimization to prevent pruning unused function
  // outputs.
  item.fetch = {"mul_1", "mul_2", "mul_3", "mul_4", "mul_5", "mul_6"};

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Make sure that MyMul was specialized once per unique context.
  EXPECT_EQ(4, output.library().function_size());

  // And graph nodes calling specialized functions.
  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "mul_1" && ++count) {
      EXPECT_EQ("MyMul_specialized_for_mul_1_at_tf_graph", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("xf", node.input(0));
      EXPECT_EQ("yf", node.input(1));

    } else if (node.name() == "mul_2" && ++count) {
      EXPECT_EQ("MyMul_specialized_for_mul_1_at_tf_graph", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("yf", node.input(0));
      EXPECT_EQ("xf", node.input(1));

    } else if (node.name() == "mul_3" && ++count) {
      EXPECT_EQ("MyMul_specialized_for_mul_3_at_tf_graph", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("xi", node.input(0));
      EXPECT_EQ("yi", node.input(1));

    } else if (node.name() == "mul_4" && ++count) {
      EXPECT_EQ("MyMul_specialized_for_mul_4_at_tf_graph", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("xf", node.input(0));
      EXPECT_EQ("^init", node.input(1));

    } else if (node.name() == "mul_5" && ++count) {
      EXPECT_EQ("MyMul_specialized_for_mul_4_at_tf_graph", node.op());
      ASSERT_EQ(3, node.input_size());
      EXPECT_EQ("yf", node.input(0));
      gtl::FlatSet<string> expected_ctrl = {"^init", "^xf"};
      gtl::FlatSet<string> actual_ctrl = {node.input(1), node.input(2)};
      EXPECT_EQ(expected_ctrl, actual_ctrl);

    } else if (node.name() == "mul_6" && ++count) {
      EXPECT_EQ("MyMul_specialized_for_mul_6_at_tf_graph", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("xf", node.input(0));
      EXPECT_EQ("^init", node.input(1));
    }
  }
  EXPECT_EQ(6, count);

  // And that graph evaluation yields the same result.
  Tensor pi = test::AsScalar<float>(3.14f);
  Tensor four = test::AsScalar<int32>(4);
  item.feed = {{"xf", pi}, {"yf", pi}, {"xi", four}, {"yi", four}};

  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);

  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  test::ExpectTensorEqual<float>(tensors_expected[1], tensors[1]);
  test::ExpectTensorEqual<int32>(tensors_expected[2], tensors[2]);
  test::ExpectTensorEqual<float>(tensors_expected[3], tensors[3]);
  test::ExpectTensorEqual<float>(tensors_expected[4], tensors[4]);
  test::ExpectTensorEqual<float>(tensors_expected[5], tensors[5]);
}

TEST_F(FunctionOptimizerTest, SpecializeFunctionForUsedOutputTensors) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);

  // MyFunc computes x*y three times and has three output values.
  FunctionDef my_func = FunctionDefHelper::Create(
      "MyFunc", {"x:T", "y:T"}, {"z1:T", "z2:T", "z3:T"}, {"T: {float, int32}"},
      {{{"output1"}, "Mul", {"x", "y"}, {{"T", "$T"}}},
       {{"output2"}, "Mul", {"x", "y"}, {{"T", "$T"}}},
       {{"output3"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z1", "output1:z:0"}, {"z2", "output2:z:0"}, {"z3", "output3:z:0"}});
  (*my_func.mutable_attr())["_noinline"].set_b(true);
  std::vector<FunctionDef> function_library = {my_func};

  GrapplerItem item;
  item.id = "tf_graph";
  item.graph = test::function::GDef(
      {NDef("init", "NoOp", {}, {}, kDevice),

       // Float placeholders.
       NDef("xf", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("yf", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),

       // Specialization #1: DT_FLOAT type parameter. All outputs used.
       NDef("fn1", "MyFunc", {"xf", "yf"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("use_fn1_0", "Identity", {"fn1:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("use_fn1_1", "Identity", {"fn1:1"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("use_fn1_2", "Identity", {"fn1:2"}, {{"T", DT_FLOAT}}, kDevice),

       // Specialization #2: DT_FLOAT type parameter. Only first output used.
       NDef("fn2", "MyFunc", {"xf", "yf"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("use_fn2_0", "Identity", {"fn2:0"}, {{"T", DT_FLOAT}}, kDevice),

       // Specialization #3: DT_FLOAT type parameter. Only second output used.
       NDef("fn3", "MyFunc", {"xf", "yf"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("use_fn3_1", "Identity", {"fn3:1"}, {{"T", DT_FLOAT}}, kDevice),

       // Specialization #4: DT_FLOAT type parameter. Only last output used.
       NDef("fn4", "MyFunc", {"xf", "yf"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("use_fn4_2", "Identity", {"fn4:2"}, {{"T", DT_FLOAT}}, kDevice),

       // Specialization #5: DT_FLOAT type parameter. First and last outputs.
       NDef("fn5", "MyFunc", {"xf", "yf"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("use_fn5_0", "Identity", {"fn5:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("use_fn5_2", "Identity", {"fn5:2"}, {{"T", DT_FLOAT}}, kDevice),

       // Specialization #6: DT_FLOAT type parameter. Outputs not used.
       // Check that function optimizer do not fail. In practice it should be
       // pruned from the graph before passing to function optimizer.
       NDef("fn6", "MyFunc", {"xf", "yf"}, {{"T", DT_FLOAT}}, kDevice)},
      function_library);

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Make sure that MyFunc was specialized once per unique context.
  EXPECT_EQ(6, output.library().function_size());

  // And graph nodes calling specialized functions.
  int found = 0;
  for (const NodeDef& node : output.node()) {
    // All function caller nodes must be specialized.
    if (node.name() == "fn1" && ++found) {
      EXPECT_EQ("MyFunc_specialized_for_fn1_at_tf_graph", node.op());
    } else if (node.name() == "fn2" && ++found) {
      EXPECT_EQ("MyFunc_specialized_for_fn2_at_tf_graph", node.op());
    } else if (node.name() == "fn3" && ++found) {
      EXPECT_EQ("MyFunc_specialized_for_fn3_at_tf_graph", node.op());
    } else if (node.name() == "fn4" && ++found) {
      EXPECT_EQ("MyFunc_specialized_for_fn4_at_tf_graph", node.op());
    } else if (node.name() == "fn5" && ++found) {
      EXPECT_EQ("MyFunc_specialized_for_fn5_at_tf_graph", node.op());
    } else if (node.name() == "fn6" && ++found) {
      EXPECT_EQ("MyFunc_specialized_for_fn6_at_tf_graph", node.op());
    }
    // And all consumers of specialized function nodes must be mapped to new
    // output ports.
    if (node.name() == "use_fn3_1" && ++found) {
      EXPECT_EQ("fn3", node.input(0));
    } else if (node.name() == "use_fn4_2" && ++found) {
      EXPECT_EQ("fn4", node.input(0));
    } else if (node.name() == "use_fn5_0" && ++found) {
      EXPECT_EQ("fn5", node.input(0));
    } else if (node.name() == "use_fn5_2" && ++found) {
      EXPECT_EQ("fn5:1", node.input(0));
    }
  }
  EXPECT_EQ(10, found);

  // And that graph evaluation yields the same result.
  Tensor pi = test::AsScalar<float>(3.14f);
  item.fetch = {"use_fn1_0", "use_fn1_1", "use_fn1_2", "use_fn2_0",
                "use_fn3_1", "use_fn4_2", "use_fn5_0", "use_fn5_2"};
  item.feed = {{"xf", pi}, {"yf", pi}};

  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);

  ASSERT_EQ(tensors_expected.size(), tensors.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorEqual<float>(tensors_expected[i], tensors[i]);
  }
}

TEST_F(FunctionOptimizerTest, SpecializeIndirectFunctionForUsedOutputTensors) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);

  // MyFunc computes x*y three times and has three output values.
  FunctionDef my_func = FunctionDefHelper::Create(
      "MyFunc", {"x:T", "y:T"}, {"z1:T", "z2:T", "z3:T"}, {"T: {float, int32}"},
      {{{"output1"}, "Mul", {"x", "y"}, {{"T", "$T"}}},
       {{"output2"}, "Mul", {"x", "y"}, {{"T", "$T"}}},
       {{"output3"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z1", "output1:z:0"}, {"z2", "output2:z:0"}, {"z3", "output3:z:0"}});
  (*my_func.mutable_attr())["_noinline"].set_b(true);
  std::vector<FunctionDef> function_library = {my_func};

  GrapplerItem item;
  item.id = "tf_graph";
  item.graph = test::function::GDef(
      {NDef("init", "NoOp", {}, {}, kDevice),

       // Float placeholders.
       NDef("xf", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("yf", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),

       // Specialization #1: DT_FLOAT type parameter. All outputs used.
       NDef("fn1", "PartitionedCall", {"xf", "yf"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT, DT_FLOAT, DT_FLOAT}},
             {"f", FDH::FunctionRef("MyFunc", {{"T", DT_FLOAT}})}},
            kDevice),
       NDef("use_fn1_0", "Identity", {"fn1:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("use_fn1_1", "Identity", {"fn1:1"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("use_fn1_2", "Identity", {"fn1:2"}, {{"T", DT_FLOAT}}, kDevice),

       // Specialization #2: DT_FLOAT type parameter. Only first output used.
       NDef("fn2", "PartitionedCall", {"xf", "yf"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT, DT_FLOAT, DT_FLOAT}},
             {"f", FDH::FunctionRef("MyFunc", {{"T", DT_FLOAT}})}},
            kDevice),
       NDef("use_fn2_0", "Identity", {"fn2:0"}, {{"T", DT_FLOAT}}, kDevice),

       // Specialization #3: DT_FLOAT type parameter. Only second output used.
       NDef("fn3", "PartitionedCall", {"xf", "yf"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT, DT_FLOAT, DT_FLOAT}},
             {"f", FDH::FunctionRef("MyFunc", {{"T", DT_FLOAT}})}},
            kDevice),
       NDef("use_fn3_1", "Identity", {"fn3:1"}, {{"T", DT_FLOAT}}, kDevice),

       // Specialization #4: DT_FLOAT type parameter. Only last output used.
       NDef("fn4", "PartitionedCall", {"xf", "yf"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT, DT_FLOAT, DT_FLOAT}},
             {"f", FDH::FunctionRef("MyFunc", {{"T", DT_FLOAT}})}},
            kDevice),
       NDef("use_fn4_2", "Identity", {"fn4:2"}, {{"T", DT_FLOAT}}, kDevice),

       // Specialization #5: DT_FLOAT type parameter. First and last outputs.
       NDef("fn5", "PartitionedCall", {"xf", "yf"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT, DT_FLOAT, DT_FLOAT}},
             {"f", FDH::FunctionRef("MyFunc", {{"T", DT_FLOAT}})}},
            kDevice),
       NDef("use_fn5_0", "Identity", {"fn5:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("use_fn5_2", "Identity", {"fn5:2"}, {{"T", DT_FLOAT}}, kDevice),

       // Specialization #6: DT_FLOAT type parameter. Outputs not used.
       // Check that function optimizer do not fail. In practice it should be
       // pruned from the graph before passing to function optimizer.
       NDef("fn6", "PartitionedCall", {"xf", "yf"},
            {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT, DT_FLOAT, DT_FLOAT}},
             {"f", FDH::FunctionRef("MyFunc", {{"T", DT_FLOAT}})}},
            kDevice)},
      function_library);

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Make sure that MyFunc was specialized once per unique context.
  EXPECT_EQ(6, output.library().function_size());

  // And graph nodes calling specialized functions.
  int found = 0;
  for (const NodeDef& node : output.node()) {
    // All function caller nodes must be specialized.
    if (node.name() == "fn1" && ++found) {
      auto& func = AttrSlice(node).Find("f")->func();
      auto& tout = AttrSlice(node).Find("Tout")->list();
      EXPECT_EQ("PartitionedCall", node.op());
      EXPECT_EQ("MyFunc_specialized_for_fn1_at_tf_graph", func.name());
      ASSERT_EQ(3, tout.type_size());

    } else if (node.name() == "fn2" && ++found) {
      auto& func = AttrSlice(node).Find("f")->func();
      auto& tout = AttrSlice(node).Find("Tout")->list();
      EXPECT_EQ("PartitionedCall", node.op());
      EXPECT_EQ("MyFunc_specialized_for_fn2_at_tf_graph", func.name());
      ASSERT_EQ(1, tout.type_size());

    } else if (node.name() == "fn3" && ++found) {
      auto& func = AttrSlice(node).Find("f")->func();
      auto& tout = AttrSlice(node).Find("Tout")->list();
      EXPECT_EQ("PartitionedCall", node.op());
      EXPECT_EQ("MyFunc_specialized_for_fn3_at_tf_graph", func.name());
      ASSERT_EQ(1, tout.type_size());

    } else if (node.name() == "fn4" && ++found) {
      auto& func = AttrSlice(node).Find("f")->func();
      auto& tout = AttrSlice(node).Find("Tout")->list();
      EXPECT_EQ("PartitionedCall", node.op());
      EXPECT_EQ("MyFunc_specialized_for_fn4_at_tf_graph", func.name());
      ASSERT_EQ(1, tout.type_size());

    } else if (node.name() == "fn5" && ++found) {
      auto& func = AttrSlice(node).Find("f")->func();
      auto& tout = AttrSlice(node).Find("Tout")->list();
      EXPECT_EQ("PartitionedCall", node.op());
      EXPECT_EQ("MyFunc_specialized_for_fn5_at_tf_graph", func.name());
      ASSERT_EQ(2, tout.type_size());

    } else if (node.name() == "fn6" && ++found) {
      auto& func = AttrSlice(node).Find("f")->func();
      auto& tout = AttrSlice(node).Find("Tout")->list();
      EXPECT_EQ("PartitionedCall", node.op());
      EXPECT_EQ("MyFunc_specialized_for_fn6_at_tf_graph", func.name());
      ASSERT_EQ(0, tout.type_size());
    }
    // And all consumers of specialized function nodes must be mapped to new
    // output ports.
    if (node.name() == "use_fn3_1" && ++found) {
      EXPECT_EQ("fn3", node.input(0));
    } else if (node.name() == "use_fn4_2" && ++found) {
      EXPECT_EQ("fn4", node.input(0));
    } else if (node.name() == "use_fn5_0" && ++found) {
      EXPECT_EQ("fn5", node.input(0));
    } else if (node.name() == "use_fn5_2" && ++found) {
      EXPECT_EQ("fn5:1", node.input(0));
    }
  }
  EXPECT_EQ(10, found);

  // And that graph evaluation yields the same result.
  Tensor pi = test::AsScalar<float>(3.14f);
  item.fetch = {"use_fn1_0", "use_fn1_1", "use_fn1_2", "use_fn2_0",
                "use_fn3_1", "use_fn4_2", "use_fn5_0", "use_fn5_2"};
  item.feed = {{"xf", pi}, {"yf", pi}};

  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);

  ASSERT_EQ(tensors_expected.size(), tensors.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorEqual<float>(tensors_expected[i], tensors[i]);
  }
}

TEST_F(FunctionOptimizerTest, PruningUselessLibraryFunctions) {
  using test::function::NDef;
  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);
  auto func = test::function::XTimesTwo();
  (*func.mutable_attr())["_noinline"].set_b(true);
  GrapplerItem item;
  item.id = "test_graph";
  item.graph = test::function::GDef(
      {NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, "/device:CPU:0"),
       NDef("y", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, "/device:CPU:0"),
       NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, "/device:CPU:0")},
      // FunctionLib
      {
          func,
          test::function::XTimesTwoInt32(),
          test::function::XTimes16(),
      });
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  ASSERT_EQ(output.library().function().size(), 1);
  EXPECT_EQ(output.library().function(0).signature().name(),
            "XTimesTwo_specialized_for_y_at_test_graph");
}

TEST_F(FunctionOptimizerTest, PreserveSaverDefFunctions) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;
  FunctionOptimizer optimizer(RewriterConfig::DEFAULT, true);
  auto func = test::function::XTimesTwo();
  (*func.mutable_attr())["_noinline"].set_b(true);
  GrapplerItem item;
  item.id = "test_graph";
  item.graph = test::function::GDef(
      {
          NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, "/device:CPU:0"),
          NDef("y", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, "/device:CPU:0"),
          NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, "/device:CPU:0"),
          NDef("Restore", "StatefulPartitionedCall", {},
               {{"Tin", {}},
                {"Tout", {}},
                {"f", FDH::FunctionRef("RestoreFn", {})}},
               "/device:CPU:0"),
          NDef("Save", "StatefulPartitionedCall", {},
               {{"Tin", {}},
                {"Tout", {}},
                {"f", FDH::FunctionRef("SaveFn", {})}},
               "/device:CPU:0"),
      },
      // FunctionLib
      {
          func,
          test::function::XTimesTwoInt32(),
          test::function::XTimes16(),
          FDH::Create("RestoreFn", {}, {}, {}, {}, {}),
          FDH::Create("SaveFn", {}, {}, {}, {}, {}),
      });
  item.restore_op = "Restore";
  item.save_op = "Save";
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  ASSERT_EQ(output.library().function().size(), 3);
  std::vector<std::string> signature_names;
  for (const auto& function : output.library().function()) {
    signature_names.push_back(function.signature().name());
  }
  EXPECT_THAT(signature_names, ::testing::UnorderedElementsAre(
                                   "XTimesTwo_specialized_for_y_at_test_graph",
                                   "RestoreFn", "SaveFn"));
}

}  // namespace grappler
}  // namespace tensorflow
