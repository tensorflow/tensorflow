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
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kDevice[] = "/device:CPU:0";

class FunctionOptimizerTest : public GrapplerTest {
 protected:
  Tensor MakeScalarTensor(float value) {
    Tensor tensor(DT_FLOAT, {});
    tensor.scalar<float>()() = value;
    return tensor;
  }

  Tensor MakeScalarTensor(int value) {
    Tensor tensor(DT_INT32, {});
    tensor.scalar<int>()() = value;
    return tensor;
  }
};

TEST_F(FunctionOptimizerTest, SimpleFunction) {
  // Build a graph to compute y = XTimesTwo(x)
  GrapplerItem item;
  constexpr char device[] = "/device:CPU:0";
  item.graph = test::function::GDef(
      {test::function::NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}},
                            device),
       test::function::NDef("y", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, device),
       test::function::NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, device)},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "y/inlined_inputs") {
      count++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("x", node.input(0));
    } else if (node.name() == "y/x") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/inlined_inputs:0", node.input(0));
    } else if (node.name() == "y/two") {
      count++;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^y/inlined_inputs", node.input(0));
    } else if (node.name() == "y/scale") {
      count++;
      EXPECT_EQ("Cast", node.op());
      EXPECT_EQ(device, node.device());
    } else if (node.name() == "y/y") {
      count++;
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("y/x", node.input(0));
      EXPECT_EQ("y/scale", node.input(1));
    } else if (node.name() == "y") {
      count++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/y", node.input(0));
    } else if (node.name() == "z") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y", node.input(0));
    }
  }
  EXPECT_EQ(7, count);

  Tensor pi = MakeScalarTensor(3.14f);
  item.fetch = {"z"};
  item.feed.emplace_back("x", pi);
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized(item, std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, FixedTypeFunction) {
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
          {{"y"}, "Mul", {"x", "two"}, {{"T", DT_FLOAT}}},
      });

  constexpr char device[] = "/device:CPU:0";
  GrapplerItem item;
  item.graph = test::function::GDef(
      {test::function::NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}},
                            device),
       test::function::NDef("y", "XTimesTwo", {"x"}, {}, device),
       test::function::NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, device)},
      // FunctionLib
      {
          x_times_two,
      });

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "y/inlined_inputs") {
      count++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("x", node.input(0));
    } else if (node.name() == "y/x") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/inlined_inputs:0", node.input(0));
    } else if (node.name() == "y/two") {
      count++;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^y/inlined_inputs", node.input(0));
      EXPECT_EQ(device, node.device());
    } else if (node.name() == "y/y") {
      count++;
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("y/x", node.input(0));
      EXPECT_EQ("y/two", node.input(1));
    } else if (node.name() == "y") {
      count++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/y", node.input(0));
    } else if (node.name() == "z") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y", node.input(0));
    }
  }
  EXPECT_EQ(6, count);

  Tensor pi = MakeScalarTensor(3.14f);
  item.fetch = {"z"};
  item.feed.emplace_back("x", pi);
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized(item, std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, FunctionWithOutputMapping) {
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
  constexpr char device[] = "/device:CPU:0";
  item.graph = test::function::GDef(
      {test::function::NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}},
                            device),
       test::function::NDef("y", "Exp_func", {"x"}, {}, device),
       test::function::NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, device)},
      // FunctionLib
      {
          func,
      });

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "y/inlined_inputs") {
      count++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("x", node.input(0));
    } else if (node.name() == "y/in") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/inlined_inputs:0", node.input(0));
    } else if (node.name() == "y/Linear_func") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/in", node.input(0));
    } else if (node.name() == "y/Exp") {
      count++;
      EXPECT_EQ("Exp", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/Linear_func", node.input(0));
    } else if (node.name() == "y") {
      count++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/Exp", node.input(0));
    } else if (node.name() == "z") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(device, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y", node.input(0));
    }
  }
  EXPECT_EQ(6, count);

  Tensor pi = MakeScalarTensor(3.14f);
  item.fetch = {"z"};
  item.feed.emplace_back("x", pi);
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized(item, std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, FunctionWithInputForwarding) {
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
  constexpr char device[] = "/device:CPU:0";
  item.graph = test::function::GDef(
      {test::function::NDef("x0", "Placeholder", {}, {{"dtype", DT_FLOAT}},
                            device),
       test::function::NDef("x1", "Placeholder", {}, {{"dtype", DT_FLOAT}},
                            device),
       test::function::NDef("x2", "Placeholder", {}, {{"dtype", DT_FLOAT}},
                            device),
       test::function::NDef("x3", "Placeholder", {}, {{"dtype", DT_INT32}},
                            device),
       test::function::NDef("x4", "Placeholder", {}, {{"dtype", DT_FLOAT}},
                            device),
       test::function::NDef("y", "ForwardInputs",
                            {"x0", "x1", "x2", "x3", "x4"}, {}, device),
       test::function::NDef("z0", "Identity", {"y:0"}, {{"T", DT_FLOAT}},
                            device),
       test::function::NDef("z1", "Identity", {"y:1"}, {{"T", DT_FLOAT}},
                            device),
       test::function::NDef("z2", "Identity", {"y:2"}, {{"T", DT_INT32}},
                            device)},
      // FunctionLib
      {
          func,
      });

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  item.fetch = {"z0", "z1", "z2"};
  item.feed.emplace_back("x0", MakeScalarTensor(3.14f));
  item.feed.emplace_back("x1", MakeScalarTensor(2.7f));
  item.feed.emplace_back("x2", MakeScalarTensor(1.0f));
  item.feed.emplace_back("x4", MakeScalarTensor(-1.0f));
  item.feed.emplace_back("x3", MakeScalarTensor(1234));
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized(item, std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  test::ExpectTensorEqual<float>(tensors_expected[1], tensors[1]);
  test::ExpectTensorEqual<int>(tensors_expected[2], tensors[2]);
}

TEST_F(FunctionOptimizerTest, FunctionWithoutInput) {
  const Tensor kTwo = test::AsScalar<int64>(2);
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
  constexpr char device[] = "/device:CPU:0";
  item.graph = test::function::GDef(
      {test::function::NDef("y", "GenerateTwo", {}, {}, device),
       test::function::NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, device)},
      // FunctionLib
      {
          func,
      });

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // For now we won't inline the function.
  EXPECT_EQ(item.graph.DebugString(), output.DebugString());
}

TEST_F(FunctionOptimizerTest, InlineFunctionWithNestedFunctionCall) {
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
      {test::function::NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}},
                            kDevice),
       test::function::NDef("square", "MySquare", {"a"}, {{"T", DT_FLOAT}},
                            kDevice),
       test::function::NDef("outputs", "Identity", {"square:0"},
                            {{"T", DT_FLOAT}}, kDevice)},
      // FunctionLib
      {mul_func, square_func});

  GraphDef output;
  FunctionOptimizer optimizer(RewriterConfig::ON);
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "square/inlined_inputs" && count++) {
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("a", node.input(0));
    } else if (node.name() == "square/x" && count++) {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("square/inlined_inputs:0", node.input(0));
    } else if (node.name() == "square/output/inlined_inputs" && count++) {
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("square/x", node.input(0));
      EXPECT_EQ("square/x", node.input(1));
    } else if (node.name() == "square/output/x" && count++) {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("square/output/inlined_inputs:0", node.input(0));
    } else if (node.name() == "square/output/y" && count++) {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("square/output/inlined_inputs:1", node.input(0));
    } else if (node.name() == "square/output/output" && count++) {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("square/output/x", node.input(0));
      EXPECT_EQ("square/output/y", node.input(1));
    } else if (node.name() == "square/output" && count++) {
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("square/output/output", node.input(0));
    } else if (node.name() == "square" && count++) {
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("square/output", node.input(0));
    } else if (node.name() == "outputs" && count++) {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("square:0", node.input(0));
    }
  }
  EXPECT_EQ(9, count);

  item.fetch = {"outputs"};
  item.feed.emplace_back("a", MakeScalarTensor(2.0f));
  auto tensors_expected = EvaluateFetchNodes(item);

  GrapplerItem optimized(item, std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);

  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, SymbolicGradients) {
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

  FunctionOptimizer optimizer(RewriterConfig::ON);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::vector<Tensor> expected =
      EvaluateNodes(item.graph, {"out1", "out2"}, {});
  std::vector<Tensor> optimized = EvaluateNodes(output, {"out1", "out2"}, {});
  test::ExpectTensorEqual<float>(expected[0], optimized[0]);
  test::ExpectTensorEqual<float>(expected[1], optimized[1]);
}

TEST_F(FunctionOptimizerTest, SymbolicGradientsIdentity) {
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

  FunctionOptimizer optimizer(RewriterConfig::ON);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(13, output.node_size());
  EXPECT_EQ("Const", output.node(0).name());
  EXPECT_EQ("Const_1", output.node(1).name());
  EXPECT_EQ("SymbolicGradient/FunctionInputs", output.node(2).name());
  EXPECT_EQ("SymbolicGradient", output.node(3).name());
  EXPECT_EQ("SymbolicGradient/SymbolicGradient/Identity",
            output.node(4).name());
  EXPECT_EQ("SymbolicGradient/Func/_0", output.node(5).name());
  EXPECT_EQ("SymbolicGradient/Func/_1", output.node(6).name());
  EXPECT_EQ("SymbolicGradient/Func/_2", output.node(7).name());
  EXPECT_EQ("SymbolicGradient/SymbolicGradient/Func/_1/dx",
            output.node(8).name());
  EXPECT_EQ("SymbolicGradient/Func/_3", output.node(9).name());
  EXPECT_EQ("SymbolicGradient/Func/_4", output.node(10).name());
  EXPECT_EQ("SymbolicGradient/Func/_5", output.node(11).name());
  EXPECT_EQ("out", output.node(12).name());
  for (int i = 2; i < 4; ++i) {
    EXPECT_EQ("IdentityN", output.node(i).op());
  }
  for (int i = 4; i < 11; ++i) {
    EXPECT_EQ("Identity", output.node(i).op());
  }

  std::vector<Tensor> expected = EvaluateNodes(item.graph, {"out"}, {});
  std::vector<Tensor> optimized = EvaluateNodes(output, {"out"}, {});
  test::ExpectTensorEqual<float>(expected[0], optimized[0]);
}

TEST_F(FunctionOptimizerTest, SymbolicGradientsNoInlineFunc) {
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

  FunctionOptimizer optimizer(RewriterConfig::ON);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  // The optimizer should succeed but the graphs should be the same.
  TF_EXPECT_OK(status);
  CompareGraphs(item.graph, output);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
