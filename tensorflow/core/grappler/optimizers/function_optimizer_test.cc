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
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace tensorflow {
namespace grappler {

namespace {
constexpr char kDevice[] = "/device:CPU:0";
}  // namespace

class FunctionOptimizerTest : public GrapplerTest {
 protected:
  void DisableFunctionSpecialization(FunctionOptimizer* optimizer) {
    optimizer->options_.enable_function_specialization = false;
  }
};

TEST_F(FunctionOptimizerTest, InlineFunction_SimpleFunction) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);

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

  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "y/inlined_inputs") {
      count++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("x", node.input(0));
    } else if (node.name() == "y/x") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/inlined_inputs:0", node.input(0));
    } else if (node.name() == "y/two") {
      count++;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^y/inlined_inputs", node.input(0));
    } else if (node.name() == "y/scale") {
      count++;
      EXPECT_EQ("Cast", node.op());
      EXPECT_EQ(kDevice, node.device());
    } else if (node.name() == "y/y") {
      count++;
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("y/x", node.input(0));
      EXPECT_EQ("y/scale", node.input(1));
    } else if (node.name() == "y") {
      count++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/y", node.input(0));
    } else if (node.name() == "z") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y", node.input(0));
    }
  }
  EXPECT_EQ(7, count);

  Tensor pi = test::AsScalar<float>(3.14f);
  item.fetch = {"z"};
  item.feed.emplace_back("x", pi);
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, InlineFunction_SkipErrorsIfGraphNotModified) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);

  // Standard XTimesTwo() function.
  FunctionDef x_times_two = test::function::XTimesTwo();

  // Function with sequence of tensors as an input (currently not supported).
  FunctionDef my_identity_n = FunctionDefHelper::Create(
      // Name
      "MyIdentityN",
      // Args
      {"x: N*T"},
      // Return values
      {"out: N*T"},
      // Attrs
      {"N:int", "T:{float, double, int32, int64}"},
      // Nodes (just forward inputs through IdentityN)
      {
          {{"Id"}, "IdentityN", {"x"}, {{"T", "$T"}, {"N", "$N"}}},
      },
      // Output mapping
      {{"out", "Id:output:0"}});

  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("y1", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("y2", "MyIdentityN", {"x"}, {{"T", DT_FLOAT}, {"N", 1}}, kDevice),
       NDef("z1", "Identity", {"y1:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("z2", "Identity", {"y2:0"}, {{"T", DT_FLOAT}}, kDevice)},
      // FunctionLib
      {x_times_two, my_identity_n});

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Verify that only MyIdentityN is in the function library after optimization.
  ASSERT_EQ(1, output.library().function().size());
  EXPECT_EQ("MyIdentityN", output.library().function(0).signature().name());

  // And that XTimesTwo was successfully inlined.
  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "y1/inlined_inputs") {
      found++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("x", node.input(0));
    } else if (node.name() == "y1") {
      found++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y1/y", node.input(0));
    } else if (node.name() == "y2") {
      found++;
      EXPECT_EQ("MyIdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("x", node.input(0));
    }
  }
  EXPECT_EQ(3, found);

  Tensor pi = test::AsScalar<float>(3.14f);
  item.fetch = {"z1"};
  item.feed.emplace_back("x", pi);
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, InlineFunction_FixedTypeFunction) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);

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

  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "y/inlined_inputs") {
      count++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("x", node.input(0));
    } else if (node.name() == "y/x") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/inlined_inputs:0", node.input(0));
    } else if (node.name() == "y/two") {
      count++;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^y/inlined_inputs", node.input(0));
      EXPECT_EQ(kDevice, node.device());
    } else if (node.name() == "y/y") {
      count++;
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("y/x", node.input(0));
      EXPECT_EQ("y/two", node.input(1));
    } else if (node.name() == "y") {
      count++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/y", node.input(0));
    } else if (node.name() == "z") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y", node.input(0));
    } else if (node.name() == "y/enter") {
      count++;
      EXPECT_TRUE(IsEnter(node));
      const string frame_name = node.attr().at("frame_name").s();
      EXPECT_EQ("y/frame", frame_name);
    }
  }
  EXPECT_EQ(7, count);

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

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);

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

  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "y/inlined_inputs") {
      count++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("x", node.input(0));
    } else if (node.name() == "y/in") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/inlined_inputs:0", node.input(0));
    } else if (node.name() == "y/Linear_func") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/in", node.input(0));
    } else if (node.name() == "y/Exp") {
      count++;
      EXPECT_EQ("Exp", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/Linear_func", node.input(0));
    } else if (node.name() == "y") {
      count++;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y/Exp", node.input(0));
    } else if (node.name() == "z") {
      count++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y", node.input(0));
    }
  }
  EXPECT_EQ(6, count);

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

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);

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

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);
  DisableFunctionSpecialization(&optimizer);  // do not specialize noinline func

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
  item.graph = test::function::GDef(
      {NDef("y", "GenerateTwo", {}, {}, kDevice),
       NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, kDevice)},
      // FunctionLib
      {
          func,
      });

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // For now we won't inline the function.
  EXPECT_EQ(item.graph.DebugString(), output.DebugString());
}

TEST_F(FunctionOptimizerTest, InlineFunction_FunctionWithNestedFunctionCall) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);

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

  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "square/inlined_inputs" && ++count) {
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("a", node.input(0));
    } else if (node.name() == "square/x" && ++count) {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("square/inlined_inputs:0", node.input(0));
    } else if (node.name() == "square/output/inlined_inputs" && ++count) {
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("square/x", node.input(0));
      EXPECT_EQ("square/x", node.input(1));
    } else if (node.name() == "square/output/x" && ++count) {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("square/output/inlined_inputs:0", node.input(0));
    } else if (node.name() == "square/output/y" && ++count) {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("square/output/inlined_inputs:1", node.input(0));
    } else if (node.name() == "square/output/output" && ++count) {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("square/output/x", node.input(0));
      EXPECT_EQ("square/output/y", node.input(1));
    } else if (node.name() == "square/output" && ++count) {
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("square/output/output", node.input(0));
    } else if (node.name() == "square" && ++count) {
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("square/output", node.input(0));
    } else if (node.name() == "outputs" && ++count) {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(kDevice, node.device());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("square:0", node.input(0));
    }
  }
  EXPECT_EQ(9, count);

  item.fetch = {"outputs"};
  item.feed.emplace_back("a", test::AsScalar<float>(2.0f));
  auto tensors_expected = EvaluateFetchNodes(item);

  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);

  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, InlineSymbolicGradient_TestFunc) {
  FunctionOptimizer optimizer(RewriterConfig::ON);

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

  std::vector<Tensor> expected =
      EvaluateNodes(item.graph, {"out1", "out2"}, {});
  std::vector<Tensor> optimized = EvaluateNodes(output, {"out1", "out2"}, {});
  test::ExpectTensorEqual<float>(expected[0], optimized[0]);
  test::ExpectTensorEqual<float>(expected[1], optimized[1]);
}

TEST_F(FunctionOptimizerTest, InlineSymbolicGradient_IdentityFunc) {
  FunctionOptimizer optimizer(RewriterConfig::ON);

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

TEST_F(FunctionOptimizerTest, InlineSymbolicGradientNoInlineFunc) {
  FunctionOptimizer optimizer(RewriterConfig::ON);

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

  FunctionOptimizer optimizer(RewriterConfig::AGGRESSIVE);

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
      // Function library.
      {mul_func} /* Function library */);

  GraphDef optimized_graph;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));

  GraphDef expected = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),

       // Function must be inlined and all nodes placed on a valid device.
       NDef("c/x", "Identity", {"a:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("c/y", "Identity", {"b:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("c/mul", "Mul", {"c/x", "c/y"}, {{"T", DT_FLOAT}}, kDevice),

       NDef("d", "Identity", {"c/mul:0"}, {{"T", DT_FLOAT}}, kDevice)},
      // Function library.
      {mul_func});

  CompareGraphs(expected, optimized_graph);

  Tensor pi = test::AsScalar<float>(3.14f);
  item.feed.emplace_back("a", pi);
  item.feed.emplace_back("b", pi);

  GrapplerItem optimized = item.WithGraph(std::move(optimized_graph));
  auto tensors_expected = EvaluateFetchNodes(item);
  auto tensors = EvaluateFetchNodes(optimized);
  ASSERT_EQ(tensors_expected.size(), 1);
  ASSERT_EQ(tensors.size(), tensors_expected.size());
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(FunctionOptimizerTest, InlineIndirectFunctionWithControlDependencies) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  FunctionOptimizer optimizer(RewriterConfig::AGGRESSIVE);

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
       {{"mul"}, "Mul", {"x", "y", "^add"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "mul:z:0"}});

  // Build a graph to compute:
  //   a = Placeholder
  //   b = Placeholder
  //   v = VarHandleOp(init = a)
  //   f1 = MyMul(a, b, v)
  //   f2 = MyMul(f1, f1, v)
  //   return [f2, v]
  GrapplerItem item;
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
       NDef("v", "VarHandleOp", {}, {{"dtype", DT_FLOAT}, {"shape", scalar}}),
       NDef("init_v", "AssignVariableOp", {"v", "a"}, {{"dtype", DT_FLOAT}},
            kDevice),

       // Function body of a first function call inlined into the graph.
       NDef("f1/x", "Identity", {"a:0", "^init_v"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("f1/y", "Identity", {"b:0", "^init_v"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("f1/v", "Identity", {"v:0", "^init_v"}, {{"T", DT_RESOURCE}},
            kDevice),
       NDef("f1/one", "Const", {"^f1/x"},
            {{"dtype", DT_FLOAT}, {"value", kOne}}, kDevice),
       NDef("f1/add", "AssignAddVariableOp", {"f1/v", "f1/one"},
            {{"dtype", DT_FLOAT}}, kDevice),
       NDef("f1/mul", "Mul", {"f1/x", "f1/y", "^f1/add"}, {{"T", DT_FLOAT}},
            kDevice),

       // Function body of a second function call also inlined into the graph,
       // and input nodes read directly from the inlined nodes of the first
       // function call.
       NDef("f2/x", "Identity", {"f1/mul:0", "^f1/add"}, {{"T", DT_FLOAT}},
            kDevice),
       NDef("f2/y", "Identity", {"f1/mul:0", "^f1/add"}, {{"T", DT_FLOAT}},
            kDevice),
       NDef("f2/v", "Identity", {"v:0", "^f1/add"}, {{"T", DT_RESOURCE}},
            kDevice),
       NDef("f2/one", "Const", {"^f2/x"},
            {{"dtype", DT_FLOAT}, {"value", kOne}}, kDevice),
       NDef("f2/add", "AssignAddVariableOp", {"f2/v", "f2/one"},
            {{"dtype", DT_FLOAT}}, kDevice),
       NDef("f2/mul", "Mul", {"f2/x", "f2/y", "^f2/add"}, {{"T", DT_FLOAT}},
            kDevice),

       // Return values read directly from inlined nodes.
       NDef("out_1", "Identity", {"f2/mul:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("out_2", "ReadVariableOp", {"v", "^f1/add", "^f2/add"},
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

  FunctionOptimizer optimizer(RewriterConfig::AGGRESSIVE);

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

  GraphDef expected = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, cpu0),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, cpu1),

       // Function must be inlined and `mul` node placed on a requested device,
       // and input `Identity` nodes must be colocated with their source nodes.
       NDef("c/x", "Identity", {"a:0"}, {{"T", DT_FLOAT}}, cpu0),
       NDef("c/y", "Identity", {"b:0"}, {{"T", DT_FLOAT}}, cpu1),
       NDef("c/mul", "Mul", {"c/x", "c/y"}, {{"T", DT_FLOAT}}, cpu1),

       NDef("d", "Identity", {"c/mul:0"}, {{"T", DT_FLOAT}}, cpu0)},
      // Function library.
      {mul_func});

  CompareGraphs(expected, optimized_graph);
}

TEST_F(FunctionOptimizerTest, InlineIndirectFunctionWithoutSideEffects) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  FunctionOptimizer optimizer(RewriterConfig::AGGRESSIVE);

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
  item.fetch = {"out"};
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),

       // Call function first time.
       NDef("f1", "PartitionedCall", {"a", "b"},
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

       // Function body of a first function call inlined into the graph.
       NDef("f1/x", "Identity", {"a:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("f1/y", "Identity", {"b:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("f1/mul", "Mul", {"f1/x", "f1/y"}, {{"T", DT_FLOAT}}, kDevice),

       // Function body of a second function call also inlined into the graph,
       // and input nodes read directly from the inlined nodes of the first
       // function call, and control dependency edge removed.
       NDef("f2/x", "Identity", {"f1/mul:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("f2/y", "Identity", {"f1/mul:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("f2/mul", "Mul", {"f2/x", "f2/y"}, {{"T", DT_FLOAT}}, kDevice),

       // Return directly from inlined node of f2.
       NDef("out", "Identity", {"f2/mul:0"}, {{"T", DT_FLOAT}}, kDevice)},

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

TEST_F(FunctionOptimizerTest, SpecializeFunctionXTimesTwo) {
  using test::function::NDef;

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);

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

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);

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

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);

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

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);

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

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);

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

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);

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
      EXPECT_EQ("fn3:0", node.input(0));
    } else if (node.name() == "use_fn4_2" && ++found) {
      EXPECT_EQ("fn4:0", node.input(0));
    } else if (node.name() == "use_fn5_0" && ++found) {
      EXPECT_EQ("fn5:0", node.input(0));
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

  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);

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
      EXPECT_EQ("fn3:0", node.input(0));
    } else if (node.name() == "use_fn4_2" && ++found) {
      EXPECT_EQ("fn4:0", node.input(0));
    } else if (node.name() == "use_fn5_0" && ++found) {
      EXPECT_EQ("fn5:0", node.input(0));
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
  FunctionOptimizer optimizer(RewriterConfig::DEFAULT);
  DisableFunctionSpecialization(&optimizer);
  auto func = test::function::XTimesTwo();
  (*func.mutable_attr())["_noinline"].set_b(true);
  GrapplerItem item;
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

  EXPECT_EQ(output.library().function().size(), 1);
  EXPECT_EQ(output.library().function(0).signature().name(), "XTimesTwo");
}

}  // namespace grappler
}  // namespace tensorflow
