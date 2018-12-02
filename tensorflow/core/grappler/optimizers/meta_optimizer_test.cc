/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"

#include "absl/strings/substitute.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kDevice[] = "/device:CPU:0";

class TestOptimizer : public CustomGraphOptimizer {
 public:
  static void SetOptimized(const bool flag_value) { optimized_ = flag_value; }
  static bool IsOptimized() { return optimized_; }

  TestOptimizer() {}
  string name() const override { return "test_optimizer"; }

  Status Init(const tensorflow::RewriterConfig_CustomGraphOptimizer* config =
                  nullptr) override {
    return Status::OK();
  }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override {
    optimized_ = true;
    *optimized_graph = item.graph;
    return Status::OK();
  }

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override {}

 private:
  static bool optimized_;
};

bool TestOptimizer::optimized_;

REGISTER_GRAPH_OPTIMIZER(TestOptimizer);

class TestGraphOptimizer : public TestOptimizer {
 public:
  string name() const override { return "test_graph_optimizer"; }
};

REGISTER_GRAPH_OPTIMIZER(TestGraphOptimizer);

class TestOptimizerWithParams : public TestOptimizer {
 public:
  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    CHECK(config != nullptr);
    return Status::OK();
  }
};

REGISTER_GRAPH_OPTIMIZER(TestOptimizerWithParams);

// Record various properties of the GrapplerItems passed for optimization.
class GrapplerItemPropertiesAccumulator : public CustomGraphOptimizer {
 public:
  static void SetAllowedOptimizations(
      gtl::FlatMap<string, GrapplerItem::AllowedOptimizations>*
          allowed_optimizations) {
    allowed_optimizations_ = allowed_optimizations;
  }
  static void ResetAllowedOptimizations() { allowed_optimizations_ = nullptr; }

  GrapplerItemPropertiesAccumulator() {}
  string name() const override {
    return "grappler_item_properties_accumulator";
  }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return Status::OK();
  }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override {
    *optimized_graph = item.graph;
    if (allowed_optimizations_) {
      allowed_optimizations_->insert({item.id, item.allowed_optimizations});
    }
    return Status::OK();
  }

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override {}

 private:
  static gtl::FlatMap<string, GrapplerItem::AllowedOptimizations>*
      allowed_optimizations_;
};

gtl::FlatMap<string, GrapplerItem::AllowedOptimizations>*
    GrapplerItemPropertiesAccumulator::allowed_optimizations_;

REGISTER_GRAPH_OPTIMIZER(GrapplerItemPropertiesAccumulator);

class MetaOptimizerTest : public GrapplerTest {};

TEST_F(MetaOptimizerTest, RunsCustomOptimizer) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  TestOptimizer::SetOptimized(false);
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("TestOptimizer");
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, RunsCustomOptimizerWithParams) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  TestOptimizer::SetOptimized(false);
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("TestOptimizerWithParams");
  auto* custom_config = rewriter_config.add_custom_optimizers();
  custom_config->set_name("TestOptimizerWithParams");
  (*custom_config->mutable_parameter_map())["foo"] = AttrValue();

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, RunsCustomOptimizerAndCustomGraphOptimizer) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  TestOptimizer::SetOptimized(false);
  TestGraphOptimizer::SetOptimized(false);
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("TestOptimizer");
  auto customGraphOptimizer = rewriter_config.add_custom_optimizers();
  customGraphOptimizer->set_name("TestGraphOptimizer");
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestOptimizer::IsOptimized());
  EXPECT_TRUE(TestGraphOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, RunOptimizersTwice) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
}

TEST_F(MetaOptimizerTest, RunToggleOptimizersAndCustomGraphOptimizerTwice) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  auto customGraphOptimizer = rewriter_config.add_custom_optimizers();
  customGraphOptimizer->set_name("TestGraphOptimizer");
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestGraphOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, OptimizeFunctionLibrary) {
  using test::function::NDef;

  // Enable ony function optimization.
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();

  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.set_function_optimization(RewriterConfig::ON);
  rewriter_config.add_optimizers("function");
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);

  // Define function library:
  //
  //   MyMul(x, y)    = x * y
  //  *MySquare(x)    = MyMul(x, x)
  //  *MyQuadratic(x) = MySquare(MySquare(x))
  //
  //  * - marked as noinline

  FunctionDef mul_func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "mul:z:0"}});

  FunctionDef square_func = FunctionDefHelper::Create(
      "MySquare", {"x:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"my_mul"}, "MyMul", {"x", "x"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "my_mul:z:0"}});
  (*square_func.mutable_attr())["_noinline"].set_b(true);

  FunctionDef quadratic_func = FunctionDefHelper::Create(
      "MyQuadratic", {"x:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"square"}, "MySquare", {"x"}, {{"T", "$T"}}},
       {{"quadratic"}, "MySquare", {"square:z"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "quadratic:z:0"}});
  (*quadratic_func.mutable_attr())["_noinline"].set_b(true);

  // Tensorflow graph:
  //
  //   a = tf.Placeholder(tf.float);
  //   b = tf.Placeholder(tf.int32);
  //
  //   square = MySquare(a);        // a^2
  //   quadratic = MyQuadratic(b);  // b^4
  GrapplerItem item;
  item.id = "tf_graph";
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_INT32}}, kDevice),
       // Calls into function library
       NDef("square", "MySquare", {"a"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("quadratic", "MyQuadratic", {"b"}, {{"T", DT_INT32}}, kDevice),
       // Forward outputs
       NDef("out_s", "Identity", {"square:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("out_q", "Identity", {"quadratic:0"}, {{"T", DT_INT32}}, kDevice)},
      // FunctionLib
      {mul_func, square_func, quadratic_func});

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  FunctionLibraryDefinition optimized_flib(OpRegistry::Global(),
                                           output.library());

  // Specialized and optimized functions should be added to the graph.
  EXPECT_EQ(6, optimized_flib.num_functions());

  // Get a specialized function name.
  const auto specialized_name = [](const string& fn, const string& node,
                                   const string& id) {
    return absl::Substitute("$0_specialized_for_$1_at_$2", fn, node, id);
  };

  // MyQuadratic should be specialized once:
  //   0. 'quadratic' node in the main graph
  const string optimized_0 =
      specialized_name("MyQuadratic", "quadratic", "tf_graph");

  // MySquare should be specialized and optimized for 3 instantiations:
  //   1. 'square' node in the main graph
  //   2. 'square' node in the MyQuadratic specialization (not in a fetch set)
  //   3. 'quadratic' node in the MyQuadratic specialization (is in a fetch set)

  const string optimized_1 = specialized_name("MySquare", "square", "tf_graph");
  const string optimized_2 =
      specialized_name("MySquare", "square", optimized_0);
  const string optimized_3 =
      specialized_name("MySquare", "quadratic", optimized_0);

  const FunctionDef* optimized_func_0 = optimized_flib.Find(optimized_0);
  const FunctionDef* optimized_func_1 = optimized_flib.Find(optimized_1);
  const FunctionDef* optimized_func_2 = optimized_flib.Find(optimized_2);
  const FunctionDef* optimized_func_3 = optimized_flib.Find(optimized_3);

  ASSERT_NE(optimized_func_0, nullptr);
  ASSERT_NE(optimized_func_1, nullptr);
  ASSERT_NE(optimized_func_2, nullptr);
  ASSERT_NE(optimized_func_3, nullptr);

  // Graph should call optimized function.
  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "square" && ++count) {
      EXPECT_EQ(optimized_1, node.op());
    } else if (node.name() == "quadratic" && ++count) {
      EXPECT_EQ(optimized_0, node.op());
    }
  }
  EXPECT_EQ(2, count);

  // Specialized MySquare should call specialized functions.
  count = 0;
  for (const NodeDef& node : optimized_func_0->node_def()) {
    if (node.name() == "square" && ++count) {
      EXPECT_EQ(optimized_2, node.op());
    } else if (node.name() == "quadratic" && ++count) {
      EXPECT_EQ(optimized_3, node.op());
    }
  }
  EXPECT_EQ(2, count);

  const std::vector<const FunctionDef*> optimized_funcs = {
      optimized_func_1, optimized_func_2, optimized_func_3};

  // MyMul should be inlined into all optimized versions of MySquare.
  for (const FunctionDef* optimized_func : optimized_funcs) {
    count = 0;
    for (const NodeDef& node : optimized_func->node_def()) {
      if (node.name() == "my_mul/inlined_inputs" && ++count) {
        EXPECT_EQ("IdentityN", node.op());
        EXPECT_EQ(2, node.input_size());
        EXPECT_EQ("x:0", node.input(0));
        EXPECT_EQ("x:0", node.input(1));
      } else if (node.name() == "my_mul/x" && ++count) {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ(1, node.input_size());
        EXPECT_EQ("my_mul/inlined_inputs:output:0", node.input(0));
      } else if (node.name() == "my_mul/y" && ++count) {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ(1, node.input_size());
        EXPECT_EQ("my_mul/inlined_inputs:output:1", node.input(0));
      } else if (node.name() == "my_mul/mul" && ++count) {
        EXPECT_EQ("Mul", node.op());
        EXPECT_EQ(2, node.input_size());
        EXPECT_EQ("my_mul/x:output:0", node.input(0));
        EXPECT_EQ("my_mul/y:output:0", node.input(1));
      } else if (node.name() == "my_mul" && ++count) {
        EXPECT_EQ("IdentityN", node.op());
        EXPECT_EQ(1, node.input_size());
        EXPECT_EQ("my_mul/mul:z:0", node.input(0));
      }
      EXPECT_TRUE(node.device().empty());
    }
    EXPECT_EQ(5, count);
  }

  item.fetch = {"out_s", "out_q"};
  item.feed.emplace_back("a", test::AsScalar<float>(2.0f));
  item.feed.emplace_back("b", test::AsScalar<int>(4));
  auto tensors_expected = EvaluateFetchNodes(item);

  GrapplerItem optimized(item, std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);

  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  test::ExpectTensorEqual<int>(tensors_expected[1], tensors[1]);
}

TEST_F(MetaOptimizerTest, OptimizeFunctionLibraryPruneFunctionBody) {
  using test::function::NDef;

  // Enable function optimization and pruning.
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();

  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.set_function_optimization(RewriterConfig::ON);
  rewriter_config.add_optimizers("function");
  rewriter_config.add_optimizers("pruning");
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);

  // MyFunc defines two Mul nodes inside function body and two corresponding
  // function outputs.
  FunctionDef my_func = FunctionDefHelper::Create(
      "MyFunc", {"x:T", "y:T"}, {"z1:T", "z2:T"}, {"T: {float, double}"},
      {{{"mul1"}, "Mul", {"x", "y"}, {{"T", "$T"}}},
       {{"mul2"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z1", "mul1:z:0"}, {"z2", "mul2:z:0"}});
  (*my_func.mutable_attr())["_noinline"].set_b(true);

  // Tensorflow graph:
  //
  //   a = tf.Placeholder(tf.float);
  //   b = tf.Placeholder(tf.int32);
  //
  //   fn1 = MyFunc(a, b);
  //   fn2 = MyFunc(a, b);
  //
  // Fetch: fn1:0 and fn2:1 via Identity nodes.
  GrapplerItem item;
  item.id = "tf_graph";
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       // Calls into function library
       NDef("fn1", "MyFunc", {"a", "b"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("fn2", "MyFunc", {"a", "b"}, {{"T", DT_FLOAT}}, kDevice),
       // Read outputs of function call nodes
       NDef("out_fn1", "Identity", {"fn1:0"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("out_fn2", "Identity", {"fn2:1"}, {{"T", DT_FLOAT}}, kDevice)},
      // FunctionLib
      {my_func});

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  FunctionLibraryDefinition optimized_flib(OpRegistry::Global(),
                                           output.library());

  // Specialized and optimized functions should be added to the graph.
  EXPECT_EQ(2, optimized_flib.num_functions());

  // Expected names of the specialized and optimized functions.
  const string optimized_fn1 = "MyFunc_specialized_for_fn1_at_tf_graph";
  const string optimized_fn2 = "MyFunc_specialized_for_fn2_at_tf_graph";

  const FunctionDef* optimized_func_fn1 = optimized_flib.Find(optimized_fn1);
  const FunctionDef* optimized_func_fn2 = optimized_flib.Find(optimized_fn2);

  ASSERT_NE(optimized_func_fn1, nullptr);
  ASSERT_NE(optimized_func_fn2, nullptr);

  // Graph should call optimized function.
  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "fn1" && ++count) {
      EXPECT_EQ(optimized_fn1, node.op());
    } else if (node.name() == "fn2" && ++count) {
      EXPECT_EQ(optimized_fn2, node.op());
    }
  }
  EXPECT_EQ(2, count);

  // Specialized MyFuncs should have just one Mul node and single output arg.

  // 1. Specialized for fn1:0.
  ASSERT_EQ(1, optimized_func_fn1->node_def_size());
  EXPECT_EQ(1, optimized_func_fn1->signature().output_arg_size());
  EXPECT_EQ("z1", optimized_func_fn1->signature().output_arg(0).name());
  EXPECT_EQ("mul1", optimized_func_fn1->node_def(0).name());

  // 2. Specialized for fn2:1.
  ASSERT_EQ(1, optimized_func_fn2->node_def_size());
  EXPECT_EQ(1, optimized_func_fn2->signature().output_arg_size());
  EXPECT_EQ("z2", optimized_func_fn2->signature().output_arg(0).name());
  EXPECT_EQ("mul2", optimized_func_fn2->node_def(0).name());

  // Verify that output tensors are equal.
  item.fetch = {"out_fn1", "out_fn2"};
  item.feed.emplace_back("a", test::AsScalar<float>(2.0f));
  item.feed.emplace_back("b", test::AsScalar<float>(3.123f));
  auto tensors_expected = EvaluateFetchNodes(item);

  GrapplerItem optimized(item, std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);

  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  test::ExpectTensorEqual<float>(tensors_expected[1], tensors[1]);
}

TEST_F(MetaOptimizerTest, OptimizeFunctionLibraryWithRestrictions) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  // We will record what type of optimizations meta optimizer allows for each
  // GrapplerItem (main graph and graphs for each function).
  gtl::FlatMap<string, GrapplerItem::AllowedOptimizations>
      allowed_optimizations;
  GrapplerItemPropertiesAccumulator::SetAllowedOptimizations(
      &allowed_optimizations);

  // Just record properties of optimized Grappler items.
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();

  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.add_optimizers("GrapplerItemPropertiesAccumulator");
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);

  // Define simple function library with two identical mul functions.
  FunctionDef mul_func_1 = FunctionDefHelper::Create(
      "MyMul1", {"x:float", "y:float"}, {"z:float"}, {},
      {{{"mul"}, "Mul", {"x", "y"}, {}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "mul:z:0"}});

  FunctionDef mul_func_2 = FunctionDefHelper::Create(
      "MyMul2", {"x:float", "y:float"}, {"z:float"}, {},
      {{{"mul"}, "Mul", {"x", "y"}, {}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "mul:z:0"}});

  // Tensorflow graph:
  //
  //   x0 = tf.Placeholder(tf.float);
  //   x1 = tf.Placeholder(tf.float);
  //   dy = tf.Placeholder(tf.float);
  //
  //   mul_1 = MyMul1(x0, x1);
  //   mul_2 = MyMul2(x0, x1);
  //   dx = SymbolicGradient({x0, x1, dy}, f=MyMul2)
  GrapplerItem item;
  item.id = "main";
  item.graph = test::function::GDef(
      {NDef("x0", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("x1", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("dy", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       // Calls into function library
       NDef("mul_1", "MyMul1", {"x0", "x1"}, {}, kDevice),
       NDef("mul_2", "MyMul2", {"x0", "x1"}, {}, kDevice),
       // Symbolic gradient of a MyMul2
       NDef("dx", "SymbolicGradient", {"x0", "x1", "dy"},
            {{"f", FDH::FunctionRef("MyMul2", {})},
             {"Tin", DataTypeSlice{DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT, DT_FLOAT}}},
            kDevice)},
      // FunctionLib
      {mul_func_1, mul_func_2});
  item.fetch = {"mul_1", "mul_2", "dx"};

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Our custom optimizer must be called for the main graph and for the two
  // functions.
  ASSERT_EQ(allowed_optimizations.size(), 3);

  auto allowed_optimizations_main =
      gtl::FindOrNull(allowed_optimizations, "main");
  ASSERT_NE(allowed_optimizations_main, nullptr);
  EXPECT_TRUE(allowed_optimizations_main->non_differentiable_rewrites);

  auto allowed_optimizations_my_mul_1 =
      gtl::FindOrNull(allowed_optimizations, "MyMul1");
  ASSERT_NE(allowed_optimizations_my_mul_1, nullptr);
  EXPECT_TRUE(allowed_optimizations_my_mul_1->non_differentiable_rewrites);

  auto allowed_optimizations_my_mul_2 =
      gtl::FindOrNull(allowed_optimizations, "MyMul2");
  ASSERT_NE(allowed_optimizations_my_mul_2, nullptr);
  EXPECT_FALSE(allowed_optimizations_my_mul_2->non_differentiable_rewrites);
}

class SleepingOptimizer : public CustomGraphOptimizer {
 public:
  SleepingOptimizer() {}
  string name() const override { return "test_optimizer"; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return Status::OK();
  }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override {
    *optimized_graph = item.graph;
    optimized_graph->add_node();
    sleep(1);
    return Status::OK();
  }

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override {}
};

REGISTER_GRAPH_OPTIMIZER(SleepingOptimizer);

TEST_F(MetaOptimizerTest, OptimizerTimesOut) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  ConfigProto config;
  RewriterConfig& rewriter_config =
      *config.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("SleepingOptimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_timeout_ms(1500);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);

  GraphDef output;
  const Status status =
      RunMetaOptimizer(item, config, nullptr, nullptr, &output);
  EXPECT_EQ(status.error_message(), "meta_optimizer exceeded deadline.");
  // Make sure the graph was reverted to the original regardless of when the
  // optimizer timed out.
  CompareGraphs(item.graph, output);
}

TEST_F(MetaOptimizerTest, OptimizerDoesNotTimeOut) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  ConfigProto config;
  RewriterConfig& rewriter_config =
      *config.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("SleepingOptimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_timeout_ms(1500);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);
  GraphDef output;
  const Status status =
      RunMetaOptimizer(item, config, nullptr, nullptr, &output);
  TF_EXPECT_OK(status);
  EXPECT_EQ(item.graph.node_size() + 1, output.node_size());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
