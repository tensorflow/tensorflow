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

#include <atomic>

#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"

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
  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Init(const tensorflow::RewriterConfig_CustomGraphOptimizer*
                        config = nullptr) override {
    return absl::OkStatus();
  }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* optimized_graph) override {
    optimized_ = true;
    *optimized_graph = item.graph;
    return absl::OkStatus();
  }

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
  absl::Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    CHECK(config != nullptr);
    return absl::OkStatus();
  }
};

REGISTER_GRAPH_OPTIMIZER(TestOptimizerWithParams);

// Record various properties of the GrapplerItems passed for optimization.
class GrapplerItemPropertiesAccumulator : public CustomGraphOptimizer {
 public:
  static void SetOptimizationOptions(
      gtl::FlatMap<string, GrapplerItem::OptimizationOptions>*
          optimization_options) {
    optimization_options_ = optimization_options;
  }
  static void ResetOptimizationOptions() { optimization_options_ = nullptr; }

  GrapplerItemPropertiesAccumulator() {}
  string name() const override {
    return "grappler_item_properties_accumulator";
  }
  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return absl::OkStatus();
  }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* optimized_graph) override {
    *optimized_graph = item.graph;
    if (optimization_options_) {
      optimization_options_->insert({item.id, item.optimization_options()});
    }
    return absl::OkStatus();
  }

 private:
  static gtl::FlatMap<string, GrapplerItem::OptimizationOptions>*
      optimization_options_;
};

gtl::FlatMap<string, GrapplerItem::OptimizationOptions>*
    GrapplerItemPropertiesAccumulator::optimization_options_;

REGISTER_GRAPH_OPTIMIZER(GrapplerItemPropertiesAccumulator);

class MetaOptimizerTest : public GrapplerTest {};

TEST_F(MetaOptimizerTest, RunsCustomOptimizer) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  TestOptimizer::SetOptimized(false);
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("TestOptimizer");
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const absl::Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, RunsCustomOptimizerWithParams) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

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
  const absl::Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, RunsCustomOptimizerAndCustomGraphOptimizer) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

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
  const absl::Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestOptimizer::IsOptimized());
  EXPECT_TRUE(TestGraphOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, RunsPluginOptimizer) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"/device:GPU:0"});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  TestOptimizer::SetOptimized(false);
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.set_min_graph_nodes(-1);

  const auto creator = []() { return new TestOptimizer; };
  ConfigList config_list;
  config_list.disable_model_pruning = true;
  PluginGraphOptimizerRegistry::RegisterPluginOptimizerOrDie(creator, "GPU",
                                                             config_list);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const absl::Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, RunOptimizersTwice) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const absl::Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
}

TEST_F(MetaOptimizerTest, RunToggleOptimizersAndCustomGraphOptimizerTwice) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  auto customGraphOptimizer = rewriter_config.add_custom_optimizers();
  customGraphOptimizer->set_name("TestGraphOptimizer");
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const absl::Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestGraphOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, OptimizeFunctionLibrary) {
  using test::function::NDef;

  // Enable only function optimization.
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
      /*ret_def=*/
      {{"z", "mul:z:0"}});

  FunctionDef square_func = FunctionDefHelper::Create(
      "MySquare", {"x:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"my_mul"}, "MyMul", {"x", "x"}, {{"T", "$T"}}}},
      /*ret_def=*/
      {{"z", "my_mul:z:0"}});
  (*square_func.mutable_attr())["_noinline"].set_b(true);

  FunctionDef quadratic_func = FunctionDefHelper::Create(
      "MyQuadratic", {"x:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"square"}, "MySquare", {"x"}, {{"T", "$T"}}},
       {{"quadratic"}, "MySquare", {"square:z"}, {{"T", "$T"}}}},
      /*ret_def=*/
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
      /*funcs=*/
      {mul_func, square_func, quadratic_func});

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  FunctionLibraryDefinition optimized_flib(OpRegistry::Global(),
                                           output.library());

  // Specialized and optimized functions should be added to the graph.
  EXPECT_EQ(3, optimized_flib.num_functions());

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
  //   1.  'square' node in the main graph
  //   2.  'square' node in the MyQuadratic specialization
  //   3*. 'quadratic' node in the MyQuadratic specialization
  //        has identical instantiation context to #2

  const string optimized_1 = specialized_name("MySquare", "square", "tf_graph");
  const string optimized_2 =
      specialized_name("MySquare", "square", optimized_0);

  const FunctionDef* optimized_func_0 = optimized_flib.Find(optimized_0);
  const FunctionDef* optimized_func_1 = optimized_flib.Find(optimized_1);
  const FunctionDef* optimized_func_2 = optimized_flib.Find(optimized_2);

  ASSERT_NE(optimized_func_0, nullptr);
  ASSERT_NE(optimized_func_1, nullptr);
  ASSERT_NE(optimized_func_2, nullptr);

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
      EXPECT_EQ(optimized_2, node.op());
    }
  }
  EXPECT_EQ(2, count);

  const std::vector<const FunctionDef*> optimized_funcs = {optimized_func_1,
                                                           optimized_func_2};

  // MyMul should be inlined into all optimized versions of MySquare.
  for (const FunctionDef* optimized_func : optimized_funcs) {
    count = 0;
    for (const NodeDef& node : optimized_func->node_def()) {
      if (node.name() == "Func/my_mul/input/_0" && ++count) {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ(1, node.input_size());
        EXPECT_EQ("x", node.input(0));
      } else if (node.name() == "Func/my_mul/input/_1" && ++count) {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ(1, node.input_size());
        EXPECT_EQ("x", node.input(0));
      } else if (node.name() == "my_mul/mul" && ++count) {
        EXPECT_EQ("Mul", node.op());
        EXPECT_EQ(2, node.input_size());
        EXPECT_EQ("Func/my_mul/input/_0:output:0", node.input(0));
        EXPECT_EQ("Func/my_mul/input/_1:output:0", node.input(1));
      }
      EXPECT_TRUE(node.device().empty());
    }
    EXPECT_EQ(3, count);
    ASSERT_EQ(1, optimized_func->ret().size());
    EXPECT_EQ("Func/my_mul/output/_2:output:0", optimized_func->ret().at("z"));
  }

  item.fetch = {"out_s", "out_q"};
  item.feed.emplace_back("a", test::AsScalar<float>(2.0f));
  item.feed.emplace_back("b", test::AsScalar<int>(4));
  auto tensors_expected = EvaluateFetchNodes(item);

  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);

  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  test::ExpectTensorEqual<int>(tensors_expected[1], tensors[1]);
}

TEST_F(MetaOptimizerTest, OptimizeFunctionLibraryPruneUnusedOutputs) {
  using test::function::NDef;

  ConfigProto config_proto;
  MetaOptimizer optimizer(nullptr, config_proto);

  // MyMul computes x*y three times and has three output values.
  FunctionDef my_mul = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z0:T", "z1:T", "z2:T"}, {"T: {float, int32}"},
      {{{"output0"}, "Mul", {"x", "y"}, {{"T", "$T"}}},
       {{"output1"}, "Mul", {"x", "y"}, {{"T", "$T"}}},
       {{"output2"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /*ret_def=*/
      {{"z0", "output0:z:0"}, {"z1", "output1:z:0"}, {"z2", "output2:z:0"}});

  // Call MyMyl and forward all three outputs.
  FunctionDef my_fwd = FunctionDefHelper::Create(
      "Fwd", {"x:T", "y:T"}, {"z0:T", "z1:T", "z2:T"}, {"T: {float, int32}"},
      {{{"output"}, "MyMul", {"x", "y"}, {{"T", "$T"}}}},
      /*ret_def=*/
      {{"z0", "output:z0:0"}, {"z1", "output:z1:0"}, {"z2", "output:z2:0"}});

  // Mark both functions as `_noinline` to trigger specialization.
  (*my_mul.mutable_attr())["_noinline"].set_b(true);
  (*my_fwd.mutable_attr())["_noinline"].set_b(true);
  /*funcs=*/
  std::vector<FunctionDef> function_library = {my_mul, my_fwd};

  // Tensorflow graph:
  //   a = Placeholder[T=float]
  //   b = Placeholder[T=float]
  //   fwd = Fwd(a, b)
  //
  // Fetch fwd:2 via Identity node.
  GrapplerItem item;
  item.id = "tf_graph";
  item.fetch = {"ret"};
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("fwd", "Fwd", {"a", "b"}, {{"T", DT_FLOAT}}, kDevice),
       NDef("ret", "Identity", {"fwd:2"}, {{"T", DT_FLOAT}}, kDevice)},
      function_library);

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  FunctionLibraryDefinition optimized_flib(OpRegistry::Global(),
                                           output.library());

  // Specialized functions should be added to the graph.
  EXPECT_EQ(2, optimized_flib.num_functions());

  // Expected names of the specialized functions.
  const string specialized_my_fwd = "Fwd_specialized_for_fwd_at_tf_graph";
  const string specialized_my_mul =
      absl::StrCat("MyMul_specialized_for_output_at_", specialized_my_fwd);

  // Specialized MyMul should have just one output argument.
  FunctionDef expected_my_mul = FunctionDefHelper::Create(
      specialized_my_mul, {"x:float", "y:float"}, {"z2:float"}, {},
      {{{"output2"}, "Mul", {"x", "y"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z2", "output2:z:0"}});

  // Specialized Fwd should also have just one output argument.
  FunctionDef expected_my_fwd = FunctionDefHelper::Create(
      specialized_my_fwd, {"x:float", "y:float"}, {"z2:float"}, {},
      {{{"output"}, specialized_my_mul, {"x", "y"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z2", "output:z2:0"}});

  const FunctionDef* my_mul_spec = optimized_flib.Find(specialized_my_mul);
  const FunctionDef* my_fwd_spec = optimized_flib.Find(specialized_my_fwd);

  ASSERT_NE(my_mul_spec, nullptr);
  ASSERT_NE(my_fwd_spec, nullptr);

  CompareFunctions(expected_my_mul, *my_mul_spec);
  CompareFunctions(expected_my_fwd, *my_fwd_spec);

  item.feed.emplace_back("a", test::AsScalar<float>(2.0f));
  item.feed.emplace_back("b", test::AsScalar<float>(4.0f));
  auto tensors_expected = EvaluateFetchNodes(item);

  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);

  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
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
      /*ret_def=*/
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
      /*funcs=*/
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

  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);

  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
  test::ExpectTensorEqual<float>(tensors_expected[1], tensors[1]);
}

TEST_F(MetaOptimizerTest, OptimizeFunctionLibraryWithRestrictions) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  // We will record what type of optimizations meta optimizer allows for each
  // GrapplerItem (main graph and graphs for each function).
  gtl::FlatMap<string, GrapplerItem::OptimizationOptions> optimization_options;
  GrapplerItemPropertiesAccumulator::SetOptimizationOptions(
      &optimization_options);

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
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z", "mul:z:0"}});

  FunctionDef mul_func_2 = FunctionDefHelper::Create(
      "MyMul2", {"x:float", "y:float"}, {"z:float"}, {},
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
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
      /*funcs=*/
      {mul_func_1, mul_func_2});
  item.fetch = {"mul_1", "mul_2", "dx"};

  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  // Our custom optimizer must be called for the main graph and for the two
  // functions.
  ASSERT_EQ(optimization_options.size(), 3);

  auto optimization_options_main =
      gtl::FindOrNull(optimization_options, "main");
  ASSERT_NE(optimization_options_main, nullptr);
  EXPECT_TRUE(optimization_options_main->allow_non_differentiable_rewrites);

  auto optimization_options_my_mul_1 =
      gtl::FindOrNull(optimization_options, "MyMul1");
  ASSERT_NE(optimization_options_my_mul_1, nullptr);
  EXPECT_TRUE(optimization_options_my_mul_1->allow_non_differentiable_rewrites);

  auto optimization_options_my_mul_2 =
      gtl::FindOrNull(optimization_options, "MyMul2");
  ASSERT_NE(optimization_options_my_mul_2, nullptr);
  EXPECT_FALSE(
      optimization_options_my_mul_2->allow_non_differentiable_rewrites);
}

class SleepingOptimizer : public CustomGraphOptimizer {
 public:
  SleepingOptimizer() {}
  string name() const override { return "test_optimizer"; }
  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return absl::OkStatus();
  }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* optimized_graph) override {
    *optimized_graph = item.graph;
    Env::Default()->SleepForMicroseconds(1000000);
    GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
    optimized_graph->add_node();
    return absl::OkStatus();
  }
};

REGISTER_GRAPH_OPTIMIZER(SleepingOptimizer);

TEST_F(MetaOptimizerTest, OptimizerTimesOut) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ConfigProto config;
  RewriterConfig& rewriter_config =
      *config.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("SleepingOptimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_timeout_ms(500);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);

  GraphDef output;
  GraphDef original = item.graph;
  const absl::Status status =
      RunMetaOptimizer(std::move(item), config, nullptr, nullptr, &output);
  EXPECT_EQ(status.message(), "meta_optimizer exceeded deadline.");
  // Make sure the graph was reverted to the original regardless of when the
  // optimizer timed out.
  CompareGraphs(original, output);
}

TEST_F(MetaOptimizerTest, MetaOptimizerTimesOut) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ConfigProto config;
  RewriterConfig& rewriter_config =
      *config.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("SleepingOptimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_timeout_ms(1500);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);

  GraphDef output;
  const int original_node_size = item.graph.node_size();
  const absl::Status status =
      RunMetaOptimizer(std::move(item), config, nullptr, nullptr, &output);
  EXPECT_EQ(status.message(), "meta_optimizer exceeded deadline.");
  // The meta optimizer should manage to finish one iteration.
  EXPECT_EQ(original_node_size + 1, output.node_size());
}

TEST_F(MetaOptimizerTest, OptimizerDoesNotTimeOut) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ConfigProto config;
  RewriterConfig& rewriter_config =
      *config.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.add_optimizers("SleepingOptimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_timeout_ms(2500);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  GraphDef output;
  const int original_node_size = item.graph.node_size();
  const absl::Status status =
      RunMetaOptimizer(std::move(item), config, nullptr, nullptr, &output);
  TF_EXPECT_OK(status);
  // The meta optimizer should manage to finish two iterations.
  EXPECT_EQ(original_node_size + 2, output.node_size());
}

TEST_F(MetaOptimizerTest, RunPostOptimizationVerifiersOnValidGraph) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ConfigProto config_proto;
  auto& post_optimization_verifier_config =
      *config_proto.mutable_graph_options()
           ->mutable_rewrite_options()
           ->mutable_post_optimization_verifier_config();
  post_optimization_verifier_config.set_structure_verifier(VerifierConfig::ON);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const absl::Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
}

TEST_F(MetaOptimizerTest, RunInterOptimizerVerifiersOnValidGraph) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {kDevice});
  GrapplerItem item;
  ASSERT_TRUE(fake_input.NextItem(&item));

  ConfigProto config_proto;
  auto& inter_optimizer_verifier_config =
      *config_proto.mutable_graph_options()
           ->mutable_rewrite_options()
           ->mutable_inter_optimizer_verifier_config();
  inter_optimizer_verifier_config.set_structure_verifier(VerifierConfig::ON);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const absl::Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
}

TEST_F(MetaOptimizerTest, RunPostOptimizationVerifiersOnInvalidGraph) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  gtl::FlatMap<string, GrapplerItem::OptimizationOptions> optimization_options;
  GrapplerItemPropertiesAccumulator::SetOptimizationOptions(
      &optimization_options);

  // Define simple function library with two identical mul functions.
  FunctionDef mul_func_1 =
      FunctionDefHelper::Create("MyMul1", {"x:float", "y:float"}, {"z:float"},
                                {}, {{{"mul"}, "Mul", {"x", "y"}, {}}},
                                /*ret_def=*/
                                {{"z", "mul:z:0"}});

  FunctionDef mul_func_2 =
      FunctionDefHelper::Create("MyMul2", {"x:float", "y:float"}, {"z:float"},
                                {}, {{{"mul"}, "Mul", {"x", "y"}, {}}},
                                /*ret_def=*/
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
      /*funcs=*/
      {mul_func_1, mul_func_2});
  item.fetch = {"mul_1", "mul_2", "dx"};

  GraphDef output;

  // Call Optimize with post optimization verifiers.
  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();

  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.add_optimizers("GrapplerItemPropertiesAccumulator");
  rewriter_config.set_min_graph_nodes(-1);
  auto& post_optimization_verifier_config =
      *config_proto.mutable_graph_options()
           ->mutable_rewrite_options()
           ->mutable_post_optimization_verifier_config();
  post_optimization_verifier_config.set_structure_verifier(VerifierConfig::ON);

  MetaOptimizer optimizer_with_post_verifiers(nullptr, config_proto);
  absl::Status status =
      optimizer_with_post_verifiers.Optimize(nullptr, item, &output);
  EXPECT_TRUE(errors::IsInvalidArgument(status));
  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "NodeDef expected inputs 'float' do not match 3 inputs specified"));
}

TEST_F(MetaOptimizerTest, RunInterOptimizerVerifiersOnInvalidGraph) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;

  gtl::FlatMap<string, GrapplerItem::OptimizationOptions> optimization_options;
  GrapplerItemPropertiesAccumulator::SetOptimizationOptions(
      &optimization_options);

  // Define simple function library with two identical mul functions.
  FunctionDef mul_func_1 =
      FunctionDefHelper::Create("MyMul1", {"x:float", "y:float"}, {"z:float"},
                                {}, {{{"mul"}, "Mul", {"x", "y"}, {}}},
                                /*ret_def=*/
                                {{"z", "mul:z:0"}});

  FunctionDef mul_func_2 =
      FunctionDefHelper::Create("MyMul2", {"x:float", "y:float"}, {"z:float"},
                                {}, {{{"mul"}, "Mul", {"x", "y"}, {}}},
                                /*ret_def=*/
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
       NDef("x1", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       // Calls into function library
       NDef("mul_1", "MyMul1", {"x0", "x1"}, {}, kDevice),
       NDef("mul_2", "MyMul2", {"x0", "x1"}, {}, kDevice),
       // Symbolic gradient of a MyMul2
       NDef("dx", "SymbolicGradient", {"x0", "x1", "dy"},
            {{"f", FDH::FunctionRef("MyMul2", {})},
             {"Tin", DataTypeSlice{DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT, DT_FLOAT}}},
            kDevice)},
      /*funcs=*/
      {mul_func_1, mul_func_2});
  item.fetch = {"mul_1", "mul_2", "dx"};

  GraphDef output;

  // Call Optimize with post optimization verifiers.
  ConfigProto config_proto;
  // Call Optimize with inter optimizer verifiers.
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.add_optimizers("GrapplerItemPropertiesAccumulator");
  rewriter_config.set_min_graph_nodes(-1);
  auto& inter_optimizer_verifier_config =
      *config_proto.mutable_graph_options()
           ->mutable_rewrite_options()
           ->mutable_inter_optimizer_verifier_config();
  inter_optimizer_verifier_config.set_structure_verifier(VerifierConfig::ON);

  MetaOptimizer optimizer_with_inter_verifiers(nullptr, config_proto);
  absl::Status status =
      optimizer_with_inter_verifiers.Optimize(nullptr, item, &output);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "NodeDef expected inputs 'float' do not match 3 inputs specified"));
}

TEST_F(MetaOptimizerTest, CompressConstants) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Tensor zeros_t(DT_FLOAT, TensorShape({64}));
  Tensor ones_t(DT_FLOAT, TensorShape({64}));
  for (int i = 0; i < 64; ++i) {
    zeros_t.flat<float>()(i) = 0.0f;
    ones_t.flat<float>()(i) = 1.0f;
  }
  Output zeros = ops::Const(scope.WithOpName("zeros"), zeros_t);
  Output host_ones = ops::Const(scope.WithOpName("host_ones"), ones_t);
  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  ASSERT_EQ(item.graph.node(1).name(), "host_ones");
  // There is not C++ api for HostConst, so we manually change the node type
  // here.
  item.graph.mutable_node(1)->set_op("HostConst");
  item.fetch = {"zeros", "host_ones"};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, {});

  ConfigProto config_proto;
  auto& rewriter_config =
      *config_proto.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config.set_min_graph_nodes(-1);
  MetaOptimizer optimizer(/*cpu_device=*/nullptr, config_proto);
  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(/*cluster=*/nullptr, item, &output));

  bool found_zeros = false;
  bool found_host_ones = false;
  ASSERT_EQ(output.node_size(), 2);
  for (const auto& node : output.node()) {
    if (node.name() == "zeros") {
      found_zeros = true;
      EXPECT_EQ(node.op(), "Const");
      const TensorProto& zeroes_t = node.attr().at("value").tensor();
      EXPECT_EQ(zeroes_t.float_val_size(), 0);
    } else if (node.name() == "host_ones") {
      found_host_ones = true;
      EXPECT_EQ(node.op(), "HostConst");
      const TensorProto& ones_t = node.attr().at("value").tensor();
      EXPECT_EQ(ones_t.float_val_size(), 1);
      EXPECT_EQ(ones_t.float_val(0), 1.0f);
    }
  }

  EXPECT_TRUE(found_zeros);
  EXPECT_TRUE(found_host_ones);

  auto tensors = EvaluateNodes(output, item.fetch, {});
  ASSERT_EQ(tensors.size(), 2);
  ASSERT_EQ(tensors_expected.size(), 2);
  for (int i = 0; i < 2; ++i) {
    test::ExpectTensorEqual<float>(tensors[i], tensors_expected[i]);
  }
}

TEST_F(MetaOptimizerTest, TestTFGRemoveDeadArguments) {
  using test::function::NDef;

  gtl::FlatMap<string, GrapplerItem::OptimizationOptions> optimization_options;
  GrapplerItemPropertiesAccumulator::SetOptimizationOptions(
      &optimization_options);

  // Define a simple function library with one branch function.
  //   def branch_func(x, y):
  //     z = tf.Mul(x, x)
  //     return z
  FunctionDef case_func = FunctionDefHelper::Create(
      "branch_func", {"x:float", "y:float"}, {"z:float"}, {},
      {{{"mul"}, "Mul", {"x", "x"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z", "mul:z:0"}});

  // Tensorflow graph:
  //
  //   idx = tf.Placeholder(tf.int32);
  //   x = tf.Placeholder(tf.float);
  //   y = tf.Placeholder(tf.float);
  //
  //   case = tf.Case(idx, x, y, branches=[branch_func])
  GrapplerItem item;
  item.id = "main";

  AttrValue branches;
  branches.mutable_list()->add_func()->set_name("branch_func");
  AttrValue output_shapes;
  output_shapes.mutable_list()->add_shape();
  item.graph = test::function::GDef(
      {NDef("idx", "Placeholder", {}, {{"dtype", DT_INT32}}, kDevice),
       NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("y", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       // Calls into function library
       NDef("case", "Case", {"idx", "x", "y"},
            {{"branches", std::move(branches)},
             {"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
             {"Tout", DataTypeSlice{DT_FLOAT}},
             {"output_shapes", std::move(output_shapes)}},
            kDevice)},
      /*funcs=*/
      {case_func});
  item.fetch = {"case"};

  GraphDef output;
  ConfigProto config_proto;
  // Disable conditional code motion to prevent placeholders from being sunk
  // into branch_func (would result in zero parameters for it).
  config_proto.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_experimental_conditional_code_motion(RewriterConfig::OFF);

  MetaOptimizer optimizer(nullptr, config_proto);
  absl::Status status = optimizer.Optimize(nullptr, item, &output);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(output.library().function_size(), 1);
  // One of the arguments was removed.
  auto& func = output.library().function(0);
  EXPECT_EQ(func.signature().input_arg_size(), 1);
  EXPECT_EQ(func.signature().input_arg(0).name(), "x_tfg_result_0");
}

TEST_F(MetaOptimizerTest, TestTFGControlFlowSink) {
  using test::function::NDef;

  gtl::FlatMap<string, GrapplerItem::OptimizationOptions> optimization_options;
  GrapplerItemPropertiesAccumulator::SetOptimizationOptions(
      &optimization_options);

  // Define a branch function.
  //   def branch_func(x, y):
  //     z = tf.Mul(x, y)
  //     return z
  FunctionDef case_func = FunctionDefHelper::Create(
      "branch_func", {"x:float", "y:float"}, {"z:float"}, {},
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z", "mul:z:0"}});

  // Define a function with a control-flow op.
  //   def Foo(idx, a, b):
  //     x_foo = Add(a, b)
  //     y_foo = Mul(a, b)
  //     case = Case(idx, x_foo, y_foo, branches=[branch_func[)
  //     return case
  AttrValue branches;
  branches.mutable_list()->add_func()->set_name("branch_func");
  AttrValue output_shapes;
  output_shapes.mutable_list()->add_shape();
  FunctionDef foo_func = FunctionDefHelper::Create(
      "Foo", {"idx:int32", "a:float", "b:float"}, {"c:float"}, {},
      {{{"add"}, "Add", {"a", "b"}, {{"T", DT_FLOAT}}},
       {{"mul"}, "Mul", {"a", "b"}, {{"T", DT_FLOAT}}},
       {{"case"},
        "Case",
        {"idx", "add:z:0", "mul:z:0"},
        {{"branches", std::move(branches)},
         {"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT}},
         {"Tout", DataTypeSlice{DT_FLOAT}},
         {"output_shapes", std::move(output_shapes)}}}},
      /*ret_def=*/
      {{"c", "case:output:0"}});
  (*foo_func.mutable_attr())["_noinline"].set_b(true);

  // Tensorflow graph:
  //
  //   idx = tf.Placeholder(tf.int32);
  //   a = tf.Placeholder(tf.float);
  //   b = tf.Placeholder(tf.float);
  //
  //   foo_val = Foo(idx, a, b)
  GrapplerItem item;
  item.id = "main";

  item.graph = test::function::GDef(
      {NDef("idx", "Placeholder", {}, {{"dtype", DT_INT32}}, kDevice),
       NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       NDef("b", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       // Calls into function library
       NDef("foo", "Foo", {"idx", "a", "b"}, {}, kDevice)},
      /*funcs=*/
      {case_func, foo_func});
  item.fetch = {"foo"};

  GraphDef output;
  ConfigProto config_proto;

  MetaOptimizer optimizer(nullptr, config_proto);
  absl::Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_EQ(output.library().function_size(), 2);

  const FunctionDef* optimized_foo_func = nullptr;
  const FunctionDef* specialized_branch_func = nullptr;
  for (const FunctionDef& func : output.library().function()) {
    if (func.signature().name() == "Foo")
      optimized_foo_func = &func;
    else if (absl::StartsWith(func.signature().name(), "branch_func"))
      specialized_branch_func = &func;
  }
  ASSERT_TRUE(optimized_foo_func);
  EXPECT_EQ(optimized_foo_func->node_def_size(), 1);
  ASSERT_TRUE(specialized_branch_func);
  EXPECT_EQ(specialized_branch_func->node_def_size(), 3);
}

// Tests for checking expected behavior when skipping tf.data functions in
// meta optimizer.

// Custom optimizer which counts the number of calls of its method `Optimize`
// across all class instances.
class TfDataTestOptimizer : public CustomGraphOptimizer {
 public:
  static void InitCount() { count_ = 0; }
  static int GetCount() { return count_; }

  TfDataTestOptimizer() = default;
  ~TfDataTestOptimizer() override = default;
  TfDataTestOptimizer(const TfDataTestOptimizer&) = delete;
  TfDataTestOptimizer& operator=(const TfDataTestOptimizer& other) = delete;

  std::string name() const override { return "tf_data_test_optimizer"; }
  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return absl::OkStatus();
  }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* optimized_graph) override {
    ++count_;
    *optimized_graph = item.graph;
    return absl::OkStatus();
  }

 private:
  static std::atomic<int> count_;
};

std::atomic<int> TfDataTestOptimizer::count_;

REGISTER_GRAPH_OPTIMIZER(TfDataTestOptimizer);

// Type for specifying how the inner function is nested inside the outer
// function.
enum class FuncNestingType {
  CallFromNode = 0,
  CallFromAttr = 1,
  CallFromList = 2
};

// Test fixture for parametrized testing.
class TfDataTestFixture
    : public ::testing::TestWithParam<std::tuple<bool, bool, FuncNestingType>> {
 protected:
  void SetUp() override {
    is_inner_func_tf_data_ = std::get<0>(GetParam());
    is_outer_func_tf_data_ = std::get<1>(GetParam());
    func_nesting_type_ = std::get<2>(GetParam());
  }
  // Controls which of the functions is flagged as tf.data function.
  bool is_inner_func_tf_data_ = false;
  bool is_outer_func_tf_data_ = false;
  // Controls how the inner function is nested inside the outer function.
  FuncNestingType func_nesting_type_ = FuncNestingType::CallFromNode;
};

// Helper functions for setting up the call of `inner_func` inside of
// `outer_func`.

void SetUpCallFromNode(FunctionDef& outer_func) {
  // Call `inner_func` from a node in `outer_func`.
  outer_func = FunctionDefHelper::Create(
      "outer_func", {"x:float"}, {"z:float"}, {},
      /*node_def=*/
      {{{"inner_func"}, "inner_func", {"x", "x"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z", "inner_func:z:0"}});
}

void SetUpCallFromAttr(FunctionDef& outer_func) {
  // Call `inner_func` from an attribute in a node in `outer_func`.
  outer_func = FunctionDefHelper::Create(
      "outer_func", {"x:float"}, {"z:float"}, {},
      /*node_def=*/
      {{{"identity"},
        "Identity",
        {"x"},
        {{"T", DT_FLOAT},
         {"f", FunctionDefHelper::FunctionRef("inner_func", {})}}}},
      /*ret_def=*/
      {{"z", "x"}});
}

void SetUpCallFromList(FunctionDef& outer_func) {
  // Call `inner_func` from a list attribute in a node in `outer_func`.
  outer_func = FunctionDefHelper::Create(
      "outer_func", {"x:float"}, {"z:float"}, {},
      /*node_def=*/
      {{{"identity"}, "Identity", {"x"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z", "x"}});

  // Add a list containing `inner_func` to the `identity` node.
  // `list_value` will be deallocated automatically since it is passed as
  // allocated list below.
  AttrValue_ListValue* list_value =
      (*outer_func.mutable_node_def(0)->mutable_attr())["list"].mutable_list();
  NameAttrList* entry = list_value->add_func();
  entry->set_name("inner_func");
}

TEST_P(TfDataTestFixture, TfDataTests) {
  using test::function::NDef;

  // Define function library with `outer_func` and `inner_func`.

  FunctionDef inner_func = FunctionDefHelper::Create(
      "inner_func", {"x:float", "y:float"}, {"z:float"}, {},
      /*node_def=*/
      {{{"mul"}, "Mul", {"x", "y"}, {{"T", DT_FLOAT}}}},
      /*ret_def=*/
      {{"z", "mul:z:0"}});
  (*inner_func.mutable_attr())[data::kTFDataFunction].set_b(
      is_inner_func_tf_data_);

  FunctionDef outer_func;
  switch (func_nesting_type_) {
    case FuncNestingType::CallFromNode:
      SetUpCallFromNode(outer_func);
      break;
    case FuncNestingType::CallFromAttr:
      SetUpCallFromAttr(outer_func);
      break;
    case FuncNestingType::CallFromList:
      SetUpCallFromList(outer_func);
      break;
    default:
      break;
  }
  (*outer_func.mutable_attr())[data::kTFDataFunction].set_b(
      is_outer_func_tf_data_);

  // Tensorflow graph:
  //
  //   a = tf.Placeholder(tf.float);
  //   result = outer_func(a);
  GrapplerItem item;
  item.id = "tf_graph";
  item.graph = test::function::GDef(
      {NDef("a", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
       // Calls into function library
       NDef("outer_func_node", "outer_func", {"a"}, {{"T", DT_FLOAT}}, kDevice),
       // Forward outputs
       NDef("out_s", "Identity", {"outer_func_node:0"}, {{"T", DT_FLOAT}},
            kDevice)},
      /*funcs=*/
      {inner_func, outer_func});

  // Use only custom optimizer which counts its calls.
  TfDataTestOptimizer::InitCount();
  ConfigProto config_proto;
  auto& rewriter_config =
      *(config_proto.mutable_graph_options()->mutable_rewrite_options());
  rewriter_config.add_optimizers("TfDataTestOptimizer");
  rewriter_config.set_min_graph_nodes(-1);
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::ONE);

  MetaOptimizer optimizer(nullptr, config_proto);
  GraphDef output;
  const absl::Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // We expect one graph optimization + one optimization for each non-tf.data
  // function. Note that if `outer_func` is flagged as a tf.data function, then
  // `inner_func` is implicitly also considered a tf.data function because it is
  // called from `outer_func`.
  int expected_count = 3;
  if (is_outer_func_tf_data_)
    expected_count = 1;
  else if (is_inner_func_tf_data_)
    expected_count = 2;
  EXPECT_EQ(TfDataTestOptimizer::GetCount(), expected_count);

  // We expect that the tf.data-attribute has been propagated from `outer_func`
  // to its callee `inner_func` if the value is `true`. Otherwise, the attribute
  // values should be unchanged.
  FunctionLibraryDefinition flib(OpRegistry::Global(), output.library());
  const FunctionDef* outer_func_after_opt = flib.Find("outer_func");
  const FunctionDef* inner_func_after_opt = flib.Find("inner_func");

  EXPECT_EQ(data::IsTFDataFunction(*outer_func_after_opt),
            is_outer_func_tf_data_);
  if (is_outer_func_tf_data_ || is_inner_func_tf_data_) {
    EXPECT_EQ(data::IsTFDataFunction(*inner_func_after_opt), true);
  } else {
    EXPECT_EQ(data::IsTFDataFunction(*inner_func_after_opt), false);
  }
}

INSTANTIATE_TEST_SUITE_P(
    MetaOptimizerTest, TfDataTestFixture,
    ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                       ::testing::Values(FuncNestingType::CallFromNode,
                                         FuncNestingType::CallFromAttr,
                                         FuncNestingType::CallFromList)),
    [](const ::testing::TestParamInfo<TfDataTestFixture::ParamType>& info) {
      bool is_inner_func_tf_data = std::get<0>(info.param);
      bool is_outer_func_tf_data = std::get<1>(info.param);
      FuncNestingType func_nesting_type = std::get<2>(info.param);

      std::string test_name;
      if (is_inner_func_tf_data && is_outer_func_tf_data)
        test_name = "both_funcs_tf_data";
      else if (is_inner_func_tf_data)
        test_name = "inner_func_tf_data";
      else if (is_outer_func_tf_data)
        test_name = "outer_func_tf_data";
      else
        test_name = "no_func_tf_data";
      switch (func_nesting_type) {
        case FuncNestingType::CallFromNode:
          test_name += "_call_from_node";
          break;
        case FuncNestingType::CallFromAttr:
          test_name += "_call_from_attribute";
          break;
        case FuncNestingType::CallFromList:
          test_name += "_call_from_list";
          break;
        default:
          break;
      }
      return test_name;
    });

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
