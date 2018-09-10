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

class MetaOptimizerTest : public GrapplerTest {};

TEST_F(MetaOptimizerTest, RunsCustomOptimizer) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  TestOptimizer::SetOptimized(false);
  RewriterConfig rewriter_config;
  rewriter_config.add_optimizers("TestOptimizer");
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, rewriter_config);
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
  RewriterConfig rewriter_config;
  rewriter_config.add_optimizers("TestOptimizer");
  auto customGraphOptimizer = rewriter_config.add_custom_optimizers();
  customGraphOptimizer->set_name("TestGraphOptimizer");
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, rewriter_config);
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

  RewriterConfig rewriter_config;
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, rewriter_config);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
}

TEST_F(MetaOptimizerTest, RunToggleOptimizersAndCustomGraphOptimizerTwice) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  RewriterConfig rewriter_config;
  auto customGraphOptimizer = rewriter_config.add_custom_optimizers();
  customGraphOptimizer->set_name("TestGraphOptimizer");
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, rewriter_config);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestGraphOptimizer::IsOptimized());
}

TEST_F(MetaOptimizerTest, OptimizeFunctionLibrary) {
  using test::function::NDef;

  // Enable ony function optimization.
  RewriterConfig rewriter_config;
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);
  rewriter_config.set_function_optimization(RewriterConfig::ON);
  rewriter_config.add_optimizers("function");
  rewriter_config.set_min_graph_nodes(-1);

  MetaOptimizer optimizer(nullptr, rewriter_config);

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
  EXPECT_EQ(5, optimized_flib.num_functions());

  // MyQuadratic should be specialized once:
  //   0. 'quadratic' node in the main graph
  const string optimized_0 = "MyQuadratic_specialized_for_quadratic";

  // MySquare should be specialized and optimized for 3 instantiations:
  //   1.  'square' node in the main graph
  //   2.  'square' node in the MyQuadratic specialization
  //   3*. 'quadratic' node in the MyQuadratic specialization
  //        has identical instantiation context to #2

  const string optimized_1 = "MySquare_specialized_for_square";
  const string optimized_2 = "MySquare_specialized_for_square_1";

  const FunctionDef* optimized_func_0 = optimized_flib.Find(optimized_0);
  const FunctionDef* optimized_func_1 = optimized_flib.Find(optimized_1);
  const FunctionDef* optimized_func_2 = optimized_flib.Find(optimized_2);

  ASSERT_NE(optimized_func_0, nullptr);
  ASSERT_NE(optimized_func_1, nullptr);
  ASSERT_NE(optimized_func_2, nullptr);

  // Graph should call optimized function.
  int count = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "square" && count++) {
      EXPECT_EQ("MySquare_specialized_for_square", node.op());
    } else if (node.name() == "quadratic" && count++) {
      EXPECT_EQ("MyQuadratic_specialized_for_quadratic", node.op());
    }
  }
  EXPECT_EQ(2, count);

  // Specialized MySquare should call specialized functions.
  count = 0;
  for (const NodeDef& node : optimized_func_0->node_def()) {
    if (node.name() == "square" && count++) {
      EXPECT_EQ(optimized_2, node.op());
    } else if (node.name() == "quadratic" && count++) {
      // Share specialized function with the 'square' node.
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
      if (node.name() == "my_mul/inlined_inputs" && count++) {
        EXPECT_EQ("IdentityN", node.op());
        EXPECT_EQ(2, node.input_size());
        EXPECT_EQ("x:0", node.input(0));
        EXPECT_EQ("x:0", node.input(1));
      } else if (node.name() == "my_mul/x" && count++) {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ(1, node.input_size());
        EXPECT_EQ("my_mul/inlined_inputs:output:0", node.input(0));
      } else if (node.name() == "my_mul/y" && count++) {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ(1, node.input_size());
        EXPECT_EQ("my_mul/inlined_inputs:output:1", node.input(0));
      } else if (node.name() == "my_mul/mul" && count++) {
        EXPECT_EQ("Mul", node.op());
        EXPECT_EQ(2, node.input_size());
        EXPECT_EQ("my_mul/x:output:0", node.input(0));
        EXPECT_EQ("my_mul/y:output:0", node.input(1));
      } else if (node.name() == "my_mul" && count++) {
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

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
