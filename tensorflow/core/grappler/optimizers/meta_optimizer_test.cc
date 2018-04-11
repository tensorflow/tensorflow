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
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class TestOptimizer : public CustomGraphOptimizer {
 public:
  static void SetOptimized(const bool flag_value) { optimized_ = flag_value; }
  static bool IsOptimized() { return optimized_; }

  TestOptimizer() {}
  string name() const override { return "test_optimizer"; }

  Status Init() override { return Status::OK(); }

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

TEST(MetaOptimizerTest, RunsCustomOptimizer) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  TestOptimizer::SetOptimized(false);
  RewriterConfig rewriter_config;
  rewriter_config.add_optimizers("TestOptimizer");

  MetaOptimizer optimizer(nullptr, rewriter_config);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_TRUE(TestOptimizer::IsOptimized());
}

TEST(MetaOptimizerTest, RunOptimizersTwice) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  RewriterConfig rewriter_config;
  rewriter_config.set_meta_optimizer_iterations(RewriterConfig::TWO);

  MetaOptimizer optimizer(nullptr, rewriter_config);
  GraphDef output;
  const Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
