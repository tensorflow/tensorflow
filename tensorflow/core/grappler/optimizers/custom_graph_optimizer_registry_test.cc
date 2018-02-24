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

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

static const char* kTestOptimizerName = "Test";

class TestGraphOptimizer : public CustomGraphOptimizer {
 public:
  Status Init() override { return Status::OK(); }
  string name() const override { return kTestOptimizerName; }
  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override {
    return Status::OK();
  }
  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override {}
};

REGISTER_GRAPH_OPTIMIZER_AS(TestGraphOptimizer, "StaticRegister");

TEST(CustomGraphOptimizerRegistryTest, DynamicRegistration) {
  std::vector<string> optimizers =
      CustomGraphOptimizerRegistry::GetRegisteredOptimizers();
  std::unique_ptr<const CustomGraphOptimizer> test_optimizer;
  ASSERT_EQ(
      0, std::count(optimizers.begin(), optimizers.end(), "DynamicRegister"));
  test_optimizer =
      CustomGraphOptimizerRegistry::CreateByNameOrNull("DynamicRegister");
  EXPECT_EQ(nullptr, test_optimizer);
  CustomGraphOptimizerRegistry::RegisterOptimizerOrDie(
      []() { return new TestGraphOptimizer; }, "DynamicRegister");
  optimizers = CustomGraphOptimizerRegistry::GetRegisteredOptimizers();
  ASSERT_EQ(
      1, std::count(optimizers.begin(), optimizers.end(), "DynamicRegister"));
  test_optimizer =
      CustomGraphOptimizerRegistry::CreateByNameOrNull("DynamicRegister");
  ASSERT_NE(nullptr, test_optimizer);
  EXPECT_EQ(kTestOptimizerName, test_optimizer->name());
}

TEST(CustomGraphOptimizerRegistryTest, StaticRegistration) {
  const std::vector<string> optimizers =
      CustomGraphOptimizerRegistry::GetRegisteredOptimizers();
  EXPECT_EQ(1,
            std::count(optimizers.begin(), optimizers.end(), "StaticRegister"));
  std::unique_ptr<const CustomGraphOptimizer> test_optimizer =
      CustomGraphOptimizerRegistry::CreateByNameOrNull("StaticRegister");
  ASSERT_NE(nullptr, test_optimizer);
  EXPECT_EQ(kTestOptimizerName, test_optimizer->name());
}

TEST(GraphOptimizerRegistryTest, CrashesOnDuplicateRegistration) {
  const auto creator = []() { return new TestGraphOptimizer; };
  EXPECT_DEATH(CustomGraphOptimizerRegistry::RegisterOptimizerOrDie(
                   creator, "StaticRegister"),
               "twice");
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
