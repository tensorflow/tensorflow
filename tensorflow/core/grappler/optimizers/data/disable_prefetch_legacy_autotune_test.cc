/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/disable_prefetch_legacy_autotune.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

using test::function::NDef;

absl::Status OptimizeWithDisablePrefetchLegacyAutotune(const GrapplerItem &item,
                                                       GraphDef *output,
                                                       bool autotune) {
  DisablePrefetchLegacyAutotune optimizer;
  RewriterConfig_CustomGraphOptimizer config;
  if (autotune) {
    (*config.mutable_parameter_map())["autotune"].set_s("true");
  } else {
    (*config.mutable_parameter_map())["autotune"].set_s("false");
  }
  TF_RETURN_IF_ERROR(optimizer.Init(&config));
  return optimizer.Optimize(nullptr, item, output);
}

class RewriteTest : public ::testing::TestWithParam<bool> {};

TEST_P(RewriteTest, DisablePrefetchLegacyAutotune) {
  const bool autotune = GetParam();
  GrapplerItem item;

  item.graph = test::function::GDef({
      NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
      NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
      NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
      NDef("range", "RangeDataset", {"start", "stop", "step"},
           {{"output_shapes", absl::Span<const TensorShape>{}},
            {"output_types", absl::Span<const DataType>{}}}),
      NDef("prefetch1", "PrefetchDataset", {"range"},
           {{"legacy_autotune", true}}),
      NDef("prefetch2", "PrefetchDataset", {"prefetch1"},
           {{"legacy_autotune", false}}),
      NDef("prefetch3", "PrefetchDataset", {"prefetch2"}, {}),
  });

  GraphDef output;
  TF_ASSERT_OK(
      OptimizeWithDisablePrefetchLegacyAutotune(item, &output, autotune));

  NodeDef prefetch_node1 =
      output.node(graph_utils::FindGraphNodeWithName("prefetch1", output));
  EXPECT_EQ(prefetch_node1.attr().at("legacy_autotune").b(), !autotune);
  NodeDef prefetch_node2 =
      output.node(graph_utils::FindGraphNodeWithName("prefetch2", output));
  EXPECT_FALSE(prefetch_node2.attr().at("legacy_autotune").b());
  NodeDef prefetch_node3 =
      output.node(graph_utils::FindGraphNodeWithName("prefetch3", output));
  if (autotune) {
    EXPECT_FALSE(prefetch_node3.attr().at("legacy_autotune").b());
  } else {
    EXPECT_TRUE(prefetch_node3.attr().find("legacy_autotune") ==
                prefetch_node3.attr().end());
  }
}

INSTANTIATE_TEST_SUITE_P(Test, RewriteTest, ::testing::Values(false, true));

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
