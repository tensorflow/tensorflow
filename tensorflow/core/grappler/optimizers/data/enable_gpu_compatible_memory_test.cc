/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/enable_gpu_compatible_memory.h"

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

constexpr char kOptionsDataset[] = "OptionsDataset";
constexpr char kParallelMapDataset[] = "ParallelMapDatasetV2";
constexpr char kPrefetchDataset[] = "PrefetchDataset";

Status Optimize(EnableGPUCompatibleMemory &optimizer, const GrapplerItem &item,
                GraphDef *output, bool autotune) {
  RewriterConfig_CustomGraphOptimizer config;
  if (autotune) {
    (*config.mutable_parameter_map())["autotune"].set_s("true");
  } else {
    (*config.mutable_parameter_map())["autotune"].set_s("false");
  }
  TF_RETURN_IF_ERROR(optimizer.Init(&config));
  return optimizer.Optimize(nullptr, item, output);
}

Status OptimizeWithEnableGPUCompatibleMemory(const GrapplerItem &item, GraphDef *output,
                                  bool autotune) {
  EnableGPUCompatibleMemory optimizer;
  return Optimize(optimizer, item, output, autotune);
}

class EnableGPUCompatibleMemoryParameterizedTest : public ::testing::TestWithParam<bool> {
};

TEST_P(RewriteTest, DisablePrefetchLegacyAutotune) {
  const bool autotune = GetParam();
  GrapplerItem item;

  item.graph = test::function::GDef({
      NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
      NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
      NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
      NDef("range", "RangeDataset", {"start", "stop", "step"},
           {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
            {"output_types", gtl::ArraySlice<DataType>{}}}),
      NDef("prefetch1", "PrefetchDataset", {"range"},
           {{"legacy_autotune", true}}),
  });

  GraphDef output;
  TF_ASSERT_OK(
      OptimizeWithEnableGPUCompatibleMemory(item, &output, autotune));

  NodeDef prefetch_node1 =
      output.node(graph_utils::FindGraphNodeWithName("prefetch1", output));
  EXPECT_EQ(prefetch_node1.attr().at("legacy_autotune").b(), !autotune);
  // TODO (kushanam): enable when the use_gpu_compat_allocator attribute is added 
  // to the Dataset ops
  // NodeDef range =
  //     output.node(graph_utils::FindGraphNodeWithName("range", output));
  // EXPECT_TRUE(prefetch_node2.attr().at("use_gpu_compat_allocator").b());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
