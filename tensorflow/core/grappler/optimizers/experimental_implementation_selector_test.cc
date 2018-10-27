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
#include "tensorflow/core/grappler/optimizers/experimental_implementation_selector.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char CpuDevice[] = "/device:CPU:0";
constexpr char GpuDevice[] = "/device:GPU:0";

class ExperimentalImplementationSelectorTest : public GrapplerTest {};

TEST_F(ExperimentalImplementationSelectorTest, NoUpdate) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {CpuDevice});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  std::unique_ptr<CustomGraphOptimizer> optimizer(
      new ExperimentalImplementationSelector);
  ASSERT_NE(nullptr, optimizer);
  TF_ASSERT_OK(optimizer->Init());

  GraphDef output;
  const Status status = optimizer->Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // This is a trivial graph so there is nothing to update.
  EXPECT_EQ(item.graph.node_size(), output.node_size());
}

TEST_F(ExperimentalImplementationSelectorTest, SwapImplementation) {
  using test::function::NDef;
  auto cpu_def = test::function::XTimesTwo();
  auto* func_attr = cpu_def.mutable_attr();
  (*func_attr)["experimental_api_implements"].set_s("times_two");
  (*func_attr)["experimental_api_preferred_device"].set_s("CPU");

  auto gpu_def = test::function::XAddX();
  auto* func2_attr = gpu_def.mutable_attr();
  (*func2_attr)["experimental_api_implements"].set_s("times_two");
  (*func2_attr)["experimental_api_preferred_device"].set_s("GPU");

  ExperimentalImplementationSelector optimizer;
  GraphDef output;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, GpuDevice),
       NDef("y1", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, GpuDevice),
       NDef("z1", "Identity", {"y1"}, {{"T", DT_FLOAT}}, GpuDevice),
       NDef("y2", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, CpuDevice),
       NDef("z2", "Identity", {"y2"}, {{"T", DT_FLOAT}}, CpuDevice)},
      // FunctionLib
      {cpu_def, gpu_def});

  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(output.node_size(), 5);
  for (const NodeDef& node : output.node()) {
    if (node.name() == "y1") {
      // Make sure the implementation has been swapped to use the GPU version.
      EXPECT_EQ("XAddX", node.op());
    } else if (node.name() == "y2") {
      // Make sure the implementation is not changed.
      EXPECT_EQ("XTimesTwo", node.op());
    }
  }
}

TEST_F(ExperimentalImplementationSelectorTest, SwapImplementationEval) {
  using test::function::NDef;
  auto cpu_def = test::function::XTimesTwo();
  auto* func_attr = cpu_def.mutable_attr();
  (*func_attr)["experimental_api_implements"].set_s("random_boost");
  (*func_attr)["experimental_api_preferred_device"].set_s("CPU");

  auto gpu_def = test::function::XTimesFour();
  auto* func2_attr = gpu_def.mutable_attr();
  (*func2_attr)["experimental_api_implements"].set_s("random_boost");
  (*func2_attr)["experimental_api_preferred_device"].set_s("GPU");

  ExperimentalImplementationSelector optimizer;
  GraphDef output;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, CpuDevice),
       NDef("y", "XTimesFour", {"x"}, {{"T", DT_FLOAT}}, CpuDevice),
       NDef("z", "Identity", {"y"}, {{"T", DT_FLOAT}}, CpuDevice)},
      // FunctionLib
      {cpu_def, gpu_def});

  const Tensor input = test::AsScalar<float>(1.0f);
  item.fetch = {"z"};
  item.feed.emplace_back("x", input);

  const auto four_times_boosted_tensor = EvaluateFetchNodes(item);
  test::ExpectTensorEqual<float>(four_times_boosted_tensor[0],
                                 test::AsScalar<float>(4.0f));

  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));
  GrapplerItem optimized(item, std::move(output));
  const auto twice_boosted_tensor = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(twice_boosted_tensor[0],
                                 test::AsScalar<float>(2.0f));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
