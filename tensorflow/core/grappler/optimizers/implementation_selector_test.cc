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
#include "tensorflow/core/grappler/optimizers/implementation_selector.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
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
constexpr char TpuDevice[] = "/device:TPU_REPLICATED_CORE";

class ImplementationSelectorTest : public GrapplerTest {};

TEST_F(ImplementationSelectorTest, NoUpdate) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {CpuDevice});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  std::unique_ptr<CustomGraphOptimizer> optimizer(new ImplementationSelector);
  ASSERT_NE(nullptr, optimizer);
  TF_ASSERT_OK(optimizer->Init());

  GraphDef output;
  const Status status = optimizer->Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // This is a trivial graph so there is nothing to update.
  EXPECT_EQ(item.graph.node_size(), output.node_size());
}

TEST_F(ImplementationSelectorTest, SelectDeviceIndex) {
  using test::function::NDef;
  ImplementationSelector optimizer;
  GraphDef output;
  GrapplerItem item;
  AttrValue device_names;
  device_names.mutable_list()->add_s("CPU");
  device_names.mutable_list()->add_s("GPU");
  item.graph = test::function::GDef(
      {NDef("x", "DeviceIndex", {}, {{"device_names", device_names}},
            CpuDevice),
       NDef("case", "Case", {"x"}, {{"T", DT_FLOAT}}, GpuDevice)});

  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  for (const NodeDef& node : output.node()) {
    if (node.name() == "x") {
      // Rewrite DeviceIndex op to a Const op with value of GPU index 1.
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.attr().at("value").tensor().int_val(0));
    }
  }
}

TEST_F(ImplementationSelectorTest, SelectDeviceIndexStatelessCase) {
  using test::function::NDef;
  ImplementationSelector optimizer;
  GraphDef output;
  GrapplerItem item;
  AttrValue device_names;
  device_names.mutable_list()->add_s("CPU");
  device_names.mutable_list()->add_s("GPU");
  item.graph = test::function::GDef(
      {NDef("x", "DeviceIndex", {}, {{"device_names", device_names}},
            CpuDevice),
       NDef("case", "StatelessCase", {"x"}, {{"T", DT_FLOAT}}, GpuDevice)});

  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  for (const NodeDef& node : output.node()) {
    if (node.name() == "x") {
      // Rewrite DeviceIndex op to a Const op with value of GPU index 1.
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.attr().at("value").tensor().int_val(0));
    }
  }
}

TEST_F(ImplementationSelectorTest, SelectDeviceIndexMultiOps) {
  using test::function::NDef;
  ImplementationSelector optimizer;
  GraphDef output;
  GrapplerItem item;
  AttrValue device_names;
  device_names.mutable_list()->add_s("CPU");
  device_names.mutable_list()->add_s("TPU_REPLICATED_CORE");
  device_names.mutable_list()->add_s("GPU");
  item.graph = test::function::GDef(
      {NDef("x", "DeviceIndex", {}, {{"device_names", device_names}},
            CpuDevice),
       NDef("case", "Case", {"x"}, {{"T", DT_FLOAT}}, GpuDevice),
       NDef("y", "DeviceIndex", {}, {{"device_names", device_names}},
            GpuDevice),
       NDef("case_y", "Case", {"y"}, {{"T", DT_FLOAT}}, TpuDevice)});

  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));
  for (const NodeDef& node : output.node()) {
    if (node.name() == "x") {
      // Rewrite DeviceIndex op to a Const op with value of GPU index 1.
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(2, node.attr().at("value").tensor().int_val(0));
    }
    if (node.name() == "y") {
      // Rewrite DeviceIndex op to a Const op with value of CPU index 0.
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.attr().at("value").tensor().int_val(0));
    }
  }
}

TEST_F(ImplementationSelectorTest, SelectDeviceIndexNotFound) {
  using test::function::NDef;
  ImplementationSelector optimizer;
  GraphDef output;
  GrapplerItem item;
  AttrValue device_names;
  device_names.mutable_list()->add_s("CPU");
  device_names.mutable_list()->add_s("GPU");
  item.graph = test::function::GDef(
      {NDef("x", "DeviceIndex", {}, {{"device_names", device_names}},
            CpuDevice),
       NDef("case", "Case", {"x"}, {{"T", DT_FLOAT}}, TpuDevice)});

  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  for (const NodeDef& node : output.node()) {
    if (node.name() == "x") {
      // Rewrite DeviceIndex op to a Const op with value of device names length.
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(2, node.attr().at("value").tensor().int_val(0));
    }
  }
}

TEST_F(ImplementationSelectorTest, SelectDeviceIndexError) {
  using test::function::NDef;
  ImplementationSelector optimizer;
  GraphDef output;
  GrapplerItem item;
  AttrValue device_names;
  device_names.mutable_list()->add_s("CPU");
  device_names.mutable_list()->add_s("GPU");
  item.graph = test::function::GDef(
      {NDef("x", "DeviceIndex", {}, {{"device_names", device_names}},
            CpuDevice),
       NDef("case", "Case", {"x"}, {{"T", DT_FLOAT}}, "")});

  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  for (const NodeDef& node : output.node()) {
    if (node.name() == "x") {
      // Device parse has error, do not rewrite the DeviceIndexNode.
      EXPECT_EQ("DeviceIndex", node.op());
    }
  }
}

TEST_F(ImplementationSelectorTest, TwoTypesOfSwapImplementation) {
  using test::function::NDef;
  ImplementationSelector optimizer;
  GraphDef output;
  GrapplerItem item;
  // DeviceIndex op based implementation selector.
  AttrValue device_names;
  device_names.mutable_list()->add_s("CPU");
  device_names.mutable_list()->add_s("TPU_REPLICATED_CORE");
  device_names.mutable_list()->add_s("GPU");

  // Function swap based implementation selector.
  auto cpu_def = test::function::XTimesTwo();
  auto* func_attr = cpu_def.mutable_attr();
  (*func_attr)["api_implements"].set_s("times_two");
  (*func_attr)["api_preferred_device"].set_s("CPU");

  auto gpu_def = test::function::XAddX();
  auto* func2_attr = gpu_def.mutable_attr();
  (*func2_attr)["api_implements"].set_s("times_two");
  (*func2_attr)["api_preferred_device"].set_s("GPU");

  item.graph = test::function::GDef(
      {NDef("x", "DeviceIndex", {}, {{"device_names", device_names}},
            CpuDevice),
       NDef("case", "Case", {"x"}, {{"T", DT_FLOAT}}, GpuDevice),
       NDef("y", "DeviceIndex", {}, {{"device_names", device_names}},
            GpuDevice),
       NDef("case_y", "Case", {"y"}, {{"T", DT_FLOAT}}, TpuDevice),
       NDef("y1", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, GpuDevice),
       NDef("z1", "Identity", {"y1"}, {{"T", DT_FLOAT}}, GpuDevice),
       NDef("y2", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, CpuDevice),
       NDef("z2", "Identity", {"y2"}, {{"T", DT_FLOAT}}, CpuDevice)},
      // FunctionLib
      {cpu_def, gpu_def});

  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));
  for (const NodeDef& node : output.node()) {
    if (node.name() == "x") {
      // Rewrite DeviceIndex op to a Const op with value of GPU index 1.
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(2, node.attr().at("value").tensor().int_val(0));
    }
    if (node.name() == "y") {
      // Rewrite DeviceIndex op to a Const op with value of CPU index 0.
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.attr().at("value").tensor().int_val(0));
    }
    if (node.name() == "y1") {
      // Make sure the implementation has been swapped to use the GPU version.
      EXPECT_EQ("XAddX", node.op());
    } else if (node.name() == "y2") {
      // Make sure the implementation is not changed.
      EXPECT_EQ("XTimesTwo", node.op());
    }
  }
}

TEST_F(ImplementationSelectorTest, NoSwapWithImplementsOnly) {
  using test::function::NDef;
  ImplementationSelector optimizer;
  GraphDef output;
  GrapplerItem item;
  // DeviceIndex op based implementation selector.
  AttrValue device_names;
  device_names.mutable_list()->add_s("CPU");
  device_names.mutable_list()->add_s("TPU_REPLICATED_CORE");
  device_names.mutable_list()->add_s("GPU");

  // Api_implements exists, api_preferred_device does not, no swap.
  auto cpu_def = test::function::XTimesTwo();
  auto* func_attr = cpu_def.mutable_attr();
  (*func_attr)["api_implements"].set_s("times_two");

  auto gpu_def = test::function::XAddX();
  auto* func2_attr = gpu_def.mutable_attr();
  (*func2_attr)["api_implements"].set_s("times_two");

  item.graph = test::function::GDef(
      {NDef("x", "DeviceIndex", {}, {{"device_names", device_names}},
            CpuDevice),
       NDef("case", "Case", {"x"}, {{"T", DT_FLOAT}}, GpuDevice),
       NDef("y", "DeviceIndex", {}, {{"device_names", device_names}},
            GpuDevice),
       NDef("case_y", "Case", {"y"}, {{"T", DT_FLOAT}}, TpuDevice),
       NDef("y1", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, GpuDevice),
       NDef("z1", "Identity", {"y1"}, {{"T", DT_FLOAT}}, GpuDevice),
       NDef("y2", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, CpuDevice),
       NDef("z2", "Identity", {"y2"}, {{"T", DT_FLOAT}}, CpuDevice)},
      // FunctionLib
      {cpu_def, gpu_def});

  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));
  for (const NodeDef& node : output.node()) {
    if (node.name() == "x") {
      // Rewrite DeviceIndex op to a Const op with value of GPU index 1.
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(2, node.attr().at("value").tensor().int_val(0));
    }
    if (node.name() == "y") {
      // Rewrite DeviceIndex op to a Const op with value of CPU index 0.
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.attr().at("value").tensor().int_val(0));
    }
    if (node.name() == "y1") {
      // api_implements only, no preferred device, no swap.
      EXPECT_EQ("XTimesTwo", node.op());
    } else if (node.name() == "y2") {
      // Make sure the implementation is not changed.
      EXPECT_EQ("XTimesTwo", node.op());
    }
  }
}

TEST_F(ImplementationSelectorTest, SwapImplementation) {
  using test::function::NDef;
  auto cpu_def = test::function::XTimesTwo();
  auto* func_attr = cpu_def.mutable_attr();
  (*func_attr)["api_implements"].set_s("times_two");
  (*func_attr)["api_preferred_device"].set_s("CPU");

  auto gpu_def = test::function::XAddX();
  auto* func2_attr = gpu_def.mutable_attr();
  (*func2_attr)["api_implements"].set_s("times_two");
  (*func2_attr)["api_preferred_device"].set_s("GPU");

  ImplementationSelector optimizer;
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

TEST_F(ImplementationSelectorTest, SwapImplementationTpu) {
  using test::function::NDef;
  auto cpu_def = test::function::XTimesTwo();
  auto* func_attr = cpu_def.mutable_attr();
  (*func_attr)["api_implements"].set_s("times_two");
  (*func_attr)["api_preferred_device"].set_s("CPU");

  auto tpu_def = test::function::XAddX();
  auto* func2_attr = tpu_def.mutable_attr();
  (*func2_attr)["api_implements"].set_s("times_two");
  (*func2_attr)["api_preferred_device"].set_s("TPU_REPLICATED_CORE");

  ImplementationSelector optimizer;
  GraphDef output;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, TpuDevice),
       NDef("y1", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, TpuDevice),
       NDef("z1", "Identity", {"y1"}, {{"T", DT_FLOAT}}, TpuDevice),
       NDef("y2", "XTimesTwo", {"x"}, {{"T", DT_FLOAT}}, CpuDevice),
       NDef("z2", "Identity", {"y2"}, {{"T", DT_FLOAT}}, CpuDevice)},
      // FunctionLib
      {cpu_def, tpu_def});

  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(output.node_size(), 5);
  for (const NodeDef& node : output.node()) {
    if (node.name() == "y1") {
      // Make sure the implementation has been swapped to use the TPU version.
      EXPECT_EQ("XAddX", node.op());
    } else if (node.name() == "y2") {
      // Make sure the implementation is not changed.
      EXPECT_EQ("XTimesTwo", node.op());
    }
  }
}

TEST_F(ImplementationSelectorTest, SwapImplementationEval) {
  using test::function::NDef;
  auto cpu_def = test::function::XTimesTwo();
  auto* func_attr = cpu_def.mutable_attr();
  (*func_attr)["api_implements"].set_s("random_boost");
  (*func_attr)["api_preferred_device"].set_s("CPU");

  auto gpu_def = test::function::XTimesFour();
  auto* func2_attr = gpu_def.mutable_attr();
  (*func2_attr)["api_implements"].set_s("random_boost");
  (*func2_attr)["api_preferred_device"].set_s("GPU");

  ImplementationSelector optimizer;
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
  GrapplerItem optimized = item.WithGraph(std::move(output));
  const auto twice_boosted_tensor = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(twice_boosted_tensor[0],
                                 test::AsScalar<float>(2.0f));
}

TEST_F(ImplementationSelectorTest, SwapImplementationWithGradient) {
  using test::function::NDef;
  using FDH = FunctionDefHelper;
  // boost_1 returns the doubled input and a const as the internal state, the
  // state will be feed to gradient function to mimic the behavior of backward
  // function of defun that use internal states as extra inputs.
  FunctionDef boost_1 = FDH::Create(
      "Boost1", {"x:float"}, {"z:float", "s:float"}, {},
      {{{"boost"}, "Add", {"x", "x"}, {{"T", DT_FLOAT}}},
       FDH::Const("one", 1.0f)},
      /* Mapping between function returns and function node outputs. */
      {{"z", "boost:z:0"}, {"s", "one:output:0"}});
  auto* boost_1_attr = boost_1.mutable_attr();
  (*boost_1_attr)["api_implements"].set_s("random_boost");
  (*boost_1_attr)["api_preferred_device"].set_s("CPU");
  (*boost_1_attr)["backward_function_name"].set_s("BoostCpuGradient");

  FunctionDef boost_1_gradient = FDH::Create(
      "Boost1Gradient", {"x:float", "s:float"}, {"dx:float"}, {},
      {FDH::Const("two", 2.0f),
       {{"grad"}, "Mul", {"x", "two:output:0"}, {{"T", DT_FLOAT}}}},
      /* Mapping between function returns and function node outputs. */
      {{"dx", "grad:z:0"}});
  auto* boost_1_grad_attr = boost_1_gradient.mutable_attr();
  (*boost_1_grad_attr)["api_implements"].set_s("random_boost");
  (*boost_1_grad_attr)["api_preferred_device"].set_s("CPU");
  (*boost_1_grad_attr)["forward_function_name"].set_s("BoostCpu");

  // boost_2 return the input * 4, and with two extra internal states.
  FunctionDef boost_2_func = FDH::Create(
      "Boost2", {"x:float"}, {"z:float", "s1:float", "s2:float"}, {},
      {FDH::Const("four", 4.0f),
       {{"boost"}, "Mul", {"x", "four:output:0"}, {{"T", DT_FLOAT}}},
       FDH::Const("one", 1.0f),
       FDH::Const("two", 2.0f)},
      /* Mapping between function returns and function node outputs. */
      {{"z", "boost:z:0"}, {"s1", "one:output:0"}, {"s2", "two:output:0"}});
  auto* boost_2_attr = boost_2_func.mutable_attr();
  (*boost_2_attr)["api_implements"].set_s("random_boost");
  (*boost_2_attr)["api_preferred_device"].set_s("GPU");
  (*boost_2_attr)["backward_function_name"].set_s("BoostGpuGradient");

  FunctionDef boost_2_gradient = FDH::Create(
      "Boost2Gradient", {"x:float", "s1:float", "s2:float"}, {"dx:float"}, {},
      {FDH::Const("four", 4.0f),
       {{"grad"}, "Mul", {"x", "four:output:0"}, {{"T", DT_FLOAT}}}},
      /* Mapping between function returns and function node outputs. */
      {{"dx", "grad:z:0"}});
  auto* boost_2_grad_attr = boost_2_gradient.mutable_attr();
  (*boost_2_grad_attr)["api_implements"].set_s("random_boost");
  (*boost_2_grad_attr)["api_preferred_device"].set_s("GPU");
  (*boost_2_grad_attr)["forward_function_name"].set_s("BoostGpu");

  // Define the forward function with f = boost2 function but with CPU device.
  // Expect the grappler plugin to swap f and attributes to use the boost1.
  const auto forward =
      NDef("lstm/StatefulPartitionedCall", "StatefulPartitionedCall", {"input"},
           {{"Tin", DataTypeSlice{DT_FLOAT}},
            {"Tout", DataTypeSlice{DT_FLOAT, DT_FLOAT, DT_FLOAT}},
            {"f", FDH::FunctionRef("Boost2")}},
           CpuDevice);
  const auto backward =
      NDef("gradient/lstm/StatefulPartitionedCall", "StatefulPartitionedCall",
           {"input", "lstm/StatefulPartitionedCall:1",
            "lstm/StatefulPartitionedCall:2"},
           {{"Tin", DataTypeSlice{DT_FLOAT, DT_FLOAT, DT_FLOAT}},
            {"Tout", DataTypeSlice{DT_FLOAT}},
            {"f", FDH::FunctionRef("Boost2Gradient")}},
           CpuDevice);

  ImplementationSelector optimizer;
  GraphDef output;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("input", "Placeholder", {}, {{"dtype", DT_FLOAT}}, CpuDevice),
       forward, backward,
       NDef("output", "Identity", {"lstm/StatefulPartitionedCall:0"},
            {{"T", DT_FLOAT}}, CpuDevice)},
      // FunctionLib
      {boost_1, boost_1_gradient, boost_2_func, boost_2_gradient});

  const Tensor input = test::AsScalar<float>(1.0f);
  item.fetch = {"output"};
  item.feed.emplace_back("input", input);

  const auto four_times_boosted_tensor = EvaluateFetchNodes(item);
  test::ExpectTensorEqual<float>(four_times_boosted_tensor[0],
                                 test::AsScalar<float>(4.0f));

  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));
  GrapplerItem optimized = item.WithGraph(std::move(output));
  const auto twice_boosted_tensor = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(twice_boosted_tensor[0],
                                 test::AsScalar<float>(2.0f));
}
}  // namespace
}  // namespace grappler
}  // namespace tensorflow
