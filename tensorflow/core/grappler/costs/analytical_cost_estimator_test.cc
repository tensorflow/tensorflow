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

#include "tensorflow/core/grappler/costs/virtual_scheduler.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/analytical_cost_estimator.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

class AnalyticalCostEstimatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initializes cluster_ and placer_.
    std::unordered_map<string, DeviceProperties> devices;
    DeviceProperties cpu_device;
    cpu_device.set_type("CPU");
    cpu_device.set_num_cores(4);
    cpu_device.set_frequency(2600);
    cpu_device.set_bandwidth(24 * 1024 * 1024);
    devices["/job:localhost/replica:0/task:0/cpu:0"] = cpu_device;
    DeviceProperties gpu_device;
    gpu_device.set_type("GPU");
    gpu_device.set_num_cores(12);
    gpu_device.set_frequency(1100);
    gpu_device.set_bandwidth(180 * 1024 * 1024);
    (*gpu_device.mutable_environment())["architecture"] = "6";
    devices["/job:localhost/replica:0/task:0/device:GPU:0"] = gpu_device;

    cluster_.reset(new VirtualCluster(devices));
  }

  GrapplerItem CreateMiniGraph() {
    const int batch = 1;
    const int width = 28;
    const int height = 28;
    const int num_channels = 1;
    const int num_labels = 10;
    const int kernel_size = 3;
    const int conv_filters = 32;

    Scope s = Scope::NewRootScope();
    auto images = ops::RandomUniform(
        s.WithOpName("image"), {batch, width, height, num_channels}, DT_FLOAT);
    auto labels = ops::RandomUniform(s.WithOpName("label"), {batch, num_labels},
                                     DT_FLOAT);
    auto w = ops::Variable(
        s.WithOpName("W"),
        {kernel_size, kernel_size, num_channels, conv_filters}, DT_FLOAT);
    auto b = ops::Variable(s.WithOpName("B"), {conv_filters}, DT_FLOAT);
    auto conv =
        ops::Conv2D(s.WithOpName("conv"), images, w, {1, 1, 1, 1}, "SAME");
    auto bias = ops::Add(s.WithOpName("bias"), conv, b);
    auto relu = ops::Relu(s.WithOpName("relu"), bias);
    auto flat_shape = ops::Const(s.WithOpName("flat_shape"),
                                 {batch, width * height * conv_filters});
    auto flat = ops::Reshape(s.WithOpName("flat"), relu, flat_shape);

    auto w2 =
        ops::Variable(s.WithOpName("W2"),
                      {width * height * conv_filters, num_labels}, DT_FLOAT);
    auto b2 = ops::Variable(s.WithOpName("B2"), {num_labels}, DT_FLOAT);
    auto matmul = ops::MatMul(s.WithOpName("matmul"), flat, w2);
    auto logits = ops::Add(s.WithOpName("logits"), matmul, b2);
    auto softmax = ops::Softmax(s.WithOpName("softmax"), logits);
    auto lsm = ops::Log(s.WithOpName("lsm"), softmax);

    GrapplerItem item;
    item.fetch.push_back("lsm");
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    return item;
  }

  std::unique_ptr<VirtualCluster> cluster_;
};

TEST_F(AnalyticalCostEstimatorTest, SimpleTest) {
  GrapplerItem item = CreateMiniGraph();

  AnalyticalCostEstimator estimator(cluster_.get(), /*use_static_shapes=*/true,
                                    /*use_aggressive_shape_inference=*/true);
  TF_ASSERT_OK(estimator.Initialize(item));

  RunMetadata run_metadata;
  Costs summary;
  TF_ASSERT_OK(estimator.PredictCosts(item.graph, &run_metadata, &summary));

  EXPECT_EQ(Costs::NanoSeconds(9158), summary.execution_time);
  // Note there are totally 17 nodes (RandomUniform creates 2 nodes), but
  // grappler will not process "label", therefore we have 15 here instead
  EXPECT_EQ(15, summary.num_ops_total);

  // Make this estimate accurate:
  // TODO(http://b/70031255): Accurate estimator for RandomUniform op needed
  //
  // Change to EXPECT_FALSE when the above TODOs are done:
  EXPECT_TRUE(summary.inaccurate);
  EXPECT_EQ(0, summary.num_ops_with_unknown_shapes);
}

}  // end namespace grappler
}  // end namespace tensorflow
