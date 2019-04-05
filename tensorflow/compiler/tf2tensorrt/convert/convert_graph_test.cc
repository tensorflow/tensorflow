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

#include "tensorflow/compiler/tf2tensorrt/convert/convert_graph.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"  // NOLINT
#include "tensorflow/core/public/session.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace convert {

// TODO(laigd): put this into some test utils file.
void ExpectStatus(Status status, error::Code code = error::OK,
                  const char* substr = nullptr) {
  EXPECT_EQ(code, status.code())
      << status << " vs expected error code \"" << error::Code_Name(code)
      << "\" and message \"" << substr << "\"";
  if (substr) {
    EXPECT_THAT(status.error_message(), ::testing::HasSubstr(substr)) << status;
  }
}

TEST(TrtCandidateSelector, Basics) {
  // Create a graph containing both TRT-compatible and TRT-incompatible nodes
  // and use it to test TrtCandidateSelector::IsTensorRTCandidate().
  const std::vector<int32> input_shape_array{2, 2};
  TensorShape input_shape;
  TF_EXPECT_OK(TensorShapeUtils::MakeShape(input_shape_array, &input_shape));

  Scope s = Scope::NewRootScope();
  ops::Placeholder::Attrs feed_attrs;
  TF_EXPECT_OK(
      TensorShapeUtils::MakeShape(input_shape_array, &feed_attrs.shape_));

  // Compatible input.
  auto feed = ops::Placeholder(s.WithOpName("feed"), DT_FLOAT, feed_attrs);
  auto const_1 = ops::Const(s.WithOpName("const_1"), 1.0f, input_shape);

  // Compatible MatMul.
  auto matmul = ops::MatMul(s.WithOpName("matmul"), feed, const_1);

  // Incompatible MatMul.
  ops::MatMul::Attrs matmul_attrs;
  matmul_attrs.transpose_a_ = true;
  auto incompatible_matmul = ops::MatMul(s.WithOpName("incompatible_matmul"),
                                         feed, const_1, matmul_attrs);

  // Unsupported op.
  auto unsupported_op = ops::Erf(s.WithOpName("sin"), feed);

  // Incompatible input.
  auto incompatible_feed = ops::Placeholder(s.WithOpName("feed"), DT_DOUBLE);
  auto const_2 = ops::Const(s.WithOpName("const_2"), 1.0, input_shape);
  // Compatible op with incompatible input.
  auto matmul_with_incompatible_input =
      ops::MatMul(s.WithOpName("matmul_with_incompatible_input"),
                  incompatible_feed, const_2);

  // Quantize ops.
  auto quantize_attrs = ops::FakeQuantWithMinMaxArgs::Min(-6.0f).Max(6.0f);
  auto quantize = ops::FakeQuantWithMinMaxArgs(s.WithOpName("quantize"), feed,
                                               quantize_attrs);

  // Get GrapplerItem and GraphProperties.
  grappler::GrapplerItem item;
  TF_EXPECT_OK(s.ToGraphDef(&item.graph));
  Tensor feed_tensor(DT_FLOAT, input_shape);
  item.feed.push_back(std::make_pair("feed", feed_tensor));
  grappler::GraphProperties graph_properties(item);
  TF_EXPECT_OK(graph_properties.InferStatically(true));

  for (const TrtPrecisionMode precision_mode :
       {TrtPrecisionMode::FP32, TrtPrecisionMode::INT8}) {
    TrtCandidateSelector selector(graph_properties, precision_mode);
    TF_EXPECT_OK(selector.IsTensorRTCandidate(matmul.operation.node()));
    ExpectStatus(
        selector.IsTensorRTCandidate(incompatible_matmul.operation.node()),
        error::INVALID_ARGUMENT,
        "transpose_a is not supported for TensorRT FullyConnected "
        "(op: MatMul), at: incompatible_matmul");
    ExpectStatus(selector.IsTensorRTCandidate(unsupported_op.operation.node()),
                 error::UNIMPLEMENTED, "Op type Erf is not supported");
    ExpectStatus(
        selector.IsTensorRTCandidate(
            matmul_with_incompatible_input.operation.node()),
        error::INTERNAL,
        "Failed to convert input with index 0 to a TRT_TensorOrWeights");
    if (precision_mode == TrtPrecisionMode::INT8) {
      TF_EXPECT_OK(selector.IsTensorRTCandidate(quantize.operation.node()));
    } else {
      ExpectStatus(selector.IsTensorRTCandidate(quantize.operation.node()),
                   error::UNIMPLEMENTED,
                   "Op type FakeQuantWithMinMaxArgs is not supported");
    }
  }
}

class FakeCluster : public grappler::Cluster {
 public:
  FakeCluster() : Cluster(0) {}

  void SetDeviceSet(const DeviceSet* device_set) { device_set_ = device_set; }

  const DeviceSet* GetDeviceSet() const override { return device_set_; }

  string type() const override { return ""; }
  Status Provision() override { return Status::OK(); }
  Status Initialize(const grappler::GrapplerItem& item) override {
    return Status::OK();
  }
  Status Run(const GraphDef& graph_def,
             const std::vector<std::pair<string, Tensor>>& feed,
             const std::vector<string>& fetch, RunMetadata* metadata) override {
    return Status::OK();
  }

 private:
  const DeviceSet* device_set_;
};

TEST(ConvertGraphTest, GetDeviceAndAllocator) {
  ConversionParams params;
  EngineInfo engine_info;
  {
    // params.cluster is not set, and no gpu device is available.
    auto result = GetDeviceAndAllocator(params, engine_info);
    EXPECT_EQ(-1, result.first);
    EXPECT_EQ(nullptr, result.second);
  }

  // Create a session with two (virtual) gpu device.
  SessionOptions options;
  ConfigProto* config = &options.config;
  GPUOptions* gpu_options = config->mutable_gpu_options();
  auto virtual_devices =
      gpu_options->mutable_experimental()->add_virtual_devices();
  virtual_devices->add_memory_limit_mb(200);
  virtual_devices->add_memory_limit_mb(200);
  std::unique_ptr<Session> session(NewSession(options));

  {
    // params.cluster is not set, should find and return first gpu id and
    // corresponding allocator.
    auto result = GetDeviceAndAllocator(params, engine_info);
    EXPECT_EQ(0, result.first);
    EXPECT_NE(nullptr, result.second);
    EXPECT_EQ("GPU_0_bfc", result.second->Name());
  }

  FakeCluster cluster;
  params.cluster = &cluster;
  {
    // params.cluster->GetDeviceSet() returns null, should find and return first
    // gpu id and corresponding allocator.
    auto result = GetDeviceAndAllocator(params, engine_info);
    EXPECT_EQ(0, result.first);
    EXPECT_NE(nullptr, result.second);
    EXPECT_EQ("GPU_0_bfc", result.second->Name());
  }

  // Build the DeviceSet.
  DeviceSet device_set;
  const DeviceMgr* device_mgr = nullptr;
  TF_ASSERT_OK(session->LocalDeviceManager(&device_mgr));
  for (auto d : device_mgr->ListDevices()) {
    device_set.AddDevice(d);
  }
  cluster.SetDeviceSet(&device_set);
  {
    // engine_info.device is not set, should find and return first gpu id and
    // corresponding allocator.
    auto result = GetDeviceAndAllocator(params, engine_info);
    EXPECT_EQ(0, result.first);
    EXPECT_NE(nullptr, result.second);
    EXPECT_EQ("GPU_0_bfc", result.second->Name());
  }

  engine_info.device = "/GPU:1";
  {
    // Set to use second device.
    auto result = GetDeviceAndAllocator(params, engine_info);
    EXPECT_EQ(0, result.first);
    EXPECT_NE(nullptr, result.second);
    EXPECT_EQ("GPU_1_bfc", result.second->Name());
  }

  engine_info.device = "/GPU:3";
  {
    // Set to use nonexistent device.
    auto result = GetDeviceAndAllocator(params, engine_info);
    EXPECT_EQ(-1, result.first);
    EXPECT_EQ(nullptr, result.second);
  }
}

class ConvertAfterShapesTest : public ::testing::Test {
 public:
  Status RunConvertAfterShape(Scope s, GraphDef* output_graph_def) {
    // Create GraphProperties.
    grappler::GrapplerItem item;
    TF_EXPECT_OK(s.ToGraphDef(&item.graph));
    grappler::GraphProperties graph_properties(item);
    TF_EXPECT_OK(graph_properties.InferStatically(true));

    // Construct ConversionParams.
    const std::vector<string> output_names{"output"};
    ConversionParams params;
    params.input_graph_def = &item.graph;
    params.output_names = &output_names;
    params.max_workspace_size_bytes = 8 << 20;
    params.output_graph_def = output_graph_def;
    params.minimum_segment_size = 2;
    params.graph_properties = &graph_properties;
    params.use_calibration = false;

    return ConvertAfterShapes(params);
  }
};

TEST_F(ConvertAfterShapesTest, DirectlyConnectedEngines) {
  // Create the graph. There will be two TRTEngineOps after the conversion, and
  // the upstream TRTEngineOp will have two output connections from the same
  // node:port inside the op to the downstream TRTEngineOp. Then, if it adds the
  // downstream TRTEngineOp first, when adding the upstream op it'll need to
  // update the same output connection twice. This test ensures the correctness
  // of the conversion under such condition.
  Scope s = Scope::NewRootScope();
  auto input = ops::Placeholder(s.WithOpName("input"), DT_FLOAT,
                                ops::Placeholder::Shape({2, 1}));
  // We purposefully choose the name of the root node of each segment, so it'll
  // process the segment in the downstream first, then, when it tries to update
  // the edge between the two TRTEngineOps, it'll try to add the same edge
  // multiple times.
  auto segment_root_1 = ops::Identity(s.WithOpName("segment_root_b"), input);
  auto add1 = ops::Add(s.WithOpName("add1"), segment_root_1, segment_root_1);
  // Add incompatible reshapes that change the batch dimension.
  auto incompatible =
      ops::Reshape(s.WithOpName("reshape1"), add1, Input({1, 2}));
  incompatible =
      ops::Reshape(s.WithOpName("reshape2"), incompatible, Input({2, 1}));

  auto add2 = ops::Add(s.WithOpName("add2"), incompatible, add1);
  auto segment_root_2 = ops::Identity(s.WithOpName("segment_root_a"), add1);
  auto add3 = ops::Add(s.WithOpName("add3"), add2, segment_root_2);
  ops::Identity(s.WithOpName("output"), add3);

  GraphDef output_graph_def;
  TF_EXPECT_OK(RunConvertAfterShape(s, &output_graph_def));

  int num_trt_ops = 0;
  for (const NodeDef& node : output_graph_def.node()) {
    if (node.name() == "TRTEngineOp_1") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("input", node.input(0));
      ++num_trt_ops;
    } else if (node.name() == "TRTEngineOp_0") {
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("TRTEngineOp_1", node.input(0));
      EXPECT_EQ("reshape2", node.input(1));
      ++num_trt_ops;
    }
  }
  EXPECT_EQ(2, num_trt_ops);
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
