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

#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

namespace {

void CreateConstOp(const string& name, std::initializer_list<int64> dims,
                   NodeDef* node) {
  Tensor tensor(DT_FLOAT, TensorShape(dims));
  for (int64_t i = 0; i < tensor.NumElements(); ++i)
    tensor.flat<float>()(i) = i / 10.0f;
  TF_CHECK_OK(NodeDefBuilder(name, "Const")
                  .Attr("dtype", DT_FLOAT)
                  .Attr("value", tensor)
                  .Finalize(node));
}

void CreateConstSizesOp(const string& name, const std::vector<int32>& sizes,
                        NodeDef* node) {
  TensorShape shape;
  shape.AddDim(sizes.size());
  Tensor tensor(DT_INT32, shape);
  for (int64_t i = 0; i < tensor.NumElements(); ++i)
    tensor.flat<int32>()(i) = sizes[i];
  TF_CHECK_OK(NodeDefBuilder(name, "Const")
                  .Attr("dtype", DT_INT32)
                  .Attr("value", tensor)
                  .Finalize(node));
}

// Helper method for converting shapes vector to TensorProperty.
OpInfo::TensorProperties ShapeToTensorProperty(const std::vector<int>& shapes,
                                               const DataType& data_type) {
  OpInfo::TensorProperties prop;
  prop.set_dtype(data_type);
  for (int shape : shapes) prop.mutable_shape()->add_dim()->set_size(shape);
  return prop;
}

TEST(UtilsTest, ConvOpInfo) {
  int batch = 32;
  int rows = 7;
  int cols = 9;
  int filter_rows = 3;
  int filter_cols = 3;
  int out_rows = 7;
  int out_cols = 9;
  int in_depth = 3;
  int out_depth = 5;
  int stride = 1;

  std::unordered_map<string, const NodeDef*> name_to_node;
  GraphDef graph;
  NodeDef* input = graph.add_node();
  name_to_node["input"] = input;
  CreateConstOp("input", {batch, rows, cols, in_depth}, input);
  NodeDef* filter = graph.add_node();
  name_to_node["filter"] = filter;
  CreateConstOp("filter", {filter_rows, filter_cols, in_depth, out_depth},
                filter);
  NodeDef* output_backprop = graph.add_node();
  name_to_node["output_backprop"] = output_backprop;
  CreateConstOp("output_backprop", {batch, out_rows, out_cols, out_depth},
                output_backprop);
  NodeDef* input_sizes = graph.add_node();
  name_to_node["input_sizes"] = input;
  CreateConstSizesOp("input_sizes",
                     std::vector<int32>({batch, rows, cols, in_depth}),
                     input_sizes);
  NodeDef* filter_sizes = graph.add_node();
  name_to_node["filter_sizes"] = filter_sizes;
  CreateConstSizesOp(
      "filter_sizes",
      std::vector<int32>({filter_rows, filter_cols, in_depth, out_depth}),
      filter_sizes);

  TensorShape paddings_shape({4, 2});
  Tensor paddings_tensor(DT_INT32, paddings_shape);
  for (int64_t i = 0; i < paddings_tensor.NumElements(); ++i) {
    paddings_tensor.flat<int32>()(i) = 0;
  }
  TF_CHECK_OK(NodeDefBuilder("paddings", "Const")
                  .Attr("dtype", DT_INT32)
                  .Attr("value", paddings_tensor)
                  .Finalize(graph.add_node()));

  // Now add the convolution op
  NodeDef* conv = graph.add_node();
  TF_CHECK_OK(NodeDefBuilder("conv2d", "Conv2D")
                  .Input("input", 0, DT_FLOAT)
                  .Input("filter", 0, DT_FLOAT)
                  .Attr("strides", {1, stride, stride, 1})
                  .Attr("padding", "SAME")
                  .Finalize(conv));

  NodeDef* conv_bp_in = graph.add_node();
  TF_CHECK_OK(NodeDefBuilder("conv2d_bp_in", "Conv2DBackpropInput")
                  .Input("input_sizes", 0, DT_INT32)
                  .Input("filter", 0, DT_FLOAT)
                  .Input("output_backprop", 0, DT_FLOAT)
                  .Attr("strides", {1, stride, stride, 1})
                  .Attr("padding", "SAME")
                  .Finalize(conv_bp_in));

  NodeDef* conv_bp_filter = graph.add_node();
  TF_CHECK_OK(NodeDefBuilder("conv2d_bp_filter", "Conv2DBackpropFilter")
                  .Input("input", 0, DT_FLOAT)
                  .Input("filter_sizes", 0, DT_INT32)
                  .Input("output_backprop", 0, DT_FLOAT)
                  .Attr("strides", {1, stride, stride, 1})
                  .Attr("padding", "SAME")
                  .Finalize(conv_bp_filter));

  for (const auto& node : graph.node()) {
    if (node.name().find("conv2d") != 0) {
      continue;
    }
    std::vector<OpInfo::TensorProperties> inputs;
    inputs.resize(node.input_size());
    OpInfo info = BuildOpInfoWithoutDevice(node, name_to_node, inputs);
    if (node.name() == "conv2d") {
      EXPECT_EQ(2, info.inputs_size());
    } else if (node.name() == "conv2dbp_in") {
      EXPECT_EQ(3, info.inputs_size());
    } else if (node.name() == "conv2d_bp_filter") {
      EXPECT_EQ(3, info.inputs_size());
    }
  }
}

TEST(UtilsTest, TestSkipControlInput) {
  GraphDef graph;
  TF_CHECK_OK(NodeDefBuilder("constant", "Const")
                  .Attr("dtype", DT_INT32)
                  .Finalize(graph.add_node()));
  TF_CHECK_OK(NodeDefBuilder("constfold", "NoOp")
                  .ControlInput("constant")
                  .Finalize(graph.add_node()));

  std::unordered_map<string, const NodeDef*> name_to_node;
  for (const auto& node : graph.node()) {
    name_to_node[node.name()] = &node;
  }

  bool node_found = false;
  for (const auto& node : graph.node()) {
    if (node.name() == "constfold") {
      std::vector<OpInfo::TensorProperties> inputs;
      OpInfo info = BuildOpInfoWithoutDevice(node, name_to_node, inputs);
      node_found = true;
      EXPECT_EQ(0, info.inputs_size());
    }
  }
  EXPECT_TRUE(node_found);
}

TEST(UtilsTest, CalculateTensorSize) {
  // Test normal usage.
  EXPECT_EQ(DataTypeSize(DT_FLOAT) * 1,
            CalculateTensorSize(ShapeToTensorProperty({1}, DT_FLOAT)));
  EXPECT_EQ(DataTypeSize(DT_FLOAT) * 4 * 4,
            CalculateTensorSize(ShapeToTensorProperty({4, 4}, DT_FLOAT)));
  EXPECT_EQ(DataTypeSize(DT_HALF) * 10 * 10 * 10,
            CalculateTensorSize(ShapeToTensorProperty({10, 10, 10}, DT_HALF)));
  EXPECT_EQ(
      DataTypeSize(DT_FLOAT) * 100 * 7 * 8 * 99,
      CalculateTensorSize(ShapeToTensorProperty({100, 7, 8, 99}, DT_FLOAT)));

  // Test unknown rank: assumes the tensor to be a scalar.
  OpInfo::TensorProperties t = ShapeToTensorProperty({100, 7, 8, 99}, DT_FLOAT);
  t.mutable_shape()->set_unknown_rank(true);
  EXPECT_EQ(DataTypeSize(DT_FLOAT) * 1, CalculateTensorSize(t));

  // Test unknown shape: assumes unknown shape (-1) to have size 1.
  EXPECT_EQ(
      DataTypeSize(DT_FLOAT) * 1 * 7 * 8 * 99,
      CalculateTensorSize(ShapeToTensorProperty({-1, 7, 8, 99}, DT_FLOAT)));
  EXPECT_EQ(
      DataTypeSize(DT_FLOAT) * 1 * 7 * 1 * 99,
      CalculateTensorSize(ShapeToTensorProperty({-1, 7, -1, 99}, DT_FLOAT)));
}

TEST(UtilsTest, CalculateOutputSize) {
  // Create a set of tensor properties.
  std::vector<OpInfo::TensorProperties> output = {
      ShapeToTensorProperty({4, 4}, DT_FLOAT),          // 0
      ShapeToTensorProperty({-1, 7, -1, 99}, DT_FLOAT)  // 1
  };

  // Test valid outputs.
  EXPECT_EQ(DataTypeSize(DT_FLOAT) * 4 * 4, CalculateOutputSize(output, 0));
  EXPECT_EQ(DataTypeSize(DT_FLOAT) * 1 * 7 * 1 * 99,
            CalculateOutputSize(output, 1));

  // port_num -1 is for control dependency: hard coded 4B.
  EXPECT_EQ(4, CalculateOutputSize(output, -1));

  // Invalid port_num (though it may be an error) shall yield zero
  // output size.
  EXPECT_EQ(0, CalculateOutputSize(output, 2));
}

// Class for testing TensorSizeHistogram.
class TestTensorSizeHistogram : public TensorSizeHistogram {
 public:
  FRIEND_TEST(TensorSizeHistogramTest, Constructor);
  FRIEND_TEST(TensorSizeHistogramTest, Index);
  FRIEND_TEST(TensorSizeHistogramTest, Add);
  FRIEND_TEST(TensorSizeHistogramTest, Merge);
};

TEST(TensorSizeHistogramTest, Constructor) {
  TestTensorSizeHistogram hist;
  EXPECT_EQ(0, hist.NumElem());
  EXPECT_EQ(0, hist.SumElem());
  EXPECT_LT(1000000000, hist.Min());  // Initially, min_ is a very large value.
  EXPECT_EQ(0, hist.Max());
  EXPECT_EQ(0.0, hist.Average());
  const auto& buckets = hist.GetBuckets();
  for (const auto& bucket : buckets) {
    EXPECT_EQ(0, bucket);
  }
}

TEST(TensorSizeHistogramTest, Index) {
  TestTensorSizeHistogram hist;
  EXPECT_EQ(0, hist.Index(0));
  EXPECT_EQ(1, hist.Index(1));
  EXPECT_EQ(2, hist.Index(2));
  EXPECT_EQ(2, hist.Index(3));
  EXPECT_EQ(3, hist.Index(4));
  EXPECT_EQ(3, hist.Index(5));
  EXPECT_EQ(3, hist.Index(6));
  EXPECT_EQ(3, hist.Index(7));
  EXPECT_EQ(4, hist.Index(8));
  EXPECT_EQ(4, hist.Index(15));
  EXPECT_EQ(5, hist.Index(16));
  EXPECT_EQ(5, hist.Index(31));
  EXPECT_EQ(6, hist.Index(32));
  EXPECT_EQ(11, hist.Index(1025));
}

TEST(TensorSizeHistogramTest, Add) {
  TestTensorSizeHistogram hist;
  hist.Add(1037);
  hist.Add(1038);
  hist.Add(1039);

  const auto& buckets = hist.GetBuckets();
  EXPECT_EQ(3, hist.NumElem());
  EXPECT_EQ(1037 + 1038 + 1039, hist.SumElem());
  EXPECT_DOUBLE_EQ(1038.0, hist.Average());
  EXPECT_EQ(1037, hist.Min());
  EXPECT_EQ(1039, hist.Max());
  EXPECT_EQ(3, buckets.at(11));
}

TEST(TensorSizeHistogramTest, Merge) {
  TestTensorSizeHistogram hist1;
  const auto& buckets = hist1.GetBuckets();
  hist1.Add(1037);
  hist1.Add(1038);
  hist1.Add(1039);

  TestTensorSizeHistogram hist2(hist1);
  hist1.Merge(hist2);
  EXPECT_EQ(6, hist1.NumElem());
  EXPECT_EQ(2 * (1037 + 1038 + 1039), hist1.SumElem());
  EXPECT_DOUBLE_EQ(1038.0, hist1.Average());
  EXPECT_EQ(1037, hist1.Min());
  EXPECT_EQ(1039, hist1.Max());
  EXPECT_EQ(6, buckets.at(11));

  TestTensorSizeHistogram hist3;
  hist3.Add(1);
  hist3.Add(2);
  hist3.Add(4);

  hist1.Merge(hist3);
  EXPECT_EQ(9, hist1.NumElem());
  EXPECT_EQ(2 * (1037 + 1038 + 1039) + 1 + 2 + 4, hist1.SumElem());
  EXPECT_DOUBLE_EQ((2 * (1037 + 1038 + 1039) + 1 + 2 + 4) / 9.0,
                   hist1.Average());
  EXPECT_EQ(1, hist1.Min());
  EXPECT_EQ(1039, hist1.Max());
  EXPECT_EQ(1, buckets.at(1));
  EXPECT_EQ(1, buckets.at(2));
  EXPECT_EQ(1, buckets.at(3));
  EXPECT_EQ(6, buckets.at(11));
}

TEST(DeviceClassTest, GetDeviceClass) {
  EXPECT_EQ(
      "Channel: /ps/CPU -> /worker/GPU",
      GetDeviceClass("Channel_from_/job_ps/replica_0/task_0/device_CPU_0_to_"
                     "/job_worker/replica_7/task_0/device_GPU_7"));
  EXPECT_EQ(
      "Channel: /worker_train/CPU -> /ps/GPU",
      GetDeviceClass(
          "Channel_from_/job_worker_train/replica_0/task_0/device_CPU_0_to_"
          "/job_ps/replica_7/task_0/device_GPU_7"));
}

TEST(DeviceClassTest, GetDeviceClassForNonChannelDevice) {
  EXPECT_EQ("Unclassified",
            GetDeviceClassForNonChannelDevice("SOMETHING_WEIRD_DEVICE_NAME"));
  EXPECT_EQ("/worker/GPU", GetDeviceClassForNonChannelDevice(
                               "/job:worker/replica:0/task:0/device:GPU:0"));
  EXPECT_EQ("/worker/CPU", GetDeviceClassForNonChannelDevice(
                               "/job:worker/replica:0/task:0/device:CPU:0"));
  EXPECT_EQ("/worker_train/CPU", GetDeviceClassForNonChannelDevice(
                                     "/job:worker_train/replica:7/CPU:0"));
  EXPECT_EQ("//GPU", GetDeviceClassForNonChannelDevice("/device:GPU:7"));
}

}  // namespace

}  // end namespace grappler
}  // end namespace tensorflow
