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

#include "tensorflow/core/grappler/costs/op_level_cost_estimator.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {

namespace {
// Wrangles the minimum number of proto fields to set up a matrix.
void DescribeMatrix(int rows, int columns, OpInfo* op_features) {
  auto input = op_features->add_inputs();
  auto shape = input->mutable_shape();
  auto shape_rows = shape->add_dim();
  shape_rows->set_size(rows);
  auto shape_columns = shape->add_dim();
  shape_columns->set_size(columns);
  input->set_dtype(DT_FLOAT);
}

void SetCpuDevice(OpInfo* op_features) {
  auto device = op_features->mutable_device();
  device->set_type("CPU");
  device->set_num_cores(10);
  device->set_bandwidth(10000000);  // 10000000 KB/s = 10 GB/s
  device->set_frequency(1000);      // 1000 Mhz = 1 GHz
}

// Returns an OpInfo for MatMul with the minimum set of fields set up.
OpContext DescribeMatMul(int m, int n, int l, int k) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("MatMul");

  DescribeMatrix(m, l, &op_context.op_info);
  DescribeMatrix(k, n, &op_context.op_info);
  return op_context;
}

// Wrangles the minimum number of proto fields to set up an input of
// arbitrary rank and type.
void DescribeArbitraryRankInput(const std::vector<int>& dims, DataType dtype,
                                OpInfo* op_info) {
  auto input = op_info->add_inputs();
  input->set_dtype(dtype);
  auto shape = input->mutable_shape();
  for (auto d : dims) {
    shape->add_dim()->set_size(d);
  }
}

// Wrangles the minimum number of proto fields to set up an output of
// arbitrary rank and type.
void DescribeArbitraryRankOutput(const std::vector<int>& dims, DataType dtype,
                                 OpInfo* op_info) {
  auto output = op_info->add_outputs();
  output->set_dtype(dtype);
  auto shape = output->mutable_shape();
  for (auto d : dims) {
    shape->add_dim()->set_size(d);
  }
}

// Returns an OpInfo for a BatchMatMul
OpContext DescribeBatchMatMul(const std::vector<int>& dims_a,
                              const std::vector<int>& dims_b) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("BatchMatMul");

  DescribeArbitraryRankInput(dims_a, DT_FLOAT, &op_context.op_info);
  DescribeArbitraryRankInput(dims_b, DT_FLOAT, &op_context.op_info);
  return op_context;
}

// Wrangles the minimum number of proto fields to set up a 1D Tensor for cost
// estimation purposes.
void DescribeTensor1D(int dim0, OpInfo::TensorProperties* tensor) {
  auto shape = tensor->mutable_shape();
  shape->add_dim()->set_size(dim0);
  tensor->set_dtype(DT_FLOAT);
}

// Wrangles the minimum number of proto fields to set up a 4D Tensor for cost
// estimation purposes.
void DescribeTensor4D(int dim0, int dim1, int dim2, int dim3,
                      OpInfo::TensorProperties* tensor) {
  auto shape = tensor->mutable_shape();
  shape->add_dim()->set_size(dim0);
  shape->add_dim()->set_size(dim1);
  shape->add_dim()->set_size(dim2);
  shape->add_dim()->set_size(dim3);
  tensor->set_dtype(DT_FLOAT);
}

// DescribeConvolution constructs an OpContext for a Conv2D applied to an input
// tensor with shape (batch, ix, iy, iz1) and a kernel tensor with shape
// (kx, ky, iz2, oz).
OpContext DescribeConvolution(int batch, int ix, int iy, int iz1, int iz2,
                              int kx, int ky, int oz) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("Conv2D");

  DescribeTensor4D(batch, ix, iy, iz1, op_context.op_info.add_inputs());
  DescribeTensor4D(kx, ky, iz2, oz, op_context.op_info.add_inputs());

  return op_context;
}

// Describe DepthwiseConvolution constructs an OpContext for a
// DepthwiseConv2dNative applied to an input
// tensor with shape (batch, ix, iy, iz1) and a kernel tensor with shape
// (kx, ky, iz2, cm). cm is channel multiplier

OpContext DescribeDepthwiseConv2dNative(int batch, int ix, int iy, int iz1,
                                        int iz2, int kx, int ky, int cm) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("DepthwiseConv2dNative");

  DescribeTensor4D(batch, ix, iy, iz1, op_context.op_info.add_inputs());
  DescribeTensor4D(kx, ky, iz2, cm, op_context.op_info.add_inputs());

  return op_context;
}

// DescribeFusedConv2DBiasActivation constructs an OpContext for a
// FusedConv2DBiasActivation applied to a convolution input tensor with shape
// (batch, ix, iy, iz1), a kernel tensor with shape (kx, ky, iz2, oz), a
// bias tensor with shape (oz), a side input tensor with shape
// (batch, ox, oy, oz) if has_side_input is set, and two scaling tensors with
// shape (1).
//
// Note that this assumes the NHWC data format.
OpContext DescribeFusedConv2DBiasActivation(int batch, int ix, int iy, int iz1,
                                            int iz2, int kx, int ky, int ox,
                                            int oy, int oz,
                                            bool has_side_input) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("FusedConv2DBiasActivation");
  DescribeTensor4D(batch, ix, iy, iz1, op_context.op_info.add_inputs());
  DescribeTensor4D(kx, ky, iz2, oz, op_context.op_info.add_inputs());
  DescribeTensor1D(oz, op_context.op_info.add_inputs());

  // Add the side_input, if any.
  auto side_input = op_context.op_info.add_inputs();
  if (has_side_input) {
    DescribeTensor4D(batch, ox, oy, oz, side_input);
  }

  // Add the scaling tensors.
  DescribeTensor1D(1, op_context.op_info.add_inputs());
  DescribeTensor1D(1, op_context.op_info.add_inputs());

  return op_context;
}

// DescribeUnaryOp constructs an OpContext for the given operation applied to
// a 4-tensor with shape (size1, 1, 1, 1).
OpContext DescribeUnaryOp(const string& op, int size1) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op(op);

  DescribeTensor4D(size1, 1, 1, 1, op_context.op_info.add_inputs());
  DescribeTensor4D(size1, 1, 1, 1, op_context.op_info.add_outputs());

  return op_context;
}

// DescribeBinaryOp constructs an OpContext for the given operation applied to
// a 4-tensor with dimensions (size1, 1, 1, 1) and a 4-tensor with dimensions
// (2 * size1, size2, 1, 1).
//
// The choice of dimension here is arbitrary, and is used strictly to test the
// cost model for applying elementwise operations to tensors with unequal
// dimension values.
OpContext DescribeBinaryOp(const string& op, int size1, int size2) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op(op);

  DescribeTensor4D(size1, 1, 1, 1, op_context.op_info.add_inputs());
  DescribeTensor4D(2 * size1, size2, 1, 1, op_context.op_info.add_inputs());
  DescribeTensor4D(2 * size1, size2, 1, 1, op_context.op_info.add_outputs());

  return op_context;
}

// DescribeBiasAdd constructs an OpContext for a BiasAdd applied to a 4-tensor
// with dimensions (1, 1, size2, size1) and a bias with dimension (size1),
// according to the constraint that the bias must be 1D with size equal to that
// of the last dimension of the input value.
OpContext DescribeBiasAdd(int size1, int size2) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("BiasAdd");

  DescribeTensor4D(1, 1, size2, size1, op_context.op_info.add_inputs());
  DescribeTensor1D(size1, op_context.op_info.add_inputs());
  DescribeTensor4D(1, 1, size2, size1, op_context.op_info.add_outputs());

  return op_context;
}

int GetOutputSize(const int x, const int k, const int s,
                  const string& padding) {
  if (padding == "SAME") {
    return (x + s - 1) / s;
  } else {
    return (x - k + s) / s;
  }
}

std::vector<int> GetPoolingOutputSize(const std::vector<int>& input,
                                      const std::vector<int>& ksize,
                                      const std::vector<int>& strides,
                                      const string& data_format,
                                      const string& padding) {
  // h, w, and c indices: default with NHWC.
  int h_index = 1;
  int w_index = 2;
  int c_index = 3;
  if (data_format == "NCHW") {
    h_index = 2;
    w_index = 3;
    c_index = 1;
  }
  // Extract parameters.
  int n = input[0];
  int h = input[h_index];
  int w = input[w_index];
  int c = input[c_index];
  int sx = strides[h_index];
  int sy = strides[w_index];
  int kx = ksize[h_index];
  int ky = ksize[w_index];

  // Output activation size: default with VALID padding.
  int ho = GetOutputSize(h, kx, sx, padding);
  int wo = GetOutputSize(w, ky, sy, padding);

  std::vector<int> output;
  if (data_format == "NHWC") {
    output = {n, ho, wo, c};
  } else {
    output = {n, c, ho, wo};
  }
  return output;
}

// Helper functions for testing GetTensorShapeProtoFromTensorProto().
void GetTensorProto(const DataType dtype, const std::vector<int64>& shape,
                    const std::vector<int64> values, const bool tensor_content,
                    TensorProto* tensor_proto) {
  tensor_proto->Clear();
  TensorProto temp_tensor_proto;
  temp_tensor_proto.set_dtype(dtype);
  for (const auto& x : shape) {
    temp_tensor_proto.mutable_tensor_shape()->add_dim()->set_size(x);
  }
  for (const auto& x : values) {
    if (dtype == DT_INT64) {
      temp_tensor_proto.add_int64_val(x);
    } else if (dtype == DT_INT32 || dtype == DT_INT16 || dtype == DT_INT8 ||
               dtype == DT_UINT8) {
      temp_tensor_proto.add_int_val(x);
    } else if (dtype == DT_UINT32) {
      temp_tensor_proto.add_uint32_val(x);
    } else if (dtype == DT_UINT64) {
      temp_tensor_proto.add_uint64_val(x);
    } else {
      CHECK(false) << "Unsupported dtype: " << dtype;
    }
  }
  Tensor tensor(dtype);
  CHECK(tensor.FromProto(temp_tensor_proto));
  if (tensor_content) {
    tensor.AsProtoTensorContent(tensor_proto);
  } else {
    tensor.AsProtoField(tensor_proto);
  }
}

OpContext DescribePoolingOp(const string& op_name, const std::vector<int>& x,
                            const std::vector<int>& ksize,
                            const std::vector<int>& strides,
                            const string& data_format, const string& padding) {
  OpContext op_context;
  auto& op_info = op_context.op_info;
  SetCpuDevice(&op_info);
  op_info.set_op(op_name);

  const std::vector<int> y =
      GetPoolingOutputSize(x, ksize, strides, data_format, padding);
  if (op_name == "AvgPool" || op_name == "MaxPool") {
    // input: x, output: y.
    DescribeTensor4D(x[0], x[1], x[2], x[3], op_info.add_inputs());
    DescribeTensor4D(y[0], y[1], y[2], y[3], op_info.add_outputs());
  } else if (op_name == "AvgPoolGrad") {
    // input: x's shape, y_grad, output: x_grad.
    DescribeArbitraryRankInput({4}, DT_INT32, &op_info);
    auto* tensor_proto = op_info.mutable_inputs(0)->mutable_value();
    GetTensorProto(DT_INT32, {4}, {x[0], x[1], x[2], x[3]},
                   /*tensor_content=*/false, tensor_proto);
    DescribeTensor4D(y[0], y[1], y[2], y[3], op_info.add_inputs());
    DescribeTensor4D(x[0], x[1], x[2], x[3], op_info.add_outputs());
  } else if (op_name == "MaxPoolGrad") {
    // input: x, y, y_grad, output: x_grad.
    DescribeTensor4D(x[0], x[1], x[2], x[3], op_info.add_inputs());
    DescribeTensor4D(y[0], y[1], y[2], y[3], op_info.add_inputs());
    DescribeTensor4D(y[0], y[1], y[2], y[3], op_info.add_inputs());
    DescribeTensor4D(x[0], x[1], x[2], x[3], op_info.add_outputs());
  }
  auto* attr = op_info.mutable_attr();
  SetAttrValue(data_format, &(*attr)["data_format"]);
  SetAttrValue(padding, &(*attr)["padding"]);
  SetAttrValue(strides, &(*attr)["strides"]);
  SetAttrValue(ksize, &(*attr)["ksize"]);
  return op_context;
}

OpContext DescribeFusedBatchNorm(const bool is_training, const bool is_grad,
                                 const std::vector<int>& x,
                                 const string& data_format) {
  // First, get MaxPool op info with unit stride and unit window.
  OpContext op_context = DescribePoolingOp("MaxPool", x, {1, 1, 1, 1},
                                           {1, 1, 1, 1}, data_format, "SAME");
  auto& op_info = op_context.op_info;
  // Override op name.
  if (is_grad) {
    op_info.set_op("FusedBatchNormGrad");
  } else {
    op_info.set_op("FusedBatchNorm");
  }

  // Add additional input output tensors.
  if (is_grad) {
    DescribeTensor4D(x[0], x[1], x[2], x[3], op_info.add_inputs());
  }
  int num_1d_inputs = is_grad ? 3 : 4;
  for (int i = 0; i < num_1d_inputs; i++) {
    auto* tensor = op_info.add_inputs();
    auto* shape = tensor->mutable_shape();
    shape->add_dim()->set_size(x[3]);
    tensor->set_dtype(DT_FLOAT);
  }
  for (int i = 0; i < 4; i++) {
    auto* tensor = op_info.add_outputs();
    auto* shape = tensor->mutable_shape();
    shape->add_dim()->set_size(x[3]);
    tensor->set_dtype(DT_FLOAT);
  }

  // Delete unnecessary attr.
  auto* attr = op_context.op_info.mutable_attr();
  attr->erase("ksize");
  attr->erase("strides");
  attr->erase("padding");

  // Additional attrs for FusedBatchNorm.
  SetAttrValue(is_training, &(*attr)["is_training"]);

  return op_context;
}
}  // namespace

class OpLevelCostEstimatorTest : public ::testing::Test {
 protected:
  Costs PredictCosts(const OpContext& op_context) const {
    return estimator_.PredictCosts(op_context);
  }

  int64 CountMatMulOperations(const OpInfo& op_features,
                              bool* found_unknown_shapes) const {
    return estimator_.CountMatMulOperations(op_features, found_unknown_shapes);
  }

  int64 CountBatchMatMulOperations(const OpInfo& op_features,
                                   bool* found_unknown_shapes) const {
    return estimator_.CountBatchMatMulOperations(op_features,
                                                 found_unknown_shapes);
  }

  void SetComputeMemoryOverlap(bool value) {
    estimator_.compute_memory_overlap_ = value;
  }

  void ValidateOpDimensionsFromImputs(const int n, const int h, const int w,
                                      const int c, const int kx, const int ky,
                                      const int sx, const int sy,
                                      const string& data_format,
                                      const string& padding) {
    OpContext op_context;
    int ho;
    int wo;
    if (data_format == "NHWC") {
      op_context = DescribePoolingOp("MaxPool", {n, h, w, c}, {1, kx, ky, 1},
                                     {1, sx, sy, 1}, "NHWC", padding);
      ho = op_context.op_info.outputs(0).shape().dim(1).size();
      wo = op_context.op_info.outputs(0).shape().dim(2).size();
    } else {
      op_context = DescribePoolingOp("MaxPool", {n, c, h, w}, {1, 1, kx, ky},
                                     {1, 1, sx, sy}, "NCHW", padding);
      ho = op_context.op_info.outputs(0).shape().dim(2).size();
      wo = op_context.op_info.outputs(0).shape().dim(3).size();
    }

    bool found_unknown_shapes;
    auto dims = OpLevelCostEstimator::OpDimensionsFromInputs(
        op_context.op_info.inputs(0).shape(), op_context.op_info,
        &found_unknown_shapes);
    Padding padding_enum;
    if (padding == "VALID") {
      padding_enum = Padding::VALID;
    } else {
      padding_enum = Padding::SAME;
    }
    EXPECT_EQ(n, dims.batch);
    EXPECT_EQ(h, dims.ix);
    EXPECT_EQ(w, dims.iy);
    EXPECT_EQ(c, dims.iz);
    EXPECT_EQ(kx, dims.kx);
    EXPECT_EQ(ky, dims.ky);
    EXPECT_EQ(sx, dims.sx);
    EXPECT_EQ(sy, dims.sy);
    EXPECT_EQ(ho, dims.ox);
    EXPECT_EQ(wo, dims.oy);
    EXPECT_EQ(c, dims.oz);
    EXPECT_EQ(padding_enum, dims.padding);
  }

  OpLevelCostEstimator estimator_;
};

TEST_F(OpLevelCostEstimatorTest, TestGatherCosts) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("Gather");

  // Huge first input shouldn't affect Gather execution and memory costs.
  DescribeArbitraryRankInput({10000000, 10}, DT_FLOAT, &op_context.op_info);
  DescribeArbitraryRankInput({16}, DT_INT64, &op_context.op_info);
  DescribeArbitraryRankOutput({16, 10}, DT_FLOAT, &op_context.op_info);

  auto cost = estimator_.PredictCosts(op_context);
  EXPECT_EQ(Costs::Duration(130), cost.memory_time);
  EXPECT_EQ(Costs::Duration(16), cost.compute_time);
  EXPECT_EQ(Costs::Duration(146), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, TestGatherCostsWithoutOutput) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("Gather");

  // Huge first input shouldn't affect Gather execution and memory costs.
  DescribeArbitraryRankInput({10000000, 10}, DT_FLOAT, &op_context.op_info);
  DescribeArbitraryRankInput({16}, DT_INT64, &op_context.op_info);

  auto cost = estimator_.PredictCosts(op_context);
  EXPECT_EQ(Costs::Duration(0), cost.memory_time);
  EXPECT_EQ(Costs::Duration(0), cost.compute_time);
  EXPECT_EQ(Costs::Duration(0), cost.execution_time);
  EXPECT_TRUE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, TestSliceCosts) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("Slice");

  // Huge first input shouldn't affect Slice execution and memory costs.
  DescribeArbitraryRankInput({10000000, 10}, DT_FLOAT, &op_context.op_info);
  DescribeArbitraryRankInput({2}, DT_INT64, &op_context.op_info);
  DescribeArbitraryRankInput({2}, DT_INT64, &op_context.op_info);
  DescribeArbitraryRankOutput({10, 10}, DT_FLOAT, &op_context.op_info);

  auto cost = estimator_.PredictCosts(op_context);
  EXPECT_EQ(Costs::Duration(81), cost.memory_time);
  EXPECT_EQ(Costs::Duration(10), cost.compute_time);
  EXPECT_EQ(Costs::Duration(91), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, BiasAddExecutionTime) {
  auto cost = PredictCosts(DescribeBiasAdd(1000, 10));
  EXPECT_EQ(Costs::Duration(8400), cost.memory_time);
  EXPECT_EQ(Costs::Duration(1000), cost.compute_time);
  EXPECT_EQ(Costs::Duration(9400), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, Conv2DExecutionTime) {
  auto cost = PredictCosts(DescribeConvolution(16, 19, 19, 48, 48, 5, 5, 256));
  EXPECT_EQ(Costs::Duration(233780), cost.memory_time);
  EXPECT_EQ(Costs::Duration(354877440), cost.compute_time);
  EXPECT_EQ(Costs::Duration(355111220), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, DepthwiseConv2dNativeExecutionTime) {
  auto cost =
      PredictCosts(DescribeDepthwiseConv2dNative(16, 19, 19, 48, 48, 5, 5, 3));
  EXPECT_EQ(Costs::Duration(112340), cost.memory_time);
  EXPECT_EQ(Costs::Duration(4158720), cost.compute_time);
  EXPECT_EQ(Costs::Duration(4271060), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, DummyExecutionTime) {
  auto cost = PredictCosts(DescribeBinaryOp("Dummy", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(0), cost.compute_time);
  EXPECT_EQ(Costs::Duration(2000), cost.execution_time);
  EXPECT_TRUE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, ExecutionTimeSumOrMax) {
  SetComputeMemoryOverlap(true);
  auto cost = PredictCosts(DescribeBinaryOp("Dummy", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(0), cost.compute_time);
  EXPECT_EQ(Costs::Duration(2000), cost.execution_time);  // max(2000, 200)
  EXPECT_TRUE(cost.inaccurate);
  SetComputeMemoryOverlap(false);  // Set it back to default.
}

TEST_F(OpLevelCostEstimatorTest, FusedConv2DBiasActivationExecutionTime) {
  auto cost = PredictCosts(DescribeFusedConv2DBiasActivation(
      16, 19, 19, 48, 48, 5, 5, 19, 19, 256, /* has_side_input = */ true));
  EXPECT_EQ(Costs::Duration(1416808), cost.memory_time);
  EXPECT_EQ(Costs::Duration(355616770), cost.compute_time);
  EXPECT_EQ(Costs::Duration(357033578), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest,
       FusedConv2DBiasActivationNoSideInputExecutionTime) {
  auto cost = PredictCosts(DescribeFusedConv2DBiasActivation(
      16, 19, 19, 48, 48, 5, 5, 19, 19, 256, /* has_side_input = */ false));
  EXPECT_EQ(Costs::Duration(825345), cost.memory_time);
  EXPECT_EQ(Costs::Duration(355321038), cost.compute_time);
  EXPECT_EQ(Costs::Duration(356146383), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, MulExecutionTime) {
  auto cost = PredictCosts(DescribeBinaryOp("Mul", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(200), cost.compute_time);
  EXPECT_EQ(Costs::Duration(2200), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, MulBroadcastExecutionTime) {
  auto cost = PredictCosts(DescribeBinaryOp("Mul", 1000, 2));
  EXPECT_EQ(Costs::Duration(3600), cost.memory_time);
  EXPECT_EQ(Costs::Duration(400), cost.compute_time);
  EXPECT_EQ(Costs::Duration(4000), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, ModExecutionTime) {
  auto cost = PredictCosts(DescribeBinaryOp("Mod", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(1600), cost.compute_time);
  EXPECT_EQ(Costs::Duration(3600), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, ReluExecutionTime) {
  auto cost = PredictCosts(DescribeUnaryOp("Relu", 1000));
  EXPECT_EQ(Costs::Duration(800), cost.memory_time);
  EXPECT_EQ(Costs::Duration(100), cost.compute_time);
  EXPECT_EQ(Costs::Duration(900), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, UnknownOrPartialShape) {
  EXPECT_FALSE(PredictCosts(DescribeMatMul(2, 4, 7, 7)).inaccurate);
  EXPECT_TRUE(PredictCosts(DescribeMatMul(-1, 4, 7, 7)).inaccurate);
  EXPECT_TRUE(PredictCosts(DescribeMatMul(2, 4, -1, 7)).inaccurate);

  EXPECT_FALSE(PredictCosts(DescribeConvolution(16, 19, 19, 48, 48, 5, 5, 256))
                   .inaccurate);
  EXPECT_TRUE(PredictCosts(DescribeConvolution(16, -1, 19, 48, 48, 5, 5, 256))
                  .inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, BatchMatMul) {
  EXPECT_TRUE(PredictCosts(DescribeBatchMatMul({}, {})).inaccurate);
  EXPECT_TRUE(PredictCosts(DescribeBatchMatMul({2, 4}, {})).inaccurate);
  EXPECT_FALSE(PredictCosts(DescribeBatchMatMul({2, 4}, {4, 2})).inaccurate);
  EXPECT_FALSE(
      PredictCosts(DescribeBatchMatMul({1, 2, 4}, {1, 4, 2})).inaccurate);
  EXPECT_FALSE(
      PredictCosts(DescribeBatchMatMul({2, 4}, {1, 3, 4, 2})).inaccurate);
  bool matmul_inaccurate = false;
  bool batch_matmul_inaccurate = false;
  EXPECT_EQ(
      CountMatMulOperations(DescribeMatMul(2, 2, 4, 4).op_info,
                            &matmul_inaccurate),
      CountBatchMatMulOperations(DescribeBatchMatMul({2, 4}, {4, 2}).op_info,
                                 &batch_matmul_inaccurate));
  EXPECT_EQ(matmul_inaccurate, batch_matmul_inaccurate);
  EXPECT_EQ(10 * CountMatMulOperations(DescribeMatMul(2, 2, 4, 4).op_info,
                                       &matmul_inaccurate),
            CountBatchMatMulOperations(
                DescribeBatchMatMul({10, 2, 4}, {-1, 10, 4, 2}).op_info,
                &batch_matmul_inaccurate));
  EXPECT_NE(matmul_inaccurate, batch_matmul_inaccurate);
  EXPECT_EQ(20 * CountMatMulOperations(DescribeMatMul(2, 2, 4, 4).op_info,
                                       &matmul_inaccurate),
            CountBatchMatMulOperations(
                DescribeBatchMatMul({2, 10, 2, 4}, {-1, 10, 4, 2}).op_info,
                &batch_matmul_inaccurate));
  EXPECT_NE(matmul_inaccurate, batch_matmul_inaccurate);
}

void ExpectTensorShape(const std::vector<int64>& expected,
                       const TensorShapeProto& tensor_shape_proto) {
  TensorShape tensor_shape_expected(expected);
  TensorShape tensor_shape(tensor_shape_proto);

  LOG(INFO) << "Expected: " << tensor_shape_expected.DebugString();
  LOG(INFO) << "TensorShape: " << tensor_shape.DebugString();
  EXPECT_TRUE(tensor_shape_expected == tensor_shape);
}

TEST_F(OpLevelCostEstimatorTest, GetTensorShapeProtoFromTensorProto) {
  TensorProto tensor_proto;
  TensorShapeProto tensor_shape_proto;

  // Dimension larger than max value; should fail while converting to Tensor
  // class.
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(255);
  EXPECT_FALSE(
      GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));

  tensor_proto.Clear();
  // Expect only 1D shape.
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(1);
  tensor_proto.mutable_tensor_shape()->add_dim()->set_size(2);
  EXPECT_FALSE(
      GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));

  // Expect only handle integer data types.
  GetTensorProto(DT_FLOAT, {}, {}, /*tensor_content=*/false, &tensor_proto);
  EXPECT_FALSE(
      GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));

  // Check GetTensorShapeProtoFromTensorProto() resturns correct values.
  {
    std::vector<int64> shape_expected = {10, 20, 30, 40};
    GetTensorProto(DT_INT32, {4}, shape_expected, /*tensor_content=*/false,
                   &tensor_proto);
    EXPECT_TRUE(
        GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));
    ExpectTensorShape(shape_expected, tensor_shape_proto);
  }

  {
    std::vector<int64> shape_expected = {40, 20, 90, 40};
    GetTensorProto(DT_INT64, {4}, shape_expected, /*tensor_content=*/false,
                   &tensor_proto);
    EXPECT_TRUE(
        GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));
    ExpectTensorShape(shape_expected, tensor_shape_proto);
  }

  {
    std::vector<int64> shape_expected = {10, 20, 30, 40};
    GetTensorProto(DT_INT32, {4}, shape_expected, /*tensor_content=*/true,
                   &tensor_proto);
    EXPECT_TRUE(
        GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));
    ExpectTensorShape(shape_expected, tensor_shape_proto);
  }

  {
    std::vector<int64> shape_expected = {40, 20, 90, 40};
    GetTensorProto(DT_INT64, {4}, shape_expected, /*tensor_content=*/true,
                   &tensor_proto);
    EXPECT_TRUE(
        GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));
    ExpectTensorShape(shape_expected, tensor_shape_proto);
  }
}

TEST_F(OpLevelCostEstimatorTest, OpDimensionsFromInputs) {
  std::vector<string> paddings = {"VALID", "SAME"};
  std::vector<string> formats = {"NHWC", "NCHW"};
  for (const auto& p : paddings) {
    for (const auto& f : formats) {
      // n, h, w, c, kx, ky, sx, sy, data_format, padding.
      ValidateOpDimensionsFromImputs(10, 20, 20, 100, 3, 3, 2, 2, f, p);
      ValidateOpDimensionsFromImputs(10, 20, 20, 100, 1, 1, 3, 3, f, p);
      ValidateOpDimensionsFromImputs(10, 200, 200, 100, 5, 5, 3, 3, f, p);
      ValidateOpDimensionsFromImputs(10, 14, 14, 3840, 3, 3, 2, 2, f, p);
    }
  }
}

TEST_F(OpLevelCostEstimatorTest, PredictMaxPool) {
  auto predict_max_pool = [this](const int n, const int in, const int c,
                                 const int k, const int s,
                                 const string& padding) -> Costs {
    OpContext op_context = DescribePoolingOp(
        "MaxPool", {n, in, in, c}, {1, k, k, 1}, {1, s, s, 1}, "NHWC", padding);
    return estimator_.PredictCosts(op_context);
  };

  {
    // Typical 3xz3 window with 2x2 stride.
    auto costs = predict_max_pool(10, 20, 384, 3, 2, "SAME");
    EXPECT_EQ(Costs::Duration(1075200), costs.execution_time);
    EXPECT_EQ(Costs::Duration(307200), costs.compute_time);
    EXPECT_EQ(Costs::Duration(768000), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }
  {
    // 1x1 window with 2x2 stride: used for shortcut in resnet-50.
    auto costs = predict_max_pool(10, 20, 384, 1, 2, "SAME");
    EXPECT_EQ(Costs::Duration(499200), costs.execution_time);
    EXPECT_EQ(Costs::Duration(38400), costs.compute_time);
    EXPECT_EQ(Costs::Duration(460800), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }
  {
    // 2x2 window with 3x3 stride.
    auto costs = predict_max_pool(10, 20, 384, 2, 3, "VALID");
    EXPECT_EQ(Costs::Duration(561792), costs.execution_time);
    EXPECT_EQ(Costs::Duration(56448), costs.compute_time);
    EXPECT_EQ(Costs::Duration(505344), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }
}

TEST_F(OpLevelCostEstimatorTest, PredictMaxPoolGrad) {
  auto predict_max_pool_grad = [this](const int n, const int in, const int c,
                                      const int k, const int s,
                                      const string& padding) -> Costs {
    OpContext op_context =
        DescribePoolingOp("MaxPoolGrad", {n, in, in, c}, {1, k, k, 1},
                          {1, s, s, 1}, "NHWC", padding);
    return estimator_.PredictCosts(op_context);
  };

  {
    // Typical 3xz3 window with 2x2 stride.
    auto costs = predict_max_pool_grad(10, 20, 384, 3, 2, "SAME");
    EXPECT_EQ(Costs::Duration(1996800), costs.execution_time);
    EXPECT_EQ(Costs::Duration(614400), costs.compute_time);
    EXPECT_EQ(Costs::Duration(1382400), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }
  {
    // 1x1 window with 2x2 stride: used for shortcut in resnet-50.
    auto costs = predict_max_pool_grad(10, 20, 384, 1, 2, "SAME");
    EXPECT_EQ(Costs::Duration(1536000), costs.execution_time);
    EXPECT_EQ(Costs::Duration(153600), costs.compute_time);
    EXPECT_EQ(Costs::Duration(1382400), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }
  {
    // 2x2 window with 3x3 stride.
    auto costs = predict_max_pool_grad(10, 20, 384, 2, 3, "VALID");
    EXPECT_EQ(Costs::Duration(1514112), costs.execution_time);
    EXPECT_EQ(Costs::Duration(210048), costs.compute_time);
    EXPECT_EQ(Costs::Duration(1304064), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }
}

TEST_F(OpLevelCostEstimatorTest, PredictAvgPool) {
  auto predict_avg_pool = [this](const int n, const int in, const int c,
                                 const int k, const int s,
                                 const string& padding) -> Costs {
    OpContext op_context = DescribePoolingOp(
        "AvgPool", {n, in, in, c}, {1, k, k, 1}, {1, s, s, 1}, "NHWC", padding);
    return estimator_.PredictCosts(op_context);
  };

  {
    // Typical 3xz3 window with 2x2 stride.
    auto costs = predict_avg_pool(10, 20, 384, 3, 2, "SAME");
    EXPECT_EQ(Costs::Duration(1113600), costs.execution_time);
    EXPECT_EQ(Costs::Duration(345600), costs.compute_time);
    EXPECT_EQ(Costs::Duration(768000), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }
  {
    // 1x1 window with 2x2 stride: used for shortcut in resnet-50.
    auto costs = predict_avg_pool(10, 20, 384, 1, 2, "SAME");
    EXPECT_EQ(Costs::Duration(499200), costs.execution_time);
    EXPECT_EQ(Costs::Duration(38400), costs.compute_time);
    EXPECT_EQ(Costs::Duration(460800), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }
  {
    // 2x2 window with 3x3 stride.
    auto costs = predict_avg_pool(10, 20, 384, 2, 3, "VALID");
    EXPECT_EQ(Costs::Duration(580608), costs.execution_time);
    EXPECT_EQ(Costs::Duration(75264), costs.compute_time);
    EXPECT_EQ(Costs::Duration(505344), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }
}

TEST_F(OpLevelCostEstimatorTest, PredictAvgPoolGrad) {
  auto predict_avg_pool_grad = [this](const int n, const int in, const int c,
                                      const int k, const int s,
                                      const string& padding) -> Costs {
    OpContext op_context =
        DescribePoolingOp("AvgPoolGrad", {n, in, in, c}, {1, k, k, 1},
                          {1, s, s, 1}, "NHWC", padding);
    return estimator_.PredictCosts(op_context);
  };

  {
    // Typical 3xz3 window with 2x2 stride.
    auto costs = predict_avg_pool_grad(10, 20, 384, 3, 2, "SAME");
    EXPECT_EQ(Costs::Duration(1305602), costs.execution_time);
    EXPECT_EQ(Costs::Duration(537600), costs.compute_time);
    EXPECT_EQ(Costs::Duration(768002), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }
  {
    // 1x1 window with 2x2 stride: used for shortcut in resnet-50.
    auto costs = predict_avg_pool_grad(10, 20, 384, 1, 2, "SAME");
    EXPECT_EQ(Costs::Duration(960002), costs.execution_time);
    EXPECT_EQ(Costs::Duration(192000), costs.compute_time);
    EXPECT_EQ(Costs::Duration(768002), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }
  {
    // 2x2 window with 3x3 stride.
    auto costs = predict_avg_pool_grad(10, 20, 384, 2, 3, "VALID");
    EXPECT_EQ(Costs::Duration(862082), costs.execution_time);
    EXPECT_EQ(Costs::Duration(172416), costs.compute_time);
    EXPECT_EQ(Costs::Duration(689666), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }
}

TEST_F(OpLevelCostEstimatorTest, PredictFusedBatchNorm) {
  auto predict_fused_bn = [this](const int n, const int in, const int c,
                                 const bool is_training) -> Costs {
    OpContext op_context = DescribeFusedBatchNorm(
        is_training, /*is_grad=*/false, {n, in, in, c}, "NHWC");
    return estimator_.PredictCosts(op_context);
  };

  {
    auto costs = predict_fused_bn(10, 20, 96, /*is_training=*/true);
    EXPECT_EQ(Costs::Duration(614737), costs.execution_time);
    EXPECT_EQ(Costs::Duration(153706), costs.compute_time);
    EXPECT_EQ(Costs::Duration(461031), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }

  {
    auto costs = predict_fused_bn(10, 20, 32, /*is_training=*/true);
    EXPECT_EQ(Costs::Duration(204913), costs.execution_time);
    EXPECT_EQ(Costs::Duration(51236), costs.compute_time);
    EXPECT_EQ(Costs::Duration(153677), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }

  {
    auto costs = predict_fused_bn(10, 20, 96, /*is_training=*/false);
    EXPECT_EQ(Costs::Duration(384154), costs.execution_time);
    EXPECT_EQ(Costs::Duration(76800), costs.compute_time);
    EXPECT_EQ(Costs::Duration(307354), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }

  {
    auto costs = predict_fused_bn(10, 20, 32, /*is_training=*/false);
    EXPECT_EQ(Costs::Duration(128052), costs.execution_time);
    EXPECT_EQ(Costs::Duration(25600), costs.compute_time);
    EXPECT_EQ(Costs::Duration(102452), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }
}

TEST_F(OpLevelCostEstimatorTest, PredictFusedBatchNormGrad) {
  auto predict_fused_bn_grad = [this](const int n, const int in,
                                      const int c) -> Costs {
    OpContext op_context = DescribeFusedBatchNorm(
        /*is_training=*/false, /*is_grad=*/true, {n, in, in, c}, "NHWC");
    return estimator_.PredictCosts(op_context);
  };

  {
    auto costs = predict_fused_bn_grad(10, 20, 96);
    EXPECT_EQ(Costs::Duration(1037050), costs.execution_time);
    EXPECT_EQ(Costs::Duration(422496), costs.compute_time);
    EXPECT_EQ(Costs::Duration(614554), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }

  {
    auto costs = predict_fused_bn_grad(128, 7, 384);
    EXPECT_EQ(Costs::Duration(6503809), costs.execution_time);
    EXPECT_EQ(Costs::Duration(2649677), costs.compute_time);
    EXPECT_EQ(Costs::Duration(3854132), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
  }
}
}  // end namespace grappler
}  // end namespace tensorflow
