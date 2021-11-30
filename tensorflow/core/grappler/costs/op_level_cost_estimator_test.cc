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

#include <unordered_set>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {

namespace {

// TODO(dyoon): Consider to use this Test class for all the test cases, and then
// remove friend in the OpLevelCostEstimator class header.
class TestOpLevelCostEstimator : public OpLevelCostEstimator {
 public:
  TestOpLevelCostEstimator() {
    compute_memory_overlap_ = true;
    device_info_ = DeviceInfo();
  }
  ~TestOpLevelCostEstimator() override {}

  void SetDeviceInfo(const DeviceInfo& device_info) {
    device_info_ = device_info;
  }

  void SetComputeMemoryOverlap(bool value) { compute_memory_overlap_ = value; }

 protected:
  DeviceInfo GetDeviceInfo(const DeviceProperties& device) const override {
    return device_info_;
  }

  DeviceInfo device_info_;
};

void ExpectZeroCost(const Costs& cost) {
  EXPECT_TRUE(cost.inaccurate);
  EXPECT_EQ(cost.compute_time, Costs::Duration::zero());
  EXPECT_EQ(cost.execution_time, Costs::Duration::zero());
  EXPECT_EQ(cost.memory_time, Costs::Duration::zero());
}

// Wrangles the minimum number of proto fields to set up a matrix.
void DescribeMatrix(int rows, int columns, OpInfo* op_info) {
  auto input = op_info->add_inputs();
  auto shape = input->mutable_shape();
  auto shape_rows = shape->add_dim();
  shape_rows->set_size(rows);
  auto shape_columns = shape->add_dim();
  shape_columns->set_size(columns);
  input->set_dtype(DT_FLOAT);
}

void SetCpuDevice(OpInfo* op_info) {
  auto device = op_info->mutable_device();
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

// Returns an OpInfo for a SparseTensorDenseMatMul
OpContext DescribeSparseTensorDenseMatMul(const int nnz_a,
                                          const std::vector<int>& dims_b,
                                          const std::vector<int>& dims_out) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("SparseTensorDenseMatMul");

  DescribeArbitraryRankInput({nnz_a, 2}, DT_INT64, &op_context.op_info);
  DescribeArbitraryRankInput({nnz_a}, DT_FLOAT, &op_context.op_info);
  DescribeArbitraryRankInput({2}, DT_INT64, &op_context.op_info);
  DescribeArbitraryRankInput(dims_b, DT_FLOAT, &op_context.op_info);
  DescribeArbitraryRankOutput(dims_out, DT_FLOAT, &op_context.op_info);
  return op_context;
}

// Returns an OpInfo for an XlaEinsum
OpContext DescribeXlaEinsum(const std::vector<int>& dims_a,
                            const std::vector<int>& dims_b,
                            const string& equation) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("XlaEinsum");
  AttrValue equation_attribute;
  equation_attribute.set_s(equation);
  (*op_context.op_info.mutable_attr())["equation"] = equation_attribute;
  if (!dims_a.empty())
    DescribeArbitraryRankInput(dims_a, DT_FLOAT, &op_context.op_info);
  if (!dims_b.empty())
    DescribeArbitraryRankInput(dims_b, DT_FLOAT, &op_context.op_info);
  return op_context;
}

// Returns an OpInfo for an Einsum
OpContext DescribeEinsum(const std::vector<int>& dims_a,
                         const std::vector<int>& dims_b,
                         const string& equation) {
  OpContext op_context = DescribeXlaEinsum(dims_a, dims_b, equation);
  op_context.op_info.set_op("Einsum");
  return op_context;
}

void DescribeDummyTensor(OpInfo::TensorProperties* tensor) {
  // Intentionally leave the tensor shape and type information missing.
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

// Wrangles the minimum number of proto fields to set up a 4D Tensor for cost
// estimation purposes.
void DescribeTensor5D(int dim0, int dim1, int dim2, int dim3, int dim4,
                      OpInfo::TensorProperties* tensor) {
  auto shape = tensor->mutable_shape();
  shape->add_dim()->set_size(dim0);
  shape->add_dim()->set_size(dim1);
  shape->add_dim()->set_size(dim2);
  shape->add_dim()->set_size(dim3);
  shape->add_dim()->set_size(dim4);
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
// shape (1). If a vectorized channel format is chosen (NCHW_VECT_C, e.g.) we'll
// default to 4 (the vector size most often used with this format on NVIDIA
// platforms) for the major channel size, and divide the input channel size by
// that amount.
//
// Note that this assumes the NHWC data format.
OpContext DescribeFusedConv2DBiasActivation(int batch, int ix, int iy, int iz1,
                                            int iz2, int kx, int ky, int ox,
                                            int oy, int oz, bool has_side_input,
                                            const string& data_format,
                                            const string& filter_format) {
  const int kVecWidth = 4;
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("FusedConv2DBiasActivation");
  auto* attr_data_format = op_context.op_info.mutable_attr();
  SetAttrValue(data_format, &(*attr_data_format)["data_format"]);
  auto* attr_filter_format = op_context.op_info.mutable_attr();
  SetAttrValue(filter_format, &(*attr_filter_format)["filter_format"]);
  if (data_format == "NHWC") {
    DescribeTensor4D(batch, ix, iy, iz1, op_context.op_info.add_inputs());
  } else if (data_format == "NCHW") {
    DescribeTensor4D(batch, iz1, ix, iy, op_context.op_info.add_inputs());
  } else {
    // Use the NCHW_VECT_C format.
    EXPECT_EQ(data_format, "NCHW_VECT_C");
    EXPECT_EQ(iz1 % kVecWidth, 0);
    DescribeTensor5D(batch, iz1 / kVecWidth, ix, iy, kVecWidth,
                     op_context.op_info.add_inputs());
  }
  if (filter_format == "HWIO") {
    DescribeTensor4D(kx, ky, iz2, oz, op_context.op_info.add_inputs());
  } else if (filter_format == "OIHW") {
    DescribeTensor4D(oz, iz2, kx, ky, op_context.op_info.add_inputs());
  } else {
    EXPECT_EQ(filter_format, "OIHW_VECT_I");
    EXPECT_EQ(iz2 % kVecWidth, 0);
    // Use the OIHW_VECT_I format.
    DescribeTensor5D(oz, iz2 / kVecWidth, kx, ky, kVecWidth,
                     op_context.op_info.add_inputs());
  }
  DescribeTensor1D(oz, op_context.op_info.add_inputs());

  // Add the side_input, if any.
  auto side_input = op_context.op_info.add_inputs();
  if (has_side_input) {
    if (data_format == "NHWC") {
      DescribeTensor4D(batch, ox, oy, oz, side_input);
    } else if (data_format == "NCHW") {
      DescribeTensor4D(batch, oz, ox, oy, side_input);
    } else {
      // Use the NCHW_VECT_C format.
      EXPECT_EQ(data_format, "NCHW_VECT_C");
      EXPECT_EQ(oz % kVecWidth, 0);
      DescribeTensor5D(batch, oz / kVecWidth, ox, oy, kVecWidth, side_input);
    }
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
void GetTensorProto(const DataType dtype, const std::vector<int64_t>& shape,
                    const std::vector<int64_t> values,
                    const bool tensor_content, TensorProto* tensor_proto) {
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
  using BatchMatMulDimensions = OpLevelCostEstimator::BatchMatMulDimensions;

  Costs PredictCosts(const OpContext& op_context) const {
    return estimator_.PredictCosts(op_context);
  }

  int64_t CountMatMulOperations(const OpInfo& op_info,
                                bool* found_unknown_shapes) const {
    return estimator_.CountMatMulOperations(op_info, found_unknown_shapes);
  }

  int64_t CountBatchMatMulOperations(const OpInfo& op_info,
                                     bool* found_unknown_shapes) const {
    return estimator_.CountBatchMatMulOperations(op_info, found_unknown_shapes);
  }

  int64_t CountBatchMatMulOperations(const OpInfo& op_info,
                                     BatchMatMulDimensions* batch_mat_mul,
                                     bool* found_unknown_shapes) const {
    return estimator_.CountBatchMatMulOperations(op_info, batch_mat_mul,
                                                 found_unknown_shapes);
  }

  void SetComputeMemoryOverlap(bool value) {
    estimator_.compute_memory_overlap_ = value;
  }

  void ValidateOpDimensionsFromInputs(const int n, const int h, const int w,
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
    TF_ASSERT_OK_AND_ASSIGN(
        auto dims, OpLevelCostEstimator::OpDimensionsFromInputs(
                       op_context.op_info.inputs(0).shape(), op_context.op_info,
                       &found_unknown_shapes));
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

  StatusOr<OpLevelCostEstimator::ConvolutionDimensions>
  CallOpDimensionsFromInputs(const int n, const int h, const int w, const int c,
                             const int kx, const int ky, const int sx,
                             const int sy, const string& data_format,
                             const string& padding) {
    OpContext op_context;

    const std::vector<int> x = {n, h, w, c};
    const std::vector<int> ksize = {1, kx, ky, 1};
    std::vector<int> strides;
    if (data_format == "NHWC") {
      strides = {1, sy, sx, 1};
    } else {
      strides = {1, 1, sy, sx};
    }

    auto& op_info = op_context.op_info;
    SetCpuDevice(&op_info);
    op_info.set_op("MaxPool");

    DescribeTensor4D(x[0], x[1], x[2], x[3], op_info.add_inputs());
    auto* attr = op_info.mutable_attr();
    SetAttrValue(data_format, &(*attr)["data_format"]);
    SetAttrValue(padding, &(*attr)["padding"]);
    SetAttrValue(strides, &(*attr)["strides"]);
    SetAttrValue(ksize, &(*attr)["ksize"]);
    bool found_unknown_shapes;
    return OpLevelCostEstimator::OpDimensionsFromInputs(
        op_context.op_info.inputs(0).shape(), op_context.op_info,
        &found_unknown_shapes);
  }

  OpLevelCostEstimator estimator_;
};

class OpLevelBatchMatMulCostEstimatorTest
    : public OpLevelCostEstimatorTest,
      public ::testing::WithParamInterface<const char*> {
 protected:
  // Returns an OpInfo for a BatchMatMul
  OpContext DescribeBatchMatMul(const std::vector<int>& dims_a,
                                const std::vector<int>& dims_b) {
    OpContext op_context;
    SetCpuDevice(&op_context.op_info);
    op_context.op_info.set_op(GetParam());

    DescribeArbitraryRankInput(dims_a, DT_FLOAT, &op_context.op_info);
    DescribeArbitraryRankInput(dims_b, DT_FLOAT, &op_context.op_info);
    return op_context;
  }

  int64_t CountBatchMatMulOperations(const OpInfo& op_info,
                                     bool* found_unknown_shapes) const {
    return OpLevelCostEstimatorTest::CountBatchMatMulOperations(
        op_info, found_unknown_shapes);
  }

  int64_t CountBatchMatMulDimProduct(const OpInfo& op_info,
                                     bool* found_unknown_shapes) const {
    BatchMatMulDimensions batch_mat_mul;

    batch_mat_mul.matmul_dims.n = 0;
    batch_mat_mul.matmul_dims.m = 0;
    batch_mat_mul.matmul_dims.k = 0;

    OpLevelCostEstimatorTest::CountBatchMatMulOperations(
        op_info, &batch_mat_mul, found_unknown_shapes);
    int dimension_product = 1;
    for (auto dim : batch_mat_mul.batch_dims) dimension_product *= dim;

    dimension_product *= batch_mat_mul.matmul_dims.n;
    dimension_product *= batch_mat_mul.matmul_dims.m;
    dimension_product *= batch_mat_mul.matmul_dims.k;

    return dimension_product;
  }
};

TEST_F(OpLevelCostEstimatorTest, TestPersistentOpCosts) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  std::unordered_set<string> persistent_ops = {
      "Const",       "Variable",       "VariableV2", "AutoReloadVariable",
      "VarHandleOp", "ReadVariableOp",
  };
  // Minimum cost for all persistent ops.
  for (const auto& op : persistent_ops) {
    op_context.op_info.set_op(op);
    auto cost = estimator_.PredictCosts(op_context);
    EXPECT_EQ(Costs::Duration(0), cost.memory_time);
    EXPECT_EQ(Costs::Duration(1), cost.compute_time);
    EXPECT_EQ(Costs::Duration(1), cost.execution_time);
    EXPECT_EQ(cost.num_ops_total, 1);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
    EXPECT_EQ(cost.temporary_memory, 0);
    EXPECT_EQ(cost.persistent_memory, 0);
  }
}

TEST_F(OpLevelCostEstimatorTest, TestGatherCosts) {
  std::vector<std::string> gather_ops = {"Gather", "GatherNd", "GatherV2"};

  for (const auto& op : gather_ops) {
    OpContext op_context;
    SetCpuDevice(&op_context.op_info);
    op_context.op_info.set_op(op);

    // Huge first input shouldn't affect Gather execution and memory costs.
    DescribeArbitraryRankInput({10000000, 10}, DT_FLOAT, &op_context.op_info);
    DescribeArbitraryRankInput({16}, DT_INT64, &op_context.op_info);
    DescribeArbitraryRankOutput({16, 10}, DT_FLOAT, &op_context.op_info);

    auto cost = estimator_.PredictCosts(op_context);
    EXPECT_EQ(Costs::Duration(130), cost.memory_time);
    EXPECT_EQ(Costs::Duration(16), cost.compute_time);
    EXPECT_EQ(Costs::Duration(146), cost.execution_time);
    EXPECT_EQ(cost.num_ops_total, 1);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
    EXPECT_EQ(cost.temporary_memory, 0);
    EXPECT_EQ(cost.persistent_memory, 0);
  }
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
  EXPECT_EQ(1, cost.num_ops_total);
  EXPECT_TRUE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
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
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, TestStridedSliceCosts) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("StridedSlice");

  // Huge first input shouldn't affect StridedSlice execution and memory costs.
  DescribeArbitraryRankInput({10000000, 10}, DT_FLOAT, &op_context.op_info);
  DescribeArbitraryRankInput({2}, DT_INT64, &op_context.op_info);
  DescribeArbitraryRankInput({2}, DT_INT64, &op_context.op_info);
  DescribeArbitraryRankInput({2}, DT_INT64, &op_context.op_info);
  DescribeArbitraryRankOutput({10, 10}, DT_FLOAT, &op_context.op_info);

  auto cost = estimator_.PredictCosts(op_context);
  EXPECT_EQ(Costs::Duration(81), cost.memory_time);
  EXPECT_EQ(Costs::Duration(10), cost.compute_time);
  EXPECT_EQ(Costs::Duration(91), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, TestScatterOps) {
  std::vector<string> scatter_ops = {"ScatterAdd",   "ScatterDiv", "ScatterMax",
                                     "ScatterMin",   "ScatterMul", "ScatterSub",
                                     "ScatterUpdate"};
  for (const auto& op : scatter_ops) {
    // Test updates.shape = indices.shape + ref.shape[1:]
    {
      OpContext op_context;
      SetCpuDevice(&op_context.op_info);
      op_context.op_info.set_op(op);
      // Huge first dimension in input shouldn't affect Scatter execution and
      // memory costs.
      DescribeArbitraryRankInput({10000000, 10}, DT_FLOAT, &op_context.op_info);
      DescribeArbitraryRankInput({16}, DT_INT64, &op_context.op_info);
      DescribeArbitraryRankInput({16, 10}, DT_FLOAT, &op_context.op_info);
      DescribeArbitraryRankOutput({10000000, 10}, DT_FLOAT,
                                  &op_context.op_info);

      auto cost = estimator_.PredictCosts(op_context);
      EXPECT_EQ(Costs::Duration(205), cost.memory_time);
      EXPECT_EQ(Costs::Duration(16), cost.compute_time);
      EXPECT_EQ(Costs::Duration(221), cost.execution_time);
      EXPECT_EQ(cost.num_ops_total, 1);
      EXPECT_FALSE(cost.inaccurate);
      EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
      EXPECT_EQ(cost.temporary_memory, 0);
      EXPECT_EQ(cost.persistent_memory, 0);
    }

    // Test updates.shape = [] and INT32 indices
    {
      OpContext op_context;
      SetCpuDevice(&op_context.op_info);
      op_context.op_info.set_op(op);
      // Huge first dimension in input shouldn't affect Scatter execution and
      // memory costs.
      DescribeArbitraryRankInput({10000000, 10}, DT_FLOAT, &op_context.op_info);
      DescribeArbitraryRankInput({16}, DT_INT32, &op_context.op_info);
      DescribeArbitraryRankInput({}, DT_FLOAT, &op_context.op_info);
      DescribeArbitraryRankOutput({10000000, 10}, DT_FLOAT,
                                  &op_context.op_info);

      auto cost = estimator_.PredictCosts(op_context);
      EXPECT_EQ(Costs::Duration(135), cost.memory_time);
      EXPECT_EQ(Costs::Duration(16), cost.compute_time);
      EXPECT_EQ(Costs::Duration(151), cost.execution_time);
      EXPECT_EQ(1, cost.num_ops_total);
      EXPECT_FALSE(cost.inaccurate);
      EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);
    }
  }
}

TEST_F(OpLevelCostEstimatorTest, BiasAddExecutionTime) {
  auto cost = PredictCosts(DescribeBiasAdd(1000, 10));
  EXPECT_EQ(Costs::Duration(8400), cost.memory_time);
  EXPECT_EQ(Costs::Duration(1000), cost.compute_time);
  EXPECT_EQ(Costs::Duration(9400), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, Conv2DExecutionTime) {
  auto cost = PredictCosts(DescribeConvolution(16, 19, 19, 48, 48, 5, 5, 256));
  EXPECT_EQ(Costs::Duration(233780), cost.memory_time);
  EXPECT_EQ(Costs::Duration(354877440), cost.compute_time);
  EXPECT_EQ(Costs::Duration(355111220), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, InvalidConv2DConfig) {
  // Convolution ops.
  const std::vector<std::string> conv_ops = {
      "Conv2D",
      "Conv2DBackpropFilter",
      "Conv2DBackpropInput",
      "DepthwiseConv2dNative",
      "DepthwiseConv2dNativeBackpropFilter",
      "DepthwiseConv2dNativeBackpropInput",
  };
  // A valid Conv2D config.
  const std::vector<int> valid_conv_config = {16, 19, 19, 48, 48, 5, 5, 256};
  for (const auto& op : conv_ops) {
    // Test with setting one value in conv config to zero.
    // PredictCosts() should return zero costs.
    for (int i = 0; i < valid_conv_config.size(); ++i) {
      std::vector<int> conv_config(valid_conv_config);
      conv_config[i] = 0;
      auto op_context = DescribeConvolution(
          conv_config[0], conv_config[1], conv_config[2], conv_config[3],
          conv_config[4], conv_config[5], conv_config[6], conv_config[7]);
      op_context.op_info.set_op(op);
      auto cost = PredictCosts(op_context);
      EXPECT_EQ(Costs::Duration(0), cost.memory_time);
      EXPECT_EQ(Costs::Duration(0), cost.compute_time);
      EXPECT_EQ(Costs::Duration(0), cost.execution_time);
      EXPECT_EQ(1, cost.num_ops_total);
      EXPECT_TRUE(cost.inaccurate);
      EXPECT_EQ(1, cost.num_ops_with_unknown_shapes);
    }
  }
}

TEST_F(OpLevelCostEstimatorTest, DepthwiseConv2dNativeExecutionTime) {
  auto cost =
      PredictCosts(DescribeDepthwiseConv2dNative(16, 19, 19, 48, 48, 5, 5, 3));
  EXPECT_EQ(Costs::Duration(112340), cost.memory_time);
  EXPECT_EQ(Costs::Duration(4158720), cost.compute_time);
  EXPECT_EQ(Costs::Duration(4271060), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, DummyExecutionTime) {
  auto cost = PredictCosts(DescribeBinaryOp("Dummy", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(0), cost.compute_time);
  EXPECT_EQ(Costs::Duration(2000), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_TRUE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, ExecutionTimeSumOrMax) {
  SetComputeMemoryOverlap(true);
  auto cost = PredictCosts(DescribeBinaryOp("Dummy", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(0), cost.compute_time);
  EXPECT_EQ(Costs::Duration(2000), cost.execution_time);  // max(2000, 200)
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_TRUE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
  SetComputeMemoryOverlap(false);  // Set it back to default.
}

TEST_F(OpLevelCostEstimatorTest,
       FusedConv2DBiasActivationNCHW_HWIO_NoSideInput) {
  auto cost = PredictCosts(DescribeFusedConv2DBiasActivation(
      16, 19, 19, 48, 48, 5, 5, 19, 19, 256, /* has_side_input = */ false,
      "NCHW", "HWIO"));
  EXPECT_EQ(Costs::Duration(825345), cost.memory_time);
  EXPECT_EQ(Costs::Duration(355321037), cost.compute_time);
  EXPECT_EQ(Costs::Duration(356146382), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, FusedConv2DBiasActivationNCHW_HWIO) {
  auto cost = PredictCosts(DescribeFusedConv2DBiasActivation(
      16, 19, 19, 48, 48, 5, 5, 19, 19, 256, /* has_side_input = */ true,
      "NCHW", "HWIO"));
  EXPECT_EQ(Costs::Duration(1416808), cost.memory_time);
  EXPECT_EQ(Costs::Duration(355616768), cost.compute_time);
  EXPECT_EQ(Costs::Duration(357033576), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, FusedConv2DBiasActivationNCHW_OIHW) {
  auto cost = PredictCosts(DescribeFusedConv2DBiasActivation(
      16, 19, 19, 48, 48, 5, 5, 19, 19, 256, /* has_side_input = */ true,
      "NCHW", "OIHW"));
  EXPECT_EQ(Costs::Duration(1416808), cost.memory_time);
  EXPECT_EQ(Costs::Duration(355616768), cost.compute_time);
  EXPECT_EQ(Costs::Duration(357033576), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, FusedConv2DBiasActivationNHWC_HWIO) {
  auto cost = PredictCosts(DescribeFusedConv2DBiasActivation(
      16, 19, 19, 48, 48, 5, 5, 19, 19, 256, /* has_side_input = */ true,
      "NHWC", "HWIO"));
  EXPECT_EQ(Costs::Duration(1416808), cost.memory_time);
  EXPECT_EQ(Costs::Duration(355616768), cost.compute_time);
  EXPECT_EQ(Costs::Duration(357033576), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, FusedConv2DBiasActivationNHWC_OIHW) {
  auto cost = PredictCosts(DescribeFusedConv2DBiasActivation(
      16, 19, 19, 48, 48, 5, 5, 19, 19, 256, /* has_side_input = */ true,
      "NHWC", "OIHW"));
  EXPECT_EQ(Costs::Duration(1416808), cost.memory_time);
  EXPECT_EQ(Costs::Duration(355616768), cost.compute_time);
  EXPECT_EQ(Costs::Duration(357033576), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, FusedConv2DBiasActivationNCHW_VECT_C_OIHW) {
  auto cost = PredictCosts(DescribeFusedConv2DBiasActivation(
      16, 19, 19, 48, 48, 5, 5, 19, 19, 256, /* has_side_input = */ true,
      "NCHW_VECT_C", "OIHW"));
  EXPECT_EQ(Costs::Duration(1416808), cost.memory_time);
  EXPECT_EQ(Costs::Duration(355616768), cost.compute_time);
  EXPECT_EQ(Costs::Duration(357033576), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, FusedConv2DBiasActivationNCHW_OIHW_VECT_I) {
  auto cost = PredictCosts(DescribeFusedConv2DBiasActivation(
      16, 19, 19, 48, 48, 5, 5, 19, 19, 256, /* has_side_input = */ true,
      "NCHW", "OIHW_VECT_I"));
  EXPECT_EQ(Costs::Duration(1416808), cost.memory_time);
  EXPECT_EQ(Costs::Duration(355616768), cost.compute_time);
  EXPECT_EQ(Costs::Duration(357033576), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest,
       FusedConv2DBiasActivationNCHW_VECT_C_OIHW_VECT_I) {
  auto cost = PredictCosts(DescribeFusedConv2DBiasActivation(
      16, 19, 19, 48, 48, 5, 5, 19, 19, 256, /* has_side_input = */ true,
      "NCHW_VECT_C", "OIHW_VECT_I"));
  EXPECT_EQ(Costs::Duration(1416808), cost.memory_time);
  EXPECT_EQ(Costs::Duration(355616768), cost.compute_time);
  EXPECT_EQ(Costs::Duration(357033576), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, MulExecutionTime) {
  auto cost = PredictCosts(DescribeBinaryOp("Mul", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(200), cost.compute_time);
  EXPECT_EQ(Costs::Duration(2200), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, MulBroadcastExecutionTime) {
  auto cost = PredictCosts(DescribeBinaryOp("Mul", 1000, 2));
  EXPECT_EQ(Costs::Duration(3600), cost.memory_time);
  EXPECT_EQ(Costs::Duration(400), cost.compute_time);
  EXPECT_EQ(Costs::Duration(4000), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, ModExecutionTime) {
  auto cost = PredictCosts(DescribeBinaryOp("Mod", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(1600), cost.compute_time);
  EXPECT_EQ(Costs::Duration(3600), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, SquaredDifferenceExecutionTime) {
  auto cost = PredictCosts(DescribeBinaryOp("SquaredDifference", 1000, 2));
  EXPECT_EQ(cost.memory_time, Costs::Duration(3600));
  EXPECT_EQ(cost.compute_time, Costs::Duration(800));
  EXPECT_EQ(cost.execution_time, Costs::Duration(4400));
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, UnaryOpExecutionTime) {
  std::vector<std::pair<std::string, int>> unary_ops = {
      {"All", 1},      {"ArgMax", 1}, {"Cast", 1},  {"Max", 1},
      {"Min", 1},      {"Prod", 1},   {"Relu", 1},  {"Relu6", 1},
      {"Softmax", 43}, {"Sum", 1},    {"TopKV2", 1}};

  const int kTensorSize = 1000;
  for (auto unary_op : unary_ops) {
    OpContext op_context = DescribeUnaryOp(unary_op.first, kTensorSize);

    const int kExpectedMemoryTime = 800;
    int expected_compute_time = std::ceil(
        unary_op.second * kTensorSize /
        estimator_.GetDeviceInfo(op_context.op_info.device()).gigaops);

    auto cost = PredictCosts(op_context);
    EXPECT_EQ(cost.memory_time, Costs::Duration(kExpectedMemoryTime));
    EXPECT_EQ(cost.compute_time, Costs::Duration(expected_compute_time))
        << unary_op.first;
    EXPECT_EQ(cost.execution_time,
              Costs::Duration(expected_compute_time + kExpectedMemoryTime));
    EXPECT_EQ(cost.num_ops_total, 1);
    EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(cost.temporary_memory, 0);
    EXPECT_EQ(cost.persistent_memory, 0);
  }
}

TEST_F(OpLevelCostEstimatorTest, BinaryOpExecutionTime) {
  std::vector<std::pair<std::string, int>> binary_ops = {
      {"Select", 1},
      {"SelectV2", 1},
      {"SquaredDifference", 2},
      {"Where", 1},
  };

  const int kTensorSize1 = 1000;
  const int kTensorSize2 = 2;
  for (auto binary_op : binary_ops) {
    OpContext op_context =
        DescribeBinaryOp(binary_op.first, kTensorSize1, kTensorSize2);

    const int kExpectedMemoryTime = 3600;
    int expected_compute_time = std::ceil(
        binary_op.second * kTensorSize1 * kTensorSize2 * 2 /
        estimator_.GetDeviceInfo(op_context.op_info.device()).gigaops);

    auto cost = PredictCosts(op_context);
    EXPECT_EQ(Costs::Duration(kExpectedMemoryTime), cost.memory_time)
        << binary_op.first;
    EXPECT_EQ(Costs::Duration(expected_compute_time), cost.compute_time)
        << binary_op.first;
    EXPECT_EQ(Costs::Duration(expected_compute_time + kExpectedMemoryTime),
              cost.execution_time)
        << binary_op.first;
    EXPECT_EQ(cost.num_ops_total, 1);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
    EXPECT_EQ(cost.temporary_memory, 0);
    EXPECT_EQ(cost.persistent_memory, 0);
  }
}

TEST_F(OpLevelCostEstimatorTest, BroadcastAddExecutionTime) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("Add");

  DescribeTensor1D(100, op_context.op_info.add_inputs());
  DescribeTensor4D(1, 10, 1, 1, op_context.op_info.add_inputs());

  auto cost = PredictCosts(op_context);
  EXPECT_EQ(Costs::Duration(44), cost.memory_time);
  EXPECT_EQ(Costs::Duration(100), cost.compute_time);
  EXPECT_EQ(Costs::Duration(144), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, UnknownOrPartialShape) {
  {
    auto cost = PredictCosts(DescribeMatMul(2, 4, 7, 7));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);
  }
  {
    auto cost = PredictCosts(DescribeMatMul(-1, 4, 7, 7));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(1, cost.num_ops_with_unknown_shapes);
  }
  {
    auto cost = PredictCosts(DescribeMatMul(2, 4, -1, 7));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(1, cost.num_ops_with_unknown_shapes);
  }
  {
    auto cost =
        PredictCosts(DescribeConvolution(16, 19, 19, 48, 48, 5, 5, 256));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);
  }
  {
    auto cost =
        PredictCosts(DescribeConvolution(16, -1, 19, 48, 48, 5, 5, 256));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(1, cost.num_ops_with_unknown_shapes);
  }
}

TEST_P(OpLevelBatchMatMulCostEstimatorTest, TestBatchMatMul) {
  {
    auto cost = PredictCosts(DescribeBatchMatMul({}, {}));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(1, cost.num_ops_with_unknown_shapes);
  }
  {
    auto cost = PredictCosts(DescribeBatchMatMul({2, 4}, {}));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(1, cost.num_ops_with_unknown_shapes);
  }
  {
    auto cost = PredictCosts(DescribeBatchMatMul({2, 4}, {4, 2}));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);
  }
  {
    auto cost = PredictCosts(DescribeBatchMatMul({1, 2, 4}, {1, 4, 2}));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);
  }
  {
    auto cost = PredictCosts(DescribeBatchMatMul({2, 4}, {1, 3, 4, 2}));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);
  }
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

  // Test the count to make sure that they extracted the dimensions correctly
  int prod = CountBatchMatMulDimProduct(
      DescribeBatchMatMul({2, 4}, {1, 3, 4, 2}).op_info,
      &batch_matmul_inaccurate);
  EXPECT_EQ(prod, 16);
  EXPECT_FALSE(batch_matmul_inaccurate);

  // Exercise the bad cases of a batchMatMul.
  OpContext bad_batch = DescribeBatchMatMul({2, 4}, {4, 2});
  bad_batch.op_info.set_op("notBatchMatMul");
  prod =
      CountBatchMatMulDimProduct(bad_batch.op_info, &batch_matmul_inaccurate);

  EXPECT_EQ(prod, 0);
  EXPECT_TRUE(batch_matmul_inaccurate);

  // Exercise a transpose case of a batchMatMul
  OpContext transpose_batch = DescribeBatchMatMul({2, 4, 3, 1}, {4, 2});
  auto attr = transpose_batch.op_info.mutable_attr();
  (*attr)["adj_x"].set_b(true);
  (*attr)["adj_y"].set_b(true);

  prod = CountBatchMatMulDimProduct(transpose_batch.op_info,
                                    &batch_matmul_inaccurate);
  EXPECT_EQ(prod, 12);
}
INSTANTIATE_TEST_SUITE_P(TestBatchMatMul, OpLevelBatchMatMulCostEstimatorTest,
                         ::testing::Values("BatchMatMul", "BatchMatMulV2"));

TEST_F(OpLevelCostEstimatorTest, SparseTensorDenseMatMul) {
  // Unknown shape cases
  {
    auto cost =
        PredictCosts(DescribeSparseTensorDenseMatMul(-1, {1, 1}, {1, 1}));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(1, cost.num_ops_with_unknown_shapes);
  }
  {
    auto cost =
        PredictCosts(DescribeSparseTensorDenseMatMul(1, {-1, 1}, {1, 1}));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(1, cost.num_ops_with_unknown_shapes);
  }
  {
    auto cost =
        PredictCosts(DescribeSparseTensorDenseMatMul(1, {1, -1}, {1, -1}));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(1, cost.num_ops_with_unknown_shapes);
  }
  {
    auto cost =
        PredictCosts(DescribeSparseTensorDenseMatMul(1, {1, 1}, {-1, 1}));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(1, cost.num_ops_with_unknown_shapes);
  }
  // Known shape cases
  {
    auto cost = PredictCosts(
        DescribeSparseTensorDenseMatMul(10, {1000, 100}, {50, 100}));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);
    EXPECT_EQ(Costs::Duration(200), cost.compute_time);
    EXPECT_EQ(Costs::Duration(2422), cost.memory_time);
  }
  {
    // Same cost as above case because cost does not depend on k_dim
    auto cost = PredictCosts(
        DescribeSparseTensorDenseMatMul(10, {100000, 100}, {50, 100}));
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);
    EXPECT_EQ(Costs::Duration(200), cost.compute_time);
    EXPECT_EQ(Costs::Duration(2422), cost.memory_time);
  }
}

void ExpectTensorShape(const std::vector<int64_t>& expected,
                       const TensorShapeProto& tensor_shape_proto) {
  TensorShape tensor_shape_expected(expected);
  TensorShape tensor_shape(tensor_shape_proto);

  EXPECT_EQ(tensor_shape_expected, tensor_shape);
}

TEST_F(OpLevelCostEstimatorTest, GetTensorShapeProtoFromTensorProto) {
  TensorProto tensor_proto;
  TensorShapeProto tensor_shape_proto;

  // Dimension larger than max value; should fail while converting to
  // Tensor class.
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

  // Check GetTensorShapeProtoFromTensorProto() returns correct values.
  {
    std::vector<int64_t> shape_expected = {10, 20, 30, 40};
    GetTensorProto(DT_INT32, {4}, shape_expected,
                   /*tensor_content=*/false, &tensor_proto);
    EXPECT_TRUE(
        GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));
    ExpectTensorShape(shape_expected, tensor_shape_proto);
  }

  {
    std::vector<int64_t> shape_expected = {40, 20, 90, 40};
    GetTensorProto(DT_INT64, {4}, shape_expected,
                   /*tensor_content=*/false, &tensor_proto);
    EXPECT_TRUE(
        GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));
    ExpectTensorShape(shape_expected, tensor_shape_proto);
  }

  {
    std::vector<int64_t> shape_expected = {10, 20, 30, 40};
    GetTensorProto(DT_INT32, {4}, shape_expected,
                   /*tensor_content=*/true, &tensor_proto);
    EXPECT_TRUE(
        GetTensorShapeProtoFromTensorProto(tensor_proto, &tensor_shape_proto));
    ExpectTensorShape(shape_expected, tensor_shape_proto);
  }

  {
    std::vector<int64_t> shape_expected = {40, 20, 90, 40};
    GetTensorProto(DT_INT64, {4}, shape_expected,
                   /*tensor_content=*/true, &tensor_proto);
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
      ValidateOpDimensionsFromInputs(10, 20, 20, 100, 3, 3, 2, 2, f, p);
      ValidateOpDimensionsFromInputs(10, 20, 20, 100, 1, 1, 3, 3, f, p);
      ValidateOpDimensionsFromInputs(10, 200, 200, 100, 5, 5, 3, 3, f, p);
      ValidateOpDimensionsFromInputs(10, 14, 14, 3840, 3, 3, 2, 2, f, p);
    }
  }
}

TEST_F(OpLevelCostEstimatorTest, OpDimensionsFromInputsError) {
  std::vector<string> paddings = {"VALID", "SAME"};
  std::vector<string> formats = {"NHWC", "NCHW"};
  for (const auto& p : paddings) {
    for (const auto& f : formats) {
      // n, h, w, c, kx, ky, sx, sy, data_format, padding.
      ASSERT_THAT(
          CallOpDimensionsFromInputs(10, 14, 14, 3840, 3, 3, 0, 2, f, p),
          testing::StatusIs(
              error::INVALID_ARGUMENT,
              "Stride must be > 0 for Height and Width, but got (2, 0)"));
      ASSERT_THAT(
          CallOpDimensionsFromInputs(10, 14, 14, 3840, 3, 3, 2, 0, f, p),
          testing::StatusIs(
              error::INVALID_ARGUMENT,
              "Stride must be > 0 for Height and Width, but got (0, 2)"));
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
    EXPECT_EQ(costs.num_ops_total, 1);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(costs.num_ops_with_unknown_shapes, 0);
    EXPECT_EQ(costs.temporary_memory, 0);
    EXPECT_EQ(costs.persistent_memory, 0);
  }
  {
    // 1x1 window with 2x2 stride: used for shortcut in resnet-50.
    auto costs = predict_max_pool(10, 20, 384, 1, 2, "SAME");
    EXPECT_EQ(Costs::Duration(499200), costs.execution_time);
    EXPECT_EQ(Costs::Duration(38400), costs.compute_time);
    EXPECT_EQ(Costs::Duration(460800), costs.memory_time);
    EXPECT_EQ(1, costs.num_ops_total);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(0, costs.num_ops_with_unknown_shapes);
  }
  {
    // 2x2 window with 3x3 stride.
    auto costs = predict_max_pool(10, 20, 384, 2, 3, "VALID");
    EXPECT_EQ(Costs::Duration(561792), costs.execution_time);
    EXPECT_EQ(Costs::Duration(56448), costs.compute_time);
    EXPECT_EQ(Costs::Duration(505344), costs.memory_time);
    EXPECT_EQ(1, costs.num_ops_total);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(0, costs.num_ops_with_unknown_shapes);
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
    // Typical 3x3 window with 2x2 stride.
    auto costs = predict_max_pool_grad(10, 20, 384, 3, 2, "SAME");
    EXPECT_EQ(Costs::Duration(1996800), costs.execution_time);
    EXPECT_EQ(Costs::Duration(614400), costs.compute_time);
    EXPECT_EQ(Costs::Duration(1382400), costs.memory_time);
    EXPECT_EQ(costs.num_ops_total, 1);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(costs.num_ops_with_unknown_shapes, 0);
    EXPECT_EQ(costs.temporary_memory, 0);
    EXPECT_EQ(costs.persistent_memory, 0);
  }
  {
    // 1x1 window with 2x2 stride: used for shortcut in resnet-50.
    auto costs = predict_max_pool_grad(10, 20, 384, 1, 2, "SAME");
    EXPECT_EQ(Costs::Duration(1536000), costs.execution_time);
    EXPECT_EQ(Costs::Duration(153600), costs.compute_time);
    EXPECT_EQ(Costs::Duration(1382400), costs.memory_time);
    EXPECT_EQ(1, costs.num_ops_total);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(0, costs.num_ops_with_unknown_shapes);
  }
  {
    // 2x2 window with 3x3 stride.
    auto costs = predict_max_pool_grad(10, 20, 384, 2, 3, "VALID");
    EXPECT_EQ(Costs::Duration(1514112), costs.execution_time);
    EXPECT_EQ(Costs::Duration(210048), costs.compute_time);
    EXPECT_EQ(Costs::Duration(1304064), costs.memory_time);
    EXPECT_EQ(1, costs.num_ops_total);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(0, costs.num_ops_with_unknown_shapes);
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
    // Typical 3x3 window with 2x2 stride.
    auto costs = predict_avg_pool(10, 20, 384, 3, 2, "SAME");
    EXPECT_EQ(Costs::Duration(1113600), costs.execution_time);
    EXPECT_EQ(Costs::Duration(345600), costs.compute_time);
    EXPECT_EQ(Costs::Duration(768000), costs.memory_time);
    EXPECT_EQ(costs.num_ops_total, 1);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(costs.num_ops_with_unknown_shapes, 0);
    EXPECT_EQ(costs.temporary_memory, 0);
    EXPECT_EQ(costs.persistent_memory, 0);
  }
  {
    // 1x1 window with 2x2 stride: used for shortcut in resnet-50.
    auto costs = predict_avg_pool(10, 20, 384, 1, 2, "SAME");
    EXPECT_EQ(Costs::Duration(499200), costs.execution_time);
    EXPECT_EQ(Costs::Duration(38400), costs.compute_time);
    EXPECT_EQ(Costs::Duration(460800), costs.memory_time);
    EXPECT_EQ(1, costs.num_ops_total);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(0, costs.num_ops_with_unknown_shapes);
  }
  {
    // 2x2 window with 3x3 stride.
    auto costs = predict_avg_pool(10, 20, 384, 2, 3, "VALID");
    EXPECT_EQ(Costs::Duration(580608), costs.execution_time);
    EXPECT_EQ(Costs::Duration(75264), costs.compute_time);
    EXPECT_EQ(Costs::Duration(505344), costs.memory_time);
    EXPECT_EQ(1, costs.num_ops_total);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(0, costs.num_ops_with_unknown_shapes);
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
    EXPECT_EQ(costs.num_ops_total, 1);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(costs.num_ops_with_unknown_shapes, 0);
    EXPECT_EQ(costs.temporary_memory, 0);
    EXPECT_EQ(costs.persistent_memory, 0);
  }
  {
    // 1x1 window with 2x2 stride: used for shortcut in resnet-50.
    auto costs = predict_avg_pool_grad(10, 20, 384, 1, 2, "SAME");
    EXPECT_EQ(Costs::Duration(960002), costs.execution_time);
    EXPECT_EQ(Costs::Duration(192000), costs.compute_time);
    EXPECT_EQ(Costs::Duration(768002), costs.memory_time);
    EXPECT_EQ(1, costs.num_ops_total);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(0, costs.num_ops_with_unknown_shapes);
  }
  {
    // 2x2 window with 3x3 stride.
    auto costs = predict_avg_pool_grad(10, 20, 384, 2, 3, "VALID");
    EXPECT_EQ(Costs::Duration(862082), costs.execution_time);
    EXPECT_EQ(Costs::Duration(172416), costs.compute_time);
    EXPECT_EQ(Costs::Duration(689666), costs.memory_time);
    EXPECT_EQ(1, costs.num_ops_total);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(0, costs.num_ops_with_unknown_shapes);
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
    EXPECT_EQ(costs.num_ops_total, 1);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(costs.num_ops_with_unknown_shapes, 0);
    EXPECT_EQ(costs.temporary_memory, 0);
    EXPECT_EQ(costs.persistent_memory, 0);
  }

  {
    auto costs = predict_fused_bn(10, 20, 32, /*is_training=*/true);
    EXPECT_EQ(Costs::Duration(204913), costs.execution_time);
    EXPECT_EQ(Costs::Duration(51236), costs.compute_time);
    EXPECT_EQ(Costs::Duration(153677), costs.memory_time);
    EXPECT_EQ(1, costs.num_ops_total);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(0, costs.num_ops_with_unknown_shapes);
  }

  {
    auto costs = predict_fused_bn(10, 20, 96, /*is_training=*/false);
    EXPECT_EQ(Costs::Duration(384154), costs.execution_time);
    EXPECT_EQ(Costs::Duration(76800), costs.compute_time);
    EXPECT_EQ(Costs::Duration(307354), costs.memory_time);
    EXPECT_EQ(1, costs.num_ops_total);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(0, costs.num_ops_with_unknown_shapes);
  }

  {
    auto costs = predict_fused_bn(10, 20, 32, /*is_training=*/false);
    EXPECT_EQ(Costs::Duration(128052), costs.execution_time);
    EXPECT_EQ(Costs::Duration(25600), costs.compute_time);
    EXPECT_EQ(Costs::Duration(102452), costs.memory_time);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(1, costs.num_ops_total);
    EXPECT_EQ(0, costs.num_ops_with_unknown_shapes);
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
    EXPECT_EQ(costs.num_ops_total, 1);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(costs.num_ops_with_unknown_shapes, 0);
    EXPECT_EQ(costs.temporary_memory, 0);
    EXPECT_EQ(costs.persistent_memory, 0);
  }

  {
    auto costs = predict_fused_bn_grad(128, 7, 384);
    EXPECT_EQ(Costs::Duration(6503809), costs.execution_time);
    EXPECT_EQ(Costs::Duration(2649677), costs.compute_time);
    EXPECT_EQ(Costs::Duration(3854132), costs.memory_time);
    EXPECT_EQ(1, costs.num_ops_total);
    EXPECT_FALSE(costs.inaccurate);
    EXPECT_EQ(0, costs.num_ops_with_unknown_shapes);
  }
}

TEST_F(OpLevelCostEstimatorTest, MaybeGetMinimumShape) {
  {
    TensorShapeProto x;
    x.set_unknown_rank(true);
    bool unknown_shapes = false;
    TensorShapeProto y = MaybeGetMinimumShape(x, 4, &unknown_shapes);
    EXPECT_TRUE(unknown_shapes);
    ExpectTensorShape({1, 1, 1, 1}, y);
  }

  {
    TensorShapeProto x;
    x.set_unknown_rank(false);
    bool unknown_shapes = false;
    TensorShapeProto y = MaybeGetMinimumShape(x, 1, &unknown_shapes);
    EXPECT_FALSE(unknown_shapes);
    ExpectTensorShape({1}, y);
  }

  {
    TensorShapeProto x;
    x.set_unknown_rank(false);
    bool unknown_shapes = false;
    TensorShapeProto y = MaybeGetMinimumShape(x, 2, &unknown_shapes);
    EXPECT_FALSE(unknown_shapes);
    ExpectTensorShape({1, 1}, y);
  }

  {
    TensorShapeProto x;
    x.set_unknown_rank(false);
    x.add_dim()->set_size(10);
    x.add_dim()->set_size(20);
    bool unknown_shapes = false;
    TensorShapeProto y = MaybeGetMinimumShape(x, 2, &unknown_shapes);
    EXPECT_FALSE(unknown_shapes);
    ExpectTensorShape({10, 20}, y);

    unknown_shapes = false;
    TensorShapeProto z = MaybeGetMinimumShape(x, 4, &unknown_shapes);
    EXPECT_TRUE(unknown_shapes);
    EXPECT_EQ(4, z.dim_size());
    ExpectTensorShape({10, 20, 1, 1}, z);
  }

  {
    TensorShapeProto x;
    x.set_unknown_rank(false);
    x.add_dim()->set_size(10);
    x.add_dim()->set_size(20);
    x.add_dim()->set_size(-1);
    x.add_dim()->set_size(20);
    bool unknown_shapes = false;
    TensorShapeProto y = MaybeGetMinimumShape(x, 4, &unknown_shapes);
    EXPECT_TRUE(unknown_shapes);
    ExpectTensorShape({10, 20, 1, 20}, y);
  }

  {
    TensorShapeProto x;
    x.set_unknown_rank(false);
    x.add_dim()->set_size(10);
    x.add_dim()->set_size(20);
    x.add_dim()->set_size(30);
    x.add_dim()->set_size(20);
    bool unknown_shapes = false;
    TensorShapeProto y = MaybeGetMinimumShape(x, 2, &unknown_shapes);
    EXPECT_TRUE(unknown_shapes);
    ExpectTensorShape({10, 20}, y);
  }
}

TEST_F(OpLevelCostEstimatorTest, IntermediateRdWrBandwidth) {
  TestOpLevelCostEstimator estimator;

  // Compute limited.
  estimator.SetDeviceInfo(DeviceInfo(/*gigaops=*/1,
                                     /*gb_per_sec=*/1));
  estimator.SetComputeMemoryOverlap(true);
  auto cost = estimator.PredictCosts(
      DescribeConvolution(16, 19, 19, 48, 48, 5, 5, 256));
  EXPECT_EQ(Costs::Duration(3548774400), cost.execution_time);
  EXPECT_EQ(cost.execution_time, cost.compute_time);

  estimator.SetComputeMemoryOverlap(false);
  cost = estimator.PredictCosts(
      DescribeConvolution(16, 19, 19, 48, 48, 5, 5, 256));
  EXPECT_EQ(Costs::Duration(3551112192), cost.execution_time);
  EXPECT_EQ(cost.execution_time, cost.compute_time + cost.memory_time +
                                     cost.intermediate_memory_time);

  // Memory limited.
  estimator.SetDeviceInfo(DeviceInfo(/*gigaops=*/99999,
                                     /*gb_per_sec=*/1));
  estimator.SetComputeMemoryOverlap(true);
  cost = estimator.PredictCosts(
      DescribeConvolution(16, 19, 19, 48, 48, 5, 5, 256));
  EXPECT_EQ(Costs::Duration(2337792), cost.execution_time);
  EXPECT_EQ(cost.execution_time, cost.memory_time);

  estimator.SetComputeMemoryOverlap(false);
  cost = estimator.PredictCosts(
      DescribeConvolution(16, 19, 19, 48, 48, 5, 5, 256));
  EXPECT_EQ(Costs::Duration(2373281), cost.execution_time);
  EXPECT_EQ(cost.execution_time, cost.compute_time + cost.memory_time +
                                     cost.intermediate_memory_time);

  // Intermediate memory bandwidth limited.
  estimator.SetDeviceInfo(DeviceInfo(/*gigaops=*/99999,
                                     /*gb_per_sec=*/9999,
                                     /*intermediate_read_gb_per_sec=*/1,
                                     /*intermediate_write_gb_per_sec=*/1));
  estimator.SetComputeMemoryOverlap(true);
  cost = estimator.PredictCosts(
      DescribeConvolution(16, 19, 19, 48, 48, 5, 5, 256));
  EXPECT_EQ(Costs::Duration(2337792), cost.execution_time);
  EXPECT_EQ(cost.execution_time, cost.intermediate_memory_time);

  estimator.SetComputeMemoryOverlap(false);
  cost = estimator.PredictCosts(
      DescribeConvolution(16, 19, 19, 48, 48, 5, 5, 256));
  EXPECT_EQ(Costs::Duration(2373515), cost.execution_time);
  EXPECT_EQ(cost.execution_time, cost.compute_time + cost.memory_time +
                                     cost.intermediate_memory_time);
}

TEST_F(OpLevelCostEstimatorTest, Einsum) {
  {  // Test a simple matrix multiplication.
    auto cost = PredictCosts(DescribeEinsum({100, 50}, {100, 50}, "ik,jk->ij"));
    EXPECT_EQ(Costs::Duration(104000), cost.execution_time);
    EXPECT_EQ(Costs::Duration(100 * 50 * 100 * 2 / (1000 * 10 * 1e-3)),
              cost.compute_time);
    EXPECT_EQ(Costs::Duration(4000), cost.memory_time);
    EXPECT_EQ(cost.num_ops_total, 1);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
    EXPECT_EQ(cost.temporary_memory, 0);
    EXPECT_EQ(cost.persistent_memory, 0);

    // Einsums and XlaEinsums should be estimated similarly.
    EXPECT_EQ(PredictCosts(DescribeEinsum({100, 50}, {100, 50}, "ik,jk->ij"))
                  .execution_time,
              PredictCosts(DescribeXlaEinsum({100, 50}, {100, 50}, "ik,jk->ij"))
                  .execution_time);
  }
  {  // Test a simple batch matrix multiplication.
    auto cost = PredictCosts(
        DescribeEinsum({25, 100, 50}, {100, 50, 25}, "Bik,jkB->Bij"));
    EXPECT_EQ(Costs::Duration(25 * 104000), cost.execution_time);
    EXPECT_EQ(Costs::Duration(25 * 100 * 50 * 100 * 2 / (1000 * 10 * 1e-3)),
              cost.compute_time);
    EXPECT_EQ(Costs::Duration(25 * 4000), cost.memory_time);
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);

    // Einsums and XlaEinsums should be estimated similarly.
    EXPECT_EQ(PredictCosts(
                  DescribeEinsum({25, 100, 50}, {100, 50, 25}, "Bik,jkB->Bij"))
                  .execution_time,
              PredictCosts(DescribeXlaEinsum({25, 100, 50}, {100, 50, 25},
                                             "Bik,jkB->Bij"))
                  .execution_time);
  }
  {  // Test multiple batch dimensions.
    auto cost = PredictCosts(DescribeEinsum(
        {25, 16, 100, 50}, {16, 100, 50, 25}, "BNik,NjkB->BNij"));
    EXPECT_EQ(Costs::Duration(16 * 25 * 104000), cost.execution_time);
    EXPECT_EQ(
        Costs::Duration(16 * 25 * 100 * 50 * 100 * 2 / (1000 * 10 * 1e-3)),
        cost.compute_time);
    EXPECT_EQ(Costs::Duration(16 * 25 * 4000), cost.memory_time);
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);

    // Einsums and XlaEinsums should be estimated similarly.
    EXPECT_EQ(
        PredictCosts(DescribeEinsum({25, 16, 100, 50}, {16, 100, 50, 25},
                                    "BNik,NjkB->BNij"))
            .execution_time,
        PredictCosts(DescribeXlaEinsum({25, 16, 100, 50}, {16, 100, 50, 25},
                                       "BNik,NjkB->BNij"))
            .execution_time);
  }
  {  // Test multiple M dimensions.
    auto cost =
        PredictCosts(DescribeEinsum({25, 100, 50}, {100, 50}, "Aik,jk->Aij"));
    EXPECT_EQ(Costs::Duration(2552000), cost.execution_time);
    EXPECT_EQ(Costs::Duration(25 * 100 * 50 * 100 * 2 / (1000 * 10 * 1e-3)),
              cost.compute_time);
    EXPECT_EQ(Costs::Duration(52000), cost.memory_time);
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);

    // Einsums and XlaEinsums should be estimated similarly.
    EXPECT_EQ(
        PredictCosts(DescribeEinsum({25, 100, 50}, {100, 50}, "Aik,jk->Aij"))
            .execution_time,
        PredictCosts(DescribeXlaEinsum({25, 100, 50}, {100, 50}, "Aik,jk->Aij"))
            .execution_time);
  }
  {  // Test multiple N dimensions.
    auto cost =
        PredictCosts(DescribeEinsum({100, 50}, {25, 100, 50}, "ik,Bjk->ijB"));
    EXPECT_EQ(Costs::Duration(2552000), cost.execution_time);
    EXPECT_EQ(Costs::Duration(25 * 100 * 50 * 100 * 2 / (1000 * 10 * 1e-3)),
              cost.compute_time);
    EXPECT_EQ(Costs::Duration(52000), cost.memory_time);
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);

    // Einsums and XlaEinsums should be estimated similarly.
    EXPECT_EQ(
        PredictCosts(DescribeEinsum({100, 50}, {25, 100, 50}, "ik,Bjk->ijB"))
            .execution_time,
        PredictCosts(DescribeXlaEinsum({100, 50}, {25, 100, 50}, "ik,Bjk->ijB"))
            .execution_time);
  }
  {  // Test multiple contracting dimensions.
    auto cost = PredictCosts(
        DescribeEinsum({100, 50, 25}, {100, 50, 25}, "ikl,jkl->ij"));
    EXPECT_EQ(Costs::Duration(2600000), cost.execution_time);
    EXPECT_EQ(Costs::Duration(100 * 50 * 25 * 100 * 2 / (1000 * 10 * 1e-3)),
              cost.compute_time);
    EXPECT_EQ(Costs::Duration(100000), cost.memory_time);
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);

    // Einsums and XlaEinsums should be estimated similarly.
    EXPECT_EQ(PredictCosts(
                  DescribeEinsum({100, 50, 25}, {100, 50, 25}, "ikl,jkl->ij"))
                  .execution_time,
              PredictCosts(DescribeXlaEinsum({100, 50, 25}, {100, 50, 25},
                                             "ikl,jkl->ij"))
                  .execution_time);
  }
  {  // Test a simple matrix transpose.
    auto cost = PredictCosts(DescribeEinsum({100, 50}, {}, "ij->ji"));
    EXPECT_EQ(Costs::Duration(2000), cost.execution_time);
    EXPECT_EQ(Costs::Duration(0), cost.compute_time);
    EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);

    // Einsums and XlaEinsums should be estimated similarly.
    EXPECT_EQ(
        PredictCosts(DescribeEinsum({100, 50}, {}, "ij->ji")).execution_time,
        PredictCosts(DescribeXlaEinsum({100, 50}, {}, "ij->ji"))
            .execution_time);
  }
  {  // Test a malformed Einsum equation: Mismatch between shapes and equation.
    auto cost =
        PredictCosts(DescribeEinsum({100, 50, 25}, {50, 100}, "ik,kl->il"));
    EXPECT_EQ(Costs::Duration(52000), cost.execution_time);
    EXPECT_EQ(Costs::Duration(0), cost.compute_time);
    EXPECT_EQ(Costs::Duration(52000), cost.memory_time);
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);

    // Einsums and XlaEinsums should be estimated similarly.
    EXPECT_EQ(
        PredictCosts(DescribeEinsum({100, 50, 25}, {50, 100}, "ik,kl->il"))
            .execution_time,
        PredictCosts(DescribeXlaEinsum({100, 50, 25}, {50, 100}, "ik,kl->il"))
            .execution_time);

    cost = PredictCosts(DescribeEinsum({100, 50}, {50, 100, 25}, "ik,kl->il"));
    EXPECT_EQ(Costs::Duration(52000), cost.execution_time);
    EXPECT_EQ(Costs::Duration(0), cost.compute_time);
    EXPECT_EQ(Costs::Duration(52000), cost.memory_time);
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);

    // Einsums and XlaEinsums should be estimated similarly.
    EXPECT_EQ(
        PredictCosts(DescribeEinsum({100, 50}, {50, 100, 25}, "ik,kl->il"))
            .execution_time,
        PredictCosts(DescribeXlaEinsum({100, 50}, {50, 100, 25}, "ik,kl->il"))
            .execution_time);
  }
  {  // Test an unsupported Einsum: ellipsis
    auto cost = PredictCosts(DescribeEinsum(
        {100, 50, 25, 16}, {50, 100, 32, 12}, "ik...,kl...->il..."));
    EXPECT_EQ(Costs::Duration(1568000), cost.execution_time);
    EXPECT_EQ(Costs::Duration(0), cost.compute_time);
    EXPECT_EQ(Costs::Duration(1568000), cost.memory_time);
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);

    // Einsums and XlaEinsums should be estimated similarly.
    EXPECT_EQ(
        PredictCosts(DescribeEinsum({100, 50, 25, 16}, {50, 100, 32, 12},
                                    "ik...,kl...->il..."))
            .execution_time,
        PredictCosts(DescribeXlaEinsum({100, 50, 25, 16}, {50, 100, 32, 12},
                                       "ik...,kl...->il..."))
            .execution_time);
  }
  {  // Test a malformed/unsupported Einsum: repeated indices
    auto cost =
        PredictCosts(DescribeEinsum({100, 100, 50}, {50, 100}, "iik,kl->il"));
    EXPECT_EQ(Costs::Duration(202000), cost.execution_time);
    EXPECT_EQ(Costs::Duration(0), cost.compute_time);
    EXPECT_EQ(Costs::Duration(202000), cost.memory_time);
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(0, cost.num_ops_with_unknown_shapes);

    // Einsums and XlaEinsums should be estimated similarly.
    EXPECT_EQ(
        PredictCosts(DescribeEinsum({100, 100, 50}, {50, 100}, "iik,kl->il"))
            .execution_time,
        PredictCosts(DescribeXlaEinsum({100, 100, 50}, {50, 100}, "iik,kl->il"))
            .execution_time);
  }
  {  // Test missing shapes.
    auto cost = PredictCosts(DescribeEinsum({-1, 50}, {100, 50}, "ik,jk->ij"));
    EXPECT_EQ(Costs::Duration(3020), cost.execution_time);
    EXPECT_EQ(Costs::Duration(1 * 50 * 100 * 2 / (1000 * 10 * 1e-3)),
              cost.compute_time);
    EXPECT_EQ(Costs::Duration(2020), cost.memory_time);
    EXPECT_EQ(1, cost.num_ops_total);
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(1, cost.num_ops_with_unknown_shapes);

    // Einsums and XlaEinsums should be estimated similarly.
    EXPECT_EQ(PredictCosts(DescribeEinsum({-1, 50}, {100, 50}, "ik,jk->ij"))
                  .execution_time,
              PredictCosts(DescribeXlaEinsum({-1, 50}, {100, 50}, "ik,jk->ij"))
                  .execution_time);
  }
}

TEST_F(OpLevelCostEstimatorTest, PredictResourceVariableOps) {
  TestOpLevelCostEstimator estimator;
  estimator.SetDeviceInfo(DeviceInfo(/*gigaops=*/1, /*gb_per_sec=*/1));

  {
    OpContext op_context;
    op_context.op_info.set_op("AssignVariableOp");
    DescribeDummyTensor(op_context.op_info.add_inputs());
    DescribeTensor1D(100, op_context.op_info.add_inputs());
    auto cost = estimator.PredictCosts(op_context);
    EXPECT_EQ(Costs::Duration(400), cost.memory_time);
    EXPECT_EQ(Costs::Duration(0), cost.compute_time);
    EXPECT_EQ(Costs::Duration(400), cost.execution_time);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(cost.temporary_memory, 0);
    EXPECT_EQ(cost.persistent_memory, 0);
  }

  {
    OpContext op_context;
    op_context.op_info.set_op("AssignSubVariableOp");
    DescribeDummyTensor(op_context.op_info.add_inputs());
    DescribeTensor1D(100, op_context.op_info.add_inputs());
    auto cost = estimator.PredictCosts(op_context);
    EXPECT_EQ(Costs::Duration(400), cost.memory_time);
    EXPECT_EQ(Costs::Duration(100), cost.compute_time);
    EXPECT_EQ(Costs::Duration(400), cost.execution_time);
    EXPECT_FALSE(cost.inaccurate);
  }
}

TEST_F(OpLevelCostEstimatorTest, AddNExecutionTime) {
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("AddN");

  DescribeTensor4D(1, 10, 10, 10, op_context.op_info.add_inputs());
  DescribeTensor4D(1, 10, 10, 10, op_context.op_info.add_inputs());
  DescribeTensor4D(1, 10, 10, 10, op_context.op_info.add_inputs());

  auto cost = PredictCosts(op_context);
  EXPECT_EQ(Costs::Duration(1200), cost.memory_time);
  EXPECT_EQ(Costs::Duration(200), cost.compute_time);
  EXPECT_EQ(Costs::Duration(1400), cost.execution_time);
  EXPECT_EQ(cost.num_ops_total, 1);
  EXPECT_FALSE(cost.inaccurate);
  EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  EXPECT_EQ(cost.temporary_memory, 0);
  EXPECT_EQ(cost.persistent_memory, 0);
}

TEST_F(OpLevelCostEstimatorTest, IdentityOpExecutionTime) {
  std::vector<std::string> identity_ops = {
      "_Recv",         "_Send",        "BitCast",         "Identity",
      "Enter",         "Exit",         "IdentityN",       "Merge",
      "NextIteration", "Placeholder",  "PreventGradient", "RefIdentity",
      "Reshape",       "StopGradient", "Switch"};

  const int kTensorSize = 1000;
  for (auto identity_op : identity_ops) {
    OpContext op_context = DescribeUnaryOp(identity_op, kTensorSize);

    const int kExpectedMemoryTime = 0;
    const int kExpectedComputeTime = 1;

    auto cost = PredictCosts(op_context);
    EXPECT_EQ(Costs::Duration(kExpectedMemoryTime), cost.memory_time);
    EXPECT_EQ(Costs::Duration(kExpectedComputeTime), cost.compute_time);
    EXPECT_EQ(Costs::Duration(kExpectedComputeTime + kExpectedMemoryTime),
              cost.execution_time);
    EXPECT_EQ(cost.max_memory, kTensorSize * 4);
    EXPECT_EQ(cost.num_ops_total, 1);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
    EXPECT_EQ(cost.temporary_memory, 0);
    EXPECT_EQ(cost.persistent_memory, 0);
  }
}

TEST_F(OpLevelCostEstimatorTest, PureMemoryOpExecutionTime) {
  std::vector<std::string> reshape_ops = {
      "ConcatV2",     "DataFormatVecPermute",
      "DepthToSpace", "ExpandDims",
      "Fill",         "OneHot",
      "Pack",         "Range",
      "SpaceToDepth", "Split",
      "Squeeze",      "Transpose",
      "Tile",         "Unpack"};

  const int kTensorSize = 1000;
  for (auto reshape_op : reshape_ops) {
    OpContext op_context = DescribeUnaryOp(reshape_op, kTensorSize);

    const int kExpectedMemoryTime = 800;
    const int kExpectedComputeTime = 0;

    auto cost = PredictCosts(op_context);
    EXPECT_EQ(Costs::Duration(kExpectedMemoryTime), cost.memory_time);
    EXPECT_EQ(Costs::Duration(kExpectedComputeTime), cost.compute_time);
    EXPECT_EQ(Costs::Duration(kExpectedComputeTime + kExpectedMemoryTime),
              cost.execution_time);
    EXPECT_EQ(cost.max_memory, kTensorSize * 4);
    EXPECT_EQ(cost.num_ops_total, 1);
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
    EXPECT_EQ(cost.temporary_memory, 0);
    EXPECT_EQ(cost.persistent_memory, 0);
  }
}

TEST_F(OpLevelCostEstimatorTest, ResizeBilinearExecutionTime) {
  const int kImageDim = 255;
  const int kChannelSize = 10;
  const int kComputeLerpCost = 9;
  {
    OpContext op_context;
    SetCpuDevice(&op_context.op_info);
    op_context.op_info.set_op("ResizeBilinear");
    DescribeTensor4D(1, kImageDim, kImageDim, kChannelSize,
                     op_context.op_info.add_inputs());
    // Test with no output.
    auto cost = PredictCosts(op_context);
    ExpectZeroCost(cost);
    op_context.op_info.clear_inputs();

    DescribeTensor4D(0, 0, 0, 0, op_context.op_info.add_outputs());
    // Test with no input.
    cost = PredictCosts(op_context);
    ExpectZeroCost(cost);
  }
  {
    // Test with size 0 output.
    OpContext op_context;
    SetCpuDevice(&op_context.op_info);
    op_context.op_info.set_op("ResizeBilinear");

    DescribeTensor4D(1, kImageDim, kImageDim, kChannelSize,
                     op_context.op_info.add_inputs());
    const int kExpectedMemoryTime = kImageDim * kImageDim * 4;
    DescribeTensor4D(0, 0, 0, 0, op_context.op_info.add_outputs());

    // As the half_pixel_centers attr was not set, cost should be inaccurate
    // with 0 compute time.
    auto cost = PredictCosts(op_context);
    EXPECT_EQ(cost.compute_time, Costs::Duration(0));
    EXPECT_EQ(cost.memory_time, Costs::Duration(kExpectedMemoryTime));
    EXPECT_EQ(cost.execution_time, Costs::Duration(kExpectedMemoryTime));
    EXPECT_TRUE(cost.inaccurate);
    EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
    EXPECT_EQ(cost.temporary_memory, 0);
    EXPECT_EQ(cost.persistent_memory, 0);

    AttrValue half_pixel_centers;
    half_pixel_centers.set_b(false);
    (*op_context.op_info.mutable_attr())["half_pixel_centers"] =
        half_pixel_centers;
    cost = PredictCosts(op_context);
    // Compute time depends only on output size, so compute time is 0.
    EXPECT_EQ(cost.compute_time, Costs::Duration(0));
    EXPECT_EQ(cost.memory_time, Costs::Duration(kExpectedMemoryTime));
    EXPECT_EQ(cost.execution_time, Costs::Duration(kExpectedMemoryTime));
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  }

  // Test with non-zero output size.
  const int kOutputImageDim = 100;
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("ResizeBilinear");
  DescribeTensor4D(1, kImageDim, kImageDim, kChannelSize,
                   op_context.op_info.add_inputs());
  DescribeTensor4D(1, kOutputImageDim, kOutputImageDim, kChannelSize,
                   op_context.op_info.add_outputs());
  const int kExpectedMemoryTime =
      (kImageDim * kImageDim + kOutputImageDim * kOutputImageDim) * 4;

  {
    // Cost of calculating weights without using half_pixel_centers.
    AttrValue half_pixel_centers;
    half_pixel_centers.set_b(false);
    (*op_context.op_info.mutable_attr())["half_pixel_centers"] =
        half_pixel_centers;
    const int kInterpWeightCost = 10;
    const int num_ops =
        kInterpWeightCost * (kOutputImageDim * 2) +
        kComputeLerpCost * (kOutputImageDim * kOutputImageDim * kChannelSize);
    const int expected_compute_time = std::ceil(
        num_ops /
        estimator_.GetDeviceInfo(op_context.op_info.device()).gigaops);

    const auto cost = PredictCosts(op_context);
    EXPECT_EQ(cost.compute_time, Costs::Duration(expected_compute_time));
    EXPECT_EQ(cost.memory_time, Costs::Duration(kExpectedMemoryTime));
    EXPECT_EQ(cost.execution_time,
              Costs::Duration(kExpectedMemoryTime + expected_compute_time));
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  }

  {
    // Cost of calculating weights using half_pixel_centers.
    AttrValue half_pixel_centers;
    half_pixel_centers.set_b(true);
    (*op_context.op_info.mutable_attr())["half_pixel_centers"] =
        half_pixel_centers;
    const int kInterpWeightCost = 12;
    const int num_ops =
        kInterpWeightCost * (kOutputImageDim * 2) +
        kComputeLerpCost * (kOutputImageDim * kOutputImageDim * kChannelSize);
    const int expected_compute_time = std::ceil(
        num_ops /
        estimator_.GetDeviceInfo(op_context.op_info.device()).gigaops);

    const auto cost = PredictCosts(op_context);
    EXPECT_EQ(cost.compute_time, Costs::Duration(expected_compute_time));
    EXPECT_EQ(cost.memory_time, Costs::Duration(kExpectedMemoryTime));
    EXPECT_EQ(cost.execution_time,
              Costs::Duration(kExpectedMemoryTime + expected_compute_time));
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  }

  {
    // Cost with very large tensor.
    op_context.op_info.clear_outputs();
    // Number of elements in tensor exceeds 2^32.
    constexpr int64_t kLargeOutputImageDim = 40000;
    DescribeTensor4D(1, kLargeOutputImageDim, kLargeOutputImageDim,
                     kChannelSize, op_context.op_info.add_outputs());
    const int64_t kInterpWeightCost = 12;
    // Using half_pixel_centers.
    AttrValue half_pixel_centers;
    half_pixel_centers.set_b(true);
    (*op_context.op_info.mutable_attr())["half_pixel_centers"] =
        half_pixel_centers;

    const int64_t num_ops =
        kInterpWeightCost * (kLargeOutputImageDim * 2) +
        kComputeLerpCost *
            (kLargeOutputImageDim * kLargeOutputImageDim * kChannelSize);
    const int64_t expected_compute_time = std::ceil(
        num_ops /
        estimator_.GetDeviceInfo(op_context.op_info.device()).gigaops);

    const int64_t expected_memory_time =
        (kImageDim * kImageDim + kLargeOutputImageDim * kLargeOutputImageDim) *
        4;

    const auto cost = PredictCosts(op_context);
    EXPECT_EQ(cost.compute_time, Costs::Duration(expected_compute_time));
    EXPECT_EQ(cost.memory_time, Costs::Duration(expected_memory_time));
    EXPECT_EQ(cost.execution_time,
              Costs::Duration(expected_memory_time + expected_compute_time));
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  }
}

TEST_F(OpLevelCostEstimatorTest, CropAndResizeExecutionTime) {
  const int kImageDim = 255;
  const int kChannelSize = 10;
  const int kOutputImageDim = 100;
  const int kNumBoxes = 10;
  const int kOutputElements =
      kNumBoxes * kOutputImageDim * kOutputImageDim * kChannelSize;
  OpContext op_context;
  SetCpuDevice(&op_context.op_info);
  op_context.op_info.set_op("CropAndResize");
  DescribeTensor4D(1, kImageDim, kImageDim, kChannelSize,
                   op_context.op_info.add_inputs());
  DescribeArbitraryRankInput({kNumBoxes, 4}, DT_INT64, &op_context.op_info);
  DescribeTensor4D(kNumBoxes, kOutputImageDim, kOutputImageDim, kChannelSize,
                   op_context.op_info.add_outputs());

  // Note this is time [ns, default in Duration in Costs], not bytes;
  // whereas memory bandwidth from SetCpuDevice() is 10GB/s.
  const int kExpectedMemoryTime =
      (kImageDim * kImageDim * 4 +  // input image in float.
       kNumBoxes * 4 * 8 / 10 +     // boxes (kNumBoxes x 4) in int64.
       kNumBoxes * kOutputImageDim * kOutputImageDim * 4);  // output in float.
  // Note that input image and output image has kChannelSize dim, which is 10,
  // hence, no need to divide it by 10 (bandwidth).

  {
    // Cost of CropAndResize with bilinear interpolation.
    AttrValue method;
    method.set_s("bilinear");
    (*op_context.op_info.mutable_attr())["method"] = method;
    int num_ops = 28 * kNumBoxes + 4 * kNumBoxes * kOutputImageDim +
                  4 * kNumBoxes * kOutputImageDim * kOutputImageDim +
                  3 * kNumBoxes * kOutputImageDim +
                  3 * kNumBoxes * kOutputImageDim * kOutputImageDim +
                  13 * kOutputElements;
    const int expected_compute_time = std::ceil(
        num_ops /
        estimator_.GetDeviceInfo(op_context.op_info.device()).gigaops);

    const auto cost = PredictCosts(op_context);
    EXPECT_EQ(cost.compute_time, Costs::Duration(expected_compute_time));
    EXPECT_EQ(cost.memory_time, Costs::Duration(kExpectedMemoryTime));
    EXPECT_EQ(cost.execution_time,
              Costs::Duration(kExpectedMemoryTime + expected_compute_time));
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  }

  {
    // Cost of CropAndResize when nearest pixel is taken.
    AttrValue method;
    method.set_s("nearest");
    (*op_context.op_info.mutable_attr())["method"] = method;
    int num_ops = 28 * kNumBoxes + 4 * kNumBoxes * kOutputImageDim +
                  4 * kNumBoxes * kOutputImageDim * kOutputImageDim +
                  2 * kNumBoxes * kOutputImageDim * kOutputImageDim +
                  kOutputElements;
    const int expected_compute_time = std::ceil(
        num_ops /
        estimator_.GetDeviceInfo(op_context.op_info.device()).gigaops);

    const auto cost = PredictCosts(op_context);
    EXPECT_EQ(cost.compute_time, Costs::Duration(expected_compute_time));
    EXPECT_EQ(cost.memory_time, Costs::Duration(kExpectedMemoryTime));
    EXPECT_EQ(cost.execution_time,
              Costs::Duration(kExpectedMemoryTime + expected_compute_time));
    EXPECT_FALSE(cost.inaccurate);
    EXPECT_EQ(cost.num_ops_with_unknown_shapes, 0);
  }
}

}  // end namespace grappler
}  // end namespace tensorflow
