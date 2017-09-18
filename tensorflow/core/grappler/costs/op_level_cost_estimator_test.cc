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
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {

namespace {
// Wrangles the minimum number of proto fields to set up a matrix.
void DescribeMatrix(int rows, int columns, OpInfo *op_features) {
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
OpInfo DescribeMatMul(int m, int n, int l, int k) {
  OpInfo op_features;
  SetCpuDevice(&op_features);
  op_features.set_op("MatMul");

  DescribeMatrix(m, l, &op_features);
  DescribeMatrix(k, n, &op_features);
  return op_features;
}

// Returns an OpInfo for MatMul with unknown input shapes.
OpInfo DescribeMatMulUnknownShape() {
  OpInfo op_features;
  SetCpuDevice(&op_features);
  op_features.set_op("MatMul");

  auto input = op_features.add_inputs();
  auto shape = input->mutable_shape();
  shape->set_unknown_rank(true);

  input = op_features.add_inputs();
  shape = input->mutable_shape();
  shape->set_unknown_rank(true);

  return op_features;
}

// Wrangles the minimum number of proto fields to set up an input of
// arbitrary rank and type.
void DescribeArbitraryRankInput(const std::vector<int>& dims, DataType dtype,
                                OpInfo* op_features) {
  auto input = op_features->add_inputs();
  input->set_dtype(dtype);
  auto shape = input->mutable_shape();
  for (auto d : dims) {
    shape->add_dim()->set_size(d);
  }
}

// Returns an OpInfo for a BatchMatMul
OpInfo DescribeBatchMatMul(const std::vector<int>& dims_a,
                           const std::vector<int>& dims_b) {
  OpInfo op_features;
  SetCpuDevice(&op_features);
  op_features.set_op("BatchMatMul");

  DescribeArbitraryRankInput(dims_a, DT_FLOAT, &op_features);
  DescribeArbitraryRankInput(dims_b, DT_FLOAT, &op_features);
  return op_features;
}

// Wrangles the minimum number of proto fields to set up a 4D Tensor for cost
// estimation purposes.
void DescribeTensor4D(int dim0, int dim1, int dim2, int dim3,
                      OpInfo *op_features) {
  auto input = op_features->add_inputs();
  auto shape = input->mutable_shape();
  shape->add_dim()->set_size(dim0);
  shape->add_dim()->set_size(dim1);
  shape->add_dim()->set_size(dim2);
  shape->add_dim()->set_size(dim3);
  input->set_dtype(DT_FLOAT);
}

// Returns an OpInfo for Conv2D with the minimum set of fields set up.
OpInfo DescribeConvolution(int batch, int ix, int iy, int iz1, int iz2, int kx,
                           int ky, int oz) {
  OpInfo op_features;
  SetCpuDevice(&op_features);
  op_features.set_op("Conv2D");

  DescribeTensor4D(batch, ix, iy, iz1, &op_features);
  DescribeTensor4D(kx, ky, iz2, oz, &op_features);
  return op_features;
}

OpInfo DescribeOp(const string& op, int size1, int size2) {
  OpInfo op_features;
  SetCpuDevice(&op_features);
  op_features.set_op(op);

  DescribeTensor4D(size1, 1, 1, 1, &op_features);
  DescribeTensor4D(2 * size1, size2, 1, 1, &op_features);

  auto output = op_features.add_outputs();
  auto shape = output->mutable_shape();
  shape->add_dim()->set_size(2 * size1);
  shape->add_dim()->set_size(size2);
  shape->add_dim()->set_size(1);
  shape->add_dim()->set_size(1);
  output->set_dtype(DT_FLOAT);

  SetCpuDevice(&op_features);
  return op_features;
}
}  // namespace

class OpLevelCostEstimatorTest : public ::testing::Test {
 protected:
  Costs PredictCosts(const OpInfo& op_features) const {
    return estimator_.PredictCosts(op_features);
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

  OpLevelCostEstimator estimator_;
};

TEST_F(OpLevelCostEstimatorTest, DummyExecutionTime) {
  auto cost = PredictCosts(DescribeOp("Dummy", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(200), cost.compute_time);
  EXPECT_EQ(Costs::Duration(2200), cost.execution_time);
  EXPECT_TRUE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, ExecutionTimeSumOrMax) {
  SetComputeMemoryOverlap(true);
  auto cost = PredictCosts(DescribeOp("Dummy", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(200), cost.compute_time);
  EXPECT_EQ(Costs::Duration(2000), cost.execution_time);  // max(2000, 200)
  EXPECT_TRUE(cost.inaccurate);
  SetComputeMemoryOverlap(false);  // Set it back to default.
}

TEST_F(OpLevelCostEstimatorTest, MulExecutionTime) {
  auto cost = PredictCosts(DescribeOp("Mul", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(200), cost.compute_time);
  EXPECT_EQ(Costs::Duration(2200), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, MulBroadcastExecutionTime) {
  auto cost = PredictCosts(DescribeOp("Mul", 1000, 2));
  EXPECT_EQ(Costs::Duration(3600), cost.memory_time);
  EXPECT_EQ(Costs::Duration(400), cost.compute_time);
  EXPECT_EQ(Costs::Duration(4000), cost.execution_time);
  EXPECT_FALSE(cost.inaccurate);
}

TEST_F(OpLevelCostEstimatorTest, ModExecutionTime) {
  auto cost = PredictCosts(DescribeOp("Mod", 1000, 1));
  EXPECT_EQ(Costs::Duration(2000), cost.memory_time);
  EXPECT_EQ(Costs::Duration(1600), cost.compute_time);
  EXPECT_EQ(Costs::Duration(3600), cost.execution_time);
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
      CountMatMulOperations(DescribeMatMul(2, 2, 4, 4), &matmul_inaccurate),
      CountBatchMatMulOperations(DescribeBatchMatMul({2, 4}, {4, 2}),
                                 &batch_matmul_inaccurate));
  EXPECT_EQ(matmul_inaccurate, batch_matmul_inaccurate);
  EXPECT_EQ(10 * CountMatMulOperations(DescribeMatMul(2, 2, 4, 4),
                                       &matmul_inaccurate),
            CountBatchMatMulOperations(
                DescribeBatchMatMul({10, 2, 4}, {-1, 10, 4, 2}),
                &batch_matmul_inaccurate));
  EXPECT_NE(matmul_inaccurate, batch_matmul_inaccurate);
  EXPECT_EQ(20 * CountMatMulOperations(DescribeMatMul(2, 2, 4, 4),
                                       &matmul_inaccurate),
            CountBatchMatMulOperations(
                DescribeBatchMatMul({2, 10, 2, 4}, {-1, 10, 4, 2}),
                &batch_matmul_inaccurate));
  EXPECT_NE(matmul_inaccurate, batch_matmul_inaccurate);
}

}  // end namespace grappler
}  // end namespace tensorflow
