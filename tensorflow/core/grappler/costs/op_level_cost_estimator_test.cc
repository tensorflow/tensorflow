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

// Returns an OpInfo for MatMul with the minimum set of fields set up.
OpInfo DescribeMatMul(int m, int n, int l, int k) {
  OpInfo op_features;
  auto device = op_features.mutable_device();
  device->set_type("CPU");
  op_features.set_op("MatMul");

  DescribeMatrix(m, l, &op_features);
  DescribeMatrix(k, n, &op_features);
  return op_features;
}

// Returns an OpInfo for MatMul with unknown input shapes.
OpInfo DescribeMatMulUnknownShape() {
  OpInfo op_features;
  auto device = op_features.mutable_device();
  device->set_type("CPU");
  op_features.set_op("MatMul");

  auto input = op_features.add_inputs();
  auto shape = input->mutable_shape();
  shape->set_unknown_rank(true);

  input = op_features.add_inputs();
  shape = input->mutable_shape();
  shape->set_unknown_rank(true);

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
}

// Returns an OpInfo for Conv2D with the minimum set of fields set up.
OpInfo DescribeConvolution(int batch, int ix, int iy, int iz1, int iz2, int kx,
                           int ky, int oz) {
  OpInfo op_features;
  auto device = op_features.mutable_device();
  device->set_type("CPU");
  op_features.set_op("Conv2D");

  DescribeTensor4D(batch, ix, iy, iz1, &op_features);
  DescribeTensor4D(kx, ky, iz2, oz, &op_features);
  return op_features;
}
}  // namespace

TEST(OpLevelCostEstimatorTest, UnknownOrPartialShape) {
  OpLevelCostEstimator estimator;

  EXPECT_EQ(false,
            estimator.PredictCosts(DescribeMatMul(2, 4, 7, 7)).inaccurate);
  EXPECT_EQ(true,
            estimator.PredictCosts(DescribeMatMul(-1, 4, 7, 7)).inaccurate);
  EXPECT_EQ(true,
            estimator.PredictCosts(DescribeMatMul(2, 4, -1, 7)).inaccurate);

  EXPECT_EQ(
      false,
      estimator.PredictCosts(DescribeConvolution(16, 19, 19, 48, 48, 5, 5, 256))
          .inaccurate);
  EXPECT_EQ(
      true,
      estimator.PredictCosts(DescribeConvolution(16, -1, 19, 48, 48, 5, 5, 256))
          .inaccurate);
}

}  // end namespace grappler
}  // end namespace tensorflow
