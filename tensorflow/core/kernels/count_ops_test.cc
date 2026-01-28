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

#include <cstdint>

#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

TEST_F(OpsTestBase, DenseCountSparseOutputShapeFn) {
  ShapeInferenceTestOp op("DenseCountSparseOutput");
  INFER_OK(op, "[?];?", "[?,1];[?];[1]");
  INFER_OK(op, "[?,?];?", "[?,2];[?];[2]");
}

TEST_F(OpsTestBase, SparseCountSparseOutputShapeFn) {
  ShapeInferenceTestOp op("SparseCountSparseOutput");
  INFER_OK(op, "[?,1];?;?;?", "[?,d0_1];[?];[d0_1]");
  INFER_OK(op, "[?,2];?;?;?", "[?,d0_1];[?];[d0_1]");
}

TEST_F(OpsTestBase, RaggedCountSparseOutputShapeFn) {
  ShapeInferenceTestOp op("RaggedCountSparseOutput");
  INFER_OK(op, "?;[?];?", "[?,2];[?];[2]");
}

class DenseCountLimitTest : public OpsTestBase {
 protected:
};

TEST_F(DenseCountLimitTest, Basic) {
  TF_ASSERT_OK(NodeDefBuilder("dense_count", "DenseCountSparseOutput")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_INT64))
                   .Attr("T", DT_INT32)
                   .Attr("minlength", -1)
                   .Attr("maxlength", -1)
                   .Attr("binary_output", false)
                   .Attr("output_type", DT_INT64)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Input: [1, 2, 1, 3]
  AddInputFromArray<int32>(TensorShape({4}), {1, 2, 1, 3});
  // Weights: [] (empty)
  AddInputFromArray<int64_t>(TensorShape({0}), {});

  TF_ASSERT_OK(RunOpKernel());

  // Expected output:
  // indices: [1, 2, 3]
  // values: [2, 1, 1]
  // dense_shape: [4] (max_value + 1)

  const Tensor* indices_tensor = GetOutput(0);
  const Tensor* values_tensor = GetOutput(1);
  const Tensor* shape_tensor = GetOutput(2);

  EXPECT_EQ(indices_tensor->dim_size(0), 3);
  EXPECT_EQ(values_tensor->dim_size(0), 3);
  EXPECT_EQ(shape_tensor->NumElements(), 1);

  auto indices = indices_tensor->matrix<int64_t>();
  auto values = values_tensor->flat<int64_t>();
  auto shape = shape_tensor->flat<int64_t>();

  EXPECT_EQ(indices(0, 0), int64_t{1});
  EXPECT_EQ(values(0), int64_t{2});

  EXPECT_EQ(indices(1, 0), int64_t{2});
  EXPECT_EQ(values(1), int64_t{1});

  EXPECT_EQ(indices(2, 0), int64_t{3});
  EXPECT_EQ(values(2), int64_t{1});

  EXPECT_EQ(shape(0), int64_t{4});
}

// Test checking that large num_value_elements calculation logic doesn't crash.
// Although we cannot allocate > 2^31 elements, we can verify that the Op
// correctly handles standard inputs with the new int64 types.
TEST_F(DenseCountLimitTest, LargeBatch) {
  TF_ASSERT_OK(NodeDefBuilder("dense_count", "DenseCountSparseOutput")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_INT64))
                   .Attr("T", DT_INT32)
                   .Attr("minlength", -1)
                   .Attr("maxlength", -1)
                   .Attr("binary_output", false)
                   .Attr("output_type", DT_INT64)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Input: Matrix [2, 3]
  // [[1, 2, 1], [3, 3, 0]]
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 1, 3, 3, 0});
  // Weights: []
  AddInputFromArray<int64_t>(TensorShape({0}), {});

  TF_ASSERT_OK(RunOpKernel());

  const Tensor* indices_tensor = GetOutput(0);
  const Tensor* values_tensor = GetOutput(1);
  const Tensor* shape_tensor = GetOutput(2);

  // Batch 0: 1:2, 2:1
  // Batch 1: 0:1, 3:2
  // Total values: 4

  EXPECT_EQ(indices_tensor->dim_size(0), 4);
  EXPECT_EQ(values_tensor->dim_size(0), 4);
  EXPECT_EQ(shape_tensor->NumElements(), 2);

  auto indices = indices_tensor->matrix<int64_t>();
  auto values = values_tensor->flat<int64_t>();
  auto shape = shape_tensor->flat<int64_t>();

  // Sorted by batch, then value
  int idx = 0;
  // Batch 0
  EXPECT_EQ(indices(idx, 0), int64_t{0});
  EXPECT_EQ(indices(idx, 1), int64_t{1});
  EXPECT_EQ(values(idx), int64_t{2});
  idx++;

  EXPECT_EQ(indices(idx, 0), int64_t{0});
  EXPECT_EQ(indices(idx, 1), int64_t{2});
  EXPECT_EQ(values(idx), int64_t{1});
  idx++;

  // Batch 1
  EXPECT_EQ(indices(idx, 0), int64_t{1});
  EXPECT_EQ(indices(idx, 1), int64_t{0});
  EXPECT_EQ(values(idx), int64_t{1});
  idx++;

  EXPECT_EQ(indices(idx, 0), int64_t{1});
  EXPECT_EQ(indices(idx, 1), int64_t{3});
  EXPECT_EQ(values(idx), int64_t{2});
  idx++;

  EXPECT_EQ(shape(0), int64_t{2});  // num_batches
  EXPECT_EQ(shape(1), int64_t{4});  // max_value + 1
}

class SparseCountLimitTest : public OpsTestBase {
 protected:
};

TEST_F(SparseCountLimitTest, Basic) {
  TF_ASSERT_OK(NodeDefBuilder("sparse_count", "SparseCountSparseOutput")
                   .Input(FakeInput(DT_INT64))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_INT64))
                   .Input(FakeInput(DT_INT64))
                   .Attr("T", DT_INT32)
                   .Attr("minlength", -1)
                   .Attr("maxlength", -1)
                   .Attr("binary_output", false)
                   .Attr("output_type", DT_INT64)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Indices: [[0, 0], [0, 1], [1, 0]]
  AddInputFromArray<int64_t>(TensorShape({3, 2}), {0, 0, 0, 1, 1, 0});
  // Values: [1, 2, 1]
  AddInputFromArray<int32>(TensorShape({3}), {1, 2, 1});
  // Shape: [2, 2]
  AddInputFromArray<int64_t>(TensorShape({2}), {2, 2});
  // Weights: []
  AddInputFromArray<int64_t>(TensorShape({0}), {});

  TF_ASSERT_OK(RunOpKernel());

  // Batch 0: 1:1, 2:1
  // Batch 1: 1:1
  // Total output values: 3

  const Tensor* indices_tensor = GetOutput(0);
  const Tensor* values_tensor = GetOutput(1);
  const Tensor* shape_tensor = GetOutput(2);

  EXPECT_EQ(indices_tensor->dim_size(0), 3);
  EXPECT_EQ(values_tensor->dim_size(0), 3);

  auto indices = indices_tensor->matrix<int64_t>();
  auto values = values_tensor->flat<int64_t>();
  auto shape = shape_tensor->flat<int64_t>();

  int idx = 0;
  // Batch 0
  EXPECT_EQ(indices(idx, 0), int64_t{0});
  EXPECT_EQ(indices(idx, 1), int64_t{1});
  EXPECT_EQ(values(idx), int64_t{1});
  idx++;

  EXPECT_EQ(indices(idx, 0), int64_t{0});
  EXPECT_EQ(indices(idx, 1), int64_t{2});
  EXPECT_EQ(values(idx), int64_t{1});
  idx++;

  // Batch 1
  EXPECT_EQ(indices(idx, 0), int64_t{1});
  EXPECT_EQ(indices(idx, 1), int64_t{1});
  EXPECT_EQ(values(idx), int64_t{1});
  idx++;

  EXPECT_EQ(shape(0), int64_t{2});
  EXPECT_EQ(shape(1), int64_t{3});  // max_value + 1
}

}  // namespace
}  // namespace tensorflow
