/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <limits>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {

namespace {

using errors::IsInvalidArgument;

constexpr int32_t kInt32Min = std::numeric_limits<int32_t>::min();
constexpr int32_t kInt32Max = std::numeric_limits<int32_t>::max();

}  // namespace

class UniformQuantizedAddOpTest : public OpsTestBase {
 protected:
};

TEST_F(UniformQuantizedAddOpTest, InvalidShape) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedAdd")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_QINT32)
                   .Attr("lhs_quantization_axis", 1)
                   .Attr("rhs_quantization_axis", 0)
                   .Attr("output_quantization_axis", 1)
                   .Attr("lhs_quantization_min_val", kInt32Min)
                   .Attr("lhs_quantization_max_val", kInt32Max)
                   .Attr("rhs_quantization_min_val", kInt32Min)
                   .Attr("rhs_quantization_max_val", kInt32Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({2, 3}), {-6, -4, -2, 0, 2, 4});
  AddInputFromArray<qint32>(TensorShape({2}), {-100, 0});
  AddInputFromArray<float>(TensorShape({3}), {2, 3, 4});
  AddInputFromArray<int32>(TensorShape({3}), {-20, 0, 20});
  AddInputFromArray<float>(TensorShape({2}), {2, 3});
  AddInputFromArray<int32>(TensorShape({2}), {0, 0});
  AddInputFromArray<float>(TensorShape({3}), {2, 3, 4});
  AddInputFromArray<int32>(TensorShape({3}), {-40, 0, 40});

  EXPECT_TRUE(IsInvalidArgument(RunOpKernel()));
}

TEST_F(UniformQuantizedAddOpTest, PerChannelSameScale) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedAdd")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_QINT32)
                   .Attr("lhs_quantization_axis", 1)
                   .Attr("rhs_quantization_axis", 0)
                   .Attr("output_quantization_axis", 1)
                   .Attr("lhs_quantization_min_val", kInt32Min)
                   .Attr("lhs_quantization_max_val", kInt32Max)
                   .Attr("rhs_quantization_min_val", kInt32Min)
                   .Attr("rhs_quantization_max_val", kInt32Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({2, 3}), {-6, -4, -2, 0, 2, 4});
  AddInputFromArray<qint32>(TensorShape({3}), {-100, 0, 100});
  AddInputFromArray<float>(TensorShape({3}), {2, 3, 4});
  AddInputFromArray<int32>(TensorShape({3}), {-20, 0, 20});
  AddInputFromArray<float>(TensorShape({3}), {2, 3, 4});
  AddInputFromArray<int32>(TensorShape({3}), {0, 0, 0});
  AddInputFromArray<float>(TensorShape({3}), {2, 3, 4});
  AddInputFromArray<int32>(TensorShape({3}), {-40, 0, 40});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 3}));
  test::FillValues<qint32>(&expected, {-126, -4, 118, -120, 2, 124});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedAddOpTest, PerTensorSameScaleLhsMultiDims) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedAdd")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_QINT32)
                   .Attr("lhs_quantization_axis", -1)
                   .Attr("rhs_quantization_axis", -1)
                   .Attr("output_quantization_axis", -1)
                   .Attr("lhs_quantization_min_val", kInt32Min)
                   .Attr("lhs_quantization_max_val", kInt32Max)
                   .Attr("rhs_quantization_min_val", kInt32Min)
                   .Attr("rhs_quantization_max_val", kInt32Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({2, 3}), {-6, -4, -2, 0, 2, 4});
  AddInputFromArray<qint32>(TensorShape({3}), {-100, 0, 100});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-20});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-40});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 3}));
  test::FillValues<qint32>(&expected, {-126, -24, 78, -120, -18, 84});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedAddOpTest, PerTensorSameScaleRhsMultiDims) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedAdd")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_QINT32)
                   .Attr("lhs_quantization_axis", -1)
                   .Attr("rhs_quantization_axis", -1)
                   .Attr("output_quantization_axis", -1)
                   .Attr("lhs_quantization_min_val", kInt32Min)
                   .Attr("lhs_quantization_max_val", kInt32Max)
                   .Attr("rhs_quantization_min_val", kInt32Min)
                   .Attr("rhs_quantization_max_val", kInt32Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({3}), {-100, 0, 100});
  AddInputFromArray<qint32>(TensorShape({2, 3}), {-6, -4, -2, 0, 2, 4});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-20});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-40});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 3}));
  test::FillValues<qint32>(&expected, {-126, -24, 78, -120, -18, 84});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedAddOpTest, PerChannelDifferentScale) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedAdd")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_QINT32)
                   .Attr("lhs_quantization_axis", 1)
                   .Attr("rhs_quantization_axis", 0)
                   .Attr("output_quantization_axis", 1)
                   .Attr("lhs_quantization_min_val", kInt32Min)
                   .Attr("lhs_quantization_max_val", kInt32Max)
                   .Attr("rhs_quantization_min_val", kInt32Min)
                   .Attr("rhs_quantization_max_val", kInt32Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({2, 3}), {-6, -4, -2, 0, 2, 4});
  AddInputFromArray<qint32>(TensorShape({3}), {-100, 0, 100});
  AddInputFromArray<float>(TensorShape({3}), {2, 3, 1});
  AddInputFromArray<int32>(TensorShape({3}), {-20, 0, 20});
  AddInputFromArray<float>(TensorShape({3}), {1, 3, 2});
  AddInputFromArray<int32>(TensorShape({3}), {0, 0, 0});
  AddInputFromArray<float>(TensorShape({3}), {4, 3, 2});
  AddInputFromArray<int32>(TensorShape({3}), {-40, 0, 40});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 3}));
  test::FillValues<qint32>(&expected, {-58, -4, 129, -55, 2, 132});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedAddOpTest, PerChannelDifferentScaleBroadcastLhs) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedAdd")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_QINT32)
                   .Attr("lhs_quantization_axis", 1)
                   .Attr("rhs_quantization_axis", 1)
                   .Attr("output_quantization_axis", 1)
                   .Attr("lhs_quantization_min_val", kInt32Min)
                   .Attr("lhs_quantization_max_val", kInt32Max)
                   .Attr("rhs_quantization_min_val", kInt32Min)
                   .Attr("rhs_quantization_max_val", kInt32Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({1, 3}), {-100, 0, 100});
  AddInputFromArray<qint32>(TensorShape({2, 3}), {-6, -4, -2, 0, 2, 4});
  AddInputFromArray<float>(TensorShape({3}), {1, 3, 2});
  AddInputFromArray<int32>(TensorShape({3}), {0, 0, 0});
  AddInputFromArray<float>(TensorShape({3}), {2, 3, 1});
  AddInputFromArray<int32>(TensorShape({3}), {-20, 0, 20});
  AddInputFromArray<float>(TensorShape({3}), {4, 3, 2});
  AddInputFromArray<int32>(TensorShape({3}), {-40, 0, 40});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 3}));
  test::FillValues<qint32>(&expected, {-58, -4, 129, -55, 2, 132});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedAddOpTest, PerTensorDifferentScale) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedAdd")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_QINT32)
                   .Attr("lhs_quantization_axis", -1)
                   .Attr("rhs_quantization_axis", -1)
                   .Attr("output_quantization_axis", -1)
                   .Attr("lhs_quantization_min_val", kInt32Min)
                   .Attr("lhs_quantization_max_val", kInt32Max)
                   .Attr("rhs_quantization_min_val", kInt32Min)
                   .Attr("rhs_quantization_max_val", kInt32Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({2, 3}), {-6, -4, -2, 0, 2, 4});
  AddInputFromArray<qint32>(TensorShape({3}), {-100, 0, 100});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-20});
  AddInputFromArray<float>(TensorShape({}), {1});
  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {4});
  AddInputFromArray<int32>(TensorShape({}), {-40});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 3}));
  test::FillValues<qint32>(&expected, {-58, -32, -6, -55, -29, -3});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedAddOpTest, PerTensorSameScaleTensorAddScalar) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedAdd")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_QINT32)
                   .Attr("lhs_quantization_axis", -1)
                   .Attr("rhs_quantization_axis", -1)
                   .Attr("output_quantization_axis", -1)
                   .Attr("lhs_quantization_min_val", kInt32Min)
                   .Attr("lhs_quantization_max_val", kInt32Max)
                   .Attr("rhs_quantization_min_val", kInt32Min)
                   .Attr("rhs_quantization_max_val", kInt32Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({2, 3}), {-6, -4, -2, 0, 2, 4});
  AddInputFromArray<qint32>(TensorShape({}), {-100});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-20});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-40});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 3}));
  test::FillValues<qint32>(&expected, {-126, -124, -122, -120, -118, -116});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedAddOpTest, PerTensorSameScaleScalarAddTensor) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedAdd")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_QINT32)
                   .Attr("lhs_quantization_axis", -1)
                   .Attr("rhs_quantization_axis", -1)
                   .Attr("output_quantization_axis", -1)
                   .Attr("lhs_quantization_min_val", kInt32Min)
                   .Attr("lhs_quantization_max_val", kInt32Max)
                   .Attr("rhs_quantization_min_val", kInt32Min)
                   .Attr("rhs_quantization_max_val", kInt32Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({}), {-100});
  AddInputFromArray<qint32>(TensorShape({2, 3}), {-6, -4, -2, 0, 2, 4});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-20});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-40});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 3}));
  test::FillValues<qint32>(&expected, {-126, -124, -122, -120, -118, -116});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedAddOpTest, PerTensorSameScaleScalarAddScalar) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedAdd")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_QINT32)
                   .Attr("lhs_quantization_axis", -1)
                   .Attr("rhs_quantization_axis", -1)
                   .Attr("output_quantization_axis", -1)
                   .Attr("lhs_quantization_min_val", kInt32Min)
                   .Attr("lhs_quantization_max_val", kInt32Max)
                   .Attr("rhs_quantization_min_val", kInt32Min)
                   .Attr("rhs_quantization_max_val", kInt32Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({}), {-6});
  AddInputFromArray<qint32>(TensorShape({}), {-100});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-20});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-40});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({}));
  test::FillValues<qint32>(&expected, {-126});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedAddOpTest, TensorAddEmptyTensor) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedAdd")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_QINT32)
                   .Attr("lhs_quantization_axis", -1)
                   .Attr("rhs_quantization_axis", -1)
                   .Attr("output_quantization_axis", -1)
                   .Attr("lhs_quantization_min_val", kInt32Min)
                   .Attr("lhs_quantization_max_val", kInt32Max)
                   .Attr("rhs_quantization_min_val", kInt32Min)
                   .Attr("rhs_quantization_max_val", kInt32Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({2, 1, 1}), {-6, -12});
  AddInputFromArray<qint32>(TensorShape({2, 0, 1}), {});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-20});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-40});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 0, 1}));
  test::FillValues<qint32>(&expected, {});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedAddOpTest, ScalarAddEmptyTensor) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedAdd")
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_QINT32)
                   .Attr("lhs_quantization_axis", -1)
                   .Attr("rhs_quantization_axis", -1)
                   .Attr("output_quantization_axis", -1)
                   .Attr("lhs_quantization_min_val", kInt32Min)
                   .Attr("lhs_quantization_max_val", kInt32Max)
                   .Attr("rhs_quantization_min_val", kInt32Min)
                   .Attr("rhs_quantization_max_val", kInt32Max)
                   .Attr("output_quantization_min_val", kInt32Min)
                   .Attr("output_quantization_max_val", kInt32Max)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  AddInputFromArray<qint32>(TensorShape({}), {-6});
  AddInputFromArray<qint32>(TensorShape({2, 0, 1}), {});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-20});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<int32>(TensorShape({}), {-40});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 0, 1}));
  test::FillValues<qint32>(&expected, {});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

}  // namespace tensorflow
