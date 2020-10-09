/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <random>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

class QuantizedOpTest : public OpsTestBase {
 protected:
};

struct ParameterizedQuantizeOpTest : public OpsTestBase,
                                     public ::testing::WithParamInterface<int> {
};

TEST_F(QuantizedOpTest, QuantizeV2) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({7}),
                           {0.0, 1.0, 1.25, 1.75, 127.0, 255.0, 500.0});
  // min_range = 0
  AddInputFromArray<float>(TensorShape({1}), {0});
  // max_range = 255
  AddInputFromArray<float>(TensorShape({1}), {255.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({7}));
  // Input element 0.0 should map to 0.
  // Input element 500.0 is quantized to 255 because max_range = 255.
  test::FillValues<quint8>(&expected, {0, 1, 1, 2, 127, 255, 255});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
}

// Creates a tensor with the specified dims, using values chosen from data,
// multiplied by (1 + index) along the axis dimension.
template <typename T>
std::vector<T> ScalePerSliceAlongAxis(std::vector<int64> dims, int axis,
                                      const std::vector<T>& data) {
  uint32 seed = 123;
  std::minstd_rand rng(seed);
  int64 out_size = 1;
  for (int dim : dims) {
    out_size *= dim;
  }
  int minor_size = 1;
  for (int i = axis + 1; i < dims.size(); ++i) {
    minor_size *= dims[i];
  }
  std::vector<T> out(out_size);
  int num_slices = (axis == -1) ? 1 : dims[axis];
  for (int out_idx = 0; out_idx < out_size; ++out_idx) {
    int in_idx = rng() % data.size();
    T multiplier = ((out_idx / minor_size) % num_slices) + 1;
    out[out_idx] = data[in_idx] * multiplier;
  }
  return out;
}

TEST_P(ParameterizedQuantizeOpTest, QuantizeV2Quint8Scaled) {
  const int axis = GetParam();
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "SCALED")
                   .Attr("axis", axis)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const std::vector<int64> dims = {2, 3, 4, 5};
  int num_slices = (axis == -1) ? 1 : dims[axis];

  // Each channel contains the same 8 values multiplied by (channel + 1).
  AddInputFromArray<float>(
      TensorShape(dims),
      ScalePerSliceAlongAxis<float>(
          dims, axis, {-255.0, 0.0, 1.0, 1.25, 1.75, 64.0, 127.0, 500.0}));
  std::vector<float> min_ranges(num_slices), max_ranges(num_slices);
  for (int slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
    min_ranges[slice_idx] = (slice_idx + 1) * -255.0;
    max_ranges[slice_idx] = (slice_idx + 1) * 127.0;
  }
  AddInputFromArray<float>(TensorShape({num_slices}), min_ranges);
  AddInputFromArray<float>(TensorShape({num_slices}), max_ranges);
  TF_ASSERT_OK(RunOpKernel());
  // Input values < 0 should map to 0 even though min_range = -255, because
  // we are performing quantization by scaling to quint8.
  // Input value 0.0 should map to 0.
  // The scale factor chosen should be 255 / 127 =  2.00787
  // Output values are clipped to 255.

  Tensor expected(allocator(), DT_QUINT8, TensorShape(dims));
  test::FillValues<quint8>(
      &expected,
      ScalePerSliceAlongAxis<quint8>(dims, -1, {0, 0, 2, 3, 4, 129, 255, 255}));

  auto output_min = *GetOutput(1);
  auto output_max = *GetOutput(2);

  for (int slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
    EXPECT_EQ(output_min.flat<float>()(slice_idx), 0);
    EXPECT_EQ(output_max.flat<float>()(slice_idx), 127.0 * (slice_idx + 1));
  }

  auto output = *GetOutput(0);
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
}

TEST_F(QuantizedOpTest, QuantizeV2Quint8ScaledSmallInputRange) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "SCALED")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({3}), {-1.0, 0.0, 2.0});
  AddInputFromArray<float>(TensorShape({1}), {-1.0f});
  AddInputFromArray<float>(TensorShape({1}), {2.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({3}));
  // Input element -1.0 should map to 0 even though min_range = -1, because
  // we are performing quantization by scaling to quint8.
  // Input element 0.0 should map to 0.
  // Input element 2.0 should map to max quint8 value 255.
  test::FillValues<quint8>(&expected, {0, 0, 255});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));

  Tensor expected_output_min(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected_output_min, {0.0});
  test::ExpectTensorEqual<float>(expected_output_min, *GetOutput(1));

  Tensor expected_output_max(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected_output_max, {2.0});
  test::ExpectTensorEqual<float>(expected_output_max, *GetOutput(2));
}

TEST_P(ParameterizedQuantizeOpTest, QuantizeV2Qint8Scaled) {
  const int axis = GetParam();
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<qint8>::v())
                   .Attr("mode", "SCALED")
                   .Attr("narrow_range", false)
                   .Attr("axis", axis)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const std::vector<int64> dims = {2, 3, 4, 5};
  int num_slices = (axis == -1) ? 1 : dims[axis];

  // Each channel contains the same 7 values multiplied by (channel + 1).
  AddInputFromArray<float>(
      TensorShape(dims),
      ScalePerSliceAlongAxis<float>(
          dims, axis, {-128.0, 0.0, 1.0, 1.25, 1.75, 64.0, 127.0}));
  std::vector<float> min_ranges(num_slices), max_ranges(num_slices);
  for (int slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
    min_ranges[slice_idx] = (slice_idx + 1) * -128.0;
    max_ranges[slice_idx] = (slice_idx + 1) * 100.0;
  }
  AddInputFromArray<float>(TensorShape({num_slices}), min_ranges);
  AddInputFromArray<float>(TensorShape({num_slices}), max_ranges);
  TF_ASSERT_OK(RunOpKernel());

  // Input element 0.0 should map to 0.
  // Input element 127.0 maps to 127 instead of 100.
  // (i.e. the max_ranges[] values should be ignored because their magnitude is
  // less than the min_ranges[] values).
  Tensor expected(allocator(), DT_QINT8, TensorShape(dims));
  test::FillValues<qint8>(
      &expected,
      ScalePerSliceAlongAxis<qint8>(dims, -1, {-128, 0, 1, 1, 2, 64, 127}));

  auto output_min = *GetOutput(1);
  auto output_max = *GetOutput(2);

  for (int slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
    EXPECT_EQ(output_min.flat<float>()(slice_idx), -128.0 * (slice_idx + 1));
    EXPECT_EQ(output_max.flat<float>()(slice_idx), 127.0 * (slice_idx + 1));
  }

  auto output = *GetOutput(0);
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));
}

TEST_P(ParameterizedQuantizeOpTest, QuantizeV2Qint8ScaledNarrowRange) {
  const int axis = GetParam();
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<qint8>::v())
                   .Attr("mode", "SCALED")
                   .Attr("narrow_range", true)
                   .Attr("axis", axis)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const std::vector<int64> dims = {2, 3, 4, 5};
  int num_slices = (axis == -1) ? 1 : dims[axis];

  // Each channel contains the same 7 values multiplied by (channel + 1).
  AddInputFromArray<float>(
      TensorShape(dims),
      ScalePerSliceAlongAxis<float>(
          dims, axis, {-128.0, 0.0, 1.0, 1.25, 1.75, 64.0, 127.0}));
  std::vector<float> min_ranges(num_slices), max_ranges(num_slices);
  for (int slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
    min_ranges[slice_idx] = (slice_idx + 1) * -128.0;
    max_ranges[slice_idx] = (slice_idx + 1) * 100.0;
  }
  AddInputFromArray<float>(TensorShape({num_slices}), min_ranges);
  AddInputFromArray<float>(TensorShape({num_slices}), max_ranges);
  TF_ASSERT_OK(RunOpKernel());

  // Input element 0.0 should map to 0.
  // Input element 127.0 maps to 127 instead of 100.
  // (i.e. the max_ranges[] values should be ignored because their magnitude is
  // less than the min_ranges[] values).
  Tensor expected(allocator(), DT_QINT8, TensorShape(dims));
  test::FillValues<qint8>(
      &expected,
      ScalePerSliceAlongAxis<qint8>(dims, -1, {-127, 0, 1, 1, 2, 64, 126}));

  auto output_min = *GetOutput(1);
  auto output_max = *GetOutput(2);

  for (int slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
    EXPECT_EQ(output_min.flat<float>()(slice_idx), -128.0 * (slice_idx + 1));
    EXPECT_EQ(output_max.flat<float>()(slice_idx), 128.0 * (slice_idx + 1));
  }

  auto output = *GetOutput(0);
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));
}

// Instantiate parameterized tests for axis = -1, 1, 3.
INSTANTIATE_TEST_SUITE_P(All, ParameterizedQuantizeOpTest,
                         ::testing::Values(-1, 1, 3));

TEST_F(QuantizedOpTest, QuantizeV2Qint8ScaledSmallInputRange) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<qint8>::v())
                   .Attr("mode", "SCALED")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({3}), {-0.064, 0.0, 0.127});
  AddInputFromArray<float>(TensorShape({1}), {-0.064f});
  AddInputFromArray<float>(TensorShape({1}), {0.127f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT8, TensorShape({3}));
  // Input element 0.0 should map to 0.
  // Input element 2.0 should map to 127, max value of qint8.
  test::FillValues<qint8>(&expected, {-64, 0, 127});
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));

  Tensor expected_output_min(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected_output_min, {-0.128});
  test::ExpectTensorEqual<float>(expected_output_min, *GetOutput(1));

  Tensor expected_output_max(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected_output_max, {0.127});
  test::ExpectTensorEqual<float>(expected_output_max, *GetOutput(2));
}

TEST_F(QuantizedOpTest, QuantizeV2Qint8ScaledRoundToEven) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<qint8>::v())
                   .Attr("mode", "SCALED")
                   .Attr("round_mode", "HALF_TO_EVEN")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({7}),
                           {-126.5, 0.0, 1.0, 2.5, 3.5, 64.0, 127.0});
  AddInputFromArray<float>(TensorShape({1}), {-128.0f});
  AddInputFromArray<float>(TensorShape({1}), {-128.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT8, TensorShape({7}));
  // Input element 0.0 should map to 0.
  // Input element 127.0 maps to 127.
  test::FillValues<qint8>(&expected, {-126, 0, 1, 2, 4, 64, 127});
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));

  Tensor expected_output_min(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected_output_min, {-128.0});
  test::ExpectTensorEqual<float>(expected_output_min, *GetOutput(1));

  Tensor expected_output_max(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected_output_max, {127.0});
  test::ExpectTensorEqual<float>(expected_output_max, *GetOutput(2));
}

TEST_F(QuantizedOpTest, QuantizeV2Qint8ScaledRoundAwayFromZero) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<qint8>::v())
                   .Attr("mode", "SCALED")
                   .Attr("round_mode", "HALF_AWAY_FROM_ZERO")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({7}),
                           {-126.5, 0.0, 1.0, 2.5, 3.5, 64.0, 127.0});
  AddInputFromArray<float>(TensorShape({1}), {-128.0f});
  AddInputFromArray<float>(TensorShape({1}), {-128.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT8, TensorShape({7}));
  // Input element 0.0 should map to 0.
  // Input element 127.0 maps to 127.
  test::FillValues<qint8>(&expected, {-127, 0, 1, 3, 4, 64, 127});
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));

  Tensor expected_output_min(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected_output_min, {-128.0});
  test::ExpectTensorEqual<float>(expected_output_min, *GetOutput(1));

  Tensor expected_output_max(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected_output_max, {127.0});
  test::ExpectTensorEqual<float>(expected_output_max, *GetOutput(2));
}

TEST_F(QuantizedOpTest, QuantizeV2_32Bit) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<qint32>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const int element_count = 8;
  AddInputFromArray<float>(
      TensorShape({element_count}),
      {-500.0f, 0.0f, 1.0f, 1.25f, 1.75f, 127.0f, 255.0f, 500.0f});
  AddInputFromArray<float>(TensorShape({1}), {-256.0f});
  AddInputFromArray<float>(TensorShape({1}), {256.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({element_count}));
  test::FillValues<qint32>(&expected,
                           {
                               std::numeric_limits<int32>::min(),
                               0,
                               static_cast<int32>(1.0f * (1 << 23)),
                               static_cast<int32>(1.25f * (1 << 23)),
                               static_cast<int32>(1.75f * (1 << 23)),
                               static_cast<int32>(127.0f * (1 << 23)),
                               static_cast<int32>(255.0f * (1 << 23)),
                               std::numeric_limits<int32>::max(),
                           });
  // We expect there will be some fuzziness in the lower bits, since this is
  // converting from float.
  const int64 epsilon = 1 << 8;
  const qint32* output_data = GetOutput(0)->flat<qint32>().data();
  const qint32* expected_data = expected.flat<qint32>().data();
  for (int i = 0; i < element_count; ++i) {
    const int64 delta = output_data[i] - expected_data[i];
    EXPECT_GT(epsilon, std::abs(delta))
        << "output_data[" << i << "]=" << output_data[i] << ", expected_data["
        << i << "]=" << expected_data[i] << ", delta=" << delta;
  }
}

TEST_F(QuantizedOpTest, QuantizeV2Ports) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({6}),
                           {1.0, 1.25, 1.75, 127.0, 255.0, 500.0});
  AddInputFromArray<float>(TensorShape({1}), {0});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({6}));
  test::FillValues<quint8>(&expected, {1, 1, 2, 127, 255, 255});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  EXPECT_NEAR(0.0f, output_min, 1e-5f);
  EXPECT_NEAR(255.0f, output_max, 1e-5f);
}

TEST_F(QuantizedOpTest, QuantizeV2EqualRange) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({6}), {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  AddInputFromArray<float>(TensorShape({1}), {0.0f});
  AddInputFromArray<float>(TensorShape({1}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({6}));
  test::FillValues<quint8>(&expected, {0, 0, 0, 0, 0, 0});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  EXPECT_NEAR(0.0f, output_min, 1e-5f);
  EXPECT_LT(0.0f, output_max);
}

TEST_F(QuantizedOpTest, QuantizeV2MovesMinToIncludeZero) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({3}), {0.1, 0.2, 0.3});
  AddInputFromArray<float>(TensorShape({1}), {0.1});
  AddInputFromArray<float>(TensorShape({1}), {0.3});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({3}));
  test::FillValues<quint8>(&expected, {85, 170, 255});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  EXPECT_NEAR(0.0f, output_min, 1e-5f);
  EXPECT_NEAR(0.3f, output_max, 1e-5f);
}

TEST_F(QuantizedOpTest, QuantizeV2MovesMaxToIncludeZero) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({3}), {-0.1, -0.2, -0.3});
  AddInputFromArray<float>(TensorShape({1}), {-0.3});
  AddInputFromArray<float>(TensorShape({1}), {-0.1});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QUINT8, TensorShape({3}));
  test::FillValues<quint8>(&expected, {170, 85, 0});
  test::ExpectTensorEqual<quint8>(expected, *GetOutput(0));
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  EXPECT_NEAR(-0.3f, output_min, 1e-5f);
  EXPECT_NEAR(0.0f, output_max, 1e-5f);
}

TEST_F(QuantizedOpTest, Dequantize) {
  TF_ASSERT_OK(NodeDefBuilder("dequantize_op", "Dequantize")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<quint8>(TensorShape({6}), {1, 2, 4, 8, 16, 255});
  AddInputFromArray<float>(TensorShape({1}), {0});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({6}));
  test::FillValues<float>(&expected, {1.0, 2.0, 4.0, 8.0, 16.0, 255.0});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.5);
}

TEST_F(QuantizedOpTest, QuantizeV2DisableEnsureMinimumRange) {
  TF_ASSERT_OK(NodeDefBuilder("quantize_op", "QuantizeV2")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<qint8>::v())
                   .Attr("mode", "MIN_FIRST")
                   .Attr("ensure_minimum_range", 0.0f)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({3}), {-0.000001, 0.0, 0.000042});
  AddInputFromArray<float>(TensorShape({1}), {-0.000128});
  AddInputFromArray<float>(TensorShape({1}), {0.000127});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT8, TensorShape({3}));
  test::FillValues<qint8>(&expected, {-1, 0, 42});
  for (int i = 0; i < 3; ++i) {
    LOG(INFO) << GetOutput(0)->flat<qint8>()(i);
  }
  test::ExpectTensorEqual<qint8>(expected, *GetOutput(0));
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  LOG(INFO) << "output_min = " << output_min;
  LOG(INFO) << "output_max = " << output_max;
  EXPECT_NEAR(-0.000128f, output_min, 1e-7f);
  EXPECT_NEAR(0.000127, output_max, 1e-7f);
}

}  // end namespace tensorflow
