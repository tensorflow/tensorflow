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

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

using ::tensorflow::testing::StatusIs;
using ::testing::MatchesRegex;

class QuantizeAndDequantizeTest : public OpsTestBase {};

struct ParameterizedQuantizeAndDequantizeTest
    : public OpsTestBase,
      public ::testing::WithParamInterface<int> {};

// Convert a simple scalar tensor.
TEST_F(QuantizeAndDequantizeTest, Convert_scalar_tensor) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV2")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", true)
          .Attr("num_bits", 8)
          .Attr("range_given", false)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1}), {-3.5});
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Max

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1}));
  test::FillValues<float>(&expected, {-3.5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));

  // Ensure that the inputs haven't been changed.
  EXPECT_EQ(inputs_[1]->scalar<float>()(), 0.0);
  EXPECT_EQ(inputs_[2]->scalar<float>()(), 0.0);
}

TEST_F(QuantizeAndDequantizeTest, Convert_scalar_tensor_V3) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV3")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("signed_input", true)
          .Attr("range_given", false)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1}), {-3.5});
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Max
  AddInputFromArray<int32>(TensorShape({}), {8});    // num_bits

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1}));
  test::FillValues<float>(&expected, {-3.5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));

  // Ensure that the inputs haven't been changed.
  EXPECT_EQ(inputs_[1]->scalar<float>()(), 0.0);
  EXPECT_EQ(inputs_[2]->scalar<float>()(), 0.0);
}

// Creates a tensor with the specified dims, using values chosen from data,
// multiplied by (1 + index) along the axis dimension.
template <typename T>
std::vector<T> ScalePerSliceAlongAxis(std::vector<int64_t> dims, int axis,
                                      const std::vector<T>& data) {
  uint32 seed = 123;
  int64_t out_size = 1;
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
    int in_idx = rand_r(&seed) % data.size();
    int multiplier = ((out_idx / minor_size) % num_slices) + 1;
    out[out_idx] = data[in_idx] * multiplier;
  }
  return out;
}

// Convert a 1D tensor with signed 8 bits.
TEST_P(ParameterizedQuantizeAndDequantizeTest, Convert_4D_tensor_with_int8) {
  const int axis = GetParam();
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV2")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", true)
          .Attr("num_bits", 8)
          .Attr("range_given", false)
          .Attr("axis", axis)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const std::vector<int64_t> dims = {2, 3, 4, 5};
  // Each slice contains the same 7 values multiplied by (slice_idx + 1).
  AddInputFromArray<float>(
      TensorShape(dims),
      ScalePerSliceAlongAxis<float>(
          dims, axis, {-1, -0.5, 0, 0.3, 0.8, 0.555, 0.50390625}));

  const int num_slices = (axis == -1) ? 1 : dims[axis];
  const TensorShape range_shape =
      (axis == -1) ? TensorShape({}) : TensorShape({num_slices});
  std::vector<float> init_value(num_slices, 0.0f);
  AddInputFromArray<float>(range_shape, init_value);  // Min
  AddInputFromArray<float>(range_shape, init_value);  // Max

  // With int8, the values in the tensor are quantized to
  // {-128, -64, 0, 38, 102, 71, 64}.
  // Scale is: (slice_idx + 1) / 128
  // Then it is dequantized to:
  //    (slice_idx + 1) * {-1, -0.5, 0, 38.0/128, 102.0/128, 71.0/128, 0.5}
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape(dims));
  test::FillValues<float>(
      &expected,
      ScalePerSliceAlongAxis<float>(
          dims, axis, {-1, -0.5, 0, 38.0 / 128, 102.0 / 128, 71.0 / 128, 0.5}));

  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);

  // Ensure that the inputs haven't been changed.
  for (int slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
    EXPECT_EQ(inputs_[1]->flat<float>()(slice_idx), 0.0);
    EXPECT_EQ(inputs_[2]->flat<float>()(slice_idx), 0.0);
  }
}

// Convert a 1D tensor with signed 8 bits and round_mode half_up.
TEST_P(ParameterizedQuantizeAndDequantizeTest,
       Convert_4D_tensor_with_int8_round_half_up) {
  const int axis = GetParam();
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV2")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", true)
          .Attr("num_bits", 8)
          .Attr("range_given", false)
          .Attr("round_mode", "HALF_UP")
          .Attr("axis", axis)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const std::vector<int64_t> dims = {5, 7, 11, 13};
  // Each slice contains the same 7 values multiplied by (slice_idx + 1).
  AddInputFromArray<float>(
      TensorShape(dims),
      ScalePerSliceAlongAxis<float>(
          dims, axis, {-1, -0.5, 0, 0.3, 0.8, 0.555, 0.50390625}));

  const int num_slices = (axis == -1) ? 1 : dims[axis];
  const TensorShape range_shape =
      (axis == -1) ? TensorShape({}) : TensorShape({num_slices});
  std::vector<float> init_value(num_slices, 0.0f);
  AddInputFromArray<float>(range_shape, init_value);  // Min
  AddInputFromArray<float>(range_shape, init_value);  // Max

  // With int8, the values in the tensor are quantized to
  // {-128, -64, 0, 38, 102, 71, 65}.
  // Scale is: (slice_idx + 1) / 128
  // Then it is dequantized to:
  //   (slice_idx + 1) * {-1, -0.5, 0, 38.0/128, 102.0/128, 71.0/128, 65.0 /128}
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape(dims));
  test::FillValues<float>(&expected, ScalePerSliceAlongAxis<float>(
                                         dims, axis,
                                         {-1, -0.5, 0, 38.0 / 128, 102.0 / 128,
                                          71.0 / 128, 65.0 / 128}));
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);

  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);

  // Ensure that the inputs haven't been changed.
  for (int slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
    EXPECT_EQ(inputs_[1]->flat<float>()(slice_idx), 0.0);
    EXPECT_EQ(inputs_[2]->flat<float>()(slice_idx), 0.0);
  }
}

// Convert a 1D tensor with signed 8 bits and round_mode half_up, using
// narrow range quantization.
TEST_P(ParameterizedQuantizeAndDequantizeTest,
       Convert_4D_tensor_with_int8_round_half_up_narrow_range) {
  const int axis = GetParam();
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV2")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", true)
          .Attr("num_bits", 8)
          .Attr("range_given", false)
          .Attr("round_mode", "HALF_UP")
          .Attr("narrow_range", true)
          .Attr("axis", axis)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const std::vector<int64_t> dims = {2, 3, 4, 5};
  // Each slice contains the same 7 values multiplied by (slice_idx + 1).
  AddInputFromArray<float>(
      TensorShape(dims),
      ScalePerSliceAlongAxis<float>(
          dims, axis, {-1, -0.5, 0, 0.3, 0.8, 0.555, 0.50390625}));

  const int num_slices = (axis == -1) ? 1 : dims[axis];
  const TensorShape range_shape =
      (axis == -1) ? TensorShape({}) : TensorShape({num_slices});
  std::vector<float> init_value(num_slices, 0.0f);
  AddInputFromArray<float>(range_shape, init_value);  // Min
  AddInputFromArray<float>(range_shape, init_value);  // Max

  // With int8, the values in the tensor are quantized to
  // {-127, -63, 0, 38, 102, 70, 64}.
  // Scale is: (slice_idx + 1) / 127
  // Then it is dequantized to:
  //    (slice_idx + 1) * {-1, -63.0/127, 0, 38.0/127, 102.0/127, 70/127,
  //    64/127}
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape(dims));
  test::FillValues<float>(
      &expected,
      ScalePerSliceAlongAxis<float>(dims, axis,
                                    {-1, -63.0 / 127, 0, 38.0 / 127,
                                     102.0 / 127, 70.0 / 127, 64.0 / 127}));
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);

  // Ensure that the inputs haven't been changed.
  for (int slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
    EXPECT_EQ(inputs_[1]->flat<float>()(slice_idx), 0.0);
    EXPECT_EQ(inputs_[2]->flat<float>()(slice_idx), 0.0);
  }
}

// Convert a 1D tensor with signed 8 bits.
TEST_F(QuantizeAndDequantizeTest, Convert_1D_tensor_with_int8_V3) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV3")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("signed_input", true)
          .Attr("range_given", false)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({6}), {-1, -0.5, 0, 0.3, 0.8, 0.555});
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Max
  AddInputFromArray<int32>(TensorShape({}), {8});    // num_bits

  // With int8, the tensor is quantized to {-128, -64, 0, 38, 102, 71}.
  // Scale is: 1/128
  // Then it is dequantized to {-1, -64.0/128, 0, 38.0/128, 102.0/128, 71.0/128}
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({6}));
  test::FillValues<float>(&expected,
                          {-1, -0.5, 0, 38.0 / 128, 102.0 / 128, 71.0 / 128});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);

  // Ensure that the inputs haven't been changed.
  EXPECT_EQ(inputs_[1]->scalar<float>()(), 0.0);
  EXPECT_EQ(inputs_[2]->scalar<float>()(), 0.0);
}

// Convert a 1D tensor with signed 8 bits, using narrow range quantization.
TEST_P(ParameterizedQuantizeAndDequantizeTest,
       Convert_4D_tensor_with_int8_narrow_range_V3) {
  const int axis = GetParam();
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV3")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("signed_input", true)
          .Attr("range_given", false)
          .Attr("narrow_range", true)
          .Attr("axis", axis)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const std::vector<int64_t> dims = {2, 3, 4, 5};
  // Each slice contains the same 7 values multiplied by (slice_idx + 1).
  AddInputFromArray<float>(
      TensorShape(dims),
      ScalePerSliceAlongAxis<float>(
          dims, axis, {-1, -0.5, 0, 0.3, 0.8, 0.555, 0.50390625}));

  const int num_slices = (axis == -1) ? 1 : dims[axis];
  const TensorShape range_shape =
      (axis == -1) ? TensorShape({}) : TensorShape({num_slices});
  std::vector<float> init_value(num_slices, 0.0f);
  AddInputFromArray<float>(range_shape, init_value);  // Min
  AddInputFromArray<float>(range_shape, init_value);  // Max
  AddInputFromArray<int32>(TensorShape({}), {8});     // num_bits

  // With int8, the values in the tensor are quantized to
  // {-127, -63, 0, 38, 102, 70, 64}.
  // Scale is: (slice_idx + 1) / 127
  // Then it is dequantized to:
  //   (slice_idx + 1) * {-1, -63.0/127, 0, 38.0/127, 102.0/127, 70/127, 64/127}

  // With int8, each slice of the tensor is quantized to
  // {-127, -64, 0, 38, 102, 70, 64}.
  // Scale is: (slice_idx + 1) / 127
  // Then it is dequantized to:
  //   (slice_idx + 1) * {-1, -64.0/127, 0, 38.0/127, 102.0/127, 70/127, 64/127}
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape(dims));
  test::FillValues<float>(
      &expected,
      ScalePerSliceAlongAxis<float>(dims, axis,
                                    {-1, -64.0 / 127, 0, 38.0 / 127,
                                     102.0 / 127, 70.0 / 127, 64.0 / 127}));
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);

  // Ensure that the inputs haven't been changed.
  for (int slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
    EXPECT_EQ(inputs_[1]->flat<float>()(slice_idx), 0.0);
    EXPECT_EQ(inputs_[2]->flat<float>()(slice_idx), 0.0);
  }
}

// Verifies the Gradient.
TEST_P(ParameterizedQuantizeAndDequantizeTest, GradientV4_op) {
  const int axis = GetParam();
  TF_ASSERT_OK(NodeDefBuilder("qdq_v4_grad_op", "QuantizeAndDequantizeV4Grad")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("axis", axis)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const std::vector<int64_t> dims = {2, 3, 4, 5};
  // Input gradient. (repeating 11 values multiplied by (slice_idx + 1))
  auto gradients = ScalePerSliceAlongAxis<float>(
      dims, axis, {1, -2, -3, 4, 5, 6, -7, -8, -9, -10, 11});
  AddInputFromArray<float>(TensorShape(dims), gradients);
  // Forward op inputs. (repeating 7 values multiplied by (slice_idx + 1)).
  auto inputs = ScalePerSliceAlongAxis<float>(
      dims, axis, {-1, -0.5, 0, 0.3, 0.8, 0.55, 0.6});
  AddInputFromArray<float>(TensorShape(dims), inputs);
  const int num_slices = (axis == -1) ? 1 : dims[axis];
  const TensorShape range_shape =
      (axis == -1) ? TensorShape({}) : TensorShape({num_slices});
  std::vector<float> input_min_values(num_slices), input_max_values(num_slices);
  for (int i = 0; i < num_slices; ++i) {
    input_max_values[i] = 0.8f + i * 0.4f;
    input_min_values[i] = -input_max_values[i];
  }
  AddInputFromArray<float>(range_shape, input_min_values);
  AddInputFromArray<float>(range_shape, input_max_values);
  std::vector<float> expected_vals(inputs.size());
  int minor_size = 1;
  for (int i = axis + 1; i < dims.size(); ++i) {
    minor_size *= dims[i];
  }
  for (int i = 0; i < inputs.size(); ++i) {
    int slice_idx = (i / minor_size) % num_slices;
    expected_vals[i] = ((inputs[i] >= input_min_values[slice_idx]) &&
                        (inputs[i] <= input_max_values[slice_idx]))
                           ? gradients[i]
                           : 0;
  }
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape(dims));
  test::FillValues<float>(&expected, expected_vals);
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Instantiate parameterized tests for axis = -1, 1, 3.
INSTANTIATE_TEST_SUITE_P(All, ParameterizedQuantizeAndDequantizeTest,
                         ::testing::Values(-1, 1, 3));

// Convert a 1D tensor with signed 4 bits.
TEST_F(QuantizeAndDequantizeTest, Convert_1D_tensor_with_int4) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV2")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", true)
          .Attr("num_bits", 4)
          .Attr("range_given", false)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({6}), {-1, -0.5, 0, 0.3125, 0.8, 0.555});
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Max

  // With int4, the tensor is quantized to {-8, -4, 0, 2, 6, 4}.
  // Scale is: 1/8
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({6}));
  test::FillValues<float>(&expected, {-1, -0.5, 0, 0.25, 0.75, 0.5});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);

  // Ensure that the inputs haven't been changed.
  EXPECT_EQ(inputs_[1]->scalar<float>()(), 0.0);
  EXPECT_EQ(inputs_[2]->scalar<float>()(), 0.0);
}

// Convert a 1D tensor with signed 4 bits and round_mode hafl_up.
TEST_F(QuantizeAndDequantizeTest, Convert_1D_tensor_with_int4_round_half_up) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV2")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", true)
          .Attr("num_bits", 4)
          .Attr("range_given", false)
          .Attr("round_mode", "HALF_UP")
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({6}), {-1, -0.5, 0, 0.3125, 0.8, 0.555});
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Max

  // With int4, the tensor is quantized to {-8, -4, 0, 3, 6, 4}.
  // Scale is: 1/8
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({6}));
  test::FillValues<float>(&expected, {-1, -0.5, 0, 0.375, 0.75, 0.5});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);

  // Ensure that the inputs haven't been changed.
  EXPECT_EQ(inputs_[1]->scalar<float>()(), 0.0);
  EXPECT_EQ(inputs_[2]->scalar<float>()(), 0.0);
}

// Convert a 1D tensor with signed 4 bits.
TEST_F(QuantizeAndDequantizeTest, Convert_1D_tensor_with_int4_V3) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV3")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("signed_input", true)
          .Attr("range_given", false)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({6}), {-1, -0.5, 0, 0.3, 0.8, 0.555});
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Max
  AddInputFromArray<int32>(TensorShape({}), {4});    // num_bits

  // With int4, the tensor is quantized to {-8, -4, 0, 2, 6, 4}.
  // Scale is: 1/8
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({6}));
  test::FillValues<float>(&expected, {-1, -0.5, 0, 0.25, 0.75, 0.5});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);

  // Ensure that the inputs haven't been changed.
  EXPECT_EQ(inputs_[1]->scalar<float>()(), 0.0);
  EXPECT_EQ(inputs_[2]->scalar<float>()(), 0.0);
}

// Convert a 2D tensor with signed 8 bits with given range.
TEST_F(QuantizeAndDequantizeTest, Convert_2D_tensor_with_int8_range_given) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV2")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", true)
          .Attr("num_bits", 8)
          .Attr("range_given", true)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // Note that the last two values are saturated.
  AddInputFromArray<float>(TensorShape({2, 4}),
                           {-0.8, -0.5, 0, 0.3, 0.8, 0.555, -2, 33});
  AddInputFromArray<float>(TensorShape({}), {-1.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {1.0});   // Max

  // Note that the range is given as [-1, 1].
  // With int8, the tensor is quantized to {-102, -64, 0, 38, 102, 70, -128,
  // 127}.
  // Scale is: 1/127
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 4}));
  test::FillValues<float>(
      &expected, {-102.0 / 127, -64.0 / 127, 0, 38.0 / 127, 102.0 / 127,
                  70.0 / 127, -128.0 / 127, 1});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Convert a 2D tensor with signed 8 bits, given range and round_mode half_up.
TEST_F(QuantizeAndDequantizeTest,
       Convert_2D_tensor_with_int8_range_given_round_half_up) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV2")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", true)
          .Attr("num_bits", 8)
          .Attr("range_given", true)
          .Attr("round_mode", "HALF_UP")
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // Note that the last two values are saturated.
  AddInputFromArray<float>(TensorShape({2, 4}),
                           {-0.8, -0.5, 0, 0.3, 0.8, 0.555, -2, 33});
  AddInputFromArray<float>(TensorShape({}), {-1.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {1.0});   // Max

  // Note that the range is given as [-1, 1].
  // With int8, the tensor is quantized to {-102, -63, 0, 38, 102, 70, -128,
  // 127}.
  // Scale is: 1/127
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 4}));
  test::FillValues<float>(
      &expected, {-102.0 / 127, -63.0 / 127, 0, 38.0 / 127, 102.0 / 127,
                  70.0 / 127, -128.0 / 127, 1});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Convert a 2D tensor with signed 8 bits with given range.
TEST_F(QuantizeAndDequantizeTest, Convert_2D_tensor_with_int8_range_given_V3) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV3")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("signed_input", true)
          .Attr("range_given", true)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // Note that the last two values are saturated.
  AddInputFromArray<float>(TensorShape({2, 4}),
                           {-0.8, -0.5, 0, 0.3, 0.8, 0.555, -2, 33});
  AddInputFromArray<float>(TensorShape({}), {-1.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {1.0});   // Max
  AddInputFromArray<int32>(TensorShape({}), {8});     // num_bits

  // Note that the range is given as [-1, 1].
  // With int8, the tensor is quantized to {-102, -64, 0, 38, 102, 70, -128,
  // 127}.
  // Scale is: 1/127
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 4}));
  test::FillValues<float>(
      &expected, {-102.0 / 127, -64.0 / 127, 0, 38.0 / 127, 102.0 / 127,
                  70.0 / 127, -128.0 / 127, 1});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Convert a 4D tensor with unsigned 8 bits with given range.
TEST_F(QuantizeAndDequantizeTest, Convert_4D_tensor_with_uint8_range_given) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV2")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", false)
          .Attr("num_bits", 8)
          .Attr("range_given", true)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({2, 2, 1, 1}), {-0.5, 0, 0.3, 0.8});
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {1.0});  // Max

  // Note that the range is given as [0, 1].
  // With int8, the tensor is quantized to {0, 0, 76, 204}
  // Scale is: 1/255
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 1, 1}));
  test::FillValues<float>(&expected, {0, 0, 76.0 / 255, 204.0 / 255});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Convert a 4D tensor with unsigned 8 bits, given range and round_mode half_up.
TEST_F(QuantizeAndDequantizeTest,
       Convert_4D_tensor_with_uint8_range_given_round_half_up) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV2")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", false)
          .Attr("num_bits", 8)
          .Attr("range_given", true)
          .Attr("round_mode", "HALF_UP")
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({2, 2, 1, 1}), {-0.5, 0, 0.3, 0.8});
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {1.0});  // Max

  // Note that the range is given as [0, 1].
  // With int8, the tensor is quantized to {0, 0, 77, 204}
  // Scale is: 1/255
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 1, 1}));
  test::FillValues<float>(&expected, {0, 0, 77.0 / 255, 204.0 / 255});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Convert a 4D tensor with unsigned 8 bits with given range.
TEST_F(QuantizeAndDequantizeTest, Convert_4D_tensor_with_uint8_range_given_V3) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV3")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("signed_input", false)
          .Attr("range_given", true)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({2, 2, 1, 1}), {-0.5, 0, 0.3, 0.8});
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {1.0});  // Max
  AddInputFromArray<int32>(TensorShape({}), {8});    // num_bits

  // Note that the range is given as [0, 1].
  // With int8, the tensor is quantized to {0, 0, 76, 204}
  // Scale is: 1/255
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 1, 1}));
  test::FillValues<float>(&expected, {0, 0, 76.0 / 255, 204.0 / 255});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Convert a tensor with all 0.
TEST_F(QuantizeAndDequantizeTest, Convert_tensor_with_all_0) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV2")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("signed_input", false)
          .Attr("num_bits", 8)
          .Attr("range_given", false)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({2, 2, 1, 1}), {0, 0, 0, 0});
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Max

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 1, 1}));
  test::FillValues<float>(&expected, {0, 0, 0, 0});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Convert a tensor with all 0.
TEST_F(QuantizeAndDequantizeTest, Convert_tensor_with_all_0_V3) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_op", "QuantizeAndDequantizeV3")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("signed_input", false)
          .Attr("range_given", false)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({2, 2, 1, 1}), {0, 0, 0, 0});
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Max
  AddInputFromArray<int32>(TensorShape({}), {8});    // num_bits

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2, 1, 1}));
  test::FillValues<float>(&expected, {0, 0, 0, 0});
  test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-5);
}

// Range is invalid
TEST_F(QuantizeAndDequantizeTest, Invalid_range_given) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_Op", "QuantizeAndDequantizeV2")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("num_bits", 8)
          .Attr("range_given", true)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({2, 2, 1, 1}), {-0.5, 0, 0.3, 0.8});
  AddInputFromArray<float>(TensorShape({}), {1.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Max

  absl::Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(),
                                "Invalid range: input_min 1 > input_max 0"))
      << s;
}

// Range is invalid
TEST_F(QuantizeAndDequantizeTest, Invalid_range_given_V3) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_Op", "QuantizeAndDequantizeV3")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("range_given", true)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({2, 2, 1, 1}), {-0.5, 0, 0.3, 0.8});
  AddInputFromArray<float>(TensorShape({}), {1.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Max
  AddInputFromArray<int32>(TensorShape({}), {8});    // num_bits

  absl::Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(),
                                "Invalid range: input_min 1 > input_max 0"))
      << s;
}

// Axis is invalid
TEST_F(QuantizeAndDequantizeTest, Invalid_axis_given_V3) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantize_and_dequantize_Op", "QuantizeAndDequantizeV3")
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("range_given", false)
          .Attr("axis", static_cast<int32_t>(-2147483648))
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({2, 2, 1, 1}), {-0.5, 0, 0.3, 0.8});
  AddInputFromArray<float>(TensorShape({}), {1.0});  // Min
  AddInputFromArray<float>(TensorShape({}), {0.0});  // Max
  AddInputFromArray<int32>(TensorShape({}), {8});    // num_bits

  EXPECT_THAT(
      RunOpKernel(),
      StatusIs(
          error::INVALID_ARGUMENT,
          MatchesRegex("Axis requested is larger than input dimensions.*")));
}

#define BM_SIMPLE_QUAN_DEQUAN(DEVICE)                                    \
  static void BM_SIMPLE_QUAN_DEQUAN_##DEVICE(                            \
      ::testing::benchmark::State& state) {                              \
    auto root = Scope::NewRootScope().ExitOnError();                     \
    ops::QuantizeAndDequantizeV2(root, -3.5, -3.5, -3.5);                \
    TF_CHECK_OK(root.status());                                          \
    Graph* g = new Graph(OpRegistry::Global());                          \
    TF_CHECK_OK(root.ToGraph(g));                                        \
    test::Benchmark(#DEVICE, g, /*old_benchmark_api*/ false).Run(state); \
  }                                                                      \
  BENCHMARK(BM_SIMPLE_QUAN_DEQUAN_##DEVICE);

BM_SIMPLE_QUAN_DEQUAN(cpu);
BM_SIMPLE_QUAN_DEQUAN(gpu);

}  // namespace
}  // namespace tensorflow
