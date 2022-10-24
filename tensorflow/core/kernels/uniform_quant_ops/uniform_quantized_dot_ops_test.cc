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
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {

class UniformQuantizedDotTest : public OpsTestBase {
 protected:
};

TEST_F(UniformQuantizedDotTest, PerTensorQuantized) {
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformQuantizedDot")
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tin", DT_QINT8)
          .Attr("Tout", DT_QINT32)
          .Attr("lhs_quantization_min_val", -128)
          .Attr("lhs_quantization_max_val", 127)
          .Attr("rhs_quantization_min_val", -128)
          .Attr("rhs_quantization_max_val", 127)
          .Attr("output_quantization_min_val",
                static_cast<int32_t>(-2147483648))
          .Attr("output_quantization_max_val", static_cast<int32_t>(2147483647))
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // lhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 2}), {1, 2, 3, 4});
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // lhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {0.5});
  AddInputFromArray<int32>(TensorShape({}), {1});
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {2.0});
  AddInputFromArray<int32>(TensorShape({}), {2});
  // output scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {0.25});
  AddInputFromArray<int32>(TensorShape({}), {-20});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 3}));
  // Dequantized output [(output + 20) * 0.25] should be equal to
  // [(lhs - 1) * 0.5] * [(rhs - 2) * 2.0]
  test::FillValues<qint32>(&expected, {-12, -8, -4, -4, 16, 36});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedDotTest, PerChannelQuantized) {
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformQuantizedDot")
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tin", DT_QINT8)
          .Attr("Tout", DT_QINT32)
          .Attr("lhs_quantization_min_val", -128)
          .Attr("lhs_quantization_max_val", 127)
          .Attr("rhs_quantization_min_val", -128)
          .Attr("rhs_quantization_max_val", 127)
          .Attr("rhs_quantization_axis", 1)
          .Attr("output_quantization_min_val",
                static_cast<int32_t>(-2147483648))
          .Attr("output_quantization_max_val", static_cast<int32_t>(2147483647))
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // lhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 2}), {1, 2, 3, 4});
  // rhs (per-channel quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 3}), {1, 4, 3, 4, 7, 6});
  // lhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {0.5});
  AddInputFromArray<int32>(TensorShape({}), {1});
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({3}), {2.0, 4.0, 2.0});
  AddInputFromArray<int32>(TensorShape({3}), {2, 4, 2});
  // output scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {0.25});
  AddInputFromArray<int32>(TensorShape({}), {-20});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 3}));
  // Dequantized output [(output + 20) * 0.25] should be equal to
  // [(lhs - 1) * 0.5] * [per-channel dequantized rhs]
  test::FillValues<qint32>(&expected, {-12, 4, -4, -4, 52, 36});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedDotTest, PerTensorQuantizedEffectiveMultiplierOne) {
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformQuantizedDot")
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tin", DT_QINT8)
          .Attr("Tout", DT_QINT32)
          .Attr("lhs_quantization_min_val", -128)
          .Attr("lhs_quantization_max_val", 127)
          .Attr("rhs_quantization_min_val", -128)
          .Attr("rhs_quantization_max_val", 127)
          .Attr("output_quantization_min_val",
                static_cast<int32_t>(-2147483648))
          .Attr("output_quantization_max_val", static_cast<int32_t>(2147483647))
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // lhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 2}), {1, 2, 3, 4});
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // lhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {0.5});
  AddInputFromArray<int32>(TensorShape({}), {1});
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {0.5});
  AddInputFromArray<int32>(TensorShape({}), {2});
  // output scales and zero_points, where
  // output_scalar_scale = lhs_scalar_scale * rhs_scalar_scale
  AddInputFromArray<float>(TensorShape({}), {0.25});
  AddInputFromArray<int32>(TensorShape({}), {-4});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 3}));
  // Dequantized output [(output + 4) * 0.25] should be equal to
  // [(lhs - 1) * 0.5] * [(rhs - 2) * 2.0]
  test::FillValues<qint32>(&expected, {-2, -1, 0, 0, 5, 10});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedDotTest, PerChannelQuantizedEffectiveMultiplierOne) {
  TF_ASSERT_OK(
      NodeDefBuilder("test", "UniformQuantizedDot")
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_INT32))
          .Attr("Tin", DT_QINT8)
          .Attr("Tout", DT_QINT32)
          .Attr("lhs_quantization_min_val", -128)
          .Attr("lhs_quantization_max_val", 127)
          .Attr("rhs_quantization_min_val", -128)
          .Attr("rhs_quantization_max_val", 127)
          .Attr("rhs_quantization_axis", 1)
          .Attr("output_quantization_min_val",
                static_cast<int32_t>(-2147483648))
          .Attr("output_quantization_max_val", static_cast<int32_t>(2147483647))
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // lhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 2}), {1, 2, 3, 4});
  // rhs (per-channel quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // lhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {0.5});
  AddInputFromArray<int32>(TensorShape({}), {1});
  // rhs scales and zero_points.
  AddInputFromArray<float>(TensorShape({3}), {0.5, 1.0, 0.5});
  AddInputFromArray<int32>(TensorShape({3}), {2, 4, 2});
  // output scales and zero_points, where
  // [output_scales] = lhs_scalar_scale * [rhs_scales]
  AddInputFromArray<float>(TensorShape({3}), {0.25, 0.5, 0.25});
  AddInputFromArray<int32>(TensorShape({3}), {4, 8, 4});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 3}));
  // Per-channel dequantized output should be equal to
  // [(lhs - 1) * 0.5] * [per-channel dequantized rhs]
  test::FillValues<qint32>(&expected, {6, 9, 8, 8, 7, 18});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

TEST_F(UniformQuantizedDotTest, HybridPerTensorQuantized) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedDotHybrid")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tlhs", DT_FLOAT)
                   .Attr("Trhs", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("rhs_quantization_min_val", -128)
                   .Attr("rhs_quantization_max_val", 127)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 2}), {-32.2, -12.1, 10.7, 11.6});
  // rhs (quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // rhs output scales and zero_points.
  AddInputFromArray<float>(TensorShape({}), {2.0});
  AddInputFromArray<int32>(TensorShape({}), {2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  // output should be similar to
  // dot(lhs, [(rhs - 2) * 2.0])
  test::FillValues<float>(&expected, {16.0, -72.6, -161.2, 25.0, 69.6, 114.2});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/0.1, /*rtol=*/0.01);
}

TEST_F(UniformQuantizedDotTest, HybridPerChannelQuantized) {
  TF_ASSERT_OK(NodeDefBuilder("test", "UniformQuantizedDotHybrid")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_INT32))
                   .Attr("Tlhs", DT_FLOAT)
                   .Attr("Trhs", DT_QINT8)
                   .Attr("Tout", DT_FLOAT)
                   .Attr("rhs_quantization_min_val", -128)
                   .Attr("rhs_quantization_max_val", 127)
                   .Attr("rhs_quantization_axis", 1)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // lhs tensor.
  AddInputFromArray<float>(TensorShape({2, 2}), {-32.2, -12.1, 10.7, 11.6});
  // rhs (per-channel quantized) tensor.
  AddInputFromArray<qint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // rhs output scales and zero_points.
  AddInputFromArray<float>(TensorShape({3}), {2.0, 4.0, 2.0});
  AddInputFromArray<int32>(TensorShape({3}), {2, 4, 2});

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  // output should be similar to
  // dot(lhs, [per-channel dequantized rhs])
  test::FillValues<float>(&expected, {16.0, 209.2, -161.2, 25.0, -39.2, 114.2});
  test::ExpectClose(expected, *GetOutput(0), /*atol=*/0.1, /*rtol=*/0.01);
}

}  // namespace tensorflow
