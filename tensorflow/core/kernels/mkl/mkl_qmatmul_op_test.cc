/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#if defined(INTEL_MKL)
#define EIGEN_USE_THREADS

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class QuantizedMatMulTest : public OpsTestBase {};

// Two small matrices A of type uint8 and B of type int8  are multiplied
// and the result is added with int32 bias
TEST_F(QuantizedMatMulTest, Small_withBias) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantized_mat_mul_op", "_MklQuantizedMatMulWithBias")
          .Input(FakeInput(DT_QUINT8))
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("Toutput", DataTypeToEnum<qint32>::v())
          .Attr("_kernel", "QuantizedMklOp")
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // A matrix is:
  // |  1 |  2 |  3 |
  // |  4 |  5 |  6 |
  AddInputFromArray<quint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // B matrix is:
  // |  7 |  8 |  9 | 10 |
  // | 11 | 12 | 13 | 14 |
  // | 15 | 16 | 17 | 18 |
  AddInputFromArray<qint8>(TensorShape({3, 4}),
                           {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  AddInputFromArray<qint32>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {255.0f});
  AddInputFromArray<float>(TensorShape({}), {-127.0f});
  AddInputFromArray<float>(TensorShape({}), {127.0f});

  TF_ASSERT_OK(RunOpKernel());
  // Here are the results we expect, from hand calculations:
  // (1 * 7) + (2 * 11) + (3 * 15) = 74
  // (1 * 8) + (2 * 12) + (3 * 16) = 80
  // (1 * 9) + (2 * 13) + (3 * 17) = 86
  // (1 * 10) + (2 * 14) + (3 * 18) = 92
  // (4 * 7) + (5 * 11) + (6 * 15) = 173
  // (4 * 8) + (5 * 12) + (6 * 16) = 188
  // (4 * 9) + (5 * 13) + (6 * 17) = 203
  // (4 * 10) + (5 * 14) + (6 * 18) = 218
  // Final result after Bias addition:
  // 74  + 1 = 75 , 80  + 2 = 82 , 86  + 3 = 89 , 92  + 4 = 96,
  // 173 + 1 = 174, 188 + 2 = 190, 203 + 3 = 206, 218 + 4 = 222
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 4}));
  test::FillValues<qint32>(&expected, {75, 82, 89, 96, 174, 190, 206, 222});

  const Tensor& output = *GetOutput(0);
  test::ExpectTensorEqual<qint32>(expected, output);
}

// Two small matrices A of type uint8 and B of type int8  are multiplied
// and the result is added with neg bias as well
TEST_F(QuantizedMatMulTest, Small_withNegBias) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantized_mat_mul_op", "_MklQuantizedMatMulWithBias")
          .Input(FakeInput(DT_QUINT8))
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("Toutput", DataTypeToEnum<qint32>::v())
          .Attr("_kernel", "QuantizedMklOp")
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // A matrix is:
  // |  1 |  2 |  3 |
  // |  4 |  5 |  6 |
  AddInputFromArray<quint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // B matrix is:
  // |  7 |  8 |  9 | 10 |
  // | 11 | 12 | 13 | 14 |
  // | 15 | 16 | 17 | 18 |
  AddInputFromArray<qint8>(TensorShape({3, 4}),
                           {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  AddInputFromArray<qint32>(TensorShape({4}), {100, -200, 300, -400});
  AddInputFromArray<float>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {255.0f});
  AddInputFromArray<float>(TensorShape({}), {-127.0f});
  AddInputFromArray<float>(TensorShape({}), {127.0f});

  TF_ASSERT_OK(RunOpKernel());
  // Here are the results we expect, from hand calculations:
  // (1 * 7) + (2 * 11) + (3 * 15) = 74
  // (1 * 8) + (2 * 12) + (3 * 16) = 80
  // (1 * 9) + (2 * 13) + (3 * 17) = 86
  // (1 * 10) + (2 * 14) + (3 * 18) = 92
  // (4 * 7) + (5 * 11) + (6 * 15) = 173
  // (4 * 8) + (5 * 12) + (6 * 16) = 188
  // (4 * 9) + (5 * 13) + (6 * 17) = 203
  // (4 * 10) + (5 * 14) + (6 * 18) = 218
  // Final result after Bias addition:
  // 74+100=174, 80-200=-120, 86+300=386, 92-400=-308,
  // 173+100=273, 188-200=-12, 203+300=503, 218-400=-182
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 4}));
  test::FillValues<qint32>(&expected,
                           {174, -120, 386, -308, 273, -12, 503, -182});

  const Tensor& output = *GetOutput(0);
  test::ExpectTensorEqual<qint32>(expected, output);
}

// Two small matrices A of type uint8 (converted from signed integer)
// and B of type int8  are multiplied and the result is added with float bias
TEST_F(QuantizedMatMulTest, Small_WithNegInp) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantized_mat_mul_op", "_MklQuantizedMatMulWithBias")
          .Input(FakeInput(DT_QUINT8))
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("Toutput", DataTypeToEnum<qint32>::v())
          .Attr("input_quant_mode", "MIN_FIRST")
          .Attr("_kernel", "QuantizedMklOp")
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // The A matrix is:
  // |  -1 |  -5 |  -9 |
  // |  -2 |  -6 | -10 |
  // |  -3 |  -7 | -11 |
  // |  -4 |  -8 | -12 |
  // The input array only contains unsigned bytes, so we specify the actual
  // quantized values as Au8 = (Af32 - Min(A𝑓32)) * Qa, where
  // Qa = 255/(Max(A𝑓32) - Min(A𝑓32)). For example, -1 is represented
  // as -1 + 12, or 11 as Qa = 255/(243+12).
  AddInputFromArray<quint8>(TensorShape({4, 3}),
                            {11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0});

  // The B matrix is:
  // |   1 |   4|
  // |   2 |   5|
  // |   3 |   6|
  AddInputFromArray<qint8>(TensorShape({3, 2}), {1, 4, 2, 5, 3, 6});
  // Bias
  AddInputFromArray<float>(TensorShape({2}), {10.0f, 20.0f});
  AddInputFromArray<float>(TensorShape({}), {-12.0f});
  AddInputFromArray<float>(TensorShape({}), {243.0f});
  AddInputFromArray<float>(TensorShape({}), {-127.0f});
  AddInputFromArray<float>(TensorShape({}), {127.0f});

  TF_ASSERT_OK(RunOpKernel());
  // First calculate C = A * B,
  // so we expect to get these results for MatMul:
  // 1*-1 + 2*-5 + 3*-9 = -38
  // 4*-1 + 5*-5 + 6*-9 = -83
  // 1*-2 + 2*-6 + 3*-10 = -44
  // 4*-2 + 5*-6 + 6*-10 = -98
  // 1*-3 + 2*-7 + 3*-11 = -50
  // 4*-3 + 5*-7 + 6*-11 = -113
  // 1*-4 + 2*-8 + 3*-12 = -56
  // 4*-4 + 5*-8 + 6*-12 = -128
  // |  -38 |  -83 |
  // |  -44 |  -98 |
  // |  -50 | -113 |
  // |  -56 | -128 |
  // After Bias add {10, 20}, the expected result is
  // |  -28 |  -63 |
  // |  -34 |  -78 |
  // |  -40 |  -93 |
  // |  -46 | -108 |
  Tensor expected(allocator(), DT_QINT32, TensorShape({4, 2}));
  test::FillValues<qint32>(&expected,
                           {-28, -63, -34, -78, -40, -93, -46, -108});

  const Tensor& output = *GetOutput(0);
  test::ExpectTensorEqual<qint32>(expected, output);
}

// Two small matrices A of type uint8 and B of type int8  are multiplied
// and the result is added with int32 bias and Requantization fusion
TEST_F(QuantizedMatMulTest, Small_withBiasAndReq) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_mat_mul_op",
                              "_MklQuantizedMatMulWithBiasAndRequantize")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("Toutput", DataTypeToEnum<quint8>::v())
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // A matrix is:
  // |  1 |  2 |  3 |
  // |  4 |  5 |  6 |
  AddInputFromArray<quint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // B matrix is:
  // |  7 |  8 |  9 | 10 |
  // | 11 | 12 | 13 | 14 |
  // | 15 | 16 | 17 | 18 |
  AddInputFromArray<qint8>(TensorShape({3, 4}),
                           {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  AddInputFromArray<qint32>(TensorShape({4}), {10, -20, 30, -40});
  AddInputFromArray<float>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {255.0f});
  AddInputFromArray<float>(TensorShape({}), {-127.0f});
  AddInputFromArray<float>(TensorShape({}), {127.0f});
  AddInputFromArray<float>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {255.0f});

  TF_ASSERT_OK(RunOpKernel());
  // Here are the results we expect, from hand calculations:
  // (1 * 7) + (2 * 11) + (3 * 15) = 74
  // (1 * 8) + (2 * 12) + (3 * 16) = 80
  // (1 * 9) + (2 * 13) + (3 * 17) = 86
  // (1 * 10) + (2 * 14) + (3 * 18) = 92
  // (4 * 7) + (5 * 11) + (6 * 15) = 173
  // (4 * 8) + (5 * 12) + (6 * 16) = 188
  // (4 * 9) + (5 * 13) + (6 * 17) = 203
  // (4 * 10) + (5 * 14) + (6 * 18) = 218
  // After Bias addition
  // 74+10=84, 80-20=60, 86+30=116, 92-40=52,
  // 173+10=183, 188-20=168, 203+30=233, 218-40=178
  // After Requantize
  // requantscale = scale_int32 / scale_eightbit / static_cast<float>(1 << 23)
  // requantscale = 2^31/255/2^23 ~= 1.00392
  // 84 * 1.00392 ~= 84.329 ~= 84
  // 60 * 1.00392 ~= 60.235 ~= 60
  // 116 * 1.00392 ~= 116.454 ~= 116
  // 52 * 1.00392 ~= 52.203 ~= 52
  // 183 * 1.00392 ~= 183.717 ~= 184
  // 168 * 1.00392 ~= 168.658 ~= 169
  // 233 * 1.00392 ~= 233.913 ~= 234
  // 178 * 1.00392 ~= 178.698 ~= 179

  Tensor expected(allocator(), DT_QUINT8, TensorShape({2, 4}));
  test::FillValues<quint8>(&expected, {84, 60, 116, 52, 184, 169, 234, 179});

  const Tensor& output = *GetOutput(0);
  test::ExpectTensorEqual<quint8>(expected, output);
}

// Two small matrices A of type uint8 and B of type int8  are multiplied
// and the result is added with int32 bias and Requantization fusion
TEST_F(QuantizedMatMulTest, Small_withBiasAndDeq) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_mat_mul_op",
                              "_MklQuantizedMatMulWithBiasAndDequantize")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("Toutput", DataTypeToEnum<float>::v())
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // A matrix is:
  // |  1 |  2 |  3 |
  // |  4 |  5 |  6 |
  AddInputFromArray<quint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // B matrix is:
  // |  7 |  8 |  9 | 10 |
  // | 11 | 12 | 13 | 14 |
  // | 15 | 16 | 17 | 18 |
  AddInputFromArray<qint8>(TensorShape({3, 4}),
                           {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  AddInputFromArray<qint32>(TensorShape({4}), {10, -20, 30, -40});
  AddInputFromArray<float>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {255.0f});
  AddInputFromArray<float>(TensorShape({}), {-127.0f});
  AddInputFromArray<float>(TensorShape({}), {127.0f});
  AddInputFromArray<float>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {255.0f});

  TF_ASSERT_OK(RunOpKernel());
  // Here are the results we expect, from hand calculations:
  // (1 * 7) + (2 * 11) + (3 * 15) = 74
  // (1 * 8) + (2 * 12) + (3 * 16) = 80
  // (1 * 9) + (2 * 13) + (3 * 17) = 86
  // (1 * 10) + (2 * 14) + (3 * 18) = 92
  // (4 * 7) + (5 * 11) + (6 * 15) = 173
  // (4 * 8) + (5 * 12) + (6 * 16) = 188
  // (4 * 9) + (5 * 13) + (6 * 17) = 203
  // (4 * 10) + (5 * 14) + (6 * 18) = 218
  // After Bias addition
  // 74+10=84, 80-20=60, 86+30=116, 92-40=52,
  // 173+10=183, 188-20=168, 203+30=233, 218-40=178
  // After Dequantize
  // requantscale = scale_int32 / static_cast<float>(1 << 31)
  // requantscale = 2^31/2^31 ~= 1.0000
  // 84 * 1.00000 ~= 84
  // 60 * 1.00000 ~=  60
  // 116 * 1.00000 ~= 116
  // 52 * 1.00000 ~= 52
  // 183 * 1.00000 ~= 183
  // 168 * 1.00000 ~= 168
  // 233 * 1.00000 ~= 233
  // 178 * 1.00000 ~= 178

  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 4}));
  test::FillValues<float>(&expected, {84, 60, 116, 52, 183, 168, 233, 178});

  const Tensor& output = *GetOutput(0);
  test::ExpectTensorEqual<float>(expected, output);
}

// Two small matrices A of type uint8 and B of type int8  are multiplied
// and the result is added with float bias and then performed relu on the result
TEST_F(QuantizedMatMulTest, Small_withBiasAndRelu) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_mat_mul_op",
                              "_MklQuantizedMatMulWithBiasAndRelu")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("Toutput", DataTypeToEnum<qint32>::v())
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // A matrix is:
  // |  1 |  2 |  3 |
  // |  4 |  5 |  6 |
  AddInputFromArray<quint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // B matrix is:
  // |  7 |  8 |  9 | 10 |
  // | 11 | 12 | 13 | 14 |
  // | 15 | 16 | 17 | 18 |
  AddInputFromArray<qint8>(TensorShape({3, 4}),
                           {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  AddInputFromArray<float>(TensorShape({4}),
                           {100.0f, -200.0f, 300.0f, -400.0f});
  AddInputFromArray<float>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {255.0f});
  AddInputFromArray<float>(TensorShape({}), {-127.0f});
  AddInputFromArray<float>(TensorShape({}), {127.0f});

  TF_ASSERT_OK(RunOpKernel());
  // Here are the results we expect, from hand calculations:
  // (1 * 7) + (2 * 11) + (3 * 15) = 74
  // (1 * 8) + (2 * 12) + (3 * 16) = 80
  // (1 * 9) + (2 * 13) + (3 * 17) = 86
  // (1 * 10) + (2 * 14) + (3 * 18) = 92
  // (4 * 7) + (5 * 11) + (6 * 15) = 173
  // (4 * 8) + (5 * 12) + (6 * 16) = 188
  // (4 * 9) + (5 * 13) + (6 * 17) = 203
  // (4 * 10) + (5 * 14) + (6 * 18) = 218
  // After Bias addition
  // 74+100=174, 80-200=-120, 86+300=386, 92-400=-308,
  // 173+100=273, 188-200=-12, 203+300=503, 218-400=-182
  // After Relu
  // 174, 0, 386, 0, 273, 0, 503, 0
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 4}));
  test::FillValues<qint32>(&expected, {174, 0, 386, 0, 273, 0, 503, 0});

  const Tensor& output = *GetOutput(0);
  test::ExpectTensorEqual<qint32>(expected, output);
}

// Simple test for Matrix multiplication with Bias, Relu and
// Requantization fusion
TEST_F(QuantizedMatMulTest, Small_withBiasAndReluAndReq) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_mat_mul_op",
                              "_MklQuantizedMatMulWithBiasAndReluAndRequantize")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_QINT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("Toutput", DataTypeToEnum<quint8>::v())
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // A matrix is:
  // |  1 |  2 |  3 |
  // |  4 |  5 |  6 |
  AddInputFromArray<quint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // B matrix is:
  // |  7 |  8 |  9 | 10 |
  // | 11 | 12 | 13 | 14 |
  // | 15 | 16 | 17 | 18 |
  AddInputFromArray<qint8>(TensorShape({3, 4}),
                           {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  AddInputFromArray<qint32>(TensorShape({4}), {10, -20, 30, -40});
  AddInputFromArray<float>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {255.0f});
  AddInputFromArray<float>(TensorShape({}), {-127.0f});
  AddInputFromArray<float>(TensorShape({}), {127.0f});
  AddInputFromArray<float>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {255.0f});

  TF_ASSERT_OK(RunOpKernel());
  // Here are the results we expect, from hand calculations:
  // (1 * 7) + (2 * 11) + (3 * 15) = 74
  // (1 * 8) + (2 * 12) + (3 * 16) = 80
  // (1 * 9) + (2 * 13) + (3 * 17) = 86
  // (1 * 10) + (2 * 14) + (3 * 18) = 92
  // (4 * 7) + (5 * 11) + (6 * 15) = 173
  // (4 * 8) + (5 * 12) + (6 * 16) = 188
  // (4 * 9) + (5 * 13) + (6 * 17) = 203
  // (4 * 10) + (5 * 14) + (6 * 18) = 218
  // After Bias addition
  // 74+10=84, 80-20=60, 86+30=116, 92-40=52,
  // 173+10=183, 188-20=168, 203+30=233, 218-40=178
  // After Relu
  // 84, 60, 116, 52, 183, 168, 233, 178
  // After Requantize
  // requantscale = scale_int32 / scale_eightbit / static_cast<float>(1 << 23)
  // requantscale = 2^31/255/2^23 ~= 1.00392
  // 84 * 1.00392 ~= 84.329 ~= 84
  // 60 * 1.00392 ~= 60.235 ~= 60
  // 116 * 1.00392 ~= 116.454 ~= 116
  // 52 * 1.00392 ~= 52.203 ~= 52
  // 183 * 1.00392 ~= 183.717 ~= 184
  // 168 * 1.00392 ~= 168.658 ~= 169
  // 233 * 1.00392 ~= 233.913 ~= 234
  // 178 * 1.00392 ~= 178.698 ~= 179

  Tensor expected(allocator(), DT_QUINT8, TensorShape({2, 4}));
  test::FillValues<quint8>(&expected, {84, 60, 116, 52, 184, 169, 234, 179});

  const Tensor& output = *GetOutput(0);
  test::ExpectTensorEqual<quint8>(expected, output);
}

// Two small matrices A of type uint8 and B of type int8 are multiplied
// and the result is added with int32 bias
// For the first time B matrix will be reordered and cached which will be
// used for subsequent runs
TEST_F(QuantizedMatMulTest, Small_withWeightCached) {
  TF_ASSERT_OK(
      NodeDefBuilder("quantized_mat_mul_op", "_MklQuantizedMatMulWithBias")
          .Input(FakeInput(DT_QUINT8))
          .Input(FakeInput(DT_QINT8))
          .Input(FakeInput(DT_QINT32))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Input(FakeInput(DT_FLOAT))
          .Attr("Toutput", DataTypeToEnum<qint32>::v())
          .Attr("_kernel", "QuantizedMklOp")
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // The tensor shape of (1,3) is selected to allow the oneDNN expected
  // weight format to be made as OI rather than IO for BS > 1
  // A matrix is:
  // |  1 |  2 |  3 |
  AddInputFromArray<quint8>(TensorShape({1, 3}), {1, 2, 3});
  // B matrix is:
  // |  7 |  8 |  9 | 10 |
  // | 11 | 12 | 13 | 14 |
  // | 15 | 16 | 17 | 18 |
  AddInputFromArray<qint8>(TensorShape({3, 4}),
                           {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  AddInputFromArray<qint32>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {255.0f});
  AddInputFromArray<float>(TensorShape({}), {-127.0f});
  AddInputFromArray<float>(TensorShape({}), {127.0f});

  int64 start_time = Env::Default()->NowMicros();
  TF_ASSERT_OK(RunOpKernel());
  int64 end_time = Env::Default()->NowMicros();
  int64 total_duration_unopt = end_time - start_time;

  // Here are the results we expect, from hand calculations:
  // (1 * 7) + (2 * 11) + (3 * 15) = 74
  // (1 * 8) + (2 * 12) + (3 * 16) = 80
  // (1 * 9) + (2 * 13) + (3 * 17) = 86
  // (1 * 10) + (2 * 14) + (3 * 18) = 92
  // Final result after Bias addition:
  // 74  + 1 = 75 , 80  + 2 = 82 , 86  + 3 = 89 , 92  + 4 = 96,
  Tensor expected(allocator(), DT_QINT32, TensorShape({1, 4}));
  test::FillValues<qint32>(&expected, {75, 82, 89, 96});

  const Tensor& output = *GetOutput(0);
  test::ExpectTensorEqual<qint32>(expected, output);

  // Test for the second time to use the cached weight
  start_time = Env::Default()->NowMicros();
  TF_ASSERT_OK(RunOpKernel());
  end_time = Env::Default()->NowMicros();
  int64 total_duration_opt = end_time - start_time;
  LOG(INFO) << " Time taken by first call : " << total_duration_unopt
            << ", Time taken after Caching : " << total_duration_opt;

  // Cached call should be at least 20% faster.
  EXPECT_LT(total_duration_opt, total_duration_unopt * 0.8);

  // Compare the result with expected result
  const Tensor& output_new = *GetOutput(0);
  test::ExpectTensorEqual<qint32>(expected, output_new);
}

}  // namespace tensorflow

#endif  // INTEL_MKL && !ENABLE_ONEDNN_V3
