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

#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class QuantizedMatMulTest : public OpsTestBase {
 protected:
};

// Runs two small matrices through the operator, and leaves all the parameters
// at their default values.
TEST_F(QuantizedMatMulTest, Small_NoParams) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_mat_mul_op", "QuantizedMatMul")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("Toutput", DataTypeToEnum<qint32>::v())
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
  AddInputFromArray<quint8>(TensorShape({3, 4}),
                            {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  AddInputFromArray<float>(TensorShape({1}), {0});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});
  AddInputFromArray<float>(TensorShape({1}), {0});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});

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
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 4}));
  test::FillValues<qint32>(&expected, {74, 80, 86, 92, 173, 188, 203, 218});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

// This test multiplies two 1x1 8bit matrices, and compares the
// results with hand-calculated expectations.
TEST_F(QuantizedMatMulTest, VerySmall_WithParams) {
  // These parameters reflect a typical production usage of eight-bit matmuls
  // in an Inception-style network.
  const bool transpose_a = true;
  const int a_rows = 1;
  const int a_cols = 1;
  const int b_rows = 1;
  const int b_cols = 1;
  const bool transpose_b = false;
  TF_ASSERT_OK(NodeDefBuilder("quantized_mat_mul_op", "QuantizedMatMul")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("Toutput", DataTypeToEnum<qint32>::v())
                   .Attr("transpose_a", transpose_a)
                   .Attr("transpose_b", transpose_b)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // The A matrix is:
  // |  -1 |
  // The input array only contains unsigned bytes, so we specify the actual
  // values as n+a_offset, where a_offset is 12 above. For example that means -1
  // is represented as -1 + 12, or 11.
  // We have set the transpose_a flag to true, so the matrix is transposed, and
  // for filling the values the in-memory storage order is effectively
  // column major, rather than the default row-major.
  AddInputFromArray<quint8>(TensorShape({a_rows, a_cols}), {11});

  // The B matrix is:
  // |   1 |
  AddInputFromArray<quint8>(TensorShape({b_rows, b_cols}), {0});
  AddInputFromArray<float>(TensorShape({1}), {-12.0f});
  AddInputFromArray<float>(TensorShape({1}), {243.0f});
  AddInputFromArray<float>(TensorShape({1}), {1.0f});
  AddInputFromArray<float>(TensorShape({1}), {256.0f});
  TF_ASSERT_OK(RunOpKernel());
  // We're requesting C = A.transposed() * B,
  // so we expect to get these results:
  // 1*-1 = -1
  // | -1 |
  Tensor expected(allocator(), DT_QINT32, TensorShape({a_cols, b_cols}));
  test::FillValues<qint32>(&expected, {-1});
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

// This test multiplies two 1x1 8bit matrices, but sets an invalid quantization
// range, so we expect to get an error
TEST_F(QuantizedMatMulTest, VerySmall_BadRange) {
  // These parameters reflect a typical production usage of eight-bit matmuls
  // in an Inception-style network.
  const bool transpose_a = true;
  const int a_rows = 1;
  const int a_cols = 1;
  const int b_rows = 1;
  const int b_cols = 1;
  const bool transpose_b = false;
  TF_ASSERT_OK(NodeDefBuilder("quantized_mat_mul_op", "QuantizedMatMul")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("Toutput", DataTypeToEnum<qint32>::v())
                   .Attr("transpose_a", transpose_a)
                   .Attr("transpose_b", transpose_b)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // The A matrix is:
  // |  -1 |
  AddInputFromArray<quint8>(TensorShape({a_rows, a_cols}), {11});

  // The B matrix is:
  // |   1 |
  AddInputFromArray<quint8>(TensorShape({b_rows, b_cols}), {0});
  AddInputFromArray<float>(TensorShape({1}), {-12.0f});
  AddInputFromArray<float>(TensorShape({1}), {243.0f});
  // Here we set the range so that the min and max are equal, so we expect to
  // see an error when we run.
  AddInputFromArray<float>(TensorShape({1}), {1.0f});
  AddInputFromArray<float>(TensorShape({1}), {1.0f});
  EXPECT_EQ(::tensorflow::error::INVALID_ARGUMENT, RunOpKernel().code());
}

// This test multiplies a couple of small 8-bit matrices, and compares the
// results with hand-calculated expectations. It uses shifts and offsets to
// control the range of the outputs.
TEST_F(QuantizedMatMulTest, Small_WithParams) {
  // These parameters reflect a typical production usage of eight-bit matmuls
  // in an Inception-style network.
  const bool transpose_a = true;
  const int a_rows = 3;
  const int a_cols = 4;
  const int b_rows = 3;
  const int b_cols = 2;
  const bool transpose_b = false;
  TF_ASSERT_OK(NodeDefBuilder("quantized_mat_mul_op", "QuantizedMatMul")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("Toutput", DataTypeToEnum<qint32>::v())
                   .Attr("transpose_a", transpose_a)
                   .Attr("transpose_b", transpose_b)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // The A matrix is:
  // |  -1 |  -5 |  -9 |
  // |  -2 |  -6 | -10 |
  // |  -3 |  -7 | -11 |
  // |  -4 |  -8 | -12 |
  // The input array only contains unsigned bytes, so we specify the actual
  // values as n+a_offset, where a_offset is 12 above. For example that means -1
  // is represented as -1 + 12, or 11.
  // We have set the transpose_a flag to true, so the matrix is transposed, and
  // for filling the values the in-memory storage order is effectively
  // column major, rather than the default row-major.
  AddInputFromArray<quint8>(TensorShape({a_rows, a_cols}),
                            {
                                11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                            });

  // The B matrix is:
  // |   1 |   4|
  // |   2 |   5|
  // |   3 |   6|
  AddInputFromArray<quint8>(TensorShape({b_rows, b_cols}), {
                                                               1, 4, 2, 5, 3, 6,
                                                           });
  AddInputFromArray<float>(TensorShape({1}), {-12.0f});
  AddInputFromArray<float>(TensorShape({1}), {243.0f});
  AddInputFromArray<float>(TensorShape({1}), {0});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});
  TF_ASSERT_OK(RunOpKernel());
  // We're requesting C = A.transposed() * B,
  // so we expect to get these results:
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
  Tensor expected(allocator(), DT_QINT32, TensorShape({a_cols, b_cols}));
  test::FillValues<qint32>(&expected,
                           {
                               -38, -83, -44, -98, -50, -113, -56, -128,
                           });
  test::ExpectTensorEqual<qint32>(expected, *GetOutput(0));
}

// This test multiplies a couple of medium-sized 8-bit matrices, and tests the
// results against what we saw from running a float MatMul with equivalent
// inputs.
TEST_F(QuantizedMatMulTest, Medium_WithParams) {
  const bool transpose_a = true;
  const bool transpose_b = false;
  TF_ASSERT_OK(NodeDefBuilder("quantized_mat_mul_op", "QuantizedMatMul")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("Toutput", DataTypeToEnum<qint32>::v())
                   .Attr("transpose_a", transpose_a)
                   .Attr("transpose_b", transpose_b)
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  const int a_rows = 8;
  const int a_cols = 8;
  const float a_min = -2164.25f;
  const float a_max = 2006.27f;
  Tensor a_float(DT_FLOAT, {a_rows, a_cols});
  test::FillValues<float>(
      &a_float,
      {-1014.12, -157.382, -810.17,  1435.28,  1016.37,  219.684,  -316.054,
       -2164.25, 2006.27,  -547.444, 857.376,  404.376,  9.72115,  332.588,
       194.385,  -286.57,  26.062,   23.1125,  110.436,  247.055,  -127.683,
       -376.275, -124.81,  -846.826, -77.1507, 305.581,  -202.747, 12.9528,
       9.64886,  872.686,  40.9069,  197.816,  44.16,    -306.768, -1457.52,
       -368.939, -1049.42, -486.353, 1745.87,  95.7695,  395.773,  -254.333,
       -404.27,  787.16,   -2.44114, 199.37,   -1024.08, 784.901,  235.055,
       -42.7295, 241.498,  -245.365, 470.763,  186.159,  186.579,  -220.163,
       1304.58,  386.272,  -358.853, -755.996, 360.109,  -866.007, 55.2828,
       -508.801});
  Tensor a_quantized = FloatTensorToQuantized<quint8>(a_float, a_min, a_max);

  const int b_rows = 8;
  const int b_cols = 8;
  const float b_min = -0.739539f;
  const float b_max = 0.641057f;
  Tensor b_float(DT_FLOAT, {b_rows, b_cols});
  test::FillValues<float>(
      &b_float,
      {-0.294619, -0.0670519, 0.261507,   -0.126274, 0.127229,   -0.176945,
       -0.251223, 0.231086,   0.453694,   0.415666,  -0.288733,  0.508717,
       0.211551,  0.0435907,  -0.582383,  -0.308779, 0.0696883,  -0.438122,
       0.114,     0.433964,   0.109883,   0.284931,  -0.149661,  0.108657,
       0.458333,  -0.130231,  -0.35805,   -0.123206, -0.437968,  0.0282411,
       0.628818,  -0.0522173, -0.0233403, 0.124863,  0.217165,   0.262294,
       -0.171005, -0.254693,  -0.200433,  -0.287354, 0.488166,   -0.0354688,
       -0.118091, -0.590444,  0.491537,   -0.739539, 0.083117,   0.282482,
       0.275269,  -0.36574,   0.107476,   0.0511428, -0.136887,  -0.0149852,
       -0.259694, 0.641057,   0.264054,   -0.295126, -0.0218791, 0.361211,
       0.012448,  0.0709718,  -0.392394,  -0.434215});
  Tensor b_quantized = FloatTensorToQuantized<quint8>(b_float, b_min, b_max);

  AddInputFromArray<quint8>(a_quantized.shape(), a_quantized.flat<quint8>());
  AddInputFromArray<quint8>(b_quantized.shape(), b_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {a_min});
  AddInputFromArray<float>(TensorShape({1}), {a_max});
  AddInputFromArray<float>(TensorShape({1}), {b_min});
  AddInputFromArray<float>(TensorShape({1}), {b_max});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_float(DT_FLOAT, {a_cols, b_cols});
  test::FillValues<float>(
      &expected_float,
      {1776.82f,  421.058f,  -854.308f, 1430.65f,  503.105f,  57.2744f,
       -1514.97f, -1163.66f, -87.0979f, -394.577f, -39.4983f, -79.1938f,
       -329.029f, 313.475f,  446.929f,  -59.5855f, 350.837f,  238.655f,
       -609.21f,  350.499f,  192.238f,  847.576f,  -103.177f, 185.886f,
       -90.5335f, 200.787f,  99.1981f,  -717.076f, 763.815f,  -703.726f,
       -125.164f, 732.325f,  -51.5303f, -418.826f, 60.0783f,  -299.658f,
       231.41f,   72.0622f,  -289.244f, 663.776f,  391.177f,  294.415f,
       -484.148f, -677.932f, -180.342f, -194.764f, 761.715f,  553.061f,
       -283.355f, 321.109f,  351.269f,  1171.7f,   -857.497f, 343.804f,
       -494.599f, -844.119f, 725.237f,  586.052f,  -735.013f, -897.723f,
       -122.434f, -502.907f, 1264.6f,   -239.991f});

  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<qint32>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 15.0);
}

}  // namespace tensorflow
