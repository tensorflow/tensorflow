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

#if defined(INTEL_MKL) && defined(ENABLE_MKL)
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

class QuantizedConv2DTest : public OpsTestBase {
 protected:
  void ConfigureQuantizedConv2D(const int& stride = 1) {
    TF_ASSERT_OK(NodeDefBuilder("quantized_conv_op", "_MklQuantizedConv2D")
                     .Input(FakeInput(DT_QUINT8))  // Input
                     .Input(FakeInput(DT_QINT8))   // Filter
                     .Input(FakeInput(DT_FLOAT))   // Min input
                     .Input(FakeInput(DT_FLOAT))   // Max input
                     .Input(FakeInput(DT_FLOAT))   // Min filter
                     .Input(FakeInput(DT_FLOAT))   // Max filter
                     .Attr("Tinput", DataTypeToEnum<quint8>::v())
                     .Attr("Tfilter", DataTypeToEnum<qint8>::v())
                     .Attr("out_type", DataTypeToEnum<qint32>::v())
                     .Attr("strides", {1, stride, stride, 1})
                     .Attr("padding", "SAME")
                     .Attr("_kernel", "QuantizedMklOp")
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void RunQuantizedDepthwiseConv2DOp(const bool& bias_enabled) {
    const int depth = 2;
    const int image_width = 2;
    const int image_height = 3;
    const int image_batch_count = 1;
    // The image matrix is ('first/second' channel):
    // | 1/2  |  3/4  |
    // | 5/6  |  7/8  |
    // | 9/10 | 11/12 |
    AddInputFromArray<quint8>(
        TensorShape({image_batch_count, image_height, image_width, depth}),
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    // The filter matrix is:
    // | 1/2 |  7/8  | 13/14 |
    // | 3/4 |  9/10 | 15/16 |
    // | 5/6 | 11/12 | 17/18 |
    const int filter_size = 3;
    const int filter_count = 1;
    AddInputFromArray<qint8>(
        TensorShape({filter_size, filter_size, depth, filter_count}),
        {1, 2, 7, 8, 13, 14, 3, 4, 9, 10, 15, 16, 5, 6, 11, 12, 17, 18});

    if (bias_enabled) {
      // Bias -> float
      AddInputFromArray<float>(TensorShape({depth}), {1.0f, 1.0f});
    }

    // Image -> uint8
    AddInputFromArray<float>(TensorShape({1}), {0.0f});
    AddInputFromArray<float>(TensorShape({1}), {255.0f});

    // Filter -> int8 with symmetric range
    AddInputFromArray<float>(TensorShape({1}), {-127.0f});
    AddInputFromArray<float>(TensorShape({1}), {127.0f});

    TF_ASSERT_OK(RunOpKernel());

    // We're sliding two 3x3 filters across the 3x2 image, with accesses outside
    // the input set to zero because we're using the 'SAME' padding mode.
    // This means we should end up with this matrix:
    // | 228/300 | 132/180 |
    // | 482/596 | 266/344 |
    // | 372/452 | 180/236 |
    //
    // Similarly, after adding a bias of 1.0f across each channel, we should end
    // up with this matrix:
    // | 229/301 | 133/181 |
    // | 483/597 | 267/345 |
    // | 373/453 | 181/237 |

    // Output -> qint32
    Tensor expected(DT_QINT32, TensorShape({image_batch_count, image_height,
                                            image_width, depth}));
    if (bias_enabled) {
      test::FillValues<qint32>(&expected, {229, 301, 133, 181, 483, 597, 267,
                                           345, 373, 453, 181, 237});
    } else {
      test::FillValues<qint32>(&expected, {228, 300, 132, 180, 482, 596, 266,
                                           344, 372, 452, 180, 236});
    }

    const Tensor& output = *GetOutput(0);
    test::ExpectTensorEqual<qint32>(expected, output);
  }
};

// Output -> float
TEST_F(QuantizedConv2DTest, Small) {
  const int stride = 1;
  ConfigureQuantizedConv2D(stride);

  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;

  // Image -> uint8
  const float image_min = 0.0f;
  const float image_max = 255.0f;

  // The image matrix is:
  // |  1 |  2 |  3 |  4 |
  // |  5 |  6 |  7 |  8 |
  // |  9 | 10 | 11 | 12 |
  Tensor image_float(DT_FLOAT,
                     {image_batch_count, image_height, image_width, depth});
  test::FillValues<float>(&image_float,
                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor image_quantized =
      FloatTensorToQuantized<quint8>(image_float, image_min, image_max);

  const int filter_size = 3;
  const int filter_count = 1;

  // Filter -> int8 with symmetric range
  const float filter_min = -127.0f;
  const float filter_max = 127.0f;

  // The filter matrix is:
  // | 1 | 4 | 7 |
  // | 2 | 5 | 8 |
  // | 3 | 6 | 9 |
  Tensor filter_float(DT_FLOAT,
                      {filter_size, filter_size, depth, filter_count});
  test::FillValues<float>(&filter_float, {1, 4, 7, 2, 5, 8, 3, 6, 9});
  Tensor filter_quantized =
      FloatTensorToQuantized<qint8>(filter_float, filter_min, filter_max);

  AddInputFromArray<quint8>(image_quantized.shape(),
                            image_quantized.flat<quint8>());
  AddInputFromArray<qint8>(filter_quantized.shape(),
                           filter_quantized.flat<qint8>());
  AddInputFromArray<float>(TensorShape({1}), {image_min});
  AddInputFromArray<float>(TensorShape({1}), {image_max});
  AddInputFromArray<float>(TensorShape({1}), {filter_min});
  AddInputFromArray<float>(TensorShape({1}), {filter_max});

  TF_ASSERT_OK(RunOpKernel());

  // We're sliding the 3x3 filter across the 3x4 image, with accesses outside
  // the input set to zero because we're using the 'SAME' padding mode.
  // The calculations behind the expected output are:
  // (1*0)+(4*0)+(7*0)+(2*0)+(5*1)+(8*2)+(3*0)+(6*5)+(9*6)=105
  // (1*0)+(4*0)+(7*0)+(2*1)+(5*2)+(8*3)+(3*5)+(6*6)+(9*7)=150
  // (1*0)+(4*0)+(7*0)+(2*2)+(5*3)+(8*4)+(3*6)+(6*7)+(9*8)=183
  // (1*0)+(4*0)+(7*0)+(2*3)+(5*4)+(8*0)+(3*7)+(6*8)+(9*0)=95
  // (1*0)+(4*1)+(7*2)+(2*0)+(5*5)+(8*6)+(3*0)+(6*9)+(9*10)=235
  // (1*1)+(4*2)+(7*3)+(2*5)+(5*6)+(8*7)+(3*9)+(6*10)+(9*11)=312
  // (1*2)+(4*3)+(7*4)+(2*6)+(5*7)+(8*8)+(3*10)+(6*11)+(9*12)=357
  // (1*3)+(4*4)+(7*0)+(2*7)+(5*8)+(8*0)+(3*11)+(6*12)+(9*0)=178
  // (1*0)+(4*5)+(7*6)+(2*0)+(5*9)+(8*10)+(3*0)+(6*0)+(9*0)=187
  // (1*5)+(4*6)+(7*7)+(2*9)+(5*10)+(8*11)+(3*0)+(6*0)+(9*0)=234
  // (1*6)+(4*7)+(7*8)+(2*10)+(5*11)+(8*12)+(3*0)+(6*0)+(9*0)=261
  // (1*7)+(4*8)+(7*0)+(2*11)+(5*12)+(8*0)+(3*0)+(6*0)+(9*0)=121
  // This means we should end up with this matrix:
  // |  105  |  150  |  183  |   95  |
  // |  235  |  312  |  357  |  178  |
  // |  187  |  234  |  261  |  121  |

  // Output -> float
  const int expected_width = image_width;
  const int expected_height = image_height;
  Tensor expected_float(
      DT_FLOAT, TensorShape({image_batch_count, expected_height, expected_width,
                             filter_count}));
  test::FillValues<float>(&expected_float, {105, 150, 183, 95, 235, 312, 357,
                                            178, 187, 234, 261, 121});

  const Tensor& output = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<qint32>(output, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 1.0);
}

TEST_F(QuantizedConv2DTest, SmallS8) {
  const int stride = 1;
  const int depth = 1;
  const int image_width = 3;
  const int image_height = 3;
  const int image_batch_count = 1;

  // Image -> uint8
  const float image_min = -127.0f;
  const float image_max = 127.0f;

  TF_ASSERT_OK(NodeDefBuilder("quantized_conv_op", "_MklQuantizedConv2D")
                   .Input(FakeInput(DT_QINT8))  // Input
                   .Input(FakeInput(DT_QINT8))  // Filter
                   .Input(FakeInput(DT_FLOAT))  // Min input
                   .Input(FakeInput(DT_FLOAT))  // Max input
                   .Input(FakeInput(DT_FLOAT))  // Min filter
                   .Input(FakeInput(DT_FLOAT))  // Max filter
                   .Attr("Tinput", DataTypeToEnum<qint8>::v())
                   .Attr("Tfilter", DataTypeToEnum<qint8>::v())
                   .Attr("padding", "VALID")
                   .Attr("out_type", DataTypeToEnum<qint32>::v())
                   .Attr("strides", {1, stride, stride, 1})
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // The image matrix is:
  // | 2 |  3 |  4 |
  // | 6 | -4 | -2 |
  // | 3 |  0 |  4 |
  Tensor image_float(DT_FLOAT,
                     {image_batch_count, image_height, image_width, depth});
  test::FillValues<float>(&image_float, {2, 3, 4, 6, -4, -2, 3, 0, 4});
  Tensor image_quantized =
      FloatTensorToQuantized<qint8>(image_float, image_min, image_max);

  const int filter_size = 3;
  const int filter_count = 1;

  // Filter -> int8 with symmetric range
  const float filter_min = -127.0f;
  const float filter_max = 127.0f;

  // The filter matrix is:
  // | 1 | 4 | 2 |
  // | 0 | 5 |-1 |
  // | 3 |-1 |-3 |
  Tensor filter_float(DT_FLOAT,
                      {filter_size, filter_size, depth, filter_count});
  test::FillValues<float>(&filter_float, {1, 4, 2, 0, 5, -1, 3, -1, -3});
  Tensor filter_quantized =
      FloatTensorToQuantized<qint8>(filter_float, filter_min, filter_max);

  AddInputFromArray<qint8>(image_quantized.shape(),
                           image_quantized.flat<qint8>());
  AddInputFromArray<qint8>(filter_quantized.shape(),
                           filter_quantized.flat<qint8>());
  AddInputFromArray<float>(TensorShape({1}), {image_min});
  AddInputFromArray<float>(TensorShape({1}), {image_max});
  AddInputFromArray<float>(TensorShape({1}), {filter_min});
  AddInputFromArray<float>(TensorShape({1}), {filter_max});

  TF_ASSERT_OK(RunOpKernel());

  // Output -> float
  const int expected_width = 1;
  const int expected_height = 1;
  Tensor expected_float(
      DT_FLOAT, TensorShape({image_batch_count, expected_height, expected_width,
                             filter_count}));
  test::FillValues<float>(&expected_float, {1});

  const Tensor& output = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<qint32>(output, output_min, output_max);

  test::ExpectTensorNear<float>(expected_float, output_float, 1.0);
}
// Output -> qint32
TEST_F(QuantizedConv2DTest, Small32Bit) {
  const int stride = 1;
  ConfigureQuantizedConv2D(stride);

  // The illustrations and details regarding inputs and outputs
  // are in TEST_F(QuantizedConv2DTest, Small)
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  AddInputFromArray<quint8>(
      TensorShape({image_batch_count, image_height, image_width, depth}),
      {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120});

  const int filter_size = 3;
  const int filter_count = 1;
  AddInputFromArray<qint8>(
      TensorShape({filter_size, filter_size, depth, filter_count}),
      {10, 40, 70, 20, 50, 80, 30, 60, 90});

  // Image -> uint8
  AddInputFromArray<float>(TensorShape({1}), {0.0f});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});

  // Filter -> int8 with symmetric range
  AddInputFromArray<float>(TensorShape({1}), {-127.0f});
  AddInputFromArray<float>(TensorShape({1}), {127.0f});

  TF_ASSERT_OK(RunOpKernel());

  // Output -> qint32
  const int expected_width = image_width;
  const int expected_height = image_height;
  Tensor expected(DT_QINT32, TensorShape({image_batch_count, expected_height,
                                          expected_width, filter_count}));
  test::FillValues<qint32>(
      &expected, {10500, 15000, 18300, 9500, 23500, 31200, 35700, 17800, 18700,
                  23400, 26100, 12100});

  const Tensor& output = *GetOutput(0);
  test::ExpectTensorEqual<qint32>(expected, output);
}

// Output -> qint32
TEST_F(QuantizedConv2DTest, Small32BitWithPadding) {
  const int stride = 1;
  TF_ASSERT_OK(NodeDefBuilder("quantized_conv_op", "_MklQuantizedConv2D")
                   .Input(FakeInput(DT_QUINT8))  // Input
                   .Input(FakeInput(DT_QINT8))   // Filter
                   .Input(FakeInput(DT_FLOAT))   // Min input
                   .Input(FakeInput(DT_FLOAT))   // Max input
                   .Input(FakeInput(DT_FLOAT))   // Min filter
                   .Input(FakeInput(DT_FLOAT))   // Max filter
                   .Attr("Tinput", DataTypeToEnum<quint8>::v())
                   .Attr("Tfilter", DataTypeToEnum<qint8>::v())
                   .Attr("out_type", DataTypeToEnum<qint32>::v())
                   .Attr("strides", {1, stride, stride, 1})
                   .Attr("padding", "SAME")
                   .Attr("padding_list", {0, 0, 1, 1, 1, 1, 0, 0})
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // The illustrations and details regarding inputs and outputs
  // are in TEST_F(QuantizedConv2DTest, Small)
  const int depth = 1;
  const int image_width = 4;
  const int image_height = 3;
  const int image_batch_count = 1;
  AddInputFromArray<quint8>(
      TensorShape({image_batch_count, image_height, image_width, depth}),
      {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120});

  const int filter_size = 3;
  const int filter_count = 1;
  AddInputFromArray<qint8>(
      TensorShape({filter_size, filter_size, depth, filter_count}),
      {10, 40, 70, 20, 50, 80, 30, 60, 90});

  // Image -> uint8
  AddInputFromArray<float>(TensorShape({1}), {0.0f});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});

  // Filter -> int8 with symmetric range
  AddInputFromArray<float>(TensorShape({1}), {-127.0f});
  AddInputFromArray<float>(TensorShape({1}), {127.0f});

  TF_ASSERT_OK(RunOpKernel());

  // Output -> qint32
  const int expected_width = image_width;
  const int expected_height = image_height;
  Tensor expected(DT_QINT32, TensorShape({image_batch_count, expected_height,
                                          expected_width, filter_count}));
  test::FillValues<qint32>(
      &expected, {10500, 15000, 18300, 9500, 23500, 31200, 35700, 17800, 18700,
                  23400, 26100, 12100});

  const Tensor& output = *GetOutput(0);
  test::ExpectTensorEqual<qint32>(expected, output);
}

// Output -> qint32
TEST_F(QuantizedConv2DTest, OddPadding) {
  const int stride = 2;
  ConfigureQuantizedConv2D(stride);

  const int depth = 1;
  const int image_width = 4;
  const int image_height = 4;
  const int image_batch_count = 1;
  AddInputFromArray<quint8>(
      TensorShape({image_batch_count, image_height, image_width, depth}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

  const int filter_size = 3;
  const int filter_count = 1;
  AddInputFromArray<qint8>(
      TensorShape({filter_size, filter_size, depth, filter_count}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9});

  // Image -> uint8
  AddInputFromArray<float>(TensorShape({1}), {0.0f});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});

  // Filter -> int8 with symmetric range
  AddInputFromArray<float>(TensorShape({1}), {-127.0f});
  AddInputFromArray<float>(TensorShape({1}), {127.0f});

  TF_ASSERT_OK(RunOpKernel());

  // Output -> qint32
  const int expected_width = image_width / stride;
  const int expected_height = image_height / stride;
  Tensor expected(DT_QINT32, TensorShape({image_batch_count, expected_height,
                                          expected_width, filter_count}));
  test::FillValues<qint32>(&expected, {348, 252, 274, 175});

  const Tensor& output = *GetOutput(0);
  test::ExpectTensorEqual<qint32>(expected, output);
}

// Output -> qint32
TEST_F(QuantizedConv2DTest, OddPaddingBatch) {
  const int stride = 2;
  ConfigureQuantizedConv2D(stride);

  const int depth = 1;
  const int image_width = 4;
  const int image_height = 4;
  const int image_batch_count = 3;
  AddInputFromArray<quint8>(
      TensorShape({image_batch_count, image_height, image_width, depth}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});

  const int filter_size = 3;
  const int filter_count = 1;
  AddInputFromArray<qint8>(
      TensorShape({filter_size, filter_size, depth, filter_count}),
      {1, 2, 3, 4, 5, 6, 7, 8, 9});

  // Image -> uint8
  AddInputFromArray<float>(TensorShape({1}), {0.0f});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});

  // Filter -> int8 with symmetric range
  AddInputFromArray<float>(TensorShape({1}), {-127.0f});
  AddInputFromArray<float>(TensorShape({1}), {127.0f});

  TF_ASSERT_OK(RunOpKernel());

  // Output -> qint32
  const int expected_width = image_width / stride;
  const int expected_height = image_height / stride;
  Tensor expected(DT_QINT32, TensorShape({image_batch_count, expected_height,
                                          expected_width, filter_count}));
  test::FillValues<qint32>(
      &expected, {348, 252, 274, 175, 348, 252, 274, 175, 348, 252, 274, 175});

  const Tensor& output = *GetOutput(0);
  test::ExpectTensorEqual<qint32>(expected, output);
}

TEST_F(QuantizedConv2DTest, DepthwiseConv2D) {
  const int stride = 1;
  TF_ASSERT_OK(NodeDefBuilder("quantized_depthwise_conv_op",
                              "_MklQuantizedDepthwiseConv2D")
                   .Input(FakeInput(DT_QUINT8))  // Input
                   .Input(FakeInput(DT_QINT8))   // Filter
                   .Input(FakeInput(DT_FLOAT))   // Min input
                   .Input(FakeInput(DT_FLOAT))   // Max input
                   .Input(FakeInput(DT_FLOAT))   // Min filter
                   .Input(FakeInput(DT_FLOAT))   // Max filter
                   .Attr("Tinput", DataTypeToEnum<quint8>::v())
                   .Attr("Tfilter", DataTypeToEnum<qint8>::v())
                   .Attr("out_type", DataTypeToEnum<qint32>::v())
                   .Attr("strides", {1, stride, stride, 1})
                   .Attr("padding", "SAME")
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  RunQuantizedDepthwiseConv2DOp(false);
}

TEST_F(QuantizedConv2DTest, DepthwiseConv2DWithBias) {
  const int stride = 1;
  TF_ASSERT_OK(NodeDefBuilder("quantized_depthwise_conv_op",
                              "_MklQuantizedDepthwiseConv2DWithBias")
                   .Input(FakeInput(DT_QUINT8))  // Input
                   .Input(FakeInput(DT_QINT8))   // Filter
                   .Input(FakeInput(DT_FLOAT))   // Bias
                   .Input(FakeInput(DT_FLOAT))   // Min input
                   .Input(FakeInput(DT_FLOAT))   // Max input
                   .Input(FakeInput(DT_FLOAT))   // Min filter
                   .Input(FakeInput(DT_FLOAT))   // Max filter
                   .Attr("Tinput", DataTypeToEnum<quint8>::v())
                   .Attr("Tfilter", DataTypeToEnum<qint8>::v())
                   .Attr("out_type", DataTypeToEnum<qint32>::v())
                   .Attr("strides", {1, stride, stride, 1})
                   .Attr("padding", "SAME")
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  RunQuantizedDepthwiseConv2DOp(true);
}

TEST_F(QuantizedConv2DTest, DepthwiseConv2DWithBiasAndRelu) {
  const int stride = 1;
  TF_ASSERT_OK(NodeDefBuilder("quantized_depthwise_conv_op",
                              "_MklQuantizedDepthwiseConv2DWithBiasAndRelu")
                   .Input(FakeInput(DT_QUINT8))  // Input
                   .Input(FakeInput(DT_QINT8))   // Filter
                   .Input(FakeInput(DT_FLOAT))   // Bias
                   .Input(FakeInput(DT_FLOAT))   // Min input
                   .Input(FakeInput(DT_FLOAT))   // Max input
                   .Input(FakeInput(DT_FLOAT))   // Min filter
                   .Input(FakeInput(DT_FLOAT))   // Max filter
                   .Attr("Tinput", DataTypeToEnum<quint8>::v())
                   .Attr("Tfilter", DataTypeToEnum<qint8>::v())
                   .Attr("out_type", DataTypeToEnum<qint32>::v())
                   .Attr("strides", {1, stride, stride, 1})
                   .Attr("padding", "SAME")
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  RunQuantizedDepthwiseConv2DOp(true);
}
}  // namespace tensorflow
#endif  // INTEL_MKL && ENABLE_MKL
