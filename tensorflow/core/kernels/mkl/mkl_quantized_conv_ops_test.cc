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

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/mkl/mkl_kernel_util.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

class QuantizedConv2DTest : public OpsTestBase {
 protected:
  template <typename Tinput>
  void ConfigureQuantizedConv2D(const bool old_api, const int& stride,
                                const string& padding,
                                const std::vector<int> padding_values = {}) {
    if (old_api) {
      TF_ASSERT_OK(NodeDefBuilder("quantized_conv_op", "_MklQuantizedConv2D")
                       .Input(FakeInput(DataTypeToEnum<Tinput>::v()))  // Input
                       .Input(FakeInput(DT_QINT8))                     // Filter
                       .Input(FakeInput(DT_FLOAT))  // Min input
                       .Input(FakeInput(DT_FLOAT))  // Max input
                       .Input(FakeInput(DT_FLOAT))  // Min filter
                       .Input(FakeInput(DT_FLOAT))  // Max filter
                       .Attr("Tinput", DataTypeToEnum<Tinput>::v())
                       .Attr("Tfilter", DataTypeToEnum<qint8>::v())
                       .Attr("out_type", DataTypeToEnum<qint32>::v())
                       .Attr("strides", {1, stride, stride, 1})
                       .Attr("padding", padding)
                       .Attr("padding_list", padding_values)
                       .Attr("_kernel", "QuantizedMklOp")
                       .Finalize(node_def()));
    } else {
      TF_EXPECT_OK(
          NodeDefBuilder("quantized_conv_op", "_FusedQuantizedConv2D")
              .Attr("Thost_inputs", {DataTypeToEnum<Tinput>::v(), DT_QINT8,
                                     DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT})
              .Attr("Thost_outputs", {DT_QINT32, DT_FLOAT, DT_FLOAT})
              .Attr("Tdevice_inputs", std::vector<DataType>())
              .Attr("Tdevice_outputs", std::vector<DataType>())
              .Attr("Tinput", DataTypeToEnum<Tinput>::v())
              .Attr("Tfilter", DT_QINT8)
              .Attr("Tsummand", DT_QINT32)
              .Attr("out_type", DT_QINT32)
              .Attr("strides", {1, stride, stride, 1})
              .Attr("padding", padding)
              .Attr("explicit_paddings", padding_values)
              .Input(FakeInput())
              .Input(FakeInput())
              .Finalize(node_def()));
    }
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

  void TestSmall(const bool old_api) {
    const int stride = 1;
    const string padding = "SAME";
    ConfigureQuantizedConv2D<quint8>(old_api, stride, padding);

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
    Tensor expected_float(DT_FLOAT,
                          TensorShape({image_batch_count, expected_height,
                                       expected_width, filter_count}));
    test::FillValues<float>(&expected_float, {105, 150, 183, 95, 235, 312, 357,
                                              178, 187, 234, 261, 121});

    const Tensor& output = *GetOutput(0);
    const float output_min = GetOutput(1)->flat<float>()(0);
    const float output_max = GetOutput(2)->flat<float>()(0);
    Tensor output_float =
        QuantizedTensorToFloat<qint32>(output, output_min, output_max);
    test::ExpectTensorNear<float>(expected_float, output_float, 1.0);
  }

  void TestSmallS8(const bool old_api) {
    const int stride = 1;
    const int depth = 1;
    const int image_width = 3;
    const int image_height = 3;
    const int image_batch_count = 1;

    // Image -> uint8
    const float image_min = -127.0f;
    const float image_max = 127.0f;

    const string padding = "VALID";
    ConfigureQuantizedConv2D<qint8>(old_api, stride, padding);

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
    Tensor expected_float(DT_FLOAT,
                          TensorShape({image_batch_count, expected_height,
                                       expected_width, filter_count}));
    test::FillValues<float>(&expected_float, {1});

    const Tensor& output = *GetOutput(0);
    const float output_min = GetOutput(1)->flat<float>()(0);
    const float output_max = GetOutput(2)->flat<float>()(0);
    Tensor output_float =
        QuantizedTensorToFloat<qint32>(output, output_min, output_max);

    test::ExpectTensorNear<float>(expected_float, output_float, 1.0);
  }

  void TestSmall32Bit(const bool old_api) {
    const int stride = 1;
    const string padding = "SAME";
    ConfigureQuantizedConv2D<quint8>(old_api, stride, padding);

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
        &expected, {10500, 15000, 18300, 9500, 23500, 31200, 35700, 17800,
                    18700, 23400, 26100, 12100});

    const Tensor& output = *GetOutput(0);
    test::ExpectTensorEqual<qint32>(expected, output);
  }

  void TestSmall32BitWithPadding(const bool old_api) {
    const int stride = 1;
    const string padding = "SAME";
    ConfigureQuantizedConv2D<quint8>(old_api, stride, padding);

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
        &expected, {10500, 15000, 18300, 9500, 23500, 31200, 35700, 17800,
                    18700, 23400, 26100, 12100});

    const Tensor& output = *GetOutput(0);
    test::ExpectTensorEqual<qint32>(expected, output);
  }

  void TestOddPadding(const bool old_api) {
    const int stride = 2;
    string padding = "SAME";
    ConfigureQuantizedConv2D<quint8>(old_api, stride, padding);

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

  void TestOddPaddingBatch(const bool old_api) {
    const int stride = 2;
    const string padding = "SAME";
    ConfigureQuantizedConv2D<quint8>(old_api, stride, padding);

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
    test::FillValues<qint32>(&expected, {348, 252, 274, 175, 348, 252, 274, 175,
                                         348, 252, 274, 175});

    const Tensor& output = *GetOutput(0);
    test::ExpectTensorEqual<qint32>(expected, output);
  }

  void TestDepthwiseConv2D(const bool old_api) {
    const int stride = 1;
    if (old_api) {
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
    } else {
      TF_EXPECT_OK(NodeDefBuilder("quantized_depthwise_conv_op",
                                  "_FusedQuantizedDepthwiseConv2D")
                       .Attr("Thost_inputs", {DT_QUINT8, DT_QINT8, DT_FLOAT,
                                              DT_FLOAT, DT_FLOAT, DT_FLOAT})
                       .Attr("Thost_outputs", {DT_QINT32, DT_FLOAT, DT_FLOAT})
                       .Attr("Tdevice_inputs", std::vector<DataType>())
                       .Attr("Tdevice_outputs", std::vector<DataType>())
                       .Attr("Tinput", DT_QUINT8)
                       .Attr("Tfilter", DT_QINT8)
                       .Attr("Tsummand", DT_QINT32)
                       .Attr("out_type", DT_QINT32)
                       .Attr("strides", {1, stride, stride, 1})
                       .Attr("padding", "SAME")
                       .Input(FakeInput())
                       .Input(FakeInput())
                       .Finalize(node_def()));
    }
    TF_ASSERT_OK(InitOp());
    RunQuantizedDepthwiseConv2DOp(false);
  }

  void TestDepthwiseConv2DWithBias(const bool old_api) {
    const int stride = 1;
    if (old_api) {
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
    } else {
      TF_EXPECT_OK(
          NodeDefBuilder("quantized_depthwise_conv_op",
                         "_FusedQuantizedDepthwiseConv2D")
              .Attr("Thost_inputs", {DT_QUINT8, DT_QINT8, DT_FLOAT, DT_FLOAT,
                                     DT_FLOAT, DT_FLOAT, DT_FLOAT})
              .Attr("Thost_outputs", {DT_QINT32, DT_FLOAT, DT_FLOAT})
              .Attr("Tdevice_inputs", std::vector<DataType>())
              .Attr("Tdevice_outputs", std::vector<DataType>())
              .Attr("Tinput", DT_QUINT8)
              .Attr("Tfilter", DT_QINT8)
              .Attr("Tbias", DT_FLOAT)
              .Attr("Tsummand", DT_QINT32)
              .Attr("out_type", DT_QINT32)
              .Attr("strides", {1, stride, stride, 1})
              .Attr("padding", "SAME")
              .Attr("fused_ops", {"BiasAdd"})
              .Input(FakeInput())
              .Input(FakeInput())
              .Finalize(node_def()));
    }
    TF_ASSERT_OK(InitOp());
    RunQuantizedDepthwiseConv2DOp(true);
  }

  void TestDepthwiseConv2DWithBiasAndRelu(const bool old_api) {
    const int stride = 1;
    if (old_api) {
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
    } else {
      TF_EXPECT_OK(
          NodeDefBuilder("quantized_depthwise_conv_op",
                         "_FusedQuantizedDepthwiseConv2D")
              .Attr("Thost_inputs", {DT_QUINT8, DT_QINT8, DT_FLOAT, DT_FLOAT,
                                     DT_FLOAT, DT_FLOAT, DT_FLOAT})
              .Attr("Thost_outputs", {DT_QINT32, DT_FLOAT, DT_FLOAT})
              .Attr("Tdevice_inputs", std::vector<DataType>())
              .Attr("Tdevice_outputs", std::vector<DataType>())
              .Attr("Tinput", DT_QUINT8)
              .Attr("Tfilter", DT_QINT8)
              .Attr("Tbias", DT_FLOAT)
              .Attr("Tsummand", DT_QINT32)
              .Attr("out_type", DT_QINT32)
              .Attr("strides", {1, stride, stride, 1})
              .Attr("padding", "SAME")
              .Attr("fused_ops", {"BiasAdd", "Relu"})
              .Input(FakeInput())
              .Input(FakeInput())
              .Finalize(node_def()));
    }
    TF_ASSERT_OK(InitOp());
    RunQuantizedDepthwiseConv2DOp(true);
  }
};

// Output -> float
TEST_F(QuantizedConv2DTest, SmallOldAPI) { TestSmall(true); }

TEST_F(QuantizedConv2DTest, SmallNewAPI) { TestSmall(false); }

TEST_F(QuantizedConv2DTest, SmallS8OldAPI) { TestSmallS8(true); }

TEST_F(QuantizedConv2DTest, SmallS8NewAPI) { TestSmallS8(false); }
// Output -> qint32
TEST_F(QuantizedConv2DTest, Small32BitOldAPI) { TestSmall32Bit(true); }
TEST_F(QuantizedConv2DTest, Small32BitNewAPI) { TestSmall32Bit(false); }

// Output -> qint32
TEST_F(QuantizedConv2DTest, Small32BitWithPaddingOldAPI) {
  TestSmall32BitWithPadding(true);
}

TEST_F(QuantizedConv2DTest, Small32BitWithPaddingNewAPI) {
  TestSmall32BitWithPadding(false);
}

// Output -> qint32
TEST_F(QuantizedConv2DTest, OddPaddingOldAPI) { TestOddPadding(true); }
TEST_F(QuantizedConv2DTest, OddPaddingNewAPI) { TestOddPadding(false); }

// Output -> qint32
TEST_F(QuantizedConv2DTest, OddPaddingBatchOldAPI) {
  TestOddPaddingBatch(true);
}
TEST_F(QuantizedConv2DTest, OddPaddingBatchNewAPI) {
  TestOddPaddingBatch(false);
}

TEST_F(QuantizedConv2DTest, DepthwiseConv2DOldAPI) {
  TestDepthwiseConv2D(true);
}

TEST_F(QuantizedConv2DTest, DepthwiseConv2DNewAPI) {
  TestDepthwiseConv2D(false);
}

TEST_F(QuantizedConv2DTest, DepthwiseConv2DWithBiasOldAPI) {
  TestDepthwiseConv2DWithBias(true);
}
TEST_F(QuantizedConv2DTest, DepthwiseConv2DWithBiasNewAPI) {
  TestDepthwiseConv2DWithBias(false);
}

TEST_F(QuantizedConv2DTest, DepthwiseConv2DWithBiasAndReluOldAPI) {
  TestDepthwiseConv2DWithBiasAndRelu(true);
}
TEST_F(QuantizedConv2DTest, DepthwiseConv2DWithBiasAndReluNewAPI) {
  TestDepthwiseConv2DWithBiasAndRelu(false);
}

class QuantizedConvTest : public OpsTestBase {
 protected:
  template <typename Tinput, typename Tfilter, typename Toutput,
            typename Tbias = float, typename Tsummand = float>
  void RunQuantizedKernel(Tensor& image_float, Tensor& filter_float,
                          Tensor& bias_float, Tensor& summand_float,
                          Tensor& expected_out_float,
                          const std::vector<string>& fused_ops,
                          const float tol = 1.0) {
    bool fuse_bias = std::find(fused_ops.begin(), fused_ops.end(), "BiasAdd") !=
                     fused_ops.end();
    bool fuse_sum =
        std::find(fused_ops.begin(), fused_ops.end(), "Sum") != fused_ops.end();
    bool fuse_requantize = std::find(fused_ops.begin(), fused_ops.end(),
                                     "Requantize") != fused_ops.end();
    float image_min, image_max;
    MklTestingUtil::ComputeMinMax<float>(image_float, &image_min, &image_max);
    const float image_max_abs =
        std::max(std::abs(image_min), std::abs(image_max));
    Tensor image_quantized;
    MklTestingUtil::RunMklQuantizeOp(image_float, -image_max_abs, image_max_abs,
                                     DataTypeToEnum<Tinput>::v(), "SCALED",
                                     &image_quantized);

    float filter_min, filter_max;
    MklTestingUtil::ComputeMinMax<float>(filter_float, &filter_min,
                                         &filter_max);
    const float filter_max_abs =
        std::max(std::abs(filter_min), std::abs(filter_max));
    Tensor filter_quantized;
    MklTestingUtil::RunMklQuantizeOp(
        filter_float, -filter_max_abs, filter_max_abs,
        DataTypeToEnum<Tfilter>::v(), "SCALED", &filter_quantized);

    AddInputFromArray<Tinput>(image_quantized.shape(),
                              image_quantized.flat<Tinput>());
    AddInputFromArray<Tfilter>(filter_quantized.shape(),
                               filter_quantized.flat<Tfilter>());
    if (fuse_bias) {
      if (std::is_same<Tbias, float>::value) {
        AddInputFromArray<Tbias>(bias_float.shape(), bias_float.flat<Tbias>());
      } else {
        // Tbias needs to be INT32
        float bias_min, bias_max;
        MklTestingUtil::ComputeMinMax<float>(bias_float, &bias_min, &bias_max);
        const float bias_max_abs =
            std::max(std::abs(bias_min), std::abs(bias_max));
        Tensor bias_quantized;
        MklTestingUtil::RunMklQuantizeOp(
            bias_float, -bias_max_abs, bias_max_abs, DataTypeToEnum<Tbias>::v(),
            "SCALED", &bias_quantized);
        AddInputFromArray<Tbias>(bias_quantized.shape(),
                                 bias_quantized.flat<Tbias>());
      }
    }
    bool is_quantized_summand = false;
    float summand_max_abs = 0;
    if (fuse_sum) {
      if (std::is_same<Tsummand, float>::value) {
        AddInputFromArray<Tsummand>(summand_float.shape(),
                                    summand_float.flat<Tsummand>());
      } else {
        // Summand needs to be quantized
        is_quantized_summand = true;
        float summand_min, summand_max;
        MklTestingUtil::ComputeMinMax<float>(summand_float, &summand_min,
                                             &summand_max);
        summand_max_abs =
            std::max(std::abs(summand_min), std::abs(summand_max));
        Tensor summand_quantized;
        MklTestingUtil::RunMklQuantizeOp(
            summand_float, -summand_max_abs, summand_max_abs,
            DataTypeToEnum<Tsummand>::v(), "SCALED", &summand_quantized);
        AddInputFromArray<Tsummand>(summand_quantized.shape(),
                                    summand_quantized.flat<Tsummand>());
      }
    }
    AddInputFromArray<float>(TensorShape({}), {-image_max_abs});
    AddInputFromArray<float>(TensorShape({}), {image_max_abs});
    AddInputFromArray<float>(TensorShape({}), {-filter_max_abs});
    AddInputFromArray<float>(TensorShape({}), {filter_max_abs});

    if (is_quantized_summand) {
      AddInputFromArray<float>(TensorShape({}), {-summand_max_abs});
      AddInputFromArray<float>(TensorShape({}), {summand_max_abs});
    }
    if (fuse_requantize) {
      float expected_output_min, expected_output_max;
      MklTestingUtil::ComputeMinMax<float>(
          expected_out_float, &expected_output_min, &expected_output_max);
      const float output_max_abs = std::max(std::abs(expected_output_min),
                                            std::abs(expected_output_max));
      AddInputFromArray<float>(TensorShape({}), {-output_max_abs});
      AddInputFromArray<float>(TensorShape({}), {output_max_abs});
    }

    TF_ASSERT_OK(RunOpKernel());

    const Tensor& output = *GetOutput(0);
    const Tensor& output_min = *GetOutput(1);
    const Tensor& output_max = *GetOutput(2);
    const float output_max_value = output_max.flat<float>()(0);

    Tensor output_float;
    MklTestingUtil::RunDequantizeOp(output, output_min, output_max, "SCALED",
                                    &output_float);
    if (std::is_same<Tsummand, qint8>::value &&
        std::is_same<Toutput, quint8>::value) {
      // When summand's type is qint8 and output's type is quint8, we need to
      // clamp the expected value. Although output's type is quint8, it cannot
      // hold values larger than 127 due to limitation in the implementation.
      for (int i = 0; i < expected_out_float.NumElements(); i++) {
        float* expected_data =
            const_cast<float*>(expected_out_float.flat<float>().data());
        expected_data[i] =
            std::min(expected_data[i], output_max_value * 127.0f / 255.0f);
      }
    }
    test::ExpectTensorNear<float>(expected_out_float, output_float, tol);
  }

  void RunFloatConv(const Tensor& input_data, const Tensor& filter_data,
                    const Tensor& bias_data, const Tensor& summand_data,
                    Tensor* output, const bool is_depthwise,
                    const std::vector<string>& fused_ops, const string padding,
                    const int stride) {
    auto root = tensorflow::Scope::NewRootScope();
    auto input_data_op =
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data));

    Output out_op;
    if (is_depthwise) {
      out_op = ops::DepthwiseConv2dNative(
          root.WithOpName("conv"), input_data_op,
          ops::Const(root.WithOpName("filter"),
                     Input::Initializer(filter_data)),
          {1, stride, stride, 1}, padding);
    } else {
      out_op = ops::Conv2D(root.WithOpName("conv"), input_data_op,
                           ops::Const(root.WithOpName("filter"),
                                      Input::Initializer(filter_data)),
                           {1, stride, stride, 1}, padding);
    }

    string last_op = "";
    for (int i = 0; i < fused_ops.size(); ++i) {
      if (fused_ops[i] == "BiasAdd") {
        last_op = "with_bias";
        out_op = ops::BiasAdd(
            root.WithOpName(last_op), out_op,
            ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));
      }

      if (fused_ops[i] == "Sum") {
        last_op = "with_sum";
        out_op = ops::AddV2(root.WithOpName(last_op), out_op,
                            ops::Const(root.WithOpName("summand"),
                                       Input::Initializer(summand_data)));
      }

      if (fused_ops[i] == "Relu") {
        last_op = "with_relu";
        out_op = ops::Relu(root.WithOpName(last_op), out_op);
      }
    }

    tensorflow::GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    MklTestingUtil::RunGraph(graph_def, last_op, output);
  }

  template <typename Tinput, typename Toutput>
  void TestBiasAddFusion(bool fuse_requantize, const bool is_depthwise,
                         string activation = "", const float tol = 1.0) {
    const int stride = 1;
    const string padding = "VALID";
    std::vector<string> fused_ops = {"BiasAdd"};
    std::map<string, DataType> data_types = {
        {"Tinput", DataTypeToEnum<Tinput>::v()},
        {"Tfilter", DT_QINT8},
        {"Tbias", DT_FLOAT},
        {"Tsummand", DataTypeToEnum<Toutput>::v()},
        {"out_type", DataTypeToEnum<Toutput>::v()}};
    std::vector<DataType> input_types = {data_types["Tinput"],
                                         data_types["Tfilter"],
                                         data_types["Tbias"],
                                         DT_FLOAT,   // min_input
                                         DT_FLOAT,   // max_input
                                         DT_FLOAT,   // min_filter
                                         DT_FLOAT};  // max_filter
    if (!activation.empty()) {
      fused_ops.push_back(activation);
    }

    if (fuse_requantize) {
      fused_ops.push_back("Requantize");
      input_types.push_back(DT_FLOAT);  // min_freezed_output
      input_types.push_back(DT_FLOAT);  // max_freezed_output
    }

    TF_EXPECT_OK(
        NodeDefBuilder("quantized_conv_op",
                       is_depthwise ? "_FusedQuantizedDepthwiseConv2D"
                                    : "_FusedQuantizedConv2D")
            .Attr("Thost_inputs", input_types)
            .Attr("Thost_outputs", {data_types["out_type"], DT_FLOAT, DT_FLOAT})
            .Attr("Tdevice_inputs", std::vector<DataType>())
            .Attr("Tdevice_outputs", std::vector<DataType>())
            .Attr("Tinput", data_types["Tinput"])
            .Attr("Tfilter", data_types["Tfilter"])
            .Attr("Tbias", data_types["Tbias"])
            .Attr("Tsummand", data_types["Tsummand"])
            .Attr("out_type", data_types["out_type"])
            .Attr("strides", {1, stride, stride, 1})
            .Attr("padding", padding)
            .Attr("fused_ops", fused_ops)
            .Input(FakeInput())
            .Input(FakeInput())
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    const int image_batch = 1;
    const int image_height = 6;
    const int image_width = 6;
    const int channels = 2;

    const int filter_height = 2;
    const int filter_width = 2;
    const int filter_out_channels = 2;

    // Format is NHWC
    Tensor image_float(DT_FLOAT,
                       {image_batch, image_height, image_width, channels});
    test::FillValues<float>(
        &image_float, {4, 3, 1, 0, 4, 6, 3, 1, 2, 1, 0, 2, 6, 2, 1, 3, 1, 3,
                       6, 1, 2, 5, 3, 2, 3, 4, 1, 4, 0, 3, 3, 1, 2, 0, 1, 1,
                       3, 3, 1, 0, 2, 1, 4, 3, 3, 2, 1, 4, 1, 0, 2, 2, 5, 0,
                       3, 3, 3, 1, 0, 2, 2, 1, 3, 2, 6, 3, 4, 6, 0, 1, 3, 5});

    Tensor filter_float(
        DT_FLOAT, {filter_height, filter_width, channels, filter_out_channels});
    test::FillValues<float>(
        &filter_float, {-2, -3, 0, 3, 1, -1, 4, 2, -3, -2, -4, 0, 4, 3, 1, 2});

    Tensor bias_float(DT_FLOAT, {is_depthwise ? channels * filter_out_channels
                                              : filter_out_channels});
    if (is_depthwise) {
      test::FillValues<float>(&bias_float, {1, 2, 1, 2});
    } else {
      test::FillValues<float>(&bias_float, {1, 2});
    }

    Tensor expected_float, dummy_summand;

    RunFloatConv(image_float, filter_float, bias_float, dummy_summand,
                 &expected_float, is_depthwise, fused_ops, padding, stride);
    RunQuantizedKernel<Tinput, qint8, Toutput, float>(
        image_float, filter_float, bias_float, dummy_summand, expected_float,
        fused_ops, tol);
  }

  template <typename Tsummand, typename Toutput>
  void TestBiasAddSumActivationFusion(string activation = "") {
    const int stride = 1;
    const string padding = "VALID";
    std::vector<string> fused_ops = {"BiasAdd", "Sum"};
    std::map<string, DataType> data_types = {
        {"Tinput", DT_QINT8},
        {"Tfilter", DT_QINT8},
        {"Tbias", DT_FLOAT},
        {"Tsummand", DataTypeToEnum<Tsummand>::v()},
        {"out_type", DataTypeToEnum<Toutput>::v()}};

    // Default values are for float summand and when Requantize is not fused
    std::vector<DataType> input_types = {data_types["Tinput"],
                                         data_types["Tfilter"],
                                         data_types["Tbias"],
                                         data_types["Tsummand"],
                                         DT_FLOAT,   // min_input
                                         DT_FLOAT,   // max_input
                                         DT_FLOAT,   // min_filter
                                         DT_FLOAT};  // max_filter
    if (std::is_same<Tsummand, quint8>::value ||
        std::is_same<Tsummand, qint8>::value) {
      input_types.push_back(DT_FLOAT);  // min_summand
      input_types.push_back(DT_FLOAT);  // max_summand
    }
    if (!activation.empty()) {
      fused_ops.push_back(activation);
    }
    if (std::is_same<Toutput, qint8>::value ||
        std::is_same<Toutput, quint8>::value) {
      fused_ops.push_back("Requantize");
      input_types.push_back(DT_FLOAT);  // min_freezed_output
      input_types.push_back(DT_FLOAT);  // min_freezed_output};
    }
    TF_EXPECT_OK(
        NodeDefBuilder("quantized_conv_op", "_FusedQuantizedConv2D")
            .Attr("Thost_inputs", input_types)
            .Attr("Thost_outputs", {data_types["out_type"], DT_FLOAT, DT_FLOAT})
            .Attr("Tdevice_inputs", std::vector<DataType>())
            .Attr("Tdevice_outputs", std::vector<DataType>())
            .Attr("Tinput", data_types["Tinput"])
            .Attr("Tfilter", data_types["Tfilter"])
            .Attr("Tbias", data_types["Tbias"])
            .Attr("Tsummand", data_types["Tsummand"])
            .Attr("out_type", data_types["out_type"])
            .Attr("strides", {1, stride, stride, 1})
            .Attr("padding", padding)
            .Attr("fused_ops", fused_ops)
            .Input(FakeInput())
            .Input(FakeInput())
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    const int image_batch = 1;
    const int image_height = 4;
    const int image_width = 4;
    const int channels = 2;

    const int filter_height = 2;
    const int filter_width = 2;
    const int filter_out_channels = 2;

    // Format is NHWC
    Tensor image_float(DT_FLOAT,
                       {image_batch, image_height, image_width, channels});
    test::FillValues<float>(&image_float,
                            {2, 4, 5, 6, 1, 2, 3, 0, 1, 1, 6, 2, 6, 2, 4, 1,
                             3, 4, 3, 1, 1, 4, 0, 7, 3, 1, 5, 0, 2, 1, 3, 3});

    Tensor filter_float(
        DT_FLOAT, {filter_height, filter_width, channels, filter_out_channels});
    test::FillValues<float>(
        &filter_float, {1, -3, 0, 2, 3, -4, 0, 5, 2, 1, -1, -2, -5, 3, 4, 0});

    Tensor bias_float(DT_FLOAT, {filter_out_channels});
    test::FillValues<float>(&bias_float, {2, 4});

    Tensor summand_float(DT_FLOAT, {1, 3, 3, 2});
    test::FillValues<float>(
        &summand_float, {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    Tensor expected_float;
    RunFloatConv(image_float, filter_float, bias_float, summand_float,
                 &expected_float, /*is_depthwise*/ false, fused_ops, padding,
                 stride);

    RunQuantizedKernel<qint8, qint8, Toutput, float, Tsummand>(
        image_float, filter_float, bias_float, summand_float, expected_float,
        fused_ops);
  }
};

TEST_F(QuantizedConvTest, BiasAddFusion) {
  TestBiasAddFusion<qint8, qint32>(false, false);
}

TEST_F(QuantizedConvTest, BiasAddRequantizeFusion) {
  TestBiasAddFusion<qint8, qint8>(true, false);
}

TEST_F(QuantizedConvTest, BiasAddReluRequantizeFusion) {
  TestBiasAddFusion<qint8, qint8>(true, false, "Relu");
}

TEST_F(QuantizedConvTest, UnsignedInputBiasAddReluRequantizeFusion) {
  // We need higher tolerance for quint8 input/output
  TestBiasAddFusion<quint8, quint8>(true, false, "Relu", 4.0);
}

TEST_F(QuantizedConvTest, DWBiasAddFusion) {
  TestBiasAddFusion<qint8, qint32>(false, true);
}

TEST_F(QuantizedConvTest, DWBiasAddRequantizeFusion) {
  TestBiasAddFusion<qint8, qint8>(true, true);
}

TEST_F(QuantizedConvTest, DWBiasAddReluRequantizeFusion) {
  TestBiasAddFusion<qint8, qint8>(true, true, "Relu");
}

TEST_F(QuantizedConvTest, DWUnsignedInputBiasAddReluRequantizeFusion) {
  // We need higher tolerance for quint8 input/output
  TestBiasAddFusion<quint8, quint8>(true, true, "Relu", 4.0);
}

TEST_F(QuantizedConvTest, BiasAddSumReluRequantizeFusion) {
  TestBiasAddSumActivationFusion<quint8, quint8>("Relu");
}

TEST_F(QuantizedConvTest, BiasAddSumReluRequantizeFusionSignedSummand) {
  TestBiasAddSumActivationFusion<qint8, quint8>("Relu");
}

TEST_F(QuantizedConvTest, BiasAddSumReluFusionFloatSummand) {
  TestBiasAddSumActivationFusion<float, qint32>("Relu");
}

}  // namespace tensorflow
#endif  // INTEL_MKL && ENABLE_MKL
