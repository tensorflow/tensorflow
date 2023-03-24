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
#if defined(INTEL_MKL) && defined(ENABLE_MKL)
#define EIGEN_USE_THREADS

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

namespace tensorflow {

class QuantizedPoolingTest : public OpsTestBase {};

class QuantizedMaxPooling3DTest : public OpsTestBase {
 protected:
  void RunFloatPooling(const Tensor& input, const int& ksize, const int& stride,
                       const string& padding, Tensor* output) {
    auto root = tensorflow::Scope::NewRootScope();
    string op_name = "maxpool3d";
    auto input_op =
        ops::Const(root.WithOpName("input"), Input::Initializer(input));
    Output out_op = ops::MaxPool3D(root.WithOpName(op_name), input_op,
                                   {1, ksize, ksize, ksize, 1},
                                   {1, stride, stride, stride, 1}, padding);
    tensorflow::GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    MklTestingUtil::RunGraph(graph_def, op_name, output);
  }

  template <typename T>
  void RunQuantizedPooling(const Tensor& input_float, const int& ksize,
                           const int& stride, const string& padding,
                           Tensor* output) {
    DataType type = DataTypeToEnum<T>::v();
    TF_ASSERT_OK(
        NodeDefBuilder("quantized_max_pool_3d_op", "_QuantizedMaxPool3D")
            .Input(FakeInput(type))
            .Input(FakeInput(DT_FLOAT))
            .Input(FakeInput(DT_FLOAT))
            .Attr("T", type)
            .Attr("ksize", {1, ksize, ksize, ksize, 1})
            .Attr("strides", {1, stride, stride, stride, 1})
            .Attr("padding", padding)
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    float input_min, input_max;
    MklTestingUtil::ComputeMinMax<float>(input_float, &input_min, &input_max);

    Tensor input_quantized;
    MklTestingUtil::RunMklQuantizeOp(input_float, input_min, input_max, type,
                                     "SCALED", &input_quantized);

    AddInputFromArray<T>(input_quantized.shape(), input_quantized.flat<T>());
    AddInputFromArray<float>(TensorShape({}), {input_min});
    AddInputFromArray<float>(TensorShape({}), {input_max});

    TF_ASSERT_OK(RunOpKernel());

    Tensor& output_quantized = *GetOutput(0);
    Tensor& output_min = *GetOutput(1);
    Tensor& output_max = *GetOutput(2);

    MklTestingUtil::RunDequantizeOp(output_quantized, output_min, output_max,
                                    "SCALED", output);
  }
};

TEST_F(QuantizedMaxPooling3DTest, QUINT8_INPUT) {
  const int ksize = 2;
  const int stride = 2;
  const string padding = "SAME";

  const int input_depth = 2;
  const int input_height = 4;
  const int input_width = 4;
  const int input_channels = 2;
  Tensor input_float(
      DT_FLOAT, {1, input_depth, input_height, input_width, input_channels});
  test::FillValues<float>(
      &input_float,
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
       32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
       1,  2,  3,  4,  5,  6,  7,  8,  16, 15, 14, 13, 12, 11, 10, 9});

  Tensor expected_output;
  RunFloatPooling(input_float, ksize, stride, padding, &expected_output);

  Tensor output_float;
  RunQuantizedPooling<quint8>(input_float, ksize, stride, padding,
                              &output_float);

  test::ExpectTensorNear<float>(expected_output, output_float, 0.3);
}

TEST_F(QuantizedMaxPooling3DTest, QINT8_INPUT) {
  const int ksize = 2;
  const int stride = 2;
  const string padding = "SAME";

  const int input_depth = 2;
  const int input_height = 4;
  const int input_width = 4;
  const int input_channels = 2;
  Tensor input_float(
      DT_FLOAT, {1, input_depth, input_height, input_width, input_channels});
  test::FillValues<float>(
      &input_float,
      {1,  -2, 3,  4,   5,  6,  -7, 8,  9,   -10, 11, 12,  13, 14,  -15, -16,
       17, 18, 19, -20, 21, 22, 23, 24, -25, 26,  27, 28,  29, -30, 31,  32,
       32, 31, 30, -29, 28, 27, 26, 25, -24, 23,  22, -21, 20, 19,  18,  -17,
       1,  2,  -3, 4,   5,  -6, 7,  8,  16,  15,  14, -13, 12, -11, 10,  9});

  Tensor expected_output;
  RunFloatPooling(input_float, ksize, stride, padding, &expected_output);

  Tensor output_float;
  RunQuantizedPooling<qint8>(input_float, ksize, stride, padding,
                             &output_float);

  test::ExpectTensorNear<float>(expected_output, output_float, 0.3);
}

TEST_F(QuantizedPoolingTest, SmallAveragePooling) {
  const int ksize = 2;
  const int stride = 2;
  TF_ASSERT_OK(NodeDefBuilder("quantized_avg_pool_op", "_MklQuantizedAvgPool")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("ksize", {1, ksize, ksize, 1})
                   .Attr("strides", {1, stride, stride, 1})
                   .Attr("padding", "SAME")
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const float input_min = 0.0f;
  const float input_max = 255.0f;
  const int input_height = 4;
  const int input_width = 4;
  const int input_channels = 2;
  Tensor input_float(DT_FLOAT, {1, input_height, input_width, input_channels});
  test::FillValues<float>(
      &input_float,
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  Tensor input_quantized =
      FloatTensorToQuantized<quint8>(input_float, input_min, input_max);

  const int expected_width = input_width / stride;
  const int expected_height = input_height / stride;

  // The input pools we are averaging. (NHWC input, quantized.)
  //    0th channel       1st channel
  //    1  3 |  5  7      2  4 |  6  8
  //    9 11 | 13 15     10 12 | 14 16
  //   -------------     -------------
  //   17 19 | 21 23     18 20 | 22 24
  //   25 27 | 29 31     26 28 | 30 32
  Tensor expected_float(DT_FLOAT,
                        {1, expected_height, expected_width, input_channels});
  test::FillValues<float>(&expected_float, {6, 7, 10, 11, 22, 23, 26, 27});

  AddInputFromArray<quint8>(input_quantized.shape(),
                            input_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({}), {input_min});
  AddInputFromArray<float>(TensorShape({}), {input_max});

  TF_ASSERT_OK(RunOpKernel());

  const Tensor& output = *GetOutput(0);
  const float output_min = GetOutput(1)->scalar<float>()();
  const float output_max = GetOutput(2)->scalar<float>()();
  Tensor output_float =
      QuantizedTensorToFloat<quint8>(output, output_min, output_max);

  test::ExpectTensorNear<float>(expected_float, output_float, 0.2);
}

TEST_F(QuantizedPoolingTest, SmallMaxPooling) {
  const int ksize = 2;
  const int stride = 2;
  TF_ASSERT_OK(NodeDefBuilder("quantized_max_pool_op", "_MklQuantizedMaxPool")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("ksize", {1, ksize, ksize, 1})
                   .Attr("strides", {1, stride, stride, 1})
                   .Attr("padding", "SAME")
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const float input_min = 0.0f;
  const float input_max = 255.0f;
  const int input_height = 4;
  const int input_width = 4;
  const int input_channels = 2;
  Tensor input_float(DT_FLOAT, {1, input_height, input_width, input_channels});
  test::FillValues<float>(
      &input_float,
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32});
  Tensor input_quantized =
      FloatTensorToQuantized<quint8>(input_float, input_min, input_max);
  const int expected_width = input_width / stride;
  const int expected_height = input_height / stride;

  // The max is computed from these input pools. (NHWC input, quantized.)
  //    0th channel       1st channel
  //    1  3 |  5  7      2  4 |  6  8
  //    9 11 | 13 15     10 12 | 14 16
  //   -------------     -------------
  //   17 19 | 21 23     18 20 | 22 24
  //   25 27 | 29 31     26 28 | 30 32

  Tensor expected_float(DT_FLOAT,
                        {1, expected_height, expected_width, input_channels});
  test::FillValues<float>(&expected_float, {11, 12, 15, 16, 27, 28, 31, 32});
  AddInputFromArray<quint8>(input_quantized.shape(),
                            input_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({}), {input_min});
  AddInputFromArray<float>(TensorShape({}), {input_max});

  TF_ASSERT_OK(RunOpKernel());

  const Tensor& output = *GetOutput(0);
  const float output_min = GetOutput(1)->scalar<float>()();
  const float output_max = GetOutput(2)->scalar<float>()();
  Tensor output_float =
      QuantizedTensorToFloat<quint8>(output, output_min, output_max);

  test::ExpectTensorNear<float>(expected_float, output_float, 0.2);
}

}  // namespace tensorflow

#endif  // defined(INTEL_MKL) && defined(ENABLE_MKL)
