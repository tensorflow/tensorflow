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

#ifdef INTEL_MKL
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

// Some helper constants
static const uint8 dummy_tensor[] = {0, 0, 0, 0, 0, 0, 0, 0};
static const TensorShape dummy_shape({8});

// Helper class for converting MKL tensors to TF tensors
class ConvMklToTF : public OpsTestBase {
 public:
  template <typename T>
  void ConvertMKL2TF(DataType dtype, const Tensor& first, const Tensor& second,
                     Tensor& output) {
    // Create an MKL to TF conversion node and execute it
    TF_EXPECT_OK(NodeDefBuilder("mkl_to_tf_op", "_MklToTf")
                     .Input(FakeInput(dtype))     // Input
                     .Input(FakeInput(DT_UINT8))  // Mkl second tensor
                     .Attr("T", dtype)
                     .Attr("_kernel", "MklOp")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    AddInputFromArray<T>(first.shape(), first.flat<T>());
    AddInputFromArray<uint8>(second.shape(), second.flat<uint8>());
    TF_ASSERT_OK(RunOpKernel());

    output = *GetOutput(0);
  }
  void TestBody(){};
};

class QuantizedConv2DPerchannelTest : public OpsTestBase {};

TEST_F(QuantizedConv2DPerchannelTest, Small) {
  const int stride = 1;
  TF_ASSERT_OK(NodeDefBuilder("quantized_conv_perchannel_op",
                              "_MklQuantizedConv2DPerChannel")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   // MKL metadata tensors
                   .Input(FakeInput(DT_UINT8))
                   .Input(FakeInput(DT_UINT8))
                   .Input(FakeInput(DT_UINT8))
                   .Input(FakeInput(DT_UINT8))
                   .Input(FakeInput(DT_UINT8))
                   .Input(FakeInput(DT_UINT8))
                   // Attributes
                   .Attr("Tinput", DataTypeToEnum<quint8>::v())
                   .Attr("Tfilter", DataTypeToEnum<qint8>::v())
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("out_type", DataTypeToEnum<qint32>::v())
                   .Attr("strides", {1, stride, stride, 1})
                   .Attr("padding", "SAME")
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Image shape
  const int image_batch_count = 1;
  const int image_height = 3;
  const int image_width = 4;
  const int image_channel = 1;

  // Image is of datatype uint8
  const float image_min = 0.0f;
  const float image_max = 255.0f;

  // The image matrix is:
  // |  1 |  2 |  3 |  4 |
  // |  5 |  6 |  7 |  8 |
  // |  9 | 10 | 11 | 12 |
  Tensor image_float(
      DT_FLOAT, {image_batch_count, image_height, image_width, image_channel});
  test::FillValues<float>(&image_float,
                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  // Create image tensor
  Tensor image_quantized =
      FloatTensorToQuantized<quint8>(image_float, image_min, image_max);

  // Filter shape
  const int filter_height = 3;
  const int filter_width = 3;
  const int filter_channel = 1;
  const int filter_count = 2;

  // Filter is of datatype int8
  const float filter_min = -128.0f;
  const float filter_max = 127.0f;

  // The filter matrix is for each channel:
  // | 1 | 4 | 7 |
  // | 2 | 5 | 8 |
  // | 3 | 6 | 9 |
  Tensor filter_float(
      DT_FLOAT, {filter_height, filter_width, filter_channel, filter_count});
  test::FillValues<float>(
      &filter_float, {1, 1, 4, 4, 7, 7, 2, 2, 5, 5, 8, 8, 3, 3, 6, 6, 9, 9});

  // Create filter tensor
  Tensor filter_quantized =
      FloatTensorToQuantized<qint8>(filter_float, filter_min, filter_max);

  // Add the tensors as input to the current Op.
  AddInputFromArray<quint8>(image_quantized.shape(),
                            image_quantized.flat<quint8>());
  AddInputFromArray<qint8>(filter_quantized.shape(),
                           filter_quantized.flat<qint8>());
  AddInputFromArray<float>(TensorShape({1}), {image_min});
  AddInputFromArray<float>(TensorShape({1}), {image_max});
  AddInputFromArray<float>(TensorShape({2}), {filter_min, filter_min});
  AddInputFromArray<float>(TensorShape({2}), {filter_max, filter_max});
  AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
  AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
  AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
  AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
  AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
  AddInputFromArray<uint8>(dummy_shape, dummy_tensor);

  // Run the Op Kernel.
  TF_ASSERT_OK(RunOpKernel());

  // Get the output
  const Tensor& output = *GetOutput(0);
  const Tensor& output_mkl_metadata = *GetOutput(3);

  // Convert the output tensor in MKL to TF format.
  ConvMklToTF conv_comp;
  Tensor output_quantized;
  conv_comp.ConvertMKL2TF<qint32>(DT_QINT32, output, output_mkl_metadata,
                                  output_quantized);

  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<qint32>(output_quantized, output_min, output_max);

  // Get the Expected Output tensor.
  // We're sliding the 3x3 filter across the 3x4 image, with accesses outside
  // the input dimensions set to zero because we're using the 'SAME' padding
  // mode.
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
  // (1*7)+(4*11)+(7*0)+(2*8)+(5*12)+(8*0)+(3*0)+(6*0)+(9*0)=121

  // This means we should end up with this matrix for each channel:
  // |  105  |  150  |  183  |   95  |
  // |  235  |  312  |  357  |  178  |
  // |  187  |  234  |  261  |  121  |

  // Shape of expected (output) tensor: N x IH x IW x filter_count
  // Create the expectated output tensor
  const int expected_width = image_width;
  const int expected_height = image_height;

  Tensor expected_float(
      DT_FLOAT, TensorShape({image_batch_count, expected_height, expected_width,
                             filter_count}));

  test::FillValues<float>(
      &expected_float,
      {105, 105, 150, 150, 183, 183, 95,  95,  235, 235, 312, 312,
       357, 357, 178, 178, 187, 187, 234, 234, 261, 261, 121, 121});

  // Test whether the values are as expected.
  test::ExpectTensorNear<float>(expected_float, output_float, 2.0);
}

}  // namespace tensorflow
#endif  // INTEL_MKL