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
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class QuantizedConv2DTransposeTest : public OpsTestBase {
 protected:
};

TEST_F(QuantizedConv2DTransposeTest, Small) {
  const int stride = 1;
  TF_ASSERT_OK(NodeDefBuilder("quantized_deconv_op", "QuantizedConv2DTranspose")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("out_type", DataTypeToEnum<qint32>::v())
                   .Attr("strides", {1, stride, stride, 1})
                   .Attr("padding", "VALID")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  const int depth = 1;
  const int input_width = 2;
  const int input_height = 2;
  const int input_batch_count = 1;

  const float input_min = 0.0;
  const float input_max = 2.0;

  Tensor input_float(DT_FLOAT,
                     {input_batch_count, input_height, input_width, depth});
  test::FillValues<float>(&input_float, {1, 1, 1, 1});
  Tensor input_quantized =
      FloatTensorToQuantized<quint8>(input_float, input_min, input_max);

  const int filter_size = 3;
  const int filter_count = 1;
  const float filter_min = 1.0;
  const float filter_max = 3.0;
  Tensor filter_float(DT_FLOAT,
                      {filter_size, filter_size, depth, filter_count});
  test::FillValues<float>(&filter_float, {2, 2, 2, 2, 2, 2, 2, 2, 2});
  Tensor filter_quantized =
      FloatTensorToQuantized<quint8>(filter_float, filter_min, filter_max);

  AddInputFromArray<int32>(TensorShape({4}), {1, 4, 4, 1});
  AddInputFromArray<quint8>(filter_quantized.shape(),
                            filter_quantized.flat<quint8>());
  AddInputFromArray<quint8>(input_quantized.shape(),
                            input_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {input_min});
  AddInputFromArray<float>(TensorShape({1}), {input_max});
  AddInputFromArray<float>(TensorShape({1}), {filter_min});
  AddInputFromArray<float>(TensorShape({1}), {filter_max});
  TF_ASSERT_OK(RunOpKernel());

  const int expected_width = 4;
  const int expected_height = 4;
  Tensor expected_float(
      DT_FLOAT, TensorShape({input_batch_count, expected_height, expected_width,
                             filter_count}));
  test::FillValues<float>(&expected_float,
                          {2, 4, 4, 2, 4, 8, 8, 4, 4, 8, 8, 4, 2, 4, 4, 2});
  const Tensor& output_quantized = *GetOutput(0);
  std::cout << output_quantized.tensor<qint32, 4>() << std::endl;
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<qint32>(output_quantized, output_min, output_max);

  test::ExpectTensorNear<float>(expected_float, output_float, 1.0);
}

TEST_F(QuantizedConv2DTransposeTest, SmallWithStrideLargerThanOne) {
  const int stride = 2;
  TF_ASSERT_OK(NodeDefBuilder("quantized_deconv_op", "QuantizedConv2DTranspose")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("out_type", DataTypeToEnum<qint32>::v())
                   .Attr("strides", {1, stride, stride, 1})
                   .Attr("padding", "VALID")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  const int depth = 1;
  const int input_width = 2;
  const int input_height = 2;
  const int input_batch_count = 1;

  const float input_min = 0.0;
  const float input_max = 2.0;

  Tensor input_float(DT_FLOAT,
                     {input_batch_count, input_height, input_width, depth});
  test::FillValues<float>(&input_float, {1, 1, 1, 1});
  Tensor input_quantized =
      FloatTensorToQuantized<quint8>(input_float, input_min, input_max);

  const int filter_size = 3;
  const int filter_count = 1;
  const float filter_min = 0.0;
  const float filter_max = 2.0;
  Tensor filter_float(DT_FLOAT,
                      {filter_size, filter_size, depth, filter_count});
  test::FillValues<float>(&filter_float, {1, 1, 1, 1, 1, 1, 1, 1, 1});
  Tensor filter_quantized =
      FloatTensorToQuantized<quint8>(filter_float, filter_min, filter_max);

  AddInputFromArray<int32>(TensorShape({4}), {1, 5, 5, 1});
  AddInputFromArray<quint8>(filter_quantized.shape(),
                            filter_quantized.flat<quint8>());
  AddInputFromArray<quint8>(input_quantized.shape(),
                            input_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({1}), {input_min});
  AddInputFromArray<float>(TensorShape({1}), {input_max});
  AddInputFromArray<float>(TensorShape({1}), {filter_min});
  AddInputFromArray<float>(TensorShape({1}), {filter_max});
  TF_ASSERT_OK(RunOpKernel());

  const int expected_width = 5;
  const int expected_height = 5;
  Tensor expected_float(
      DT_FLOAT, TensorShape({input_batch_count, expected_height, expected_width,
                             filter_count}));
  test::FillValues<float>(&expected_float,
                          {1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 4,
                           2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1});
  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<qint32>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 1.0);
}

}  // tensorflow
