/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/contrib/quantization/kernels/quantization_utils.h"
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

class QuantizedConcatTest : public OpsTestBase {
 protected:
  QuantizedConcatTest() {}
};

TEST_F(QuantizedConcatTest, Small8Bit) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_concat_op", "QuantizedConcat")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(2, DT_QUINT8))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Attr("N", 2)
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const float first_min = 0.0f;
  const float first_max = 255.0f;
  const int first_batch = 2;
  const int first_height = 2;
  const int first_width = 3;
  Tensor first_float(DT_FLOAT, {first_batch, first_height, first_width});
  test::FillValues<float>(&first_float,
                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor first_quantized =
      FloatTensorToQuantized<quint8>(first_float, first_min, first_max);

  const float second_min = 0.0f;
  const float second_max = 25.0f;
  const int second_batch = 2;
  const int second_height = 2;
  const int second_width = 3;
  Tensor second_float(DT_FLOAT, {second_batch, second_height, second_width});
  test::FillValues<float>(&second_float,
                          {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  Tensor second_quantized =
      FloatTensorToQuantized<quint8>(second_float, second_min, second_max);

  const int expected_batch = first_batch + second_batch;
  Tensor expected_float(DT_FLOAT, {expected_batch, first_height, first_width});
  test::FillValues<float>(&expected_float,
                          {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<quint8>(first_quantized.shape(),
                            first_quantized.flat<quint8>());
  AddInputFromArray<quint8>(second_quantized.shape(),
                            second_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({}), {first_min});
  AddInputFromArray<float>(TensorShape({}), {second_min});
  AddInputFromArray<float>(TensorShape({}), {first_max});
  AddInputFromArray<float>(TensorShape({}), {second_max});
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<quint8>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 0.2);
}

TEST_F(QuantizedConcatTest, Small32Bit) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_concat_op", "QuantizedConcat")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(2, DT_QINT32))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Attr("N", 2)
                   .Attr("T", DataTypeToEnum<qint32>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const float first_min = 0.0f;
  const float first_max = 1200.0f;
  const int first_batch = 2;
  const int first_height = 2;
  const int first_width = 3;
  Tensor first_float(DT_FLOAT, {first_batch, first_height, first_width});
  test::FillValues<float>(&first_float, {100, 200, 300, 400, 500, 600, 700, 800,
                                         900, 1000, 1100, 1200});
  Tensor first_quantized =
      FloatTensorToQuantized<qint32>(first_float, first_min, first_max);

  const float second_min = 0.0f;
  const float second_max = 2400.0f;
  const int second_batch = 2;
  const int second_height = 2;
  const int second_width = 3;
  Tensor second_float(DT_FLOAT, {second_batch, second_height, second_width});
  test::FillValues<float>(&second_float, {1300, 1400, 1500, 1600, 1700, 1800,
                                          1900, 2000, 2100, 2200, 2300, 2400});
  Tensor second_quantized =
      FloatTensorToQuantized<qint32>(second_float, second_min, second_max);

  const int expected_batch = first_batch + second_batch;
  Tensor expected_float(DT_FLOAT, {expected_batch, first_height, first_width});
  test::FillValues<float>(
      &expected_float,
      {100,  200,  300,  400,  500,  600,  700,  800,  900,  1000, 1100, 1200,
       1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400});

  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<qint32>(first_quantized.shape(),
                            first_quantized.flat<qint32>());
  AddInputFromArray<qint32>(second_quantized.shape(),
                            second_quantized.flat<qint32>());
  AddInputFromArray<float>(TensorShape({}), {first_min});
  AddInputFromArray<float>(TensorShape({}), {second_min});
  AddInputFromArray<float>(TensorShape({}), {first_max});
  AddInputFromArray<float>(TensorShape({}), {second_max});
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<qint32>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 0.2);
}

TEST_F(QuantizedConcatTest, SecondDim8Bit) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_concat_op", "QuantizedConcat")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(2, DT_QUINT8))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Attr("N", 2)
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const float first_min = -10.0f;
  const float first_max = 150.0f;
  const int first_batch = 2;
  const int first_height = 2;
  const int first_width = 3;
  Tensor first_float(DT_FLOAT, {first_batch, first_height, first_width});
  test::FillValues<float>(&first_float,
                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor first_quantized =
      FloatTensorToQuantized<quint8>(first_float, first_min, first_max);

  const float second_min = 0.0f;
  const float second_max = 200.0f;
  const int second_batch = 2;
  const int second_height = 2;
  const int second_width = 3;
  Tensor second_float(DT_FLOAT, {second_batch, second_height, second_width});
  test::FillValues<float>(&second_float,
                          {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  Tensor second_quantized =
      FloatTensorToQuantized<quint8>(second_float, second_min, second_max);

  const int expected_height = first_height + second_height;
  Tensor expected_float(DT_FLOAT, {first_batch, expected_height, first_width});
  test::FillValues<float>(&expected_float,
                          {1, 2, 3, 4,  5,  6,  13, 14, 15, 16, 17, 18,
                           7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24});

  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<quint8>(first_quantized.shape(),
                            first_quantized.flat<quint8>());
  AddInputFromArray<quint8>(second_quantized.shape(),
                            second_quantized.flat<quint8>());
  AddInputFromArray<float>(TensorShape({}), {first_min});
  AddInputFromArray<float>(TensorShape({}), {second_min});
  AddInputFromArray<float>(TensorShape({}), {first_max});
  AddInputFromArray<float>(TensorShape({}), {second_max});
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->flat<float>()(0);
  const float output_max = GetOutput(2)->flat<float>()(0);
  Tensor output_float =
      QuantizedTensorToFloat<quint8>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 1.0);
}

}  // namespace tensorflow
