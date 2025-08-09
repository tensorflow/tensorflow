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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

using test::graph::Constant;

class QuantizedConcatTest : public OpsTestBase {
 protected:
  QuantizedConcatTest() {}

  void TestSmall8Bit(float first_min, float first_max, float second_min,
                     float second_max);
  void TestSecondDim8Bit(float first_min, float first_max, float second_min,
                         float second_max);
};

TEST_F(QuantizedConcatTest, Small8BitSameRange) {
  // Range for both is the same, so impl can use memcpy.
  TestSmall8Bit(0.0f, 255.0f, 0.0f, 255.0f);
}

TEST_F(QuantizedConcatTest, Small8BitDifferentRange) {
  TestSmall8Bit(0.0f, 255.0f, 0.0f, 25.0f);
}

void QuantizedConcatTest::TestSmall8Bit(float first_min, float first_max,
                                        float second_min, float second_max) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_concat_op", "_MklQuantizedConcatV2")
                   .Input(FakeInput(2, DT_QUINT8))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Attr("N", 2)
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("Tidx", DT_INT32)
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const int first_batch = 2;
  const int first_height = 2;
  const int first_width = 3;
  const int first_depth = 1;
  Tensor first_float(DT_FLOAT,
                     {first_batch, first_height, first_width, first_depth});
  test::FillValues<float>(&first_float,
                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor first_quantized =
      FloatTensorToQuantized<quint8>(first_float, first_min, first_max);

  const int second_batch = 2;
  const int second_height = 2;
  const int second_width = 3;
  const int second_depth = 1;
  Tensor second_float(
      DT_FLOAT, {second_batch, second_height, second_width, second_depth});
  test::FillValues<float>(&second_float,
                          {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  Tensor second_quantized =
      FloatTensorToQuantized<quint8>(second_float, second_min, second_max);

  const int expected_batch = first_batch + second_batch;
  Tensor expected_float(
      DT_FLOAT, {expected_batch, first_height, first_width, first_depth});
  test::FillValues<float>(&expected_float,
                          {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                           13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  AddInputFromArray<quint8>(first_quantized.shape(),
                            first_quantized.flat<quint8>());
  AddInputFromArray<quint8>(second_quantized.shape(),
                            second_quantized.flat<quint8>());
  AddInputFromArray<int32>(TensorShape({}), {0});
  AddInputFromArray<float>(TensorShape({}), {first_min});
  AddInputFromArray<float>(TensorShape({}), {second_min});
  AddInputFromArray<float>(TensorShape({}), {first_max});
  AddInputFromArray<float>(TensorShape({}), {second_max});
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->scalar<float>()();
  const float output_max = GetOutput(2)->scalar<float>()();
  Tensor output_float =
      QuantizedTensorToFloat<quint8>(output_quantized, output_min, output_max);
  test::ExpectTensorNear<float>(expected_float, output_float, 0.2);
}

TEST_F(QuantizedConcatTest, SecondDim8BitSameRange) {
  TestSecondDim8Bit(-10.0f, 150.0f, -10.0f, 150.0f);
}

void QuantizedConcatTest::TestSecondDim8Bit(float first_min, float first_max,
                                            float second_min,
                                            float second_max) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_concat_op", "_MklQuantizedConcatV2")
                   .Input(FakeInput(2, DT_QUINT8))
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Attr("N", 2)
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("Tidx", DT_INT32)
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const int first_batch = 2;
  const int first_height = 2;
  const int first_width = 3;
  const int first_depth = 1;
  Tensor first_float(DT_FLOAT,
                     {first_batch, first_height, first_width, first_depth});
  test::FillValues<float>(&first_float,
                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor first_quantized =
      FloatTensorToQuantized<quint8>(first_float, first_min, first_max);

  const int second_batch = 2;
  const int second_height = 2;
  const int second_width = 3;
  const int second_depth = 1;

  Tensor second_float(
      DT_FLOAT, {second_batch, second_height, second_width, second_depth});
  test::FillValues<float>(&second_float,
                          {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
  Tensor second_quantized =
      FloatTensorToQuantized<quint8>(second_float, second_min, second_max);

  const int expected_height = first_height + second_height;
  Tensor expected_float(
      DT_FLOAT, {first_batch, expected_height, first_width, first_depth});
  test::FillValues<float>(&expected_float,
                          {1, 2, 3, 4,  5,  6,  13, 14, 15, 16, 17, 18,
                           7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24});

  AddInputFromArray<quint8>(first_quantized.shape(),
                            first_quantized.flat<quint8>());
  AddInputFromArray<quint8>(second_quantized.shape(),
                            second_quantized.flat<quint8>());
  AddInputFromArray<int32>(TensorShape({}), {1});
  AddInputFromArray<float>(TensorShape({}), {first_min});
  AddInputFromArray<float>(TensorShape({}), {second_min});
  AddInputFromArray<float>(TensorShape({}), {first_max});
  AddInputFromArray<float>(TensorShape({}), {second_max});
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_quantized = *GetOutput(0);
  const float output_min = GetOutput(1)->scalar<float>()();
  const float output_max = GetOutput(2)->scalar<float>()();
  Tensor output_float =
      QuantizedTensorToFloat<quint8>(output_quantized, output_min, output_max);
  // Using the same error tolerance as in Eigen QuantizedConcat test
  test::ExpectTensorNear<float>(expected_float, output_float, 1.0);
}

}  // namespace tensorflow

#endif  // INTEL_MKL
