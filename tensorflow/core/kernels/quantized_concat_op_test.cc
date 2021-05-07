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
  void TestSmall32Bit(float first_min, float first_max, float second_min,
                      float second_max);
  void TestSecondDim8Bit(float first_min, float first_max, float second_min,
                         float second_max);
};

TEST_F(QuantizedConcatTest, Small8Bit) {
  TestSmall8Bit(0.0f, 255.0f, 0.0f, 25.0f);
}

TEST_F(QuantizedConcatTest, Small8BitSameRange) {
  // Range for both is the same, so impl can use memcpy.
  TestSmall8Bit(0.0f, 255.0f, 0.0f, 255.0f);
}

void QuantizedConcatTest::TestSmall8Bit(float first_min, float first_max,
                                        float second_min, float second_max) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_concat_op", "QuantizedConcat")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(2, DT_QUINT8))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Attr("N", 2)
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const int first_batch = 2;
  const int first_height = 2;
  const int first_width = 3;
  Tensor first_float(DT_FLOAT, {first_batch, first_height, first_width});
  test::FillValues<float>(&first_float,
                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor first_quantized =
      FloatTensorToQuantized<quint8>(first_float, first_min, first_max);

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
  TestSmall32Bit(0.0f, 1200.0f, 0.0f, 2400.0f);
}

TEST_F(QuantizedConcatTest, Small32BitSameRange) {
  TestSmall32Bit(-2400.0f, 2400.0f, -2400.0f, 2400.0f);
}

TEST_F(QuantizedConcatTest, Small32BitOneDimSameRangeAsOutput) {
  TestSmall32Bit(-2400.0f, 2400.0f, -1200.0f, 2400.0f);
}

void QuantizedConcatTest::TestSmall32Bit(float first_min, float first_max,
                                         float second_min, float second_max) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_concat_op", "QuantizedConcat")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(2, DT_QINT32))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Attr("N", 2)
                   .Attr("T", DataTypeToEnum<qint32>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const int first_batch = 2;
  const int first_height = 2;
  const int first_width = 3;
  Tensor first_float(DT_FLOAT, {first_batch, first_height, first_width});
  test::FillValues<float>(&first_float, {100, 200, 300, 400, 500, 600, 700, 800,
                                         900, 1000, 1100, 1200});
  Tensor first_quantized =
      FloatTensorToQuantized<qint32>(first_float, first_min, first_max);

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
  TestSecondDim8Bit(-10.0f, 150.0f, 0.0f, 200.0f);
}

TEST_F(QuantizedConcatTest, SecondDim8BitSameRange) {
  TestSecondDim8Bit(-10.0f, 150.0f, -10.0f, 150.0f);
}

void QuantizedConcatTest::TestSecondDim8Bit(float first_min, float first_max,
                                            float second_min,
                                            float second_max) {
  TF_ASSERT_OK(NodeDefBuilder("quantized_concat_op", "QuantizedConcat")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(2, DT_QUINT8))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Input(FakeInput(2, DT_FLOAT))
                   .Attr("N", 2)
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  const int first_batch = 2;
  const int first_height = 2;
  const int first_width = 3;
  Tensor first_float(DT_FLOAT, {first_batch, first_height, first_width});
  test::FillValues<float>(&first_float,
                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  Tensor first_quantized =
      FloatTensorToQuantized<quint8>(first_float, first_min, first_max);

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

// For the benchmark, we set up two 2-dimensional tensors, each kDim1 x 'dim'
// in size, and concat them together along "concat_dimension".
// If <same_limits> is true, then both concatenated dimensions have the same
// quantized range; otherwise, they are set to different values.
template <typename T>
static void ConcatHelper(::testing::benchmark::State& state,
                         int concat_dimension, bool same_limits, int dim2) {
  Graph* g = new Graph(OpRegistry::Global());

  DataType dt = DataTypeToEnum<T>::v();
  const int kDim1 = 100;
  TensorShape shape({kDim1, dim2});

  Tensor concat_dim = test::AsScalar<int32>(concat_dimension);
  Tensor in0(dt, shape);
  in0.flat<T>().setRandom();
  Tensor in1(dt, shape);
  in1.flat<T>().setRandom();

  Tensor mins0 = test::AsScalar<float>(-1.0);
  Tensor maxes0 = test::AsScalar<float>(1.0);
  Tensor mins1 = test::AsScalar<float>(same_limits ? -1.0 : -255.0);
  Tensor maxes1 = test::AsScalar<float>(same_limits ? 1.0 : 255.0);

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "QuantizedConcat")
                  .Input(Constant(g, concat_dim))
                  .Input({Constant(g, in0), Constant(g, in1)})
                  .Input({Constant(g, mins0), Constant(g, mins1)})
                  .Input({Constant(g, maxes0), Constant(g, maxes1)})
                  .Attr("N", 2)
                  .Attr("T", dt)
                  .Finalize(g, &node));

  test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);
  state.SetBytesProcessed(static_cast<int64>(state.iterations()) *
                          ((kDim1 * dim2) + (kDim1 * dim2)) * sizeof(T));
}

static void BM_QConcatDim0SameLimitQInt32(::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 0 /* concat_dimension */, true /* same_limits */,
                       dim2);
}

static void BM_QConcatDim1SameLimitQInt32(::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 1 /* concat_dimension */, true /* same_limits */,
                       dim2);
}

static void BM_QConcatDim0DifferLimitQInt32(
    ::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 0 /* concat_dimension */, false /* same_limits */,
                       dim2);
}

static void BM_QConcatDim1DifferLimitQInt32(
    ::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 1 /* concat_dimension */, false /* same_limits */,
                       dim2);
}

BENCHMARK(BM_QConcatDim0SameLimitQInt32)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);
BENCHMARK(BM_QConcatDim1SameLimitQInt32)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);
BENCHMARK(BM_QConcatDim0DifferLimitQInt32)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);
BENCHMARK(BM_QConcatDim1DifferLimitQInt32)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);

static void BM_QConcatDim0SameLimitQUint8(::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 0 /* concat_dimension */, true /* same_limits */,
                       dim2);
}

static void BM_QConcatDim1SameLimitQUint8(::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 1 /* concat_dimension */, true /* same_limits */,
                       dim2);
}

static void BM_QConcatDim0DifferLimitQUint8(
    ::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 0 /* concat_dimension */, false /* same_limits */,
                       dim2);
}

static void BM_QConcatDim1DifferLimitQUint8(
    ::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatHelper<qint32>(state, 1 /* concat_dimension */, false /* same_limits */,
                       dim2);
}

BENCHMARK(BM_QConcatDim0SameLimitQUint8)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);
BENCHMARK(BM_QConcatDim1SameLimitQUint8)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);
BENCHMARK(BM_QConcatDim0DifferLimitQUint8)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);
BENCHMARK(BM_QConcatDim1DifferLimitQUint8)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(20000)
    ->Arg(100000);

}  // namespace tensorflow
