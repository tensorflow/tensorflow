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

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class ReverseOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType data_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "Reverse")
                     .Input(FakeInput(data_type))
                     .Input(FakeInput())
                     .Attr("T", data_type)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  template <typename T>
  void Reverse_0() {
    MakeOp(DataTypeToEnum<T>::value);
    AddInputFromArray<T>(TensorShape({}), {3});
    AddInputFromArray<bool>(TensorShape({}), {true});
    TF_ASSERT_OK(RunOpKernel());

    Tensor* output = GetOutput(0);
    Tensor expected(allocator(), DataTypeToEnum<T>::value, TensorShape({}));
    expected.scalar<T>() = expected.scalar<T>().constant(3);
    test::ExpectTensorEqual<T>(expected, *output);
  }

  template <typename T>
  void Reverse_234() {
    MakeOp(DataTypeToEnum<T>::value);
    // Feed and run
    // [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    //  [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
    AddInputFromArray<T>(TensorShape({2, 3, 4}),
                         {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    AddInputFromArray<bool>(TensorShape({3}), {true, false, true});

    TF_ASSERT_OK(RunOpKernel());

    // Check the new state of the input
    Tensor* params_tensor = GetOutput(0);
    Tensor expected(allocator(), DataTypeToEnum<T>::value,
                    TensorShape({2, 3, 4}));
    // Should become
    // [[[15, 14, 13, 12], [19, 18, 17, 16], [23, 22, 21, 20]]
    //  [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]]]
    test::FillValues<T>(&expected,
                        {15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20,
                         3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8});
    test::ExpectTensorEqual<T>(expected, *params_tensor);
  }

  template <typename T>
  void Reverse_1234() {
    MakeOp(DataTypeToEnum<T>::value);
    // Feed and run
    // [[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    //   [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]]
    AddInputFromArray<T>(TensorShape({1, 2, 3, 4}),
                         {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    AddInputFromArray<bool>(TensorShape({4}), {true, true, false, true});

    TF_ASSERT_OK(RunOpKernel());

    // Check the new state of the input
    Tensor* params_tensor = GetOutput(0);
    Tensor expected(allocator(), DataTypeToEnum<T>::value,
                    TensorShape({1, 2, 3, 4}));
    // Should become
    // [[[[15, 14, 13, 12], [19, 18, 17, 16], [23, 22, 21, 20]]
    //   [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]]]]
    test::FillValues<T>(&expected,
                        {15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20,
                         3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8});
    test::ExpectTensorEqual<T>(expected, *params_tensor);
  }
};

TEST_F(ReverseOpTest, Reverse_0_uint8) { Reverse_0<uint8>(); }

TEST_F(ReverseOpTest, Reverse_0_int8) { Reverse_0<int8>(); }

TEST_F(ReverseOpTest, Reverse_0_uint16) { Reverse_0<uint16>(); }

TEST_F(ReverseOpTest, Reverse_0_int16) { Reverse_0<int16>(); }

TEST_F(ReverseOpTest, Reverse_0_float) { Reverse_0<float>(); }

TEST_F(ReverseOpTest, Reverse_0_int32) { Reverse_0<int32>(); }

TEST_F(ReverseOpTest, Reverse_0_int64) { Reverse_0<int64>(); }

TEST_F(ReverseOpTest, Reverse_0_double) { Reverse_0<double>(); }

TEST_F(ReverseOpTest, Reverse_0_complex64) { Reverse_0<complex64>(); }

TEST_F(ReverseOpTest, Reverse_0_complex128) { Reverse_0<complex128>(); }

TEST_F(ReverseOpTest, Reverse_234_uint8) { Reverse_234<uint8>(); }

TEST_F(ReverseOpTest, Reverse_234_int8) { Reverse_234<int8>(); }

TEST_F(ReverseOpTest, Reverse_234_uint16) { Reverse_234<uint16>(); }

TEST_F(ReverseOpTest, Reverse_234_int16) { Reverse_234<int16>(); }

TEST_F(ReverseOpTest, Reverse_234_float) { Reverse_234<float>(); }

TEST_F(ReverseOpTest, Reverse_234_int32) { Reverse_234<int32>(); }

TEST_F(ReverseOpTest, Reverse_234_int64) { Reverse_234<int64>(); }

TEST_F(ReverseOpTest, Reverse_234_double) { Reverse_234<double>(); }

TEST_F(ReverseOpTest, Reverse_234_complex64) { Reverse_234<complex64>(); }

TEST_F(ReverseOpTest, Reverse_234_complex128) { Reverse_234<complex128>(); }

TEST_F(ReverseOpTest, Reverse_1234_uint8) { Reverse_1234<uint8>(); }

TEST_F(ReverseOpTest, Reverse_1234_int8) { Reverse_1234<int8>(); }

TEST_F(ReverseOpTest, Reverse_1234_uint16) { Reverse_1234<uint16>(); }

TEST_F(ReverseOpTest, Reverse_1234_int16) { Reverse_1234<int16>(); }

TEST_F(ReverseOpTest, Reverse_1234_float) { Reverse_1234<float>(); }

TEST_F(ReverseOpTest, Reverse_1234_int32) { Reverse_1234<int32>(); }

TEST_F(ReverseOpTest, Reverse_1234_int64) { Reverse_1234<int64>(); }

TEST_F(ReverseOpTest, Reverse_1234_double) { Reverse_1234<double>(); }

TEST_F(ReverseOpTest, Reverse_1234_complex64) { Reverse_1234<complex64>(); }

TEST_F(ReverseOpTest, Reverse_1234_complex128) { Reverse_1234<complex128>(); }

static SessionOptions GetOptions(int intra_threads) {
  SessionOptions opts;
  opts.config.set_intra_op_parallelism_threads(intra_threads);
  opts.config.set_inter_op_parallelism_threads(1);
  return opts;
}

// Creates a Graph which "reduce"s a 3D float tensor of "num" elements
// into a scalar.
template <typename T>
static Graph* Reverse(const TensorShape& shape, int reverse_axis) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor data(DataTypeToEnum<T>::value, shape);
  data.flat<T>().setRandom();
  Tensor axes(DT_INT32, TensorShape({1}));
  axes.flat<int32>()(0) = reverse_axis;
  test::graph::Reverse(g, test::graph::Constant(g, data),
                       test::graph::Constant(g, axes));
  return g;
}

template <typename T>
static void RunReverseRowsBenchmark(int iters, int outer_dim, int middle_dim,
                                    int intra_threads, int channels) {
  SessionOptions opts = GetOptions(intra_threads);
  TensorShape shape{outer_dim, middle_dim, channels};
  const int64 num_items = static_cast<int64>(iters) * shape.num_elements();
  testing::ItemsProcessed(num_items);
  testing::BytesProcessed(num_items * sizeof(T));
  testing::UseRealTime();
  test::Benchmark("cpu", Reverse<T>(shape, 1), &opts).Run(iters);
}

static void BM_ReverseRowsOf1Channel_1T_float(int iters, int outer_dim,
                                              int middle_dim) {
  RunReverseRowsBenchmark<float>(iters, outer_dim, middle_dim,
                                 1 /* intra_threads */, 1 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf1Channel_1T_float)
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

static void BM_ReverseRowsOf1Channel_1T_uint8(int iters, int outer_dim,
                                              int middle_dim) {
  RunReverseRowsBenchmark<uint8>(iters, outer_dim, middle_dim,
                                 1 /* intra_threads */, 1 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf1Channel_1T_uint8)
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

static void BM_ReverseRowsOf1Channel_4T_float(int iters, int outer_dim,
                                              int middle_dim) {
  RunReverseRowsBenchmark<float>(iters, outer_dim, middle_dim,
                                 4 /* intra_threads */, 1 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf1Channel_4T_float)
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

static void BM_ReverseRowsOf1Channel_4T_uint8(int iters, int outer_dim,
                                              int middle_dim) {
  RunReverseRowsBenchmark<uint8>(iters, outer_dim, middle_dim,
                                 4 /* intra_threads */, 1 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf1Channel_4T_uint8)
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

static void BM_ReverseRowsOf3Channels_1T_float(int iters, int outer_dim,
                                               int middle_dim) {
  RunReverseRowsBenchmark<float>(iters, outer_dim, middle_dim,
                                 1 /* intra_threads */, 3 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf3Channels_1T_float)
    ->ArgPair(288, 288)
    ->ArgPair(30, 30)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

static void BM_ReverseRowsOf3Channels_1T_uint8(int iters, int outer_dim,
                                               int middle_dim) {
  RunReverseRowsBenchmark<uint8>(iters, outer_dim, middle_dim,
                                 1 /* intra_threads */, 3 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf3Channels_1T_uint8)
    ->ArgPair(288, 288)
    ->ArgPair(30, 30)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

static void BM_ReverseRowsOf3Channels_4T_float(int iters, int outer_dim,
                                               int middle_dim) {
  RunReverseRowsBenchmark<float>(iters, outer_dim, middle_dim,
                                 4 /* intra_threads */, 3 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf3Channels_4T_float)
    ->ArgPair(288, 288)
    ->ArgPair(30, 30)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

static void BM_ReverseRowsOf3Channels_4T_uint8(int iters, int outer_dim,
                                               int middle_dim) {
  RunReverseRowsBenchmark<uint8>(iters, outer_dim, middle_dim,
                                 4 /* intra_threads */, 3 /* channels */);
}
BENCHMARK(BM_ReverseRowsOf3Channels_4T_uint8)
    ->ArgPair(288, 288)
    ->ArgPair(30, 30)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

static void BM_ReverseRowsOf4Channels_1T_float(int iters, int outer_dim,
                                               int middle_dim) {
  RunReverseRowsBenchmark<float>(iters, outer_dim, middle_dim,
                                 1 /* intra_threads */, 4 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf4Channels_1T_float)
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

static void BM_ReverseRowsOf4Channels_1T_uint8(int iters, int outer_dim,
                                               int middle_dim) {
  RunReverseRowsBenchmark<uint8>(iters, outer_dim, middle_dim,
                                 1 /* intra_threads */, 4 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf4Channels_1T_uint8)
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

static void BM_ReverseRowsOf4Channels_4T_float(int iters, int outer_dim,
                                               int middle_dim) {
  RunReverseRowsBenchmark<float>(iters, outer_dim, middle_dim,
                                 4 /* intra_threads */, 4 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf4Channels_4T_float)
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

static void BM_ReverseRowsOf4Channels_4T_uint8(int iters, int outer_dim,
                                               int middle_dim) {
  RunReverseRowsBenchmark<uint8>(iters, outer_dim, middle_dim,
                                 4 /* intra_threads */, 4 /* channels */);
}

BENCHMARK(BM_ReverseRowsOf4Channels_4T_uint8)
    ->ArgPair(288, 288)
    ->ArgPair(1024, 1024)
    ->ArgPair(10 * 1024, 1024);

}  // namespace
}  // namespace tensorflow
