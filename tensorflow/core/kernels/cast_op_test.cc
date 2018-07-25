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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

using Eigen::half;

namespace tensorflow {

template <typename Src, typename Dst>
static Graph* Cast(int num) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor data(DataTypeToEnum<Src>::value,
              TensorShape({64, 64, num / (64 * 64)}));
  data.flat<Src>().setRandom();
  test::graph::Cast(g, test::graph::Constant(g, data),
                    DataTypeToEnum<Dst>::value);
  return g;
}

class CastOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType src, DataType dst, bool trunc = false) {
    if (trunc) {
      TF_EXPECT_OK(NodeDefBuilder("cast_op", "Cast")
                       .Input(FakeInput(src))
                       .Attr("SrcT", src)
                       .Attr("DstT", dst)
                       .Attr("Truncate", true)
                       .Finalize(node_def()));
    } else {
      TF_EXPECT_OK(NodeDefBuilder("cast_op", "Cast")
                       .Input(FakeInput(src))
                       .Attr("SrcT", src)
                       .Attr("DstT", dst)
                       .Finalize(node_def()));
    }

    TF_EXPECT_OK(InitOp());
  }

  template <typename INPUT, typename OUTPUT>
  void CheckCast(bool trunc = false) {
    DataType in_type = DataTypeToEnum<INPUT>::v();
    DataType out_type = DataTypeToEnum<OUTPUT>::v();
    MakeOp(in_type, out_type);
    AddInputFromArray<INPUT>(TensorShape({1, 2, 2, 1}),
                             {INPUT(1), INPUT(2), INPUT(3), INPUT(4)});
    TF_ASSERT_OK(RunOpKernel());
    Tensor expected(allocator(), out_type, TensorShape({1, 2, 2, 1}));
    test::FillValues<OUTPUT>(&expected,
                             {OUTPUT(1), OUTPUT(2), OUTPUT(3), OUTPUT(4)});
    test::ExpectTensorEqual<OUTPUT>(expected, *GetOutput(0));
  }
};

#define TEST_CAST(in, out)                                              \
  TEST_F(CastOpTest, TestCast##_##in##_##out) { CheckCast<in, out>(); } \
  TEST_F(CastOpTest, TestCast2##_##in##_##out) { CheckCast<in, out>(true); }

#define TEST_ALL_CASTS_FROM(in) \
  TEST_CAST(in, uint8);         \
  TEST_CAST(in, uint16);        \
  TEST_CAST(in, uint32);        \
  TEST_CAST(in, uint64);        \
  TEST_CAST(in, int16);         \
  TEST_CAST(in, int32);         \
  TEST_CAST(in, int64);         \
  TEST_CAST(in, half);          \
  TEST_CAST(in, float);         \
  TEST_CAST(in, double);        \
  TEST_CAST(in, bfloat16);      \
  TEST_CAST(in, quint8);        \
  TEST_CAST(in, qint8);         \
  TEST_CAST(in, qint32);        \
  TEST_CAST(in, qint16);        \
  TEST_CAST(in, quint16);

TEST_ALL_CASTS_FROM(uint8)
TEST_ALL_CASTS_FROM(uint16)
TEST_ALL_CASTS_FROM(uint32)
TEST_ALL_CASTS_FROM(uint64)
TEST_ALL_CASTS_FROM(int16)
TEST_ALL_CASTS_FROM(int32)
TEST_ALL_CASTS_FROM(int64)
TEST_ALL_CASTS_FROM(half)
TEST_ALL_CASTS_FROM(float)
TEST_ALL_CASTS_FROM(double)
TEST_ALL_CASTS_FROM(bfloat16)
TEST_ALL_CASTS_FROM(quint8)
TEST_ALL_CASTS_FROM(qint8)
TEST_ALL_CASTS_FROM(qint32)
TEST_ALL_CASTS_FROM(qint16)
TEST_ALL_CASTS_FROM(quint16)

#undef TEST_ALL_CASTS_FROM
#undef TEST_CAST

// TODO(wicke): check conversions from/to bool, and bfloat16

static void BM_cpu_float_int64(int iters, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  testing::BytesProcessed(static_cast<int64>(iters) * num *
                          (sizeof(float) + sizeof(int64)));
  testing::UseRealTime();
  test::Benchmark("cpu", Cast<float, int64>(num)).Run(iters);
}
BENCHMARK(BM_cpu_float_int64)->Arg(64 << 10)->Arg(32 << 20);

static void BM_gpu_float_int64(int iters, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  testing::BytesProcessed(static_cast<int64>(iters) * num *
                          (sizeof(float) + sizeof(int64)));
  testing::UseRealTime();
#if GOOGLE_CUDA
  test::Benchmark("gpu", Cast<float, int64>(num)).Run(iters);
#endif  // GOOGLE_CUDA
#ifdef TENSORFLOW_USE_SYCL
  test::Benchmark("sycl", Cast<float, int64>(num)).Run(iters);
#endif  // TENSORFLOW_USE_SYCL
}
BENCHMARK(BM_gpu_float_int64)->Arg(64 << 10)->Arg(32 << 20);

static void BM_cpu_bool_float(int iters, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  testing::BytesProcessed(static_cast<int64>(iters) * num *
                          (sizeof(bool) + sizeof(float)));
  testing::UseRealTime();
  test::Benchmark("cpu", Cast<bool, float>(num)).Run(iters);
}
BENCHMARK(BM_cpu_bool_float)->Arg(64 << 10)->Arg(32 << 20);

static void BM_gpu_bool_float(int iters, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  testing::BytesProcessed(static_cast<int64>(iters) * num *
                          (sizeof(bool) + sizeof(float)));
  testing::UseRealTime();
#if GOOGLE_CUDA
  test::Benchmark("gpu", Cast<bool, float>(num)).Run(iters);
#endif  // GOOGLE_CUDA
#ifdef TENSORFLOW_USE_SYCL
  test::Benchmark("sycl", Cast<bool, float>(num)).Run(iters);
#endif  // TENSORFLOW_USE_SYCL
}
BENCHMARK(BM_gpu_bool_float)->Arg(64 << 10)->Arg(32 << 20);

static void BM_cpu_float_bfloat16(int iters, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  testing::BytesProcessed(static_cast<int64>(iters) * num *
                          (sizeof(float) + sizeof(bfloat16)));
  testing::UseRealTime();
  test::Benchmark("cpu", Cast<float, bfloat16>(num)).Run(iters);
}
BENCHMARK(BM_cpu_float_bfloat16)->Arg(64 << 10)->Arg(32 << 20);

static void BM_cpu_bfloat16_float(int iters, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  testing::BytesProcessed(static_cast<int64>(iters) * num *
                          (sizeof(float) + sizeof(bfloat16)));
  testing::UseRealTime();
  test::Benchmark("cpu", Cast<bfloat16, float>(num)).Run(iters);
}
BENCHMARK(BM_cpu_bfloat16_float)->Arg(64 << 10)->Arg(32 << 20);

static void BM_cpu_float_half(int iters, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  testing::BytesProcessed(static_cast<int64>(iters) * num *
                          (sizeof(float) + sizeof(Eigen::half)));
  testing::UseRealTime();
  test::Benchmark("cpu", Cast<float, Eigen::half>(num)).Run(iters);
}
BENCHMARK(BM_cpu_float_half)->Arg(64 << 10)->Arg(32 << 20);

static void BM_cpu_half_float(int iters, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  testing::BytesProcessed(static_cast<int64>(iters) * num *
                          (sizeof(float) + sizeof(Eigen::half)));
  testing::UseRealTime();
  test::Benchmark("cpu", Cast<Eigen::half, float>(num)).Run(iters);
}
BENCHMARK(BM_cpu_half_float)->Arg(64 << 10)->Arg(32 << 20);

static void BM_gpu_float_half(int iters, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  testing::BytesProcessed(static_cast<int64>(iters) * num *
                          (sizeof(float) + sizeof(Eigen::half)));
  testing::UseRealTime();
#if GOOGLE_CUDA
  test::Benchmark("gpu", Cast<float, Eigen::half>(num)).Run(iters);
#endif  // GOOGLE_CUDA
}
BENCHMARK(BM_gpu_float_half)->Arg(64 << 10)->Arg(32 << 20);

static void BM_gpu_half_float(int iters, int num) {
  testing::ItemsProcessed(static_cast<int64>(iters) * num);
  testing::BytesProcessed(static_cast<int64>(iters) * num *
                          (sizeof(float) + sizeof(Eigen::half)));
  testing::UseRealTime();
#if GOOGLE_CUDA
  test::Benchmark("gpu", Cast<Eigen::half, float>(num)).Run(iters);
#endif  // GOOGLE_CUDA
}
BENCHMARK(BM_gpu_half_float)->Arg(64 << 10)->Arg(32 << 20);

}  // end namespace tensorflow
