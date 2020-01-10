#include "tensorflow/core/framework/fake_input.h"
#include <gtest/gtest.h>
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/tensor.h"

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
  void MakeOp(DataType src, DataType dst) {
    RequireDefaultOps();
    EXPECT_OK(NodeDefBuilder("cast_op", "Cast")
                  .Input(FakeInput(DT_INT32))
                  .Attr("SrcT", src)
                  .Attr("DstT", dst)
                  .Finalize(node_def()));
    EXPECT_OK(InitOp());
  }
};

TEST_F(CastOpTest, Int32ToUint8) {
  MakeOp(DT_INT32, DT_UINT8);
  AddInputFromArray<int32>(TensorShape({1, 2, 2, 1}), {1, 2, 3, 4});
  ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_UINT8, TensorShape({1, 2, 2, 1}));
  test::FillValues<uint8>(&expected, {1, 2, 3, 4});
  test::ExpectTensorEqual<uint8>(expected, *GetOutput(0));
}

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
  test::Benchmark("gpu", Cast<float, int64>(num)).Run(iters);
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
  test::Benchmark("gpu", Cast<bool, float>(num)).Run(iters);
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

}  // end namespace tensorflow
