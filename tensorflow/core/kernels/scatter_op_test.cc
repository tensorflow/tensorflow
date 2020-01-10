#include <functional>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {
namespace {

class ScatterUpdateOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType index_type) {
    RequireDefaultOps();
    ASSERT_OK(NodeDefBuilder("myop", "ScatterUpdate")
                  .Input(FakeInput(DT_FLOAT_REF))
                  .Input(FakeInput(index_type))
                  .Input(FakeInput(DT_FLOAT))
                  .Finalize(node_def()));
    ASSERT_OK(InitOp());
  }
};

TEST_F(ScatterUpdateOpTest, Simple_TwoD32) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected, {100, 101, 102, 0, 0, 0, 10000, 10001,
                                      10002, 0, 0, 0, 777, 778, 779});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Simple_Two64) {
  MakeOp(DT_INT64);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int64>(TensorShape({3}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected, {100, 101, 102, 0, 0, 0, 10000, 10001,
                                      10002, 0, 0, 0, 777, 778, 779});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Simple_ZeroD) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {101});
  ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected, {0, 0, 0, 101, 0});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Simple_OneD) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3}), {100, 101, 102});
  ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected, {100, 0, 102, 0, 101});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, HigherRank) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({8}), {0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({2, 3}), {0, 4, 2, 1, 3, 6});
  AddInputFromArray<float>(TensorShape({2, 3}), {10, 20, 30, 40, 50, 60});
  ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({8}));
  test::FillValues<float>(&expected, {10, 40, 30, 50, 20, 0, 60, 0});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Error_IndexOutOfRange) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 99});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("Index 99 at offset 2 in indices is out of range"))
      << s;
}

TEST_F(ScatterUpdateOpTest, Error_WrongDimsIndices) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 3}), {0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({1, 3}), {0, 4, 99});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("Must have updates.shape = indices.shape + "
                            "params.shape[1:], got "))
      << s;
}

TEST_F(ScatterUpdateOpTest, Error_MismatchedParamsAndUpdateDimensions) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 2});
  AddInputFromArray<float>(
      TensorShape({3, 4}),
      {100, 101, 102, 103, 777, 778, 779, 780, 10000, 10001, 10002, 10004});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("Must have updates.shape = indices.shape + "
                            "params.shape[1:], got "))

      << s;
}

TEST_F(ScatterUpdateOpTest, Error_MismatchedIndicesAndUpdateDimensions) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {100, 101, 102, 10000, 10001, 10002});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("Must have updates.shape = indices.shape + "
                            "params.shape[1:], got "))
      << s;
}

class ScatterUpdateBM : public ScatterUpdateOpTest {
 public:
  virtual void TestBody() {}
  void MakeBenchmarkOp(const char* op, DataType index_type) {
    ASSERT_OK(NodeDefBuilder("myop", op)
                  .Input(FakeInput(DT_FLOAT_REF))
                  .Input(FakeInput(index_type))
                  .Input(FakeInput(DT_FLOAT))
                  .Finalize(node_def()));
    TF_CHECK_OK(InitOp());
  }
};

template <typename Index>
static void BM_ScatterHelper(int iters, int embedding_size, const char* op) {
  testing::StopTiming();
  const int kRows = 10000000 / embedding_size;
  std::vector<float> values;
  for (int i = 0; i < kRows * embedding_size; i++) {
    values.push_back(i);
  }
  const int kNumUpdates = 1000;
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  std::vector<Index> indices;
  std::vector<float> updates;
  for (int i = 0; i < kNumUpdates; i++) {
    indices.push_back(rnd.Uniform(kRows));
    for (int j = 0; j < embedding_size; j++) {
      updates.push_back(i * 10 + j);
    }
  }

  ScatterUpdateBM bm;
  bm.MakeBenchmarkOp(op, DataTypeToEnum<Index>::v());
  bm.AddInputFromArray<float>(TensorShape({kRows, embedding_size}), values);
  bm.AddInputFromArray<Index>(TensorShape({kNumUpdates}), indices);
  bm.AddInputFromArray<float>(TensorShape({kNumUpdates, embedding_size}),
                              updates);
  testing::ItemsProcessed((static_cast<int64>(kNumUpdates) * embedding_size) *
                          iters);
  testing::StartTiming();
  while (iters-- > 0) {
    Status s = bm.RunOpKernel();
  }
}

static void BM_ScatterUpdateInt32(int iters, int embedding_size) {
  BM_ScatterHelper<int32>(iters, embedding_size, "ScatterUpdate");
}
static void BM_ScatterUpdateInt64(int iters, int embedding_size) {
  BM_ScatterHelper<int64>(iters, embedding_size, "ScatterUpdate");
}

static void BM_ScatterAddInt32(int iters, int embedding_size) {
  BM_ScatterHelper<int32>(iters, embedding_size, "ScatterAdd");
}
static void BM_ScatterAddInt64(int iters, int embedding_size) {
  BM_ScatterHelper<int64>(iters, embedding_size, "ScatterAdd");
}

BENCHMARK(BM_ScatterUpdateInt32)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_ScatterUpdateInt64)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK(BM_ScatterAddInt32)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_ScatterAddInt64)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

}  // namespace
}  // namespace tensorflow
