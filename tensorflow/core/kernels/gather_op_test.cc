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
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {
namespace {

class GatherOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType index_type) {
    RequireDefaultOps();
    ASSERT_OK(NodeDefBuilder("myop", "Gather")
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(index_type))
                  .Finalize(node_def()));
    ASSERT_OK(InitOp());
  }
};

TEST_F(GatherOpTest, ScalarIndices) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 1, 2, 3, 4});
  AddInputFromArray<int32>(TensorShape({}), {3});
  ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({}));
  test::FillValues<float>(&expected, {3});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, Simple_TwoD32) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 0, 2});
  ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 3}));
  test::FillValues<float>(&expected, {0, 1, 2, 12, 13, 14, 0, 1, 2, 6, 7, 8});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, Simple_TwoD64) {
  MakeOp(DT_INT64);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int64>(TensorShape({4}), {0, 4, 0, 2});
  ASSERT_OK(RunOpKernel());

  // Check the output.
  Tensor expected(allocator(), DT_FLOAT, TensorShape({4, 3}));
  test::FillValues<float>(&expected, {0, 1, 2, 12, 13, 14, 0, 1, 2, 6, 7, 8});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, HighRank) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({4}), {0, 1, 2, 3});
  AddInputFromArray<int32>(TensorShape({2, 3}), {1, 2, 0, 2, 3, 0});
  ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 3}));
  test::FillValues<float>(&expected, {1, 2, 0, 2, 3, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(GatherOpTest, Error_IndexOutOfRange) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
  AddInputFromArray<int32>(TensorShape({4}), {0, 4, 99, 2});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("Index 99 at offset 2 in Tindices is out of range"))
      << s;
}

class GatherOpForBenchmark : public GatherOpTest {
 public:
  void TestBody() override {  // not used }
  }
  void PublicMakeOp(DataType index_type) { MakeOp(index_type); }
};

static const int kSorted = 0x8000;  // Mask for arg to specify sorting vs. not

template <typename Index>
void BM_Gather(int iters, int arg) {
  testing::StopTiming();

  bool sorted = ((arg & kSorted) != 0);
  int dim = arg & ~kSorted;

  GatherOpForBenchmark t;
  t.PublicMakeOp(DataTypeToEnum<Index>::v());
  // Use a 512 MB table, regardless of dim
  const int kRows = ((1 << 29) / sizeof(float)) / dim;
  std::vector<float> data(kRows * dim, 1.0f);
  t.AddInputFromArray<float>(TensorShape({kRows, dim}), data);
  const int kLookups = 2000;
  const int kBatches = 1000000 / kLookups;
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  std::vector<std::vector<Index>> all_ids(kBatches);
  for (int i = 0; i < kBatches; ++i) {
    std::vector<Index>* ids = &all_ids[i];
    ids->resize(kLookups);
    for (int j = 0; j < kLookups; ++j) {
      (*ids)[j] = rnd.Uniform(kRows);
    }
    if (sorted) {
      sort(ids->begin(), ids->end());
    }
  }

  t.AddInput<Index>(TensorShape({kLookups}), [](int i) { return 0; });
  if (sorted) {
    testing::SetLabel("sorted by id");
  }
  testing::BytesProcessed(static_cast<int64>(iters) * kLookups * dim *
                          sizeof(float));
  testing::StartTiming();
  while (--iters > 0) {
    const std::vector<Index>& b = all_ids[iters % kBatches];
    TensorValue input = t.mutable_input(1);
    gtl::MutableArraySlice<Index> slice(&input->vec<Index>()(0),
                                        input->NumElements());
    for (int i = 0; i < kLookups; i++) {
      slice[i] = b[i];
    }
    Status s = t.RunOpKernel();
  }
}

static void BM_Gather32(int iters, int arg) { BM_Gather<int32>(iters, arg); }

static void BM_Gather64(int iters, int arg) { BM_Gather<int64>(iters, arg); }

BENCHMARK(BM_Gather32)
    ->Arg(10)
    ->Arg(10 | kSorted)
    ->Arg(20)
    ->Arg(40)
    ->Arg(63)
    ->Arg(63 | kSorted)
    ->Arg(64)
    ->Arg(64 | kSorted)
    ->Arg(65)
    ->Arg(65 | kSorted)
    ->Arg(100)
    ->Arg(100 | kSorted)
    ->Arg(127)
    ->Arg(127 | kSorted)
    ->Arg(128)
    ->Arg(128 | kSorted)
    ->Arg(129)
    ->Arg(129 | kSorted)
    ->Arg(1000)
    ->Arg(1000 | kSorted);

BENCHMARK(BM_Gather64)
    ->Arg(10)
    ->Arg(10 | kSorted)
    ->Arg(20)
    ->Arg(40)
    ->Arg(63)
    ->Arg(63 | kSorted)
    ->Arg(64)
    ->Arg(64 | kSorted)
    ->Arg(65)
    ->Arg(65 | kSorted)
    ->Arg(100)
    ->Arg(100 | kSorted)
    ->Arg(127)
    ->Arg(127 | kSorted)
    ->Arg(128)
    ->Arg(128 | kSorted)
    ->Arg(129)
    ->Arg(129 | kSorted)
    ->Arg(1000)
    ->Arg(1000 | kSorted);

}  // namespace
}  // namespace tensorflow
