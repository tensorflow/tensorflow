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

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

class InTopKOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType target_type) {
    TF_ASSERT_OK(NodeDefBuilder("in_topk", "InTopKV2")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(target_type))
                     .Input(FakeInput(target_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(InTopKOpTest, Small) {
  MakeOp(DT_INT32);
  AddInputFromArray<float>(TensorShape({3, 4}),
                           {0.1f, 0.3f, 0.2f, 0.4f,  //
                            0.1f, 0.2f, 0.3f, 0.4f,  //
                            0.4f, 0.4f, 0.4f, 0.1f});
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 0, 2});
  AddInputFromArray<int32_t>(TensorShape({}), {2});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(DT_BOOL, TensorShape({3}));
  // Row 0: class 1 (0.3) has one larger value -> in top 2.
  // Row 1: class 0 (0.1) has three larger values -> not in top 2.
  // Row 2: ties count as in top-k.
  test::FillValues<bool>(&expected, {true, false, true});
  test::ExpectTensorEqual<bool>(expected, *GetOutput(0));
}

TEST_F(InTopKOpTest, NonFiniteAndOutOfRange) {
  MakeOp(DT_INT64);
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  AddInputFromArray<float>(TensorShape({4, 3}),
                           {0.1f, 0.2f, 0.3f,  //
                            0.1f, inf, 0.3f,   //
                            nan, 0.2f, 0.3f,   //
                            0.1f, 0.2f, 0.3f});
  AddInputFromArray<int64_t>(TensorShape({4}), {2, 2, 2, 3});
  AddInputFromArray<int64_t>(TensorShape({}), {3});
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(DT_BOOL, TensorShape({4}));
  test::FillValues<bool>(&expected, {true, false, false, false});
  test::ExpectTensorEqual<bool>(expected, *GetOutput(0));
}

// Large enough to be split across worker threads.
TEST_F(InTopKOpTest, LargeMatchesReference) {
  MakeOp(DT_INT32);
  const int rows = 3000, cols = 517, k = 50;
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> val(0, 200);  // Plenty of ties.
  std::uniform_int_distribution<int> tgt(-2, cols + 1);
  std::vector<float> pred(rows * cols);
  for (auto& p : pred) p = static_cast<float>(val(gen));
  pred[17 * cols + 3] = std::numeric_limits<float>::quiet_NaN();
  pred[1234 * cols + 99] = std::numeric_limits<float>::infinity();
  std::vector<int32_t> targets(rows);
  for (auto& t : targets) t = tgt(gen);
  AddInputFromArray<float>(TensorShape({rows, cols}), pred);
  AddInputFromArray<int32_t>(TensorShape({rows}), targets);
  AddInputFromArray<int32_t>(TensorShape({}), {k});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(DT_BOOL, TensorShape({rows}));
  auto e = expected.flat<bool>();
  for (int r = 0; r < rows; ++r) {
    const int t = targets[r];
    bool ok = t >= 0 && t < cols;
    int larger = 0;
    for (int c = 0; ok && c < cols; ++c) {
      const float p = pred[r * cols + c];
      if (!std::isfinite(p)) ok = false;
      if (p > pred[r * cols + t]) ++larger;
    }
    e(r) = ok && larger < k;
  }
  test::ExpectTensorEqual<bool>(expected, *GetOutput(0));
}

template <typename T>
static Graph* InTopK(int num_targets, int num_classes, T top_k) {
  Graph* g = new Graph(OpRegistry::Global());

  DataType dtype = DataTypeToEnum<T>::value;

  Tensor predictions_t(DT_FLOAT, TensorShape({num_targets, num_classes}));
  predictions_t.flat<float>().setRandom();

  // Targets must be valid class ids; unconstrained random int64 values are
  // out of range, which makes the kernel skip every row.
  Tensor targets_t(dtype, TensorShape({num_targets}));
  for (int i = 0; i < num_targets; ++i) {
    targets_t.flat<T>()(i) = static_cast<T>(i % num_classes);
  }

  Tensor k_t(dtype, TensorShape({}));
  k_t.scalar<T>() = k_t.scalar<T>().constant(top_k);

  Node* predictions = test::graph::Constant(g, predictions_t, "predictions");
  Node* targets = test::graph::Constant(g, targets_t, "targets");
  Node* k = test::graph::Constant(g, k_t, "k");

  Node* in_topk;
  TF_CHECK_OK(NodeBuilder(g->NewName("in_topk"), "InTopKV2")
                  .Input(predictions)
                  .Input(targets)
                  .Input(k)
                  .Attr("T", dtype)
                  .Finalize(g, &in_topk));

  return g;
}

#define BM_NAME(T, TARGETS, CLASSES, K, DEVICE) \
  BM_InTopK##_##T##_##TARGETS##_##CLASSES##_##K##_##DEVICE

#define BM_InTopK(T, TARGETS, CLASSES, K, DEVICE)                      \
  static void BM_NAME(T, TARGETS, CLASSES, K,                          \
                      DEVICE)(::testing::benchmark::State & state) {   \
    test::Benchmark(#DEVICE, InTopK<T>(TARGETS, CLASSES, K),           \
                    /*old_benchmark_api=*/false)                       \
        .Run(state);                                                   \
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * \
                            TARGETS * CLASSES);                        \
  }                                                                    \
  BENCHMARK(BM_NAME(T, TARGETS, CLASSES, K, DEVICE))->UseRealTime();

BM_InTopK(int64_t, 64, 1000, 10, cpu);
BM_InTopK(int64_t, 64, 10000, 10, cpu);

// `hit` = 0: random predictions, so most rows stop scanning early once more
// than k larger predictions are seen. `hit` = 1: the target holds the largest
// prediction in its row, so every row is scanned in full (the common case for
// a trained classifier).
static Graph* InTopKScenario(int num_targets, int num_classes, int k,
                             bool hit) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor predictions_t(DT_FLOAT, TensorShape({num_targets, num_classes}));
  predictions_t.flat<float>().setRandom();
  Tensor targets_t(DT_INT32, TensorShape({num_targets}));
  auto pred = predictions_t.matrix<float>();
  for (int i = 0; i < num_targets; ++i) {
    const int t = (i * 7919) % num_classes;
    targets_t.flat<int32_t>()(i) = t;
    if (hit) pred(i, t) = 2.0f;  // setRandom() values are in [0, 1).
  }
  Tensor k_t(DT_INT32, TensorShape({}));
  k_t.scalar<int32_t>()() = k;
  Node* in_topk;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("in_topk"), "InTopKV2")
          .Input(test::graph::Constant(g, predictions_t, "predictions"))
          .Input(test::graph::Constant(g, targets_t, "targets"))
          .Input(test::graph::Constant(g, k_t, "k"))
          .Attr("T", DT_INT32)
          .Finalize(g, &in_topk));
  return g;
}

static void BM_InTopKScenario(::testing::benchmark::State& state) {
  const int rows = state.range(0);
  const int classes = state.range(1);
  const bool hit = state.range(2);
  test::Benchmark("cpu", InTopKScenario(rows, classes, 5, hit),
                  /*old_benchmark_api=*/false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * rows *
                          classes);
}

BENCHMARK(BM_InTopKScenario)
    ->UseRealTime()
    ->Args({64, 1000, 0})
    ->Args({64, 1000, 1})
    ->Args({1024, 1000, 0})
    ->Args({1024, 1000, 1})
    ->Args({4096, 1000, 1})
    ->Args({1024, 10000, 1})
    ->Args({256, 100000, 1});

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
BM_InTopK(int64_t, 64, 1000, 10, gpu);
BM_InTopK(int64_t, 64, 10000, 10, gpu);
#endif  // defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)

}  // namespace tensorflow
