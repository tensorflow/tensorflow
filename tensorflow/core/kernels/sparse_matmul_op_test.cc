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

#include "tensorflow/core/kernels/sparse_matmul_op.h"

#include <cstdint>
#include <vector>

#include "hwy/detect_compiler_arch.h"  // from @highway
#include "hwy/detect_targets.h"  // from @highway
#include "hwy/targets.h"  // from @highway
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
random::PhiloxRandom philox(1, 1);
random::SimplePhilox rnd(&philox);
using Eigen::operator==;

template <typename T>
void Sparsify(Tensor* t, float sparsity) {
  const int64_t N = t->NumElements();
  CHECK_LE(sparsity, 1);
  auto flat = t->flat<T>();
  if (sparsity == 1) {
    flat.setZero();
    return;
  }
  static const uint32_t K = 10000;
  for (int64_t i = 0; i < N; ++i) {
    if (rnd.Uniform(K) < sparsity * K) {
      flat(i) = T(0);
    } else if (flat(i) == T(0)) {
      flat(i) = T(1);
    }
  }
}

Node* SparseMatMulNode(Graph* g, Node* in0, Node* in1, bool transpose_a,
                       bool transpose_b, bool a_sparse, bool b_sparse) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "SparseMatMul")
                  .Input(in0)
                  .Input(in1)
                  .Attr("transpose_a", transpose_a)
                  .Attr("transpose_b", transpose_b)
                  .Attr("a_is_sparse", a_sparse)
                  .Attr("b_is_sparse", b_sparse)
                  .Finalize(g, &ret));
  return ret;
}

template <typename TA, typename TB>
static Graph* SparseMatMulHelper(Graph* g, int m, int n, int d,
                                 float sparsity_a, float sparsity_b,
                                 bool transpose_a, bool transpose_b) {
  bool a_sparse = (sparsity_a > 0);
  bool b_sparse = (sparsity_b > 0);

  auto left_shape = transpose_a ? TensorShape({d, m}) : TensorShape({m, d});
  Tensor left(DataTypeToEnum<TA>::value, left_shape);
  left.flat<TA>().setRandom();
  Sparsify<TA>(&left, sparsity_a);

  auto right_shape = transpose_b ? TensorShape({n, d}) : TensorShape({d, n});
  Tensor right(DataTypeToEnum<TB>::value, right_shape);
  right.flat<TB>().setRandom();
  Sparsify<TB>(&right, sparsity_b);

  SparseMatMulNode(g, test::graph::Constant(g, left),
                   test::graph::Constant(g, right), transpose_a, transpose_b,
                   a_sparse, b_sparse);
  return g;
}

template <typename TA, typename TB>
static Graph* SparseMatMul(int m, int n, int d, float sparsity_a,
                           float sparsity_b, bool transpose_a,
                           bool transpose_b) {
  Graph* g = new Graph(OpRegistry::Global());
  return SparseMatMulHelper<TA, TB>(g, m, n, d, sparsity_a, sparsity_b,
                                    transpose_a, transpose_b);
}

static Graph* ReplicatedSparseMatMul(int m, int n, int d, float sparsity_1,
                                     float sparsity_2, int copies) {
  Graph* g = new Graph(OpRegistry::Global());
  for (int i = 0; i < copies; ++i) {
    SparseMatMulHelper<float, float>(g, m, n, d, sparsity_1, sparsity_2, false,
                                     false);
  }
  return g;
}

#define BM_SPARSE(M, K, N, S1, S2, TRA, TRB, TA, TB)                           \
  static void                                                                  \
      BM_Sparse##_##M##_##K##_##N##_##S1##_##S2##_##TRA##_##TRB##_##TA##_##TB( \
          ::testing::benchmark::State& state) {                                \
    auto label = strings::Printf("tr_a: %d tr_b: %d sp_a: %0.2f sp_b: %0.2f",  \
                                 TRA, TRB, S1 / 100.0, S2 / 100.0);            \
    state.SetLabel(label);                                                     \
    auto g = SparseMatMul<TA, TB>(M, N, K, S1 / 100.0, S2 / 100.0, TRA, TRB);  \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);         \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_Sparse##_##M##_##K##_##N##_##S1##_##S2##_##TRA##_##TRB##_##TA##_##TB) \
      ->UseRealTime();

#define BM_SPARSE_REPLICATED(M, K, N, S1, S2, Copies)                          \
  static void BM_Sparse_replicated##_##M##_##K##_##N##_##S1##_##S2##_##Copies( \
      ::testing::benchmark::State& state) {                                    \
    auto label = strings::Printf("copies: %d sp_a: %0.2f sp_b: %0.2f",         \
                                 (Copies), S1 / 100.0, S2 / 100.0);            \
    state.SetLabel(label);                                                     \
    auto g =                                                                   \
        ReplicatedSparseMatMul(M, N, K, S1 / 100.0, S2 / 100.0, (Copies));     \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);         \
    state.SetItemsProcessed(state.iterations() * M * K * N * Copies * 2);      \
  }                                                                            \
  BENCHMARK(BM_Sparse_replicated##_##M##_##K##_##N##_##S1##_##S2##_##Copies)   \
      ->UseRealTime();

#define BM_SPARSE_FLOAT(M, K, N, S1, S2, TRA, TRB) \
  BM_SPARSE(M, K, N, S1, S2, TRA, TRB, float, float)
#define BM_SPARSE_BFLOAT16(M, K, N, S1, S2, TRA, TRB) \
  BM_SPARSE(M, K, N, S1, S2, TRA, TRB, bfloat16, bfloat16)
#define BM_SPARSE_FLOAT_BFLOAT16(M, K, N, S1, S2, TRA, TRB) \
  BM_SPARSE(M, K, N, S1, S2, TRA, TRB, float, bfloat16)
#define BM_SPARSE_BFLOAT16_FLOAT(M, K, N, S1, S2, TRA, TRB) \
  BM_SPARSE(M, K, N, S1, S2, TRA, TRB, bfloat16, float)

// Test sparse b
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 0, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 1, 0, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 50, 0, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 85, 0, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 99, 0, false, false);
// Test sparse a
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 50, false, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 85, false, false);
// Test transposing
BM_SPARSE_FLOAT(2048, 2048, 2048, 85, 0, true, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 85, 0, false, true);
BM_SPARSE_FLOAT(2048, 2048, 2048, 85, 0, true, true);
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 85, true, false);
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 85, false, true);
BM_SPARSE_FLOAT(2048, 2048, 2048, 0, 85, true, true);

// Test smaller sizes
BM_SPARSE_FLOAT(1024, 1024, 1024, 0, 0, false, false);
BM_SPARSE_FLOAT(1024, 1024, 1024, 1, 0, false, false);
BM_SPARSE_FLOAT(1024, 1024, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(256, 256, 256, 1, 0, false, false);
BM_SPARSE_FLOAT(512, 512, 512, 1, 0, false, false);
BM_SPARSE_FLOAT(2560, 400, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(2560, 400, 1024, 85, 0, true, false);

BM_SPARSE_FLOAT(400, 800, 2560, 85, 0, false, false);
BM_SPARSE_FLOAT(400, 2560, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(400, 1024, 256, 85, 0, false, false);
BM_SPARSE_FLOAT(400, 256, 1, 85, 0, false, false);

BM_SPARSE_REPLICATED(400, 800, 2560, 85, 0, 6);
BM_SPARSE_REPLICATED(400, 2560, 1024, 85, 0, 6);
BM_SPARSE_REPLICATED(400, 1024, 256, 85, 0, 6);
BM_SPARSE_REPLICATED(400, 256, 1, 85, 0, 6);

BM_SPARSE_FLOAT(2048, 1792, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(2048, 1024, 768, 85, 0, false, false);
BM_SPARSE_FLOAT(2048, 768, 512, 85, 0, false, false);
BM_SPARSE_FLOAT(2048, 512, 256, 85, 0, false, false);

BM_SPARSE_FLOAT(2049, 1792, 1024, 85, 0, false, false);
BM_SPARSE_FLOAT(2049, 1024, 768, 85, 0, false, false);
BM_SPARSE_FLOAT(2049, 768, 512, 85, 0, false, false);
BM_SPARSE_FLOAT(2049, 512, 256, 85, 0, false, false);

BM_SPARSE_REPLICATED(2048, 1792, 1024, 85, 0, 6);
BM_SPARSE_REPLICATED(2048, 1024, 768, 85, 0, 6);
BM_SPARSE_REPLICATED(2048, 768, 512, 85, 0, 6);
BM_SPARSE_REPLICATED(2048, 512, 256, 85, 0, 6);

// Test bfloat16
BM_SPARSE_BFLOAT16(2048, 2048, 2048, 0, 0, false, false);
BM_SPARSE_BFLOAT16(2048, 2048, 2048, 1, 0, false, false);
BM_SPARSE_BFLOAT16(2048, 2048, 2048, 85, 0, false, false);
BM_SPARSE_BFLOAT16(2048, 2048, 2048, 99, 0, false, false);
BM_SPARSE_BFLOAT16_FLOAT(2048, 2048, 2048, 85, 0, false, false);
BM_SPARSE_BFLOAT16_FLOAT(2048, 2048, 2048, 99, 0, false, false);
BM_SPARSE_FLOAT_BFLOAT16(2048, 2048, 2048, 85, 0, false, false);
BM_SPARSE_FLOAT_BFLOAT16(2048, 2048, 2048, 99, 0, false, false);

static Graph* MultiSparseMatMul(int m, int n, int d, float sparsity_1,
                                float sparsity_2, int copies) {
  Graph* g = new Graph(OpRegistry::Global());
  for (int i = 0; i < copies; ++i) {
    SparseMatMulHelper<float, float>(g, d, n, m, sparsity_1, sparsity_2, true,
                                     false);
    SparseMatMulHelper<float, float>(g, m, d, n, sparsity_2, 0, false, true);
  }
  return g;
}

// clang-format off
// NOLINTBEGIN
#define BM_SPARSE_MULTI(M, K, N, S1, S2, Copies)                              \
  static void BM_Sparse_Multi##_##M##_##K##_##N##_##S1##_##S2##_##Copies(::testing::benchmark::State& state) {                                              \
    auto label = strings::Printf("%d_%d_%d_%d_%0.2f_%0.2f", M, K, N, Copies,  \
                                 S1 / 100.0, S2 / 100.0);                     \
    state.SetLabel(label);                                                    \
    auto g = MultiSparseMatMul(M, N, K, S1 / 100.0, S2 / 100.0, Copies);      \
    test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);        \
    state.SetItemsProcessed(state.iterations() * M * K * N * 2 * 2 * Copies); \
  }                                                                           \
  BENCHMARK(BM_Sparse_Multi##_##M##_##K##_##N##_##S1##_##S2##_##Copies)       \
      ->UseRealTime();
// NOLINTEND
// clang-format on
BM_SPARSE_MULTI(1024, 2140, 4096, 0, 82, 1);
BM_SPARSE_MULTI(1024, 4096, 2048, 83, 83, 1);
BM_SPARSE_MULTI(400, 800, 2560, 85, 85, 1);
BM_SPARSE_MULTI(400, 2560, 1024, 85, 85, 1);
BM_SPARSE_MULTI(400, 1024, 256, 85, 85, 1);
BM_SPARSE_MULTI(400, 256, 1, 85, 85, 1);

BM_SPARSE_MULTI(2048, 1792, 1024, 85, 85, 1);
BM_SPARSE_MULTI(2048, 1024, 768, 85, 85, 1);
BM_SPARSE_MULTI(2048, 768, 512, 85, 85, 1);
BM_SPARSE_MULTI(2048, 512, 256, 85, 85, 1);

BM_SPARSE_MULTI(2048, 1792, 1024, 85, 85, 3);
BM_SPARSE_MULTI(2048, 1024, 768, 85, 85, 3);
BM_SPARSE_MULTI(2048, 768, 512, 85, 85, 3);
BM_SPARSE_MULTI(2048, 512, 256, 85, 85, 3);

BM_SPARSE_MULTI(2048, 1792, 1024, 85, 85, 6);
BM_SPARSE_MULTI(2048, 1024, 768, 85, 85, 6);
BM_SPARSE_MULTI(2048, 768, 512, 85, 85, 6);
BM_SPARSE_MULTI(2048, 512, 256, 85, 85, 6);

}  // end namespace tensorflow

namespace tensorflow {
namespace {

// Test on all highway targets supported by test machine
class SparseMatMulHighwayOpTest : public OpsTestBase {
 protected:
  template <typename TA, typename TB>
  void VerifyAcrossAllTargets(int m, int k, int n, bool transpose_a,
                              bool transpose_b, bool a_is_sparse,
                              bool b_is_sparse) {
    TensorShape shape_a =
        transpose_a ? TensorShape({k, m}) : TensorShape({m, k});
    TensorShape shape_b =
        transpose_b ? TensorShape({n, k}) : TensorShape({k, n});
    std::vector<TA> data_a(shape_a.num_elements());
    std::vector<TB> data_b(shape_b.num_elements());
    for (int64_t i = 0; i < shape_a.num_elements(); ++i) {
      data_a[i] = (i % 5 == 0) ? TA(0) : TA(static_cast<float>((i % 7) - 3));
    }
    for (int64_t i = 0; i < shape_b.num_elements(); ++i) {
      data_b[i] = (i % 3 == 0) ? TB(0) : TB(static_cast<float>((i % 5) - 2));
    }

    Tensor expected(allocator(), DT_FLOAT, TensorShape({m, n}));
    auto expected_mat = expected.matrix<float>();
    expected_mat.setZero();
    for (int row = 0; row < m; ++row) {
      for (int depth = 0; depth < k; ++depth) {
        const float a_val = static_cast<float>(
            transpose_a ? data_a[depth * m + row] : data_a[row * k + depth]);
        for (int col = 0; col < n; ++col) {
          const float b_val = static_cast<float>(
              transpose_b ? data_b[col * k + depth] : data_b[depth * n + col]);
          expected_mat(row, col) += a_val * b_val;
        }
      }
    }

    hwy::SetSupportedTargetsForTest(0);
    const std::vector<int64_t> targets = hwy::SupportedAndGeneratedTargets();
    ASSERT_FALSE(targets.empty());

    for (int64_t target : targets) {
      hwy::SetSupportedTargetsForTest(target);
      inputs_.clear();
      ASSERT_TRUE(NodeDefBuilder("sparse_matmul_op", "SparseMatMul")
                      .Input(FakeInput(DataTypeToEnum<TA>::v()))
                      .Input(FakeInput(DataTypeToEnum<TB>::v()))
                      .Attr("transpose_a", transpose_a)
                      .Attr("transpose_b", transpose_b)
                      .Attr("a_is_sparse", a_is_sparse)
                      .Attr("b_is_sparse", b_is_sparse)
                      .Finalize(node_def())
                      .ok());
      ASSERT_TRUE(InitOp().ok());
      AddInputFromArray<TA>(shape_a, data_a);
      AddInputFromArray<TB>(shape_b, data_b);
      ASSERT_TRUE(RunOpKernel().ok());
      test::ExpectTensorNear<float>(expected, *GetOutput(0), 1e-4);
    }
    hwy::SetSupportedTargetsForTest(0);
  }
};

TEST_F(SparseMatMulHighwayOpTest, AllSupportedTargetsMatchReference) {
  VerifyAcrossAllTargets<float, float>(67, 133, 263, false, false, true, false);
  VerifyAcrossAllTargets<bfloat16, bfloat16>(67, 133, 263, false, false, true,
                                             false);
  VerifyAcrossAllTargets<float, bfloat16>(67, 133, 128, true, true, true,
                                          false);
  VerifyAcrossAllTargets<bfloat16, float>(67, 133, 263, false, true, false,
                                          true);
}

}  // namespace
}  // namespace tensorflow
