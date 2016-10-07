/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace {

static void ExpectHasSubstr(StringPiece s, StringPiece expected) {
  EXPECT_TRUE(StringPiece(s).contains(expected))
      << "'" << s << "' does not contain '" << expected << "'";
}

class SparseDenseCDivTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp() {
    DataType value_type = tensorflow::DataTypeToEnum<T>::value;
    TF_ASSERT_OK(NodeDefBuilder("cdiv", "SparseDenseCwiseDiv")
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Attr("T", value_type)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

class SparseDenseCMulTest : public OpsTestBase {
 protected:
  template <typename T>
  void MakeOp() {
    DataType value_type = tensorflow::DataTypeToEnum<T>::value;
    TF_ASSERT_OK(NodeDefBuilder("cmul", "SparseDenseCwiseMul")
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Input(FakeInput(DT_INT64))
                     .Input(FakeInput(value_type))
                     .Attr("T", value_type)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SparseDenseCDivTest, DoNotBroadcastSparse_FewerDims) {
  MakeOp<float>();
  // [1] op [2, 1]
  AddInputFromArray<int64>(TensorShape({1, 1}), {0});       // indices
  AddInputFromArray<float>(TensorShape({1}), {1618});       // values
  AddInputFromArray<int64>(TensorShape({1}), {1});          // shape
  AddInputFromArray<float>(TensorShape({2, 1}), {17, 19});  // dense

  ExpectHasSubstr(RunOpKernel().ToString(), "broadcasts dense to sparse only");
}

TEST_F(SparseDenseCDivTest, DoNotBroadcastSparse_SameDims) {
  MakeOp<float>();
  // [1, 1] op [2, 1]
  AddInputFromArray<int64>(TensorShape({1, 2}), {0, 0});
  AddInputFromArray<float>(TensorShape({1}), {1618});
  AddInputFromArray<int64>(TensorShape({2}), {1, 1});
  AddInputFromArray<float>(TensorShape({2, 1}), {17, 19});

  ExpectHasSubstr(RunOpKernel().ToString(), "broadcasts dense to sparse only");
}

TEST_F(SparseDenseCDivTest, SameShape) {
  MakeOp<float>();
  // [    1]
  // [2    ]  cdiv [dense: same shape, all 1's]
  // [3   4]
  const auto indices_shape = TensorShape({4, 2});
  const gtl::ArraySlice<int64> indices = {0, 1, 1, 0, 2, 0, 2, 1};
  const gtl::ArraySlice<int64> shape = {3, 2};

  // Tensor dense(DT_FLOAT, TensorShape({3, 1}));
  Tensor dense(DT_FLOAT, TensorShape(shape));
  auto dense_flat = dense.flat<float>();
  dense_flat.setConstant(1.);

  AddInputFromArray<int64>(indices_shape, indices);
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int64>(TensorShape({2}), shape);
  AddInputFromArray<float>(TensorShape(shape), dense_flat);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4}));
  test::FillValues<float>(&expected, {1, 2, 3, 4});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseDenseCDivTest, BroadcastDenseSameDims) {
  // No broadcast.
  MakeOp<float>();
  // [    1]
  // [2    ]  cdiv [dense: shape [3,1], all 1's]
  // [3   4]
  const auto indices_shape = TensorShape({4, 2});
  const gtl::ArraySlice<int64> indices = {0, 1, 1, 0, 2, 0, 2, 1};
  const gtl::ArraySlice<int64> shape = {3, 2};

  Tensor dense(DT_FLOAT, TensorShape({3, 1}));
  auto dense_flat = dense.flat<float>();
  dense_flat.setConstant(1.);

  AddInputFromArray<int64>(indices_shape, indices);
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int64>(TensorShape({2}), shape);
  AddInputFromArray<float>(TensorShape({3, 1}), dense_flat);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4}));
  test::FillValues<float>(&expected, {1, 2, 3, 4});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseDenseCDivTest, BroadcastDenseFewerDims) {
  MakeOp<float>();
  // [    1]
  // [2    ]  cdiv [dense: shape [2]]
  // [3   4]
  const auto indices_shape = TensorShape({4, 2});
  const gtl::ArraySlice<int64> indices = {0, 1, 1, 0, 2, 0, 2, 1};
  const gtl::ArraySlice<int64> shape = {3, 2};

  Tensor dense(DT_FLOAT, TensorShape({2}));
  auto dense_flat = dense.flat<float>();
  dense_flat.setConstant(1.);

  AddInputFromArray<int64>(indices_shape, indices);
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int64>(TensorShape({2}), shape);
  AddInputFromArray<float>(TensorShape({2}), dense_flat);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4}));
  test::FillValues<float>(&expected, {1, 2, 3, 4});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseDenseCMulTest, BroadcastDense) {
  MakeOp<float>();
  // [    1]
  // [2    ] (shape [3,2])  cmul  [0.5  0] (shape [2])
  // [3   4]
  //
  // Result:
  // [?   0]
  // [1   ?]  where ? remains implicitly zero.
  // [1.5 0]
  const auto indices_shape = TensorShape({4, 2});
  const gtl::ArraySlice<int64> indices = {0, 1, 1, 0, 2, 0, 2, 1};
  const gtl::ArraySlice<int64> shape = {3, 2};

  Tensor dense(DT_FLOAT, TensorShape({2}));
  auto dense_flat = dense.flat<float>();
  dense_flat(0) = 0.5;
  dense_flat(1) = 0;

  AddInputFromArray<int64>(indices_shape, indices);
  AddInputFromArray<float>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<int64>(TensorShape({2}), shape);
  AddInputFromArray<float>(TensorShape({2}), dense_flat);

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, TensorShape({4}));
  test::FillValues<float>(&expected, {0, 1, 1.5, 0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

// Benchmarking code follows.

static Graph* SparseMatCMulDenseMat(Graph* g, Node* sp_indices, Node* sp_vals,
                                    Node* sp_shape, Node* dense) {
  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("SparseDenseCwiseMul"), "SparseDenseCwiseMul")
          .Input(sp_indices)
          .Input(sp_vals)
          .Input(sp_shape)
          .Input(dense)
          .Finalize(g, &ret));
  return g;
}

static Node* MakeTensor(Graph* g, int B, int M, int N) {
  Tensor data(DT_FLOAT, TensorShape({B, M, N}));
  data.flat<float>().setRandom();
  return test::graph::Constant(g, data);
}

struct ST {
  Node* indices;
  Node* vals;
  Node* shape;
};

static ST MakeSparseTensor(Graph* g, int B, int M, int N, int nnz_inner) {
  const int total_nnz = B * M * nnz_inner;
  const int kNumDims = 3;

  Tensor indices(DT_INT64, TensorShape({total_nnz, kNumDims}));
  Tensor vals(DT_FLOAT, TensorShape({total_nnz}));
  Tensor shape(DT_INT64, TensorShape({kNumDims}));
  vals.flat<float>().setRandom();
  test::FillValues(&shape, gtl::ArraySlice<int64>({B, M, N}));
  auto indices_mat = indices.matrix<int64>();

  int nnz_cnt = 0;
  std::unordered_set<int> picked;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, N - 1);

  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < M; ++j) {
      for (int k = 0; k < nnz_inner; ++k) {
        indices_mat(nnz_cnt, 0) = i;
        indices_mat(nnz_cnt, 1) = j;

        int inner = dist(gen);
        while (picked.count(inner) == 1) {
          inner = dist(gen);
        }
        picked.insert(inner);
        indices_mat(nnz_cnt, 2) = inner;

        ++nnz_cnt;
      }
    }
  }

  return ST{test::graph::Constant(g, indices), test::graph::Constant(g, vals),
            test::graph::Constant(g, shape)};
}

// [8, 4, N{nnz}] cmul [8, 4, N]
#define BM_SparseMatCMulDenseMatArgs(N, NNZ_INNER)                             \
  static void BM_SparseMatCMulDenseMat_##N##_##NNZ_INNER(int iters) {          \
    Graph* g = new Graph(OpRegistry::Global());                                \
    Node* dense = MakeTensor(g, 8, 4, N);                                      \
    ST sp = MakeSparseTensor(g, 8, 4, N, NNZ_INNER);                           \
                                                                               \
    testing::ItemsProcessed(static_cast<int64>(iters * 8 * 4 * N * 2));        \
    test::Benchmark(                                                           \
        "cpu", SparseMatCMulDenseMat(g, sp.indices, sp.vals, sp.shape, dense)) \
        .Run(iters);                                                           \
  }                                                                            \
  BENCHMARK(BM_SparseMatCMulDenseMat_##N##_##NNZ_INNER)

BM_SparseMatCMulDenseMatArgs(1048576, 1);
BM_SparseMatCMulDenseMatArgs(1048576, 8);
BM_SparseMatCMulDenseMatArgs(1048576, 32);
BM_SparseMatCMulDenseMatArgs(262144, 1);
BM_SparseMatCMulDenseMatArgs(262144, 8);
BM_SparseMatCMulDenseMatArgs(262144, 32);

}  // namespace

}  // namespace tensorflow
