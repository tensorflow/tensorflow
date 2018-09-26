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
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
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
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class SparseConcatTest : public OpsTestBase {
 protected:
  void CreateOp(DataType dtype, int n, int concat_dim) {
    TF_ASSERT_OK(NodeDefBuilder("op", "SparseConcat")
                     .Input(FakeInput(n, DT_INT64))
                     .Input(FakeInput(n, dtype))
                     .Input(FakeInput(n, DT_INT64))
                     .Attr("N", n)
                     .Attr("concat_dim", concat_dim)
                     .Finalize(node_def()));
  }
};

TEST_F(SparseConcatTest, Concat_dim0_100_str) {
  CreateOp(DT_STRING, 100, 0);
  TF_ASSERT_OK(InitOp());
  std::vector<int64> indices({0, 0, 0, 1, 3, 2, 2, 0, 1, 3, 3, 3});
  std::vector<string> values({"a", "b", "c", "d"});
  std::vector<int64> shapes(3, 4);

  std::vector<int64> out_indices;
  std::vector<string> out_values;
  std::vector<int64> out_shapes({400, 4, 4});
  out_indices.reserve(1200);
  out_values.reserve(400);
  // input and output
  for (int i = 0; i < 100; ++i) {
    AddInputFromArray<int64>(TensorShape({4, 3}), indices);
    out_indices.insert(out_indices.end(), indices.begin(), indices.end());
  }
  for (int i = 0; i < 400; ++i) {
    out_indices[i * 3] += (i / 4) * 4;
  }
  for (int i = 0; i < 100; ++i) {
    AddInputFromArray<string>(TensorShape({4}), values);
    out_values.insert(out_values.end(), values.begin(), values.end());
  }
  for (int i = 0; i < 100; ++i) {
    AddInputFromArray<int64>(TensorShape({3}), shapes);
  }

  TF_ASSERT_OK(RunOpKernel());

  Tensor expect_idx(DT_INT64, TensorShape{400, 3});
  Tensor expect_val(DT_STRING, TensorShape{400});
  Tensor expect_shape(DT_INT64, TensorShape{3});
  test::FillValues<int64>(&expect_idx, out_indices);
  test::FillValues<string>(&expect_val, out_values);
  test::FillValues<int64>(&expect_shape, out_shapes);
  test::ExpectTensorEqual<int64>(expect_idx, *GetOutput(0));
  test::ExpectTensorEqual<string>(expect_val, *GetOutput(1));
  test::ExpectTensorEqual<int64>(expect_shape, *GetOutput(2));
}

TEST_F(SparseConcatTest, Concat_dim1_100_int) {
  CreateOp(DT_INT64, 100, 1);
  TF_ASSERT_OK(InitOp());
  std::vector<int64> indices({0, 0, 0, 1, 3, 2, 2, 0, 1, 3, 3, 3});
  std::vector<int64> values({1, 2, 3, 4});
  std::vector<int64> shapes(3, 4);

  std::vector<int64> out_indices(400 * 3);
  std::vector<int64> out_values(400);
  std::vector<int64> out_shapes({4, 400, 4});
  out_indices.reserve(1200);
  out_values.reserve(400);
  // input and output
  for (int i = 0; i < 100; ++i) {
    AddInputFromArray<int64>(TensorShape({4, 3}), indices);
  }
  for (int i = 0; i < 100; ++i) {
    AddInputFromArray<int64>(TensorShape({4}), values);
  }
  for (int i = 0; i < 100; ++i) {
    AddInputFromArray<int64>(TensorShape({3}), shapes);
  }
  for (int i = 0; i < 400; ++i) {
    out_indices[i * 3] = indices[(i / 100) * 3];
    out_indices[i * 3 + 1] = indices[(i / 100) * 3 + 1] + (i % 100) * 4;
    out_indices[i * 3 + 2] = indices[(i / 100) * 3 + 2];
    out_values[i] = values[i / 100];
  }

  TF_ASSERT_OK(RunOpKernel());

  Tensor expect_idx(DT_INT64, TensorShape{400, 3});
  Tensor expect_val(DT_INT64, TensorShape{400});
  Tensor expect_shape(DT_INT64, TensorShape{3});
  test::FillValues<int64>(&expect_idx, out_indices);
  test::FillValues<int64>(&expect_val, out_values);
  test::FillValues<int64>(&expect_shape, out_shapes);
  test::ExpectTensorEqual<int64>(expect_idx, *GetOutput(0));
  test::ExpectTensorEqual<int64>(expect_val, *GetOutput(1));
  test::ExpectTensorEqual<int64>(expect_shape, *GetOutput(2));
}

TEST_F(SparseConcatTest, Concat_dim2_1000_float) {
  CreateOp(DT_FLOAT, 1000, 2);
  TF_ASSERT_OK(InitOp());
  std::vector<int64> indices({0, 0, 0, 1, 3, 2, 2, 0, 1, 3, 3, 3});
  std::vector<float> values({0.1, 0.2, 0.3, 0.4});
  std::vector<int64> shapes(3, 4);

  std::vector<int64> out_indices(4000 * 3);
  std::vector<float> out_values(4000);
  std::vector<int64> out_shapes({4, 4, 4000});
  out_indices.reserve(12000);
  out_values.reserve(4000);
  // input and output
  for (int i = 0; i < 1000; ++i) {
    AddInputFromArray<int64>(TensorShape({4, 3}), indices);
  }
  for (int i = 0; i < 1000; ++i) {
    AddInputFromArray<float>(TensorShape({4}), values);
  }
  for (int i = 0; i < 1000; ++i) {
    AddInputFromArray<int64>(TensorShape({3}), shapes);
  }
  for (int i = 0; i < 4000; ++i) {
    out_indices[i * 3] = indices[(i / 1000) * 3];
    out_indices[i * 3 + 1] = indices[(i / 1000) * 3 + 1];
    out_indices[i * 3 + 2] = indices[(i / 1000) * 3 + 2] + (i % 1000) * 4;
    out_values[i] = values[i / 1000];
  }

  TF_ASSERT_OK(RunOpKernel());

  Tensor expect_idx(DT_INT64, TensorShape{4000, 3});
  Tensor expect_val(DT_FLOAT, TensorShape{4000});
  Tensor expect_shape(DT_INT64, TensorShape{3});
  test::FillValues<int64>(&expect_idx, out_indices);
  test::FillValues<float>(&expect_val, out_values);
  test::FillValues<int64>(&expect_shape, out_shapes);
  test::ExpectTensorEqual<int64>(expect_idx, *GetOutput(0));
  test::ExpectTensorEqual<float>(expect_val, *GetOutput(1));
  test::ExpectTensorEqual<int64>(expect_shape, *GetOutput(2));
}

static void SparseConcatHelper(int iters, string Op, int len, int num,
                               int concat_dim, int nth) {
  testing::StopTiming();
  Graph* g = new Graph(OpRegistry::Global());

  std::vector<int64> indices(len * 3);
  std::vector<string> values(
      len,
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
  std::vector<int64> shapes(3, 64);
  for (int i = 0; i < len; ++i) {
    indices[i * 3] = i / 16;
    indices[i * 3 + 1] = i % 16;
    indices[i * 3 + 2] = i;
  }

  std::vector<NodeBuilder::NodeOut> t_indices;
  std::vector<NodeBuilder::NodeOut> t_values;
  std::vector<NodeBuilder::NodeOut> t_shapes;
  t_indices.reserve(num);
  t_values.reserve(num);
  t_shapes.reserve(num);
  for (int i = 0; i < num; ++i) {
    Tensor idx(DT_INT64, TensorShape({len, 3}));
    Tensor val(DT_STRING, TensorShape({len}));
    Tensor shp(DT_INT64, TensorShape({3}));
    test::FillValues<int64>(&idx, indices);
    test::FillValues<string>(&val, values);
    test::FillValues<int64>(&shp, shapes);
    t_indices.push_back(test::graph::Constant(g, idx));
    t_values.push_back(test::graph::Constant(g, val));
    t_shapes.push_back(test::graph::Constant(g, shp));
  }
  Node* node;
  TF_ASSERT_OK(NodeBuilder(g->NewName("n"), Op.c_str())
                   .Input(t_indices)
                   .Input(t_values)
                   .Input(t_shapes)
                   .Attr("N", num)
                   .Attr("concat_dim", concat_dim)
                   .Finalize(g, &node));

  testing::UseRealTime();
  testing::ItemsProcessed(static_cast<int64>(iters) * num * len);
  SessionOptions opts;
  opts.config.set_intra_op_parallelism_threads(nth);
  testing::StartTiming();
  test::Benchmark("cpu", g, &opts).Run(iters);
}

#define BM_SparseConcat(OP, LEN, N, DIM, NTH)                    \
  static void BM_##OP##_##LEN##_##N##_##DIM##_##NTH(int iters) { \
    return SparseConcatHelper(iters, #OP, LEN, N, DIM, NTH);     \
  }                                                              \
  BENCHMARK(BM_##OP##_##LEN##_##N##_##DIM##_##NTH);

BM_SparseConcat(SparseConcat, 200, 15, 0, 1);
BM_SparseConcat(SparseConcat, 200, 15, 0, 4);
BM_SparseConcat(SparseConcat, 200, 15, 1, 1);
BM_SparseConcat(SparseConcat, 200, 15, 1, 4);
BM_SparseConcat(SparseConcat, 1, 1000, 0, 1);
BM_SparseConcat(SparseConcat, 1, 1000, 0, 4);
BM_SparseConcat(SparseConcat, 1, 1000, 1, 1);
BM_SparseConcat(SparseConcat, 1, 1000, 1, 4);
BM_SparseConcat(SparseConcat, 10000, 5, 0, 1);
BM_SparseConcat(SparseConcat, 10000, 5, 0, 4);
BM_SparseConcat(SparseConcat, 10000, 5, 2, 1);
BM_SparseConcat(SparseConcat, 10000, 5, 2, 4);
BM_SparseConcat(SparseConcat, 2000, 100, 0, 1);
BM_SparseConcat(SparseConcat, 2000, 100, 0, 4);
BM_SparseConcat(SparseConcat, 2000, 100, 2, 1);
BM_SparseConcat(SparseConcat, 2000, 100, 2, 4);

}  // namespace tensorflow
