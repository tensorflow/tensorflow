/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class UnaryOpsCompositionTest : public OpsTestBase {
 protected:
  template <typename T>
  void RunComposedOp(const std::vector<string> op_names, T input, T expected) {
    TF_ASSERT_OK(NodeDefBuilder("unary_op_composition", "_UnaryOpsComposition")
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Attr("T", DataTypeToEnum<T>::v())
                     .Attr("op_names", op_names)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    TensorShape shape({});
    AddInputFromArray<T>(shape, {input});

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected_tensor(allocator(), DataTypeToEnum<T>::value, shape);
    test::FillValues<T>(&expected_tensor, {expected});
    test::ExpectClose(expected_tensor, *GetOutput(0));
  }
};

TEST_F(UnaryOpsCompositionTest, Compose_Sqrt_Sqrt_F) {
  RunComposedOp<float>({"Sqrt", "Sqrt"}, 81.0, 3.0);
}

TEST_F(UnaryOpsCompositionTest, Compose_Sqrt_Sqrt_D) {
  RunComposedOp<double>({"Sqrt", "Sqrt"}, 81.0, 3.0);
}

TEST_F(UnaryOpsCompositionTest, Compose_Sqrt_Sin_F) {
  RunComposedOp<float>({"Sqrt", "Sin"}, 81.0, std::sin(9.0f));
}

TEST_F(UnaryOpsCompositionTest, Compose_Cos_Acos_F) {
  RunComposedOp<float>({"Cos", "Acos"}, 0.5, std::acos(std::cos(0.5f)));
}

TEST_F(UnaryOpsCompositionTest, Compose_Tanh_Relu_F) {
  RunComposedOp<float>({"Tanh", "Relu"}, 0.5, std::max(0.0f, std::tanh(0.5f)));
}

TEST_F(UnaryOpsCompositionTest, Compose_Tanh_Relu_D) {
  RunComposedOp<double>({"Tanh", "Relu"}, 0.5, std::max(0.0, std::tanh(0.5)));
}

TEST_F(UnaryOpsCompositionTest, Compose_Tanh_Relu6_F) {
  RunComposedOp<float>({"Relu6"}, 11.0f, 6.0f);
}

// Performance benchmarks below.

string Function(int i) {
  std::vector<string> ops = {"Tanh", "Relu", "Sigmoid", "Sqrt", "Log", "Exp"};
  return ops[i % ops.size()];
}

// Unary ops chained together as a separate graph nodes.
static Graph* UnaryOpsChain(int tensor_size, int repeat_graph,
                            int num_functions) {
  Graph* g = new Graph(OpRegistry::Global());

  Tensor t(DT_FLOAT, TensorShape({tensor_size}));
  t.flat<float>() = t.flat<float>().setRandom();

  for (int i = 0; i < repeat_graph; ++i) {
    Node* node = test::graph::Constant(g, t);
    for (int j = 0; j < num_functions; ++j) {
      TF_CHECK_OK(NodeBuilder(g->NewName("n"), Function(j))
                      .Input(node)
                      .Attr("T", DT_FLOAT)
                      .Finalize(g, &node));
    }
  }

  return g;
}

#define BM_UnaryOpsChain(N, R, F, type)                                \
  static void BM_UnaryOpsChain##_##type##_##N##_##R##_##F(int iters) { \
    testing::ItemsProcessed(static_cast<int64>(iters) * N * R * F);    \
    test::Benchmark(#type, UnaryOpsChain(N, R, F)).Run(iters);         \
  }                                                                    \
  BENCHMARK(BM_UnaryOpsChain##_##type##_##N##_##R##_##F);

// Unary ops fused together.
static Graph* UnaryOpsCompo(int tensor_size, int repeat_graph,
                            int num_functions) {
  Graph* g = new Graph(OpRegistry::Global());

  Tensor t(DT_FLOAT, TensorShape({tensor_size}));
  t.flat<float>() = t.flat<float>().setRandom();

  std::vector<string> functions;
  for (int j = 0; j < num_functions; ++j) {
    functions.push_back(Function(j));
  }

  for (int i = 0; i < repeat_graph; ++i) {
    Node* node = test::graph::Constant(g, t);
    TF_CHECK_OK(NodeBuilder(g->NewName("n"), "_UnaryOpsComposition")
                    .Input(node)
                    .Attr("T", DT_FLOAT)
                    .Attr("op_names", functions)
                    .Finalize(g, &node));
  }

  return g;
}

#define BM_UnaryOpsCompo(N, R, F, type)                                \
  static void BM_UnaryOpsCompo##_##type##_##N##_##R##_##F(int iters) { \
    testing::ItemsProcessed(static_cast<int64>(iters) * N * R * F);    \
    test::Benchmark(#type, UnaryOpsCompo(N, R, F)).Run(iters);         \
  }                                                                    \
  BENCHMARK(BM_UnaryOpsCompo##_##type##_##N##_##R##_##F);

// BenchmarkName(tensor_size, repeat_graph, num_ops, type)

BM_UnaryOpsChain(1000, 25, 2, cpu);
BM_UnaryOpsCompo(1000, 25, 2, cpu);

BM_UnaryOpsChain(1000, 25, 5, cpu);
BM_UnaryOpsCompo(1000, 25, 5, cpu);

BM_UnaryOpsChain(1000, 25, 10, cpu);
BM_UnaryOpsCompo(1000, 25, 10, cpu);

BM_UnaryOpsChain(100000, 25, 2, cpu);
BM_UnaryOpsCompo(100000, 25, 2, cpu);

BM_UnaryOpsChain(100000, 25, 5, cpu);
BM_UnaryOpsCompo(100000, 25, 5, cpu);

BM_UnaryOpsChain(100000, 25, 10, cpu);
BM_UnaryOpsCompo(100000, 25, 10, cpu);

BM_UnaryOpsChain(1000000, 25, 2, cpu);
BM_UnaryOpsCompo(1000000, 25, 2, cpu);

BM_UnaryOpsChain(1000000, 25, 5, cpu);
BM_UnaryOpsCompo(1000000, 25, 5, cpu);

BM_UnaryOpsChain(1000000, 25, 10, cpu);
BM_UnaryOpsCompo(1000000, 25, 10, cpu);

}  // namespace
}  // end namespace tensorflow
