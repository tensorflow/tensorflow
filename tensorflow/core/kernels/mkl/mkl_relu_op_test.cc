/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#if defined(INTEL_MKL) && defined(ENABLE_MKL)

#include "absl/strings/match.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/mkl_util.h"

// Compare performance of default Tensorflow convolution kernels (Eigen) with
// MKL kernels on CPU.
// Before running these benchmarks configure OpenMP environment variables:
//   export KMP_BLOCKTIME=0
//   export OMP_NUM_THREADS=${num_threads}

namespace tensorflow {

static Graph* Activation(const string& op_name, const string& kind,
                         const TensorShape& shape) {
  auto* graph = new Graph(OpRegistry::Global());
  const string node_name = kind + "_" + op_name;
  const bool isForwardOp = !tensorflow::str_util::EndsWith(op_name, "Grad");
  const bool isDefault = (kind == "Default");

  Tensor input_t(DT_FLOAT, shape);
  input_t.flat<float>().setRandom();
  Node* input = test::graph::Constant(graph, input_t, "input");
  Node* not_mkl_shape =
      test::graph::Constant(graph, GetMklMetaTensor(), "not_mkl");

  if (isForwardOp) {
    // Default forward op.
    if (isDefault) {
      TF_CHECK_OK(NodeBuilder(graph->NewName(node_name), op_name)
                      .Input(input)
                      .Attr("T", DT_FLOAT)
                      .Finalize(graph, nullptr));
      return graph;
    }
    // MKL forward op.
    TF_CHECK_OK(NodeBuilder(graph->NewName(node_name), "_Mkl" + op_name)
                    .Input(input)
                    .Input(not_mkl_shape)
                    .Attr("T", DT_FLOAT)
                    .Attr("_kernel", "MklLayoutDependentOp")
                    .Finalize(graph, nullptr));
    return graph;
  }

  // Default backward op.
  Tensor grad_t(DT_FLOAT, shape);
  grad_t.flat<float>().setRandom();
  Node* grad = test::graph::Constant(graph, grad_t, "grad");
  if (isDefault) {
    TF_CHECK_OK(NodeBuilder(graph->NewName(node_name), op_name)
                    .Input(grad)
                    .Input(input)
                    .Attr("T", DT_FLOAT)
                    .Finalize(graph, nullptr));
    return graph;
  }

  // MKL backward op.
  TF_CHECK_OK(NodeBuilder(graph->NewName(node_name), "_Mkl" + op_name)
                  .Input(grad)
                  .Input(input)
                  .Input(not_mkl_shape)
                  .Input(not_mkl_shape)
                  .Attr("T", DT_FLOAT)
                  .Attr("_kernel", "MklLayoutDependentOp")
                  .Finalize(graph, nullptr));
  return graph;
}

#define BM_Activation(op, kind, A, B, C, D, type)                 \
  static void BM_##op##_##kind##_##type##_##A##_##B##_##C##_##D(  \
      ::testing::benchmark::State& state) {                       \
    int64 num_computed_elements = (A) * (B) * (C) * (D);          \
    int64 flops_per_iter = num_computed_elements;                 \
                                                                  \
    test::Benchmark(#type, Activation(#op, #kind, {A, B, C, D}),  \
                    /*old_benchmark_api*/ false)                  \
        .Run(state);                                              \
    state.SetItemsProcessed(state.iterations() * flops_per_iter); \
  }                                                               \
  BENCHMARK(BM_##op##_##kind##_##type##_##A##_##B##_##C##_##D)

#define BM(op, A, B, C, D, type)                \
  BM_Activation(op, Default, A, B, C, D, type); \
  BM_Activation(op, Mkl, A, B, C, D, type);

#define TEST_ALL_SIZES(OP)       \
  BM(OP, 2, 4, 8, 16, cpu);      \
  BM(OP, 3, 5, 9, 17, cpu);      \
  BM(OP, 32, 64, 128, 256, cpu); \
  BM(OP, 33, 65, 129, 257, cpu);

TEST_ALL_SIZES(Tanh)
TEST_ALL_SIZES(TanhGrad)
TEST_ALL_SIZES(Relu)
TEST_ALL_SIZES(ReluGrad)
TEST_ALL_SIZES(Elu)
TEST_ALL_SIZES(EluGrad)
TEST_ALL_SIZES(Relu6)
TEST_ALL_SIZES(Relu6Grad)
TEST_ALL_SIZES(LeakyRelu)
TEST_ALL_SIZES(LeakyReluGrad)

}  // namespace tensorflow

#endif  // INTEL_MKL && ENABLE_MKL
