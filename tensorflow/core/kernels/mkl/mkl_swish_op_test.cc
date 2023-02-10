/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#if defined(INTEL_MKL) && !defined(ENABLE_ONEDNN_V3)

#include "absl/strings/match.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/mkl/mkl_eltwise_activation_base_op.h"
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

// This is a special case, because EIGEN kernels does not have Swish Kerenls.
// Compare the performance of default tensorflow kernels (Eigen) with
// MKL kernels on CPU.
//
// Then you could use below command to test mkl and eigen performance:
// $ bazel run --action_env=TF_ENABLE_ONEDNN_OPTS=1 -c opt \
//  //tensorflow/core/kernels/mkl:mkl_swish_op_test -- --benchmark_filter=all
//

namespace tensorflow {

// --------------------------------------------------------------------------//
//  Test Swish Kernels accuracy and performance                              //
// --------------------------------------------------------------------------//
template <typename T>
static Graph* SwishGraph(const string& kind, const TensorShape& shape) {
  auto* graph = new Graph(OpRegistry::Global());

  DataType dtype = DataTypeToEnum<T>::v();
  Tensor input_t(dtype, shape);
  input_t.flat<T>().setRandom();
  Node* input = test::graph::Constant(graph, input_t, "input");
  const bool isDefault = (kind == "Default");

  Node* sigmoid;
  Node* mul;
  Node* swish;
  if (isDefault) {
    TF_CHECK_OK(NodeBuilder(graph->NewName("Default_sigmoid"), "Sigmoid")
                    .Input(input)
                    .Attr("T", dtype)
                    .Finalize(graph, &sigmoid));

    TF_CHECK_OK(NodeBuilder(graph->NewName("Default_mul"), "Mul")
                    .Input(input)
                    .Input(sigmoid)
                    .Attr("T", dtype)
                    .Finalize(graph, &mul));
    return graph;
  }
  // Mkl Swish op.
  TF_CHECK_OK(NodeBuilder(graph->NewName("Mkl_swish"), "_MklSwish")
                  .Input(input)
                  .Attr("T", dtype)
                  .Finalize(graph, &swish));
  return graph;
}

#define BM_SWISH(kind, A, B, C, D, type, T)                                \
  static void BM_SWISH_##kind##_##type##_##A##_##B##_##C##_##D##_##T(      \
      ::testing::benchmark::State& state) {                                \
    int64 num_computed_elements = (A) * (B) * (C) * (D);                   \
    int64 flops_per_iter = num_computed_elements;                          \
                                                                           \
    test::Benchmark(#type, SwishGraph<T>(#kind, {A, B, C, D})).Run(state); \
    state.SetItemsProcessed(state.iterations() * flops_per_iter);          \
  }                                                                        \
  BENCHMARK(BM_SWISH_##kind##_##type##_##A##_##B##_##C##_##D##_##T)

#define BENCHMARK_SWISH(A, B, C, D, type, T) \
  BM_SWISH(Default, A, B, C, D, type, T);    \
  BM_SWISH(Mkl, A, B, C, D, type, T);

#define BENCHMARK_DTYPE(T)                    \
  BENCHMARK_SWISH(1, 16, 16, 3, cpu, T);      \
  BENCHMARK_SWISH(16, 32, 32, 1, cpu, T);     \
  BENCHMARK_SWISH(16, 64, 64, 128, cpu, T);   \
  BENCHMARK_SWISH(32, 64, 64, 128, cpu, T);   \
  BENCHMARK_SWISH(32, 256, 256, 128, cpu, T); \
  BENCHMARK_SWISH(32, 512, 512, 128, cpu, T);

BENCHMARK_DTYPE(float)
BENCHMARK_DTYPE(bfloat16)

}  // namespace tensorflow

#endif  // INTEL_MKL && !ENABLE_ONEDNN_V3
