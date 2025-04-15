/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifdef INTEL_MKL

#include <initializer_list>

#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/mkl_testlib.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

template <typename T>
static void BM_Softmax(::testing::benchmark::State& state,
                       std::initializer_list<int64_t> dims, int num_threads,
                       const string& label, bool onednn) {
  DataType dtype = DataTypeToEnum<T>::v();
  TensorShape shape = TensorShape(dims);
  Tensor input(dtype, shape);
  input.flat<T>().setRandom();
  Graph* g = new Graph(OpRegistry::Global());
  if (onednn) {
    test::graph::oneDNNSoftmax(g, test::graph::Constant(g, input));
  } else {
    auto root = Scope::NewRootScope().ExitOnError();
    auto softmax = ops::Softmax(root, input);
    TF_CHECK_OK(root.status());
    TF_CHECK_OK(root.ToGraph(g));
  }
  SessionOptions opts;
  opts.config.set_inter_op_parallelism_threads(1);
  opts.config.set_intra_op_parallelism_threads(num_threads);
  opts.config.set_use_per_session_threads(true);
  opts.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions::L0);
  test::Benchmark("cpu", g, &opts, nullptr, nullptr, "",
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(shape.num_elements() * state.iterations());
  state.SetLabel(label);
}

// For a tensor shape {a, b, c, d}, we want to produce a token a_b_c_d
#define CONCAT_DIMS1(a) _##a
#define CONCAT_DIMS2(a, b) _##a##_##b
#define CONCAT_DIMS3(a, b, c) _##a##_##b##_##c
#define CONCAT_DIMS4(a, b, c, d) _##a##_##b##_##c##_##d
#define CONCAT_DIMS5(a, b, c, d, e) _##a##_##b##_##c##_##d##_e
#define JOIN(x, y) JOIN_AGAIN(x, y)
#define JOIN_AGAIN(x, y) x##y

// Wrapping BENCHMARK to get benchmark name appropriate, so that the argument
// expansion takes place before the expansion of BENCHMARK.
#define WRAP_BENCHMARK(FUNC) BENCHMARK(FUNC)

#define BM_oneDNN_Softmax(dtype, num_threads, label, num_dims, ...)          \
  static void JOIN(BM_oneDNN_Softmax_##dtype##_intraop_##num_threads##_dims, \
                   JOIN(CONCAT_DIMS, num_dims)(__VA_ARGS__))(                \
      ::testing::benchmark::State & state) {                                 \
    BM_Softmax<dtype>(state, {__VA_ARGS__}, num_threads, label, true);       \
  }                                                                          \
  WRAP_BENCHMARK(                                                            \
      JOIN(BM_oneDNN_Softmax_##dtype##_intraop_##num_threads##_dims,         \
           JOIN(CONCAT_DIMS, num_dims)(__VA_ARGS__)))                        \
      ->MeasureProcessCPUTime()

#define BM_Eigen_Softmax(dtype, num_threads, label, num_dims, ...)             \
  static void JOIN(BM_Eigen_Softmax_##dtype##_intraop_##num_threads##_dims,    \
                   JOIN(CONCAT_DIMS, num_dims)(__VA_ARGS__))(                  \
      ::testing::benchmark::State & state) {                                   \
    BM_Softmax<dtype>(state, {__VA_ARGS__}, num_threads, label, false);        \
  }                                                                            \
  WRAP_BENCHMARK(JOIN(BM_Eigen_Softmax_##dtype##_intraop_##num_threads##_dims, \
                      JOIN(CONCAT_DIMS, num_dims)(__VA_ARGS__)))               \
      ->MeasureProcessCPUTime()

#define BM_Softmax(dtype, num_threads, label, num_dims, ...)           \
  BM_oneDNN_Softmax(dtype, num_threads, label, num_dims, __VA_ARGS__); \
  BM_Eigen_Softmax(dtype, num_threads, label, num_dims, __VA_ARGS__);

BM_Softmax(float, 4, "float32_BERT_batch_size_1", 4, 1, 16, 384, 384);
BM_Softmax(float, 4, "float32_BERT_batch_size_16", 4, 16, 16, 384, 384);
BM_Softmax(float, 1, "float32_ImageNet_batch_size_32", 2, 32, 1008);
BM_Softmax(float, 1, "float32_ImageNet_batch_size_128", 2, 128, 1008);
BM_Softmax(float, 4, "float32_ImageNet_batch_size_32", 2, 32, 1008);
BM_Softmax(float, 4, "float32_ImageNet_batch_size_128", 2, 128, 1008);
BM_Softmax(bfloat16, 4, "bfloat16_BERT_batch_size_1", 4, 1, 16, 384, 384);
BM_Softmax(bfloat16, 4, "bfloat16_BERT_batch_size_16", 4, 16, 16, 384, 384);
BM_Softmax(bfloat16, 1, "bfloat16_ImageNet_batch_size_32", 2, 32, 1008);
BM_Softmax(bfloat16, 1, "bfloat16_ImageNet_batch_size_128", 2, 128, 1008);
BM_Softmax(bfloat16, 4, "bfloat16_ImageNet_batch_size_32", 2, 32, 1008);
BM_Softmax(bfloat16, 4, "bfloat16_ImageNet_batch_size_128", 2, 128, 1008);

}  // namespace
}  // namespace tensorflow

#endif  // INTEL_MKL
