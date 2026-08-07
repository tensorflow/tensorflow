/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Benchmark comparing the KDNN-backed _KdnnSigmoid activation kernel to the
// default CPU Sigmoid kernel. Intended to be run on Kunpeng 920 (aarch64)
// hardware with a real libkdnn.so to collect the speedup figures referenced
// in PR #124543 review feedback.
//
// This file is intentionally buildable on every platform. When IsKDNNEnabled()
// returns false (the default on x86 hosts, on machines without libkdnn.so,
// or when --define=enable_kdnn=true was not passed), the benchmark reports
// "KDNN unavailable on this platform" and skips its body without failing.
// That keeps the file CI-friendly on the standard x86/GPU matrix while still
// running for real on aarch64 + libkdnn.so.
//
// Run on a Kunpeng 920 host:
//
//   bazel test --define=enable_kdnn=true \
//     --test_arg=--benchmark_filter=BM_KdnnVsCpuSigmoid \
//     //tensorflow/core/kernels/kdnn:kdnn_sigmoid_benchmark_test
//
// Expected output (placeholder numbers; replace with real measurements once
// hardware is available):
//
//   BM_KdnnVsCpuSigmoid/kdnn/8192     1000000  ...  ~1.2 ns/elem
//   BM_KdnnVsCpuSigmoid/cpu/8192      1000000  ...  ~0.8 ns/elem
//   BM_KdnnVsCpuSigmoid/kdnn/131072    100000  ...  ~4.0 ns/elem
//   BM_KdnnVsCpuSigmoid/cpu/131072     100000  ...  ~2.5 ns/elem
//
// The numbers above are *not* real measurements; they are placeholder
// markers to make the format obvious to anyone running the benchmark. See
// third_party/KDNN/HARDWARE_VERIFICATION.md for the full verification recipe.

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {
namespace {

// Single-thread options; KDNN vector kernels are bandwidth-bound on the
// activation path, so parallel speedup is orthogonal to this comparison.
SessionOptions BmOptions() {
  SessionOptions opts;
  opts.config.set_intra_op_parallelism_threads(1);
  opts.config.set_inter_op_parallelism_threads(1);
  return opts;
}

SessionOptions* BmOptionsPtr() {
  static SessionOptions opts = BmOptions();
  return &opts;
}

// Build a single-input, single-output graph that calls the op named
// `op_name` on a 1-D float tensor of size `n`.
Graph* ActivationGraph(const string& op_name, int n) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor data(DT_FLOAT, TensorShape({n}));
  data.flat<float>().setRandom();
  Node* in = test::graph::Constant(g, data);
  // Promote the 1-D tensor to shape [n, 1]; the activation ops expect
  // a rank-2+ input.
  TensorShape shape({n, 1});
  Node* reshaped = test::graph::Reshape(g, in, test::graph::Constant(g, shape));
  test::graph::Unary(g, op_name, reshaped);
  return g;
}

// Drive one run of the benchmark against a particular (backend, size) pair.
void RunOne(::testing::benchmark::State& state, const string& backend,
            int n) {
  const string op_name = (backend == "kdnn") ? "_KdnnSigmoid" : "Sigmoid";
  if (backend == "kdnn" && !IsKDNNEnabled()) {
    state.SkipWithError(
        "KDNN unavailable on this platform. Run on aarch64 with "
        "--define=enable_kdnn=true and a libkdnn.so in KDNN_ROOT. See "
        "third_party/KDNN/HARDWARE_VERIFICATION.md for the recipe.");
    return;
  }
  Graph* run_g = ActivationGraph(op_name, n);
  test::Benchmark("cpu", run_g, BmOptionsPtr(), nullptr, nullptr, "",
                  /*old_benchmark_api=*/false)
      .Run(state);
  // ItemsProcessed counts float elements touched (input + output).
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(n) * 2);
}

void BM_KdnnVsCpuSigmoid_Kdnn_8192(::testing::benchmark::State& state) {
  RunOne(state, "kdnn", 8192);
}
void BM_KdnnVsCpuSigmoid_Cpu_8192(::testing::benchmark::State& state) {
  RunOne(state, "cpu", 8192);
}
void BM_KdnnVsCpuSigmoid_Kdnn_131072(::testing::benchmark::State& state) {
  RunOne(state, "kdnn", 131072);
}
void BM_KdnnVsCpuSigmoid_Cpu_131072(::testing::benchmark::State& state) {
  RunOne(state, "cpu", 131072);
}
BENCHMARK(BM_KdnnVsCpuSigmoid_Kdnn_8192);
BENCHMARK(BM_KdnnVsCpuSigmoid_Cpu_8192);
BENCHMARK(BM_KdnnVsCpuSigmoid_Kdnn_131072);
BENCHMARK(BM_KdnnVsCpuSigmoid_Cpu_131072);

}  // namespace
}  // namespace tensorflow