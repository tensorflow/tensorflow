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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

// We focus on the single thread performance of running ops.
static SessionOptions InitOptions() {
  SessionOptions opts;
  opts.config.set_intra_op_parallelism_threads(1);
  opts.config.set_inter_op_parallelism_threads(1);
  return opts;
}

static SessionOptions* GetOptions() {
  static SessionOptions opts = InitOptions();
  return &opts;
}

static Node* Var(Graph* g, int n) {
  return test::graph::Var(g, DT_FLOAT, TensorShape({n}));
}

static Node* Zeros(Graph* g, int n) {
  Tensor data(DT_FLOAT, TensorShape({n}));
  data.flat<float>().setZero();
  return test::graph::Constant(g, data);
}

static void MulChain(int chain_length, Graph** init_g, Graph** run_g) {
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, 1);
    test::graph::Assign(g, var, Zeros(g, 1));
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    Node* cur = Var(g, 1);
    for (int i = 0; i < chain_length - 1; ++i) {
      cur = test::graph::Multi(g, i % 2 == 0 ? "Div" : "Mul", {cur, cur});
    }
    *run_g = g;
  }
}

// Benchmark a chain of simple multiplications.
// This emphasizes per-op overhead.
static void BM_MulChain(int iters, int chain_length) {
  const int64 tot = static_cast<int64>(iters) * chain_length;
  testing::ItemsProcessed(tot);
  Graph* init;
  Graph* run;
  MulChain(chain_length, &init, &run);
  test::Benchmark("cpu", run, GetOptions(), init).Run(iters);
}
BENCHMARK(BM_MulChain)->Arg(1 << 10);

}  // end namespace tensorflow
