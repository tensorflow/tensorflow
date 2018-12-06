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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

// We focus on the single thread performance of training ops.
static SessionOptions InitSingleThreadedOptions() {
  SessionOptions opts;
  opts.config.set_intra_op_parallelism_threads(1);
  opts.config.set_inter_op_parallelism_threads(1);
  return opts;
}

static SessionOptions* GetOptions() {
  static SessionOptions opts = InitSingleThreadedOptions();
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

static Node* Random(Graph* g, int n) {
  Tensor data(DT_FLOAT, TensorShape({n}));
  data.flat<float>().setRandom();
  return test::graph::Constant(g, data);
}

static Node* Scalar(Graph* g, float val) {
  Tensor data(DT_FLOAT, TensorShape({}));
  data.flat<float>()(0) = val;
  return test::graph::Constant(g, data);
}

static void SGD(int32 n, Graph** init_g, Graph** train_g) {
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    test::graph::Assign(g, var, Zeros(g, n));
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto lr = Scalar(g, 0.01);
    auto grad = Random(g, n);
    test::graph::Multi(g, "ApplyGradientDescent", {var, lr, grad});
    *train_g = g;
  }
}

static void BM_SGD(int iters, int params) {
  const int64 tot = static_cast<int64>(iters) * params;
  testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * sizeof(float));
  Graph* init;
  Graph* train;
  SGD(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init).Run(iters);
}
BENCHMARK(BM_SGD)->Arg(128 << 10)->Arg(256 << 10);

static void Adagrad(int32 n, Graph** init_g, Graph** train_g) {
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto accum = Var(g, n);
    auto zero = Zeros(g, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, accum, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto accum = Var(g, n);
    auto lr = Scalar(g, 0.01);
    auto grad = Random(g, n);
    test::graph::Multi(g, "ApplyAdagrad", {var, accum, lr, grad});
    *train_g = g;
  }
}

static void BM_Adagrad(int iters, int params) {
  const int64 tot = static_cast<int64>(iters) * params;
  testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * sizeof(float));
  Graph* init;
  Graph* train;
  Adagrad(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init).Run(iters);
}
BENCHMARK(BM_Adagrad)->Arg(128 << 10)->Arg(256 << 10);

static void Momentum(int32 n, Graph** init_g, Graph** train_g) {
  TensorShape shape({n});
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto accum = Var(g, n);
    auto zero = Zeros(g, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, accum, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto accum = Var(g, n);
    auto lr = Scalar(g, 0.01);
    auto grad = Random(g, n);
    auto mom = Scalar(g, 0.01);
    test::graph::Multi(g, "ApplyMomentum", {var, accum, lr, grad, mom});
    *train_g = g;
  }
}

static void BM_Momentum(int iters, int params) {
  const int64 tot = static_cast<int64>(iters) * params;
  testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * sizeof(float));
  Graph* init;
  Graph* train;
  Momentum(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init).Run(iters);
}
BENCHMARK(BM_Momentum)->Arg(128 << 10)->Arg(256 << 10);

static void KerasMomentum(int32 n, Graph** init_g, Graph** train_g) {
  TensorShape shape({n});
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto accum = Var(g, n);
    auto zero = Zeros(g, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, accum, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto accum = Var(g, n);
    auto lr = Scalar(g, 0.01);
    auto grad = Random(g, n);
    auto mom = Scalar(g, 0.01);
    test::graph::Multi(g, "ApplyKerasMomentum", {var, accum, lr, grad, mom});
    *train_g = g;
  }
}

static void BM_KerasMomentum(int iters, int params) {
  const int64 tot = static_cast<int64>(iters) * params;
  testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * sizeof(float));
  Graph* init;
  Graph* train;
  KerasMomentum(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init).Run(iters);
}
BENCHMARK(BM_KerasMomentum)->Arg(128 << 10)->Arg(256 << 10);

static void Adam(int32 n, Graph** init_g, Graph** train_g) {
  TensorShape shape({n});
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto m = Var(g, n);
    auto v = Var(g, n);
    auto zero = Zeros(g, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, m, zero);
    test::graph::Assign(g, v, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto m = Var(g, n);
    auto v = Var(g, n);
    auto beta1_power = Scalar(g, 0.9);
    auto beta2_power = Scalar(g, 0.99);
    auto lr = Scalar(g, 0.01);
    auto beta1 = Scalar(g, 0.9);
    auto beta2 = Scalar(g, 0.99);
    auto epsilon = Scalar(g, 1e-8);
    auto grad = Random(g, n);
    test::graph::Multi(
        g, "ApplyAdam",
        {var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad});
    *train_g = g;
  }
}

static void BM_Adam(int iters, int params) {
  const int64 tot = static_cast<int64>(iters) * params;
  testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * sizeof(float));
  Graph* init;
  Graph* train;
  Adam(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init).Run(iters);
}
BENCHMARK(BM_Adam)->Arg(128 << 10)->Arg(256 << 10);

static void AdamWithAmsgrad(int32 n, Graph** init_g, Graph** train_g) {
  TensorShape shape({n});
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto m = Var(g, n);
    auto v = Var(g, n);
    auto zero = Zeros(g, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, m, zero);
    test::graph::Assign(g, v, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto m = Var(g, n);
    auto v = Var(g, n);
    auto vhat = Var(g, n);
    auto beta1_power = Scalar(g, 0.9);
    auto beta2_power = Scalar(g, 0.99);
    auto lr = Scalar(g, 0.01);
    auto beta1 = Scalar(g, 0.9);
    auto beta2 = Scalar(g, 0.99);
    auto epsilon = Scalar(g, 1e-8);
    auto grad = Random(g, n);
    test::graph::Multi(g, "ApplyAdamWithAmsgrad",
                       {var, m, v, vhat, beta1_power, beta2_power, lr, beta1,
                        beta2, epsilon, grad});
    *train_g = g;
  }
}

static void BM_AdamWithAmsgrad(int iters, int params) {
  const int64 tot = static_cast<int64>(iters) * params;
  testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * sizeof(float));
  Graph* init;
  Graph* train;
  AdamWithAmsgrad(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init).Run(iters);
}
BENCHMARK(BM_AdamWithAmsgrad)->Arg(128 << 10)->Arg(256 << 10);

static void RMSProp(int32 n, Graph** init_g, Graph** train_g) {
  TensorShape shape({n});
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto ms = Var(g, n);
    auto mom = Var(g, n);
    auto zero = Zeros(g, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, ms, zero);
    test::graph::Assign(g, mom, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto ms = Var(g, n);
    auto mom = Var(g, n);
    auto lr = Scalar(g, 0.01);
    auto rho = Scalar(g, 0.9);
    auto momentum = Scalar(g, 0.9);
    auto epsilon = Scalar(g, 1e-8);
    auto grad = Random(g, n);
    test::graph::Multi(g, "ApplyRMSProp",
                       {var, ms, mom, lr, rho, momentum, epsilon, grad});
    *train_g = g;
  }
}

static void BM_RMSProp(int iters, int params) {
  const int64 tot = static_cast<int64>(iters) * params;
  testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * sizeof(float));
  Graph* init;
  Graph* train;
  RMSProp(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init).Run(iters);
}
BENCHMARK(BM_RMSProp)->Arg(128 << 10)->Arg(256 << 10);

static void AddSign(int32 n, Graph** init_g, Graph** train_g) {
  TensorShape shape({n});
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto m = Var(g, n);
    auto zero = Zeros(g, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, m, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto m = Var(g, n);
    auto lr = Scalar(g, 0.01);
    auto alpha = Scalar(g, 0.1);
    auto sign_decay = Scalar(g, 0.9);
    auto beta = Scalar(g, 0.8);
    auto grad = Random(g, n);
    test::graph::Multi(g, "ApplyAddSign",
                       {var, m, lr, alpha, sign_decay, beta, grad});
    *train_g = g;
  }
}

static void BM_AddSign(int iters, int params) {
  const int64 tot = static_cast<int64>(iters) * params;
  testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * sizeof(float));
  Graph* init;
  Graph* train;
  AddSign(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init).Run(iters);
}
BENCHMARK(BM_AddSign)->Arg(128 << 10)->Arg(256 << 10);

static void PowerSign(int32 n, Graph** init_g, Graph** train_g) {
  TensorShape shape({n});
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto m = Var(g, n);
    auto zero = Zeros(g, n);
    test::graph::Assign(g, var, zero);
    test::graph::Assign(g, m, zero);
    *init_g = g;
  }
  {
    Graph* g = new Graph(OpRegistry::Global());
    auto var = Var(g, n);
    auto m = Var(g, n);
    auto lr = Scalar(g, 0.01);
    auto logbase = Scalar(g, 2);
    auto sign_decay = Scalar(g, 0.9);
    auto beta = Scalar(g, 0.8);
    auto grad = Random(g, n);
    test::graph::Multi(g, "ApplyPowerSign",
                       {var, m, lr, logbase, sign_decay, beta, grad});
    *train_g = g;
  }
}

static void BM_PowerSign(int iters, int params) {
  const int64 tot = static_cast<int64>(iters) * params;
  testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * sizeof(float));
  Graph* init;
  Graph* train;
  PowerSign(params, &init, &train);
  test::Benchmark("cpu", train, GetOptions(), init).Run(iters);
}
BENCHMARK(BM_PowerSign)->Arg(128 << 10)->Arg(256 << 10);

}  // end namespace tensorflow
