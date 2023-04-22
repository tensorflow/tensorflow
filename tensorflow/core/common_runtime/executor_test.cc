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

#include "tensorflow/core/common_runtime/executor.h"

#include <algorithm>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/common_runtime/lower_functional_ops.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class ExecutorTest : public ::testing::Test {
 protected:
  ExecutorTest()
      : device_(DeviceFactory::NewDevice("CPU", {},
                                         "/job:localhost/replica:0/task:0")),

        step_stats_collector_(&step_stats_) {
    SessionOptions options;
    thread_pool_ = ComputePool(options);
  }

  ~ExecutorTest() override {
    // There should always be exactly one Ref left on the Rendezvous
    // when the test completes.
    CHECK(rendez_->Unref());
    delete exec_;
  }

  // Resets executor_ with a new executor based on a graph 'gdef'.
  void Create(std::unique_ptr<const Graph> graph) {
    const int version = graph->versions().producer();
    LocalExecutorParams params;
    params.device = device_.get();
    params.create_kernel =
        [this, version](const std::shared_ptr<const NodeProperties>& props,
                        OpKernel** kernel) {
          return CreateNonCachedKernel(device_.get(), nullptr, props, version,
                                       kernel);
        };
    params.delete_kernel = [](OpKernel* kernel) {
      DeleteNonCachedKernel(kernel);
    };
    rendez_ = NewLocalRendezvous();
    delete exec_;
    TF_CHECK_OK(NewLocalExecutor(params, *graph, &exec_));
    runner_ = [this](std::function<void()> fn) { thread_pool_->Schedule(fn); };
  }

  Status Run(Rendezvous* rendez) {
    Executor::Args args;
    args.rendezvous = rendez;
    args.stats_collector = &step_stats_collector_;
    args.runner = runner_;
    return exec_->Run(args);
  }

  thread::ThreadPool* thread_pool_ = nullptr;
  std::unique_ptr<Device> device_;
  Executor* exec_ = nullptr;
  StepStatsCollector step_stats_collector_;
  StepStats step_stats_;
  Executor::Args::Runner runner_;
  Rendezvous* rendez_ = nullptr;
};

// A float val -> Tensor<float>
Tensor V(const float val) {
  Tensor tensor(DT_FLOAT, TensorShape({}));
  tensor.scalar<float>()() = val;
  return tensor;
}

// A int32 val -> Tensor<int32>
Tensor VI(const int32 val) {
  Tensor tensor(DT_INT32, TensorShape({}));
  tensor.scalar<int32>()() = val;
  return tensor;
}

// A bool val -> Tensor<bool>
Tensor VB(const bool val) {
  Tensor tensor(DT_BOOL, TensorShape({}));
  tensor.scalar<bool>()() = val;
  return tensor;
}

// A double val -> Tensor<double>
Tensor VD(const double val) {
  Tensor tensor(DT_DOUBLE, TensorShape({}));
  tensor.scalar<double>()() = val;
  return tensor;
}

// Tensor<float> -> a float val.
float V(const Tensor& tensor) {
  CHECK_EQ(tensor.dtype(), DT_FLOAT);
  CHECK(TensorShapeUtils::IsScalar(tensor.shape()));
  return tensor.scalar<float>()();
}

static uint64 kIncarnation = 1;  // Uses in following tests.

Rendezvous::ParsedKey Key(const string& sender, const uint64 incarnation,
                          const string& receiver, const string& name) {
  Rendezvous::ParsedKey result;
  CHECK(
      Rendezvous::ParseKey(Rendezvous::CreateKey(sender, incarnation, receiver,
                                                 name, FrameAndIter(0, 0)),
                           &result)
          .ok());
  return result;
}

#define ALICE "/job:j/replica:0/task:0/cpu:0"
#define BOB "/job:j/replica:0/task:0/device:GPU:0"

TEST_F(ExecutorTest, SimpleAdd) {
  // c = a + b
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  auto in0 = test::graph::Recv(g.get(), "a", "float", ALICE, 1, BOB);
  auto in1 = test::graph::Recv(g.get(), "b", "float", ALICE, 1, BOB);
  auto tmp = test::graph::Add(g.get(), in0, in1);
  test::graph::Send(g.get(), tmp, "c", BOB, 1, ALICE);
  Create(std::move(g));
  Rendezvous::Args args;
  TF_ASSERT_OK(rendez_->Send(Key(ALICE, kIncarnation, BOB, "a"), args, V(1.0),
                             false));  // in0 = 1.0
  TF_ASSERT_OK(rendez_->Send(Key(ALICE, kIncarnation, BOB, "b"), args, V(1.0),
                             false));  // in1 = 1.0
  TF_ASSERT_OK(Run(rendez_));
  Tensor out = V(-1);
  bool is_dead = false;
  TF_ASSERT_OK(
      rendez_->Recv(Key(BOB, kIncarnation, ALICE, "c"), args, &out, &is_dead));
  EXPECT_EQ(2.0, V(out));  // out = 1.0 + 1.0 = 2.0
}

TEST_F(ExecutorTest, SelfAdd) {
  // v0 <- a
  // v1 = v0 + v0
  // v2 = v1 + v1
  // ... ...
  // v10 = v9 + v9
  //
  // b <- v10
  // All nodes are executed by one thread.
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  auto v = test::graph::Recv(g.get(), "a", "float", ALICE, 1, BOB);
  const int N = 10;
  for (int i = 1; i <= N; ++i) {
    v = test::graph::Add(g.get(), v, v);
  }
  // out <- v10
  test::graph::Send(g.get(), v, "b", BOB, 1, ALICE);
  Create(std::move(g));
  Rendezvous::Args args;
  // a = 1.0
  TF_ASSERT_OK(
      rendez_->Send(Key(ALICE, kIncarnation, BOB, "a"), args, V(1.0), false));
  TF_ASSERT_OK(Run(rendez_));
  Tensor out = V(-1);
  bool is_dead = false;
  TF_ASSERT_OK(
      rendez_->Recv(Key(BOB, kIncarnation, ALICE, "b"), args, &out, &is_dead));
  EXPECT_EQ(1024.0, V(out));  // b=v10=2*v9=4*v8=...=1024*a=1024.0
}

// Builds a graph which adds N copies of one variable "in". I.e.,
//     a + a + a + ... + a
// The returned graph is parenthesized ramdonly. I.e.,
//     a + ((a + a) + a)
//     (a + a) + (a + a)
//     ((a + a) + a) + a
// are all possibly generated.
void BuildTree(int N, Graph* g) {
  CHECK_GT(N, 1);
  // A single input node "in".
  auto in = test::graph::Recv(g, "a", "float", ALICE, 1, BOB);
  std::vector<Node*> nodes;
  int i = 0;
  // Duplicate "in" N times. Each copies is named as l0, l1, l2, ....
  for (; i < N; ++i) {
    nodes.push_back(test::graph::Identity(g, in, 0));
  }
  random::PhiloxRandom philox(testing::RandomSeed(), 17);
  random::SimplePhilox rnd(&philox);
  while (nodes.size() > 1) {
    // Randomly pick two from nodes and add them. The resulting node
    // is named lik n10, n11, .... and is put back into "nodes".
    int x = rnd.Uniform(nodes.size());
    auto in0 = nodes[x];
    nodes[x] = nodes.back();
    nodes.resize(nodes.size() - 1);
    x = rnd.Uniform(nodes.size());
    auto in1 = nodes[x];
    // node = in0 + in1.
    nodes[x] = test::graph::Add(g, in0, in1);
  }
  // The final output node "out".
  test::graph::Send(g, nodes.back(), "b", BOB, 1, ALICE);
}

TEST_F(ExecutorTest, RandomTree) {
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  BuildTree(4096, g.get());
  Create(std::move(g));
  Rendezvous::Args args;
  TF_ASSERT_OK(
      rendez_->Send(Key(ALICE, kIncarnation, BOB, "a"), args, V(1.0), false));
  TF_ASSERT_OK(Run(rendez_));
  Tensor out = V(-1);
  bool is_dead = false;
  TF_ASSERT_OK(
      rendez_->Recv(Key(BOB, kIncarnation, ALICE, "b"), args, &out, &is_dead));
  EXPECT_EQ(4096.0, V(out));
}

void BuildConcurrentAddAssign(Graph* g) {
  auto one = test::graph::Constant(g, V(1.0));
  // A variable holds one float.
  auto var = test::graph::Var(g, DT_FLOAT, TensorShape({}));
  // Initialize the variable with 1.0.
  auto init = test::graph::Assign(g, var, one);
  // Output
  auto out = test::graph::Send(g, var, "out", ALICE, kIncarnation, BOB);
  // Have many concurrent computation. Each does v = v + 1.
  for (int i = 0; i < 1024; ++i) {
    auto add = test::graph::Add(g, var, one);
    g->AddControlEdge(init, add);  // Ensures run after init.
    auto assign = test::graph::Assign(g, var, add);
    g->AddControlEdge(assign, out);
  }
}

#ifndef THREAD_SANITIZER
TEST_F(ExecutorTest, ConcurrentAddAssign) {
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  BuildConcurrentAddAssign(g.get());
  Create(std::move(g));
  for (int iters = 0; iters < 16; ++iters) {
    Rendezvous* rendez = NewLocalRendezvous();
    TF_ASSERT_OK(Run(rendez));
    Rendezvous::Args args;
    Tensor out;
    bool is_dead;
    TF_ASSERT_OK(rendez->Recv(Key(ALICE, kIncarnation, BOB, "out"), args, &out,
                              &is_dead));
    VLOG(1) << "Get " << V(out);
    EXPECT_LE(V(out), 1025.0);
    rendez->Unref();
  }
}
#endif

TEST_F(ExecutorTest, SimpleSwitchLive) {
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  auto in0 = test::graph::Recv(g.get(), "a", "float", ALICE, 1, BOB);
  auto in1 = test::graph::Constant(g.get(), VB(false));
  auto tmp = test::graph::Switch(g.get(), in0, in1);
  test::graph::Send(g.get(), tmp, "c", BOB, 1, ALICE);
  Create(std::move(g));
  Rendezvous::Args args;
  TF_ASSERT_OK(rendez_->Send(Key(ALICE, kIncarnation, BOB, "a"), args, V(1.0),
                             false));  // in0 = 1.0
  TF_ASSERT_OK(Run(rendez_));
  Tensor out = V(-1);
  bool is_dead = false;
  TF_ASSERT_OK(
      rendez_->Recv(Key(BOB, kIncarnation, ALICE, "c"), args, &out, &is_dead));
  EXPECT_EQ(1.0, V(out));  // out = 1.0
  EXPECT_FALSE(is_dead);
}

TEST_F(ExecutorTest, SimpleSwitchDead) {
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  auto in0 = test::graph::Recv(g.get(), "a", "float", ALICE, 1, BOB);
  auto in1 = test::graph::Constant(g.get(), VB(true));
  auto tmp = test::graph::Switch(g.get(), in0, in1);
  test::graph::Send(g.get(), tmp, "c", BOB, 1, ALICE);
  Create(std::move(g));
  Rendezvous::Args args;
  TF_ASSERT_OK(rendez_->Send(Key(ALICE, kIncarnation, BOB, "a"), args, V(1.0),
                             false));  // in0 = 1.0
  TF_ASSERT_OK(Run(rendez_));
  Tensor out = V(-1);
  bool is_dead = false;
  TF_ASSERT_OK(
      rendez_->Recv(Key(BOB, kIncarnation, ALICE, "c"), args, &out, &is_dead));
  EXPECT_TRUE(is_dead);
}

TEST_F(ExecutorTest, Abort) {
  // e = a + b + c + d
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  auto in0 = test::graph::Recv(g.get(), "a", "float", ALICE, 1, BOB);
  auto in1 = test::graph::Recv(g.get(), "b", "float", ALICE, 1, BOB);
  auto in2 = test::graph::Recv(g.get(), "c", "float", ALICE, 1, BOB);
  auto in3 = test::graph::Recv(g.get(), "d", "float", ALICE, 1, BOB);
  auto add0 = test::graph::Add(g.get(), in0, in1);
  auto add1 = test::graph::Add(g.get(), in2, in3);
  auto add2 = test::graph::Add(g.get(), add0, add1);
  test::graph::Send(g.get(), add2, "e", BOB, 1, ALICE);
  Create(std::move(g));

  // Needs 4 inputs (recv). One of them is aborted.
  rendez_->Ref();
  SchedClosure([this]() {
    Env::Default()->SleepForMicroseconds(100 * 1000);
    Status s = rendez_->Send(Key(ALICE, kIncarnation, BOB, "a"),
                             Rendezvous::Args(), V(1.0), false);
    rendez_->Unref();
  });
  rendez_->Ref();
  SchedClosure([this]() {
    Env::Default()->SleepForMicroseconds(100 * 1000);
    Status s = rendez_->Send(Key(ALICE, kIncarnation, BOB, "b"),
                             Rendezvous::Args(), V(1.0), false);
    rendez_->Unref();
  });
  rendez_->Ref();
  SchedClosure([this]() {
    Env::Default()->SleepForMicroseconds(100 * 1000);
    Status s = rendez_->Send(Key(ALICE, kIncarnation, BOB, "c"),
                             Rendezvous::Args(), V(1.0), false);
    rendez_->Unref();
  });
  rendez_->Ref();
  SchedClosure([this]() {
    Env::Default()->SleepForMicroseconds(100 * 1000);
    rendez_->StartAbort(errors::Aborted(""));
    rendez_->Unref();
  });
  EXPECT_TRUE(errors::IsAborted(Run(rendez_)));
  Tensor out = V(-1);
  bool is_dead = false;
  EXPECT_TRUE(errors::IsAborted(rendez_->Recv(
      Key(BOB, kIncarnation, ALICE, "c"), Rendezvous::Args(), &out, &is_dead)));
  // At this point there can still be pending (albeit Aborted) Send
  // closures holding Refs on rendez_.  We need to wait for them, or
  // else there can be a memory leak at termination.
  while (!rendez_->RefCountIsOne()) {
  }
}

TEST_F(ExecutorTest, RecvInvalidDtype) {
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  // An input vector of type float of size 1.
  auto one = test::graph::Recv(g.get(), "one", "float", ALICE, 1, BOB);
  // A floating point variable vector of size 1.
  auto var = test::graph::Var(g.get(), DT_FLOAT, TensorShape({1}));
  // Initialize the variable with input.
  auto init = test::graph::Assign(g.get(), var, one);
  // Output
  auto* two = test::graph::Send(g.get(), var, "two", BOB, 1, ALICE);
  g->AddControlEdge(init, two);  // Ensures run after init.
  Create(std::move(g));
  Rendezvous* rendez = NewLocalRendezvous();
  // Send a double instead of float.
  TF_ASSERT_OK(rendez->Send(Key(ALICE, 1, BOB, "one"), Rendezvous::Args(),
                            VD(1.0), false));
  // Fails due to invalid dtype.
  EXPECT_TRUE(errors::IsInternal(Run(rendez)));
  Tensor output;
  bool is_dead;
  EXPECT_TRUE(errors::IsInternal(rendez->Recv(
      Key(BOB, 1, ALICE, "two"), Rendezvous::Args(), &output, &is_dead)));
  rendez->Unref();
}

TEST_F(ExecutorTest, RecvInvalidRefDtype) {
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  // A var that always produces as invalid dtype.
  auto var = test::graph::InvalidRefType(g.get(), DT_FLOAT, DT_DOUBLE);
  test::graph::Send(g.get(), var, "out", BOB, 1, ALICE);
  Create(std::move(g));
  Rendezvous* rendez = NewLocalRendezvous();
  EXPECT_TRUE(errors::IsInternal(Run(rendez)));
  Tensor output;
  bool is_dead;
  EXPECT_TRUE(errors::IsInternal(rendez->Recv(
      Key(BOB, 1, ALICE, "out"), Rendezvous::Args(), &output, &is_dead)));
  rendez->Unref();
}

TEST_F(ExecutorTest, NoInputTensors) {
  // Create a graph where none of the nodes have input tensors.
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  test::graph::Constant(g.get(), V(1.0));
  Create(std::move(g));
  TF_ASSERT_OK(Run(rendez_));
}

// Create a graph that is 'depth' deep. At each level, fan-in and fan-out a
// maximum of 'width' nodes. All nodes are no-ops and all dependencies are
// control dependencies.
static void BM_executor(::testing::benchmark::State& state) {
  const int width = state.range(0);
  const int depth = state.range(1);

  Graph* g = new Graph(OpRegistry::Global());
  random::PhiloxRandom philox(1729, 17);
  random::SimplePhilox rand(&philox);
  uint64 cur = 0;
  uint32 r = 1 + rand.Rand32() % width;
  std::vector<Node*> ready_nodes;
  for (int i = 0; i < r; ++i) {
    ready_nodes.push_back(test::graph::NoOp(g, {}));
    ++cur;
  }
  std::random_device random_device;
  std::mt19937 rng(random_device());
  for (int i = 0; i < depth; ++i) {
    std::shuffle(ready_nodes.begin(), ready_nodes.end(), rng);
    r = 1 + rand.Rand32() % (ready_nodes.size());
    std::vector<Node*> control_inputs;
    for (int j = 0; j < r; ++j) {
      control_inputs.push_back(ready_nodes.back());
      ready_nodes.pop_back();
    }
    Node* n = test::graph::NoOp(g, control_inputs);
    ++cur;
    r = 1 + rand.Rand32() % width;
    for (int j = 0; j < r; ++j) {
      ready_nodes.push_back(test::graph::NoOp(g, {n}));
      ++cur;
    }
  }

  FixupSourceAndSinkEdges(g);
  test::Benchmark("cpu", g, /*old_benchmark_api=*/false).Run(state);

  state.SetLabel(strings::StrCat("Nodes = ", cur));
  state.SetItemsProcessed(cur * static_cast<int64>(state.iterations()));
}

// Tall skinny graphs
BENCHMARK(BM_executor)->UseRealTime()->ArgPair(16, 1024);
BENCHMARK(BM_executor)->UseRealTime()->ArgPair(32, 8192);

// Short fat graphs
BENCHMARK(BM_executor)->UseRealTime()->ArgPair(1024, 16);
BENCHMARK(BM_executor)->UseRealTime()->ArgPair(8192, 32);

// Tall fat graph
BENCHMARK(BM_executor)->UseRealTime()->ArgPair(1024, 1024);

static void BM_const_identity(::testing::benchmark::State& state) {
  const int width = state.range(0);
  const int outputs_per_const = state.range(1);

  Graph* g = new Graph(OpRegistry::Global());
  for (int i = 0; i < width; ++i) {
    Tensor i_t(i);
    Node* const_node = test::graph::Constant(g, i_t);
    for (int j = 0; j < outputs_per_const; ++j) {
      test::graph::Identity(g, const_node);
    }
  }
  FixupSourceAndSinkEdges(g);
  test::Benchmark("cpu", g, /*old_benchmark_api=*/false).Run(state);
  state.SetLabel(strings::StrCat("Nodes = ", (1 + outputs_per_const) * width));
  state.SetItemsProcessed((1 + outputs_per_const) * width *
                          static_cast<int64>(state.iterations()));
}

// Graph with actual op execution.
BENCHMARK(BM_const_identity)
    ->UseRealTime()
    ->ArgPair(1, 1)
    ->ArgPair(1, 100)
    ->ArgPair(100, 1)
    ->ArgPair(100, 100);

static void BM_FeedInputFetchOutput(::testing::benchmark::State& state) {
  Graph* g = new Graph(OpRegistry::Global());
  // z = x + y: x and y are provided as benchmark inputs.  z is the
  // output of the benchmark.  Conceptually, the caller is ALICE, the
  // benchmark is BOB.
  Node* x = test::graph::Recv(g, "x", "float", ALICE, 1, BOB);
  Node* y = test::graph::Recv(g, "y", "float", ALICE, 1, BOB);
  Node* sum = test::graph::Add(g, x, y);
  Node* z = test::graph::Send(g, sum, "z", BOB, 1, ALICE);

  string x_key = test::GetRendezvousKey(x);
  string y_key = test::GetRendezvousKey(y);
  string z_key = test::GetRendezvousKey(z);

  Tensor val(DT_FLOAT, TensorShape({}));
  val.scalar<float>()() = 3.14;
  FixupSourceAndSinkEdges(g);
  test::Benchmark("cpu", g, /*old_benchmark_api=*/false)
      .RunWithRendezvousArgs({{x_key, val}, {y_key, val}}, {z_key}, state);
  state.SetItemsProcessed(static_cast<int64>(state.iterations()));
}
BENCHMARK(BM_FeedInputFetchOutput);

// Defines a graph to perform the following computation:
//
//     i = 0
//     while (i < loop_iters)
//       i += 1;
//
// ...using the functional `WhileOp` (if `lower` is false) or the
// `Switch`/`Merge`-style of control flow (if `lower` is true).
static void BM_WhileLoopHelper(::testing::benchmark::State& state,
                               int loop_iters, int loop_vars, bool lower) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  // Add test functions for cond and body.
  FunctionDefLibrary f_lib_proto;

  // Define the loop body as a function: `x = x + 1`.
  const Tensor one_t = test::AsScalar<int32>(1);

  std::vector<string> args;
  args.reserve(loop_vars);
  args.push_back("x: int32");
  for (int i = 1; i < loop_vars; ++i) {
    args.push_back(strings::StrCat("x", i, ": int32"));
  }

  std::vector<string> body_rets;
  body_rets.reserve(loop_vars);
  body_rets.push_back("y: int32");
  for (int i = 1; i < loop_vars; ++i) {
    body_rets.push_back(strings::StrCat("y", i, ": int32"));
  }

  std::vector<FunctionDefHelper::Node> body_nodes;
  body_nodes.reserve(1 + loop_vars);
  body_nodes.push_back(
      {{"one"}, "Const", {}, {{"value", one_t}, {"dtype", DT_INT32}}});
  body_nodes.push_back({{"y"}, "Add", {"x", "one"}, {{"T", DT_INT32}}});
  for (int i = 1; i < loop_vars; ++i) {
    body_nodes.push_back({{strings::StrCat("y", i)},
                          "Identity",
                          {strings::StrCat("x", i)},
                          {{"T", DT_INT32}}});
  }

  *f_lib_proto.add_function() = FunctionDefHelper::Define(
      // Name
      "XPlusOne",
      // Args
      args,
      // Return values
      body_rets,
      // Attr def
      {},
      // Nodes
      body_nodes);

  // Define the loop condition as a function: `x < loop_iters`.
  const Tensor loop_iters_t = test::AsScalar<int32>(loop_iters);
  *f_lib_proto.add_function() = FunctionDefHelper::Define(
      // Name
      "LessThanOrEqualToN",
      // Args
      args,
      // Return values
      {"z: bool"},
      // Attr def
      {},
      // Nodes
      {
          {{"N"}, "Const", {}, {{"value", loop_iters_t}, {"dtype", DT_INT32}}},
          {{"z"}, "LessEqual", {"x", "N"}, {{"T", DT_INT32}}},
      });

  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto a = ops::Const(root.WithOpName("A"), 0, {});
  Node* while_node;
  std::vector<NodeBuilder::NodeOut> inputs;
  std::vector<DataType> input_types(loop_vars, DT_INT32);
  inputs.reserve(loop_vars);
  for (int i = 0; i < loop_vars; ++i) {
    inputs.push_back(NodeBuilder::NodeOut(a.node()));
  }
  AttrValue int32_attr;
  int32_attr.set_type(DT_INT32);
  AttrValue cond_func;
  cond_func.mutable_func()->set_name("LessThanOrEqualToN");
  AttrValue body_func;
  body_func.mutable_func()->set_name("XPlusOne");
  TF_ASSERT_OK(
      NodeBuilder("while", "While", &root.graph()->flib_def())
          .Input(inputs)
          .Attr("T", input_types)
          .Attr("cond", cond_func)
          .Attr("body", body_func)
          .Attr("parallel_iterations", 100)
          .Attr(LowerFunctionalOpsPass::kLowerUsingSwitchMergeAttr, true)
          .Finalize(root.graph(), &while_node));
  auto c = ops::Identity(
      root.WithOpName("C").WithControlDependencies(Output(while_node)),
      Output(while_node));
  TF_ASSERT_OK(root.DoShapeInference(while_node));
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  if (lower) {
    FunctionLibraryDefinition flib_def(graph->flib_def());
    GraphOptimizationPassOptions opt_options;
    SessionOptions session_options;
    session_options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_do_function_inlining(true);
    opt_options.session_options = &session_options;
    opt_options.graph = &graph;
    opt_options.flib_def = &flib_def;
    LowerFunctionalOpsPass pass;
    TF_ASSERT_OK(pass.Run(opt_options));
  }

  FixupSourceAndSinkEdges(graph.get());
  test::Benchmark("cpu", graph.release(), /*old_benchmark_api=*/false)
      .Run(state);
}

static void BM_LoweredWhileLoop(::testing::benchmark::State& state) {
  const int loop_iters = state.range(0);
  const int loop_vars = state.range(1);

  BM_WhileLoopHelper(state, loop_iters, loop_vars, /* lower= */ true);
}
BENCHMARK(BM_LoweredWhileLoop)
    ->ArgPair(0, 1)
    ->ArgPair(1, 1)
    ->ArgPair(10, 1)
    ->ArgPair(100, 1)
    ->ArgPair(1000, 1)
    ->ArgPair(0, 100)
    ->ArgPair(1, 100)
    ->ArgPair(10, 100)
    ->ArgPair(100, 100)
    ->ArgPair(1000, 100);

static void BM_FunctionalWhileLoop(::testing::benchmark::State& state) {
  const int loop_iters = state.range(0);
  const int loop_vars = state.range(1);

  BM_WhileLoopHelper(state, loop_iters, loop_vars, /* lower= */ false);
}
BENCHMARK(BM_FunctionalWhileLoop)
    ->ArgPair(0, 1)
    ->ArgPair(1, 1)
    ->ArgPair(10, 1)
    ->ArgPair(100, 1)
    ->ArgPair(1000, 1)
    ->ArgPair(0, 100)
    ->ArgPair(1, 100)
    ->ArgPair(10, 100)
    ->ArgPair(100, 100)
    ->ArgPair(1000, 100);
}  // namespace tensorflow
