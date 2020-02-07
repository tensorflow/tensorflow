/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/run_handler.h"

#include <memory>
#include <vector>

#define EIGEN_USE_THREADS
#include "absl/memory/memory.h"
#include "absl/synchronization/barrier.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

TEST(RunHandlerUtilTest, TestBasicScheduling) {
  int num_threads = 2;
  int num_handlers = 10;

  std::unique_ptr<RunHandlerPool> pool(
      new RunHandlerPool(num_threads, num_threads));

  // RunHandler should always be able to run num_threads inter closures
  absl::Barrier barrier(num_threads);

  BlockingCounter counter(2 * num_handlers * num_threads);

  thread::ThreadPool test_pool(Env::Default(), "test", num_handlers);
  for (int i = 0; i < num_handlers; ++i) {
    test_pool.Schedule([&counter, &barrier, &pool, i, num_threads]() {
      auto handler = pool->Get(i);
      BlockingCounter local_counter(2 * num_threads);
      auto intra_thread_pool = handler->AsIntraThreadPoolInterface();

      for (int j = 0; j < num_threads; ++j) {
        handler->ScheduleInterOpClosure(
            [&local_counter, &counter, &barrier, i]() {
              if (i == 2) {
                barrier.Block();
              }
              counter.DecrementCount();
              local_counter.DecrementCount();
            });
        intra_thread_pool->Schedule([&local_counter, &counter]() {
          counter.DecrementCount();
          local_counter.DecrementCount();
        });
      }
      local_counter.Wait();
    });
  }
  counter.Wait();
}

SessionOptions DefaultSessionOptions() {
  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  return options;
}

std::unique_ptr<Session> CreateSession() {
  return std::unique_ptr<Session>(NewSession(DefaultSessionOptions()));
}

class RunHandlerTest : public ::testing::Test {
 public:
  void Initialize(std::initializer_list<float> a_values) {
    Graph graph(OpRegistry::Global());

    Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
    test::FillValues<float>(&a_tensor, a_values);
    Node* a = test::graph::Constant(&graph, a_tensor);
    a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
    a_ = a->name();

    Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
    test::FillValues<float>(&x_tensor, {1, 1});
    Node* x = test::graph::Constant(&graph, x_tensor);
    x->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");
    x_ = x->name();

    // y = A * x
    Node* y = test::graph::Matmul(&graph, a, x, false, false);
    y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
    y_ = y->name();

    Node* y_neg = test::graph::Unary(&graph, "Neg", y);
    y_neg_ = y_neg->name();
    y_neg->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

    Node* z = test::graph::Unary(&graph, "Identity", y_neg);
    z_ = z->name();
    z->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

    graph.ToGraphDef(&def_);

    ASSERT_EQ(setenv("TF_RUN_HANDLER_NUM_SUB_THREAD_POOL", "2", true), 0);
    ASSERT_EQ(
        setenv("TF_RUN_HANDLER_NUM_THREADS_IN_SUB_THREAD_POOL", "8,8", true),
        0);
    ASSERT_EQ(setenv("TF_RUN_HANDLER_SUB_THREAD_POOL_START_REQUEST_PERCENTAGE",
                     "0,0.4", true),
              0);
    ASSERT_EQ(setenv("TF_RUN_HANDLER_SUB_THREAD_POOL_END_REQUEST_PERCENTAGE",
                     "0.4,1", true),
              0);
    ASSERT_EQ(setenv("TF_NUM_INTEROP_THREADS", "16", true), 0);
  }

  string a_;
  string x_;
  string y_;
  string y_neg_;
  string z_;
  GraphDef def_;
};

TEST_F(RunHandlerTest, UseRunHandlerPoolEnableSubPool) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  EXPECT_EQ(::tensorflow::Status::OK(), session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;

  // Prepares RunOptions and RunMetadata
  RunOptions run_options;
  run_options.mutable_experimental()->set_use_run_handler_pool(true);

  Status s = session->Run(run_options, inputs, output_names, target_nodes,
                          &outputs, nullptr);
  EXPECT_EQ(::tensorflow::Status::OK(), s);

  ASSERT_EQ(1, outputs.size());
  // The first output should be initialized and have the correct
  // output.
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));
}

TEST_F(RunHandlerTest, TestConcurrencyUseRunHandlerPool) {
  Initialize({1, 2, 3, 4});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  EXPECT_EQ(::tensorflow::Status::OK(), session->Create(def_));

  RunOptions run_options;
  run_options.mutable_experimental()->set_use_run_handler_pool(true);

  // Fill in the input and ask for the output
  thread::ThreadPool* tp = new thread::ThreadPool(Env::Default(), "test", 4);

  // Run the graph 1000 times in 4 different threads concurrently.
  std::vector<string> output_names = {y_ + ":0"};
  auto fn = [&session, output_names, run_options]() {
    for (int i = 0; i < 1000; ++i) {
      std::vector<std::pair<string, Tensor>> inputs;
      std::vector<Tensor> outputs;
      // Run the graph
      Status s = session->Run(run_options, inputs, output_names, {}, &outputs,
                              nullptr);
      EXPECT_EQ(::tensorflow::Status::OK(), s);
      ASSERT_EQ(1, outputs.size());
      auto mat = outputs[0].matrix<float>();
      EXPECT_FLOAT_EQ(3.0, mat(0, 0));
    }
  };

  for (int i = 0; i < 4; ++i) {
    tp->Schedule(fn);
  }

  // Wait for the functions to finish.
  delete tp;
}

TEST_F(RunHandlerTest, TestWaitTimeout) {
  std::unique_ptr<RunHandlerPool> pool(new RunHandlerPool(1, 1));

  // Get the single handler in the pool.
  std::vector<std::unique_ptr<RunHandler>> blocking_handles;
  const int32 kMaxConcurrentHandlers = 128;  // Copied from run_handler.cc.
  blocking_handles.reserve(kMaxConcurrentHandlers);
  for (int i = 0; i < kMaxConcurrentHandlers; ++i) {
    blocking_handles.push_back(pool->Get(i));
  }

  // A subsequent request with a non-zero timeout will fail by returning
  // nullptr.
  auto null_handle = pool->Get(128, 1);
  EXPECT_EQ(null_handle.get(), nullptr);

  // A subsequent request with no timeout will succeed once the blocking handle
  // is returned.
  auto tp = std::make_unique<thread::ThreadPool>(Env::Default(), "test", 4);
  std::atomic<int64> release_time;

  tp->Schedule([&blocking_handles, &release_time]() {
    Env::Default()->SleepForMicroseconds(5000);
    release_time = EnvTime::NowNanos();
    blocking_handles[0].reset();
  });

  auto next_handle = pool->Get(129, 0);
  EXPECT_GT(EnvTime::NowNanos(), release_time);
  EXPECT_NE(next_handle.get(), nullptr);
}

}  // namespace
}  // namespace tensorflow
