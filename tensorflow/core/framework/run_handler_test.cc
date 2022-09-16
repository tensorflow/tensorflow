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
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/blocking_counter.h"
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

TEST(RunHandlerUtilTest, PrioritySchedulingTest) {
  int num_threads = 2;
  std::unique_ptr<RunHandlerPool> pool(
      new RunHandlerPool(num_threads, num_threads));

  RunOptions::Experimental::RunHandlerPoolOptions options =
      RunOptions::Experimental::RunHandlerPoolOptions();
  options.set_priority(2);
  auto handler1 = pool->Get(/*step_id=*/1, /*timeout_in_ms=*/0, options);
  options.set_priority(1);
  auto handler2 = pool->Get(/*step_id=*/2, /*timeout_in_ms=*/0, options);
  options.set_priority(3);
  auto handler3 = pool->Get(/*step_id=*/3, /*timeout_in_ms=*/0, options);

  // The active requests should be ordered by priorites.
  std::vector<int64_t> sorted_active_list =
      pool->GetActiveHandlerPrioritiesForTesting();
  EXPECT_EQ(sorted_active_list.size(), 3);
  EXPECT_EQ(sorted_active_list[0], 3);
  EXPECT_EQ(sorted_active_list[1], 2);
  EXPECT_EQ(sorted_active_list[2], 1);

  handler1.reset();
  options.set_priority(5);
  auto handler4 = pool->Get(/*step_id=*/4, /*timeout_in_ms=*/0, options);
  options.set_priority(4);
  auto handler5 = pool->Get(/*step_id=*/5, /*timeout_in_ms=*/0, options);
  sorted_active_list = pool->GetActiveHandlerPrioritiesForTesting();
  EXPECT_EQ(sorted_active_list.size(), 4);
  EXPECT_EQ(sorted_active_list[0], 5);
  EXPECT_EQ(sorted_active_list[1], 4);
  EXPECT_EQ(sorted_active_list[2], 3);
  EXPECT_EQ(sorted_active_list[3], 1);
}

TEST(RunHandlerThreadPool, EnqueueTask) {
  Eigen::MaxSizeVector<mutex> waiters_mu(2);
  waiters_mu.resize(2);
  Eigen::MaxSizeVector<internal::Waiter> waiters(2);
  waiters.resize(2);
  internal::RunHandlerThreadPool run_handler_thread_pool(
      /*num_blocking_threads=*/0, /*num_non_blocking_threads=*/0,
      Env::Default(), ThreadOptions(), "tf_run_handler_pool", &waiters_mu,
      &waiters);
  internal::ThreadWorkSource tws;

  int result = 0;
  std::function<void()> fn = [&result] { result = 1; };
  std::function<void()> fn2 = [&result] { result = 2; };
  run_handler_thread_pool.AddWorkToQueue(&tws, /*is_blocking=*/true, fn);
  EXPECT_EQ(tws.TaskQueueSize(/*is_blocking=*/true), 1);
  run_handler_thread_pool.AddWorkToQueue(&tws, /*is_blocking=*/true, fn2);
  EXPECT_EQ(tws.TaskQueueSize(/*is_blocking=*/true), 2);
  tws.PopBlockingTask().f->f();
  EXPECT_EQ(result, 1);
  tws.PopBlockingTask().f->f();
  EXPECT_EQ(result, 2);

  run_handler_thread_pool.AddWorkToQueue(&tws, /*is_blocking=*/false, fn);
  EXPECT_EQ(tws.TaskQueueSize(/*is_blocking=*/false), 1);
  run_handler_thread_pool.AddWorkToQueue(&tws, /*is_blocking=*/false, fn2);
  EXPECT_EQ(tws.TaskQueueSize(/*is_blocking=*/false), 2);
  tws.PopNonBlockingTask(0, true).f->f();
  EXPECT_EQ(result, 1);
  tws.PopNonBlockingTask(0, true).f->f();
  EXPECT_EQ(result, 2);
}

TEST(RunHandlerThreadPool, FindTask) {
  Eigen::MaxSizeVector<mutex> waiters_mu(2);
  waiters_mu.resize(2);
  Eigen::MaxSizeVector<internal::Waiter> waiters(2);
  waiters.resize(2);
  internal::RunHandlerThreadPool run_handler_thread_pool(
      /*num_blocking_threads=*/1, /*num_non_blocking_threads=*/0,
      Env::Default(), ThreadOptions(), "tf_run_handler_pool", &waiters_mu,
      &waiters);

  Eigen::MaxSizeVector<internal::ThreadWorkSource*> thread_work_sources(5);
  thread_work_sources.resize(5);
  for (int i = 0; i < 5; ++i) {
    thread_work_sources[i] = new internal::ThreadWorkSource();
  }

  {
    // The thread should search the task following round robin fashion.
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[2],
                                           /*is_blocking=*/true,
                                           [&result] { result = 2; });
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[2],
                                           /*is_blocking=*/true,
                                           [&result] { result = 2; });
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[3],
                                           /*is_blocking=*/true,
                                           [&result] { result = 3; });
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[3],
                                           /*is_blocking=*/true,
                                           [&result] { result = 3; });

    const auto find_blocking_task_from_all_handlers =
        [&](bool* task_from_blocking_queue, internal::Task* t) {
          internal::ThreadWorkSource* tws;
          *t = run_handler_thread_pool.FindTask(
              /*searching_range_start=*/0, /*searching_range_end=*/5,
              /*thread_id=*/0,
              /*sub_thread_pool_id=*/0, /*max_blocking_inflight=*/10,
              /*may_steal_blocking_work=*/true, thread_work_sources,
              task_from_blocking_queue, &tws);
        };
    bool task_from_blocking_queue;
    internal::Task t;
    find_blocking_task_from_all_handlers(&task_from_blocking_queue, &t);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 2);

    find_blocking_task_from_all_handlers(&task_from_blocking_queue, &t);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 3);

    find_blocking_task_from_all_handlers(&task_from_blocking_queue, &t);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 2);

    find_blocking_task_from_all_handlers(&task_from_blocking_queue, &t);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 3);
  }

  {
    // Task out of searching range cannot be found.
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[3],
                                           /*is_blocking=*/true,
                                           [&result] { result = 3; });

    const auto find_blocking_task_from_range =
        [&](bool* task_from_blocking_queue, internal::Task* t, int range_start,
            int range_end) {
          internal::ThreadWorkSource* tws;
          *t = run_handler_thread_pool.FindTask(
              range_start, range_end,
              /*thread_id=*/0,
              /*sub_thread_pool_id=*/0, /*max_blocking_inflight=*/10,
              /*may_steal_blocking_work=*/true, thread_work_sources,
              task_from_blocking_queue, &tws);
        };

    bool task_from_blocking_queue;
    internal::Task t;
    find_blocking_task_from_range(&task_from_blocking_queue, &t, 0, 3);
    EXPECT_EQ(t.f, nullptr);

    // Clean up the queue.
    find_blocking_task_from_range(&task_from_blocking_queue, &t, 0, 5);
  }

  {
    // The thread should search from start range if the current index is
    // smaller.
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[2],
                                           /*is_blocking=*/true,
                                           [&result] { result = 2; });
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[3],
                                           /*is_blocking=*/true,
                                           [&result] { result = 3; });

    const auto find_blocking_task_from_range =
        [&](bool* task_from_blocking_queue, internal::Task* t, int range_start,
            int range_end) {
          internal::ThreadWorkSource* tws;
          *t = run_handler_thread_pool.FindTask(
              range_start, range_end,
              /*thread_id=*/0,
              /*sub_thread_pool_id=*/0, /*max_blocking_inflight=*/10,
              /*may_steal_blocking_work=*/true, thread_work_sources,
              task_from_blocking_queue, &tws);
        };
    bool task_from_blocking_queue;
    internal::Task t;
    find_blocking_task_from_range(&task_from_blocking_queue, &t, 3, 5);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 3);

    find_blocking_task_from_range(&task_from_blocking_queue, &t, 0, 5);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 2);
  }

  {
    // The thread should search within the range even if the current index
    // is larger than searching_range_end;
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[2],
                                           /*is_blocking=*/true,
                                           [&result] { result = 2; });

    const auto find_blocking_task_from_range =
        [&](bool* task_from_blocking_queue, internal::Task* t, int range_start,
            int range_end) {
          internal::ThreadWorkSource* tws;
          *t = run_handler_thread_pool.FindTask(
              range_start, range_end,
              /*thread_id=*/0,
              /*sub_thread_pool_id=*/0, /*max_blocking_inflight=*/10,
              /*may_steal_blocking_work=*/true, thread_work_sources,
              task_from_blocking_queue, &tws);
        };
    bool task_from_blocking_queue;
    // Make the current index to be 3.
    internal::Task t;
    find_blocking_task_from_range(&task_from_blocking_queue, &t, 0, 5);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 2);

    // Search in a smaller range.
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[2],
                                           /*is_blocking=*/true,
                                           [&result] { result = 2; });
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[3],
                                           /*is_blocking=*/true,
                                           [&result] { result = 3; });
    find_blocking_task_from_range(&task_from_blocking_queue, &t, 0, 3);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 2);

    // Clean up the queue.
    find_blocking_task_from_range(&task_from_blocking_queue, &t, 0, 5);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 3);
  }

  {
    // We prefer blocking task for blocking threads.
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[2],
                                           /*is_blocking=*/false,
                                           [&result] { result = 2; });
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[2],
                                           /*is_blocking=*/true,
                                           [&result] { result = 2; });
    const auto blocking_thread_find_task_from_all_handler =
        [&](bool* task_from_blocking_queue, internal::Task* t) {
          internal::ThreadWorkSource* tws;
          *t = run_handler_thread_pool.FindTask(
              /*searching_range_start=*/0, /*searching_range_end=*/5,
              /*thread_id=*/0,
              /*sub_thread_pool_id=*/0, /*max_blocking_inflight=*/10,
              /*may_steal_blocking_work=*/true, thread_work_sources,
              task_from_blocking_queue, &tws);
        };
    bool task_from_blocking_queue;
    internal::Task t;
    blocking_thread_find_task_from_all_handler(&task_from_blocking_queue, &t);
    EXPECT_EQ(task_from_blocking_queue, true);
    t.f->f();
    EXPECT_EQ(result, 2);

    blocking_thread_find_task_from_all_handler(&task_from_blocking_queue, &t);
    EXPECT_EQ(task_from_blocking_queue, false);
    t.f->f();
    EXPECT_EQ(result, 2);
  }

  {
    // Nonblocking threads can only pick up non-blocking task.
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[2],
                                           /*is_blocking=*/false,
                                           [&result] { result = 2; });
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[2],
                                           /*is_blocking=*/true,
                                           [&result] { result = 2; });

    const auto find_task_from_all_handler = [&](bool* task_from_blocking_queue,
                                                internal::Task* t,
                                                bool is_blocking_thread) {
      internal::ThreadWorkSource* tws;
      *t = run_handler_thread_pool.FindTask(
          /*searching_range_start=*/0, /*searching_range_end=*/5,
          /*thread_id=*/0,
          /*sub_thread_pool_id=*/0, /*max_blocking_inflight=*/10,
          is_blocking_thread, thread_work_sources, task_from_blocking_queue,
          &tws);
    };
    bool task_from_blocking_queue;
    internal::Task t;
    find_task_from_all_handler(&task_from_blocking_queue, &t,
                               /*is_blocking_thread=*/false);
    EXPECT_EQ(task_from_blocking_queue, false);
    t.f->f();
    EXPECT_EQ(result, 2);

    find_task_from_all_handler(&task_from_blocking_queue, &t,
                               /*is_blocking_thread=*/false);
    EXPECT_EQ(t.f, nullptr);

    // Clean up the queue.
    find_task_from_all_handler(&task_from_blocking_queue, &t,
                               /*is_blocking_thread=*/true);
  }

  {
    // There is a limit for max_blocking_inflight requests.
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(thread_work_sources[2],
                                           /*is_blocking=*/true,
                                           [&result] { result = 2; });

    const auto find_task_from_all_handler = [&](bool* task_from_blocking_queue,
                                                internal::Task* t,
                                                bool is_blocking_thread) {
      internal::ThreadWorkSource* tws;
      *t = run_handler_thread_pool.FindTask(
          /*searching_range_start=*/0, /*searching_range_end=*/5,
          /*thread_id=*/0,
          /*sub_thread_pool_id=*/0, /*max_blocking_inflight=*/10,
          is_blocking_thread, thread_work_sources, task_from_blocking_queue,
          &tws);
    };

    bool task_from_blocking_queue;
    internal::Task t;
    find_task_from_all_handler(&task_from_blocking_queue, &t,
                               /*is_blocking_thread=*/false);
    EXPECT_EQ(task_from_blocking_queue, false);
    EXPECT_EQ(t.f, nullptr);

    // Clean up the queue.
    find_task_from_all_handler(&task_from_blocking_queue, &t,
                               /*is_blocking_thread=*/true);
  }

  for (int i = 0; i < 5; ++i) {
    delete thread_work_sources[i];
  }
}

TEST(RunHandlerThreadPool, RoundRobinExecution) {
  // Set up environment for 1 sub thread pool.
  setenv("TF_RUN_HANDLER_USE_SUB_THREAD_POOL", "true", true);
  setenv("TF_RUN_HANDLER_NUM_THREADS_IN_SUB_THREAD_POOL", "1", true);
  setenv("TF_RUN_HANDLER_SUB_THREAD_POOL_START_REQUEST_PERCENTAGE", "0", true);
  setenv("TF_RUN_HANDLER_SUB_THREAD_POOL_END_REQUEST_PERCENTAGE", "1", true);

  Eigen::MaxSizeVector<mutex> waiters_mu(1);
  waiters_mu.resize(1);
  Eigen::MaxSizeVector<internal::Waiter> waiters(1);
  waiters.resize(1);
  internal::RunHandlerThreadPool* run_handler_thread_pool =
      new internal::RunHandlerThreadPool(
          /*num_blocking_threads=*/1, /*num_non_blocking_threads=*/0,
          Env::Default(), ThreadOptions(), "tf_run_handler_pool", &waiters_mu,
          &waiters);
  Eigen::MaxSizeVector<internal::ThreadWorkSource*> thread_work_sources(3);
  thread_work_sources.resize(3);
  internal::ThreadWorkSource tws[3];
  for (int i = 0; i < 3; ++i) {
    tws[i].SetWaiter(1, &waiters[0], &waiters_mu[0]);
    thread_work_sources[i] = &tws[i];
  }

  int result = 0;
  mutex mu;
  bool ok_to_execute = false;
  bool ok_to_validate = false;
  condition_variable function_start;
  condition_variable function_end;
  std::vector<std::function<void()>> fns;
  for (int i = 0; i < 3; ++i) {
    fns.push_back([&result, &mu, &function_start, &function_end, &ok_to_execute,
                   &ok_to_validate, i] {
      mutex_lock l(mu);
      while (!ok_to_execute) {
        function_start.wait(l);
      }
      result = i;
      ok_to_execute = false;
      ok_to_validate = true;
      function_end.notify_one();
    });
    run_handler_thread_pool->AddWorkToQueue(&tws[i], /*is_blocking=*/true,
                                            fns[i]);
    run_handler_thread_pool->AddWorkToQueue(&tws[i], /*is_blocking=*/true,
                                            fns[i]);
  }
  run_handler_thread_pool->Start();
  run_handler_thread_pool->SetThreadWorkSources(
      /*tid=*/0, /*start_request_idx=*/0, /*version=*/1, thread_work_sources);

  // Validate the execution should be roundrobin.
  mutex_lock l(mu);
  for (int round = 0; round < 2; ++round) {
    for (int i = 0; i < 3; ++i) {
      ok_to_execute = true;
      function_start.notify_one();
      while (!ok_to_validate) {
        function_end.wait(l);
      }
      ok_to_validate = false;
      EXPECT_EQ(result, i);
    }
  }

  delete run_handler_thread_pool;
}

TEST(RunHandlerThreadPool, MultipleSubThreadPool) {
  // Set up environment for 2 sub thread pools.
  setenv("TF_RUN_HANDLER_USE_SUB_THREAD_POOL", "true", true);
  setenv("TF_RUN_HANDLER_NUM_THREADS_IN_SUB_THREAD_POOL", "2", true);
  setenv("TF_RUN_HANDLER_SUB_THREAD_POOL_START_REQUEST_PERCENTAGE", "0,0.5",
         true);
  setenv("TF_RUN_HANDLER_SUB_THREAD_POOL_END_REQUEST_PERCENTAGE", "0.5,1",
         true);

  Eigen::MaxSizeVector<mutex> waiters_mu(2);
  waiters_mu.resize(2);
  Eigen::MaxSizeVector<internal::Waiter> waiters(2);
  waiters.resize(2);
  internal::RunHandlerThreadPool* run_handler_thread_pool =
      new internal::RunHandlerThreadPool(
          /*num_blocking_threads=*/2, /*num_non_blocking_threads=*/0,
          Env::Default(), ThreadOptions(), "tf_run_handler_pool", &waiters_mu,
          &waiters);
  Eigen::MaxSizeVector<internal::ThreadWorkSource*> thread_work_sources(4);
  thread_work_sources.resize(4);
  internal::ThreadWorkSource tws[4];
  for (int i = 0; i < 4; ++i) {
    tws[i].SetWaiter(1, &waiters[i / 2], &waiters_mu[i / 2]);
    thread_work_sources[i] = &tws[i];
  }

  int result = 0;
  mutex mu;
  bool ok_to_execute = false;
  bool ok_to_validate = false;
  condition_variable function_start;
  condition_variable function_end;

  std::vector<std::function<void()>> fns;
  for (int i = 0; i < 4; ++i) {
    fns.push_back([&result, &mu, &function_start, &function_end, &ok_to_execute,
                   &ok_to_validate, i] {
      mutex_lock l(mu);
      while (!ok_to_execute) {
        function_start.wait(l);
      }
      result = i;
      ok_to_execute = false;
      ok_to_validate = true;
      function_end.notify_one();
    });
    run_handler_thread_pool->AddWorkToQueue(&tws[i], /*is_blocking=*/true,
                                            fns[i]);
    run_handler_thread_pool->AddWorkToQueue(&tws[i], /*is_blocking=*/true,
                                            fns[i]);
  }
  run_handler_thread_pool->StartOneThreadForTesting();
  run_handler_thread_pool->SetThreadWorkSources(
      /*tid=*/0, /*start_request_idx=*/0, /*version=*/1, thread_work_sources);
  run_handler_thread_pool->SetThreadWorkSources(
      /*tid=*/1, /*start_request_idx=*/0, /*version=*/1, thread_work_sources);

  // Pick task from the given sub thread pool requests in a round robin fashion.
  mutex_lock l(mu);
  for (int round = 0; round < 2; ++round) {
    for (int i = 0; i < 2; ++i) {
      ok_to_execute = true;
      function_start.notify_one();
      while (!ok_to_validate) {
        function_end.wait(l);
      }
      ok_to_validate = false;
      EXPECT_EQ(result, i);
    }
  }

  // Pick task from any task if there is no tasks from the requests in the sub
  // thread pool.
  for (int i = 0; i < 2; ++i) {
    for (int round = 0; round < 2; ++round) {
      ok_to_execute = true;
      function_start.notify_one();
      while (!ok_to_validate) {
        function_end.wait(l);
      }
      ok_to_validate = false;
      EXPECT_EQ(result, i + 2);
    }
  }

  delete run_handler_thread_pool;
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
  EXPECT_EQ(OkStatus(), session->Create(def_));
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
  EXPECT_EQ(OkStatus(), s);

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
  EXPECT_EQ(OkStatus(), session->Create(def_));

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
      EXPECT_EQ(OkStatus(), s);
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

TEST_F(RunHandlerTest, UseRunHandlerPoolEnableSubPoolWithPriority) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  EXPECT_EQ(OkStatus(), session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;

  // Prepares RunOptions and RunMetadata
  RunOptions run_options;
  run_options.mutable_experimental()->set_use_run_handler_pool(true);
  run_options.mutable_experimental()
      ->mutable_run_handler_pool_options()
      ->set_priority(1);

  Status s = session->Run(run_options, inputs, output_names, target_nodes,
                          &outputs, nullptr);
  EXPECT_EQ(OkStatus(), s);

  ASSERT_EQ(1, outputs.size());
  // The first output should be initialized and have the correct
  // output.
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));
}

TEST_F(RunHandlerTest, TestConcurrencyUseRunHandlerPoolWithPriority) {
  Initialize({1, 2, 3, 4});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  EXPECT_EQ(OkStatus(), session->Create(def_));

  // Fill in the input and ask for the output
  thread::ThreadPool* tp = new thread::ThreadPool(Env::Default(), "test", 4);

  // Run the graph 1000 times in 4 different threads concurrently.
  std::vector<string> output_names = {y_ + ":0"};
  auto fn = [&session, output_names]() {
    for (int i = 0; i < 1000; ++i) {
      RunOptions run_options;
      run_options.mutable_experimental()->set_use_run_handler_pool(true);
      run_options.mutable_experimental()
          ->mutable_run_handler_pool_options()
          ->set_priority(i % 4);
      std::vector<std::pair<string, Tensor>> inputs;
      std::vector<Tensor> outputs;
      // Run the graph
      Status s = session->Run(run_options, inputs, output_names, {}, &outputs,
                              nullptr);
      EXPECT_EQ(OkStatus(), s);
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
  const int32_t kMaxConcurrentHandlers = 128;  // Copied from run_handler.cc.
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
  std::atomic<int64_t> release_time;

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
