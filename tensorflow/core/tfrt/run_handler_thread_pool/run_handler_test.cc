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

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#define EIGEN_USE_THREADS

#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler.h"

#define EIGEN_USE_THREADS

#include "absl/memory/memory.h"
#include "absl/synchronization/barrier.h"
#include "absl/synchronization/notification.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tfrt/host_context/task_function.h"  // from @tf_runtime

namespace tfrt {
namespace tf {
namespace {

TEST(RunHandlerUtilTest, TestBasicScheduling) {
  int num_threads = 2;
  int num_handlers = 10;

  RunHandlerPool::Options options;
  options.num_intra_op_threads = num_threads;
  options.num_inter_op_threads = num_threads;
  std::unique_ptr<RunHandlerPool> pool(new RunHandlerPool(options));

  // RunHandler should always be able to run num_threads inter closures
  absl::Barrier barrier(num_threads);

  tensorflow::BlockingCounter counter(2 * num_handlers * num_threads);

  tensorflow::thread::ThreadPool test_pool(tensorflow::Env::Default(), "test",
                                           num_handlers);
  for (int i = 0; i < num_handlers; ++i) {
    test_pool.Schedule([&counter, &barrier, &pool, i, num_threads]() {
      auto handler = pool->Get(i);
      tensorflow::BlockingCounter local_counter(2 * num_threads);
      for (int j = 0; j < num_threads; ++j) {
        handler->ScheduleInterOpClosure(
            TaskFunction([&local_counter, &counter, &barrier, i]() {
              if (i == 2) {
                barrier.Block();
              }
              counter.DecrementCount();
              local_counter.DecrementCount();
            }));
        handler->ScheduleIntraOpClosure(
            TaskFunction([&local_counter, &counter]() {
              counter.DecrementCount();
              local_counter.DecrementCount();
            }));
      }
      local_counter.Wait();
    });
  }
  counter.Wait();
}

TEST(RunHandlerUtilTest, PrioritySchedulingTest) {
  int num_threads = 2;
  RunHandlerPool::Options pool_options;
  pool_options.num_intra_op_threads = num_threads;
  pool_options.num_inter_op_threads = num_threads;
  pool_options.num_threads_in_sub_thread_pool = {2};
  std::unique_ptr<RunHandlerPool> pool(new RunHandlerPool(pool_options));

  RunHandlerOptions options = RunHandlerOptions();
  options.priority = 2;
  auto handler1 = pool->Get(/*step_id=*/1, /*timeout_in_ms=*/0, options);
  options.priority = 1;
  auto handler2 = pool->Get(/*step_id=*/2, /*timeout_in_ms=*/0, options);
  options.priority = 3;
  auto handler3 = pool->Get(/*step_id=*/3, /*timeout_in_ms=*/0, options);

  // The active requests should be ordered by priorites.
  std::vector<int64_t> sorted_active_list =
      pool->GetActiveHandlerPrioritiesForTesting();
  EXPECT_EQ(sorted_active_list.size(), 3);
  EXPECT_EQ(sorted_active_list[0], 3);
  EXPECT_EQ(sorted_active_list[1], 2);
  EXPECT_EQ(sorted_active_list[2], 1);

  handler1.reset();
  options.priority = 5;
  auto handler4 = pool->Get(/*step_id=*/4, /*timeout_in_ms=*/0, options);
  options.priority = 4;
  auto handler5 = pool->Get(/*step_id=*/5, /*timeout_in_ms=*/0, options);
  sorted_active_list = pool->GetActiveHandlerPrioritiesForTesting();
  EXPECT_EQ(sorted_active_list.size(), 4);
  EXPECT_EQ(sorted_active_list[0], 5);
  EXPECT_EQ(sorted_active_list[1], 4);
  EXPECT_EQ(sorted_active_list[2], 3);
  EXPECT_EQ(sorted_active_list[3], 1);
}

TEST(RunHandlerUtilTest, IntraOpThreadPool) {
  int num_threads = 2;
  RunHandlerPool::Options pool_options;
  pool_options.num_intra_op_threads = num_threads;
  pool_options.num_inter_op_threads = num_threads;
  pool_options.num_threads_in_sub_thread_pool = {2};
  std::unique_ptr<RunHandlerPool> pool(new RunHandlerPool(pool_options));

  RunHandlerOptions options = RunHandlerOptions();
  auto handler = pool->Get(/*step_id=*/1, /*timeout_in_ms=*/0, options);
  auto* intra_pool = handler->AsIntraThreadPoolInterface();

  absl::Notification notification;
  intra_pool->Schedule([&notification]() { notification.Notify(); });
  notification.WaitForNotification();
}

class RunHandlerThreadPoolTest
    : public testing::TestWithParam<std::tuple<bool, bool>> {
 protected:
  bool adaptive_waiting_time() { return std::get<0>(GetParam()); }

  bool wait_if_no_active_request() { return std::get<1>(GetParam()); }
};

TEST_P(RunHandlerThreadPoolTest, EnqueueTask) {
  Eigen::MaxSizeVector<tensorflow::mutex> waiters_mu(2);
  waiters_mu.resize(2);
  Eigen::MaxSizeVector<internal::Waiter> waiters(2);
  waiters.resize(2);
  internal::RunHandlerThreadPool run_handler_thread_pool(
      internal::RunHandlerThreadPool::Options(
          /*num_blocking_threads=*/0, /*num_non_blocking_threads=*/0,
          /*wait_if_no_active_request=*/true,
          /*non_blocking_threads_sleep_time_micro_sec=*/250,
          /*blocking_threads_max_sleep_time_micro_sec=*/250,
          /*use_adaptive_waiting_time=*/true, /*enable_wake_up=*/true,
          /*max_concurrent_handler=*/128,
          /*num_threads_in_sub_thread_pool=*/{0},
          /*sub_thread_request_percentage=*/{1}),
      tensorflow::Env::Default(), tensorflow::ThreadOptions(),
      "tf_run_handler_pool", &waiters_mu, &waiters);
  internal::ThreadWorkSource tws;
  tws.SetWaiter(1, &waiters[0], &waiters_mu[0]);

  int result = 0;
  std::function<void()> fn = [&result] { result = 1; };
  std::function<void()> fn2 = [&result] { result = 2; };
  run_handler_thread_pool.AddWorkToQueue(&tws, /*is_blocking=*/true,
                                         TaskFunction(fn));
  EXPECT_EQ(tws.TaskQueueSize(/*is_blocking=*/true), 1);
  run_handler_thread_pool.AddWorkToQueue(&tws, /*is_blocking=*/true,
                                         TaskFunction(fn2));
  EXPECT_EQ(tws.TaskQueueSize(/*is_blocking=*/true), 2);
  tws.PopBlockingTask().f->f();
  EXPECT_EQ(result, 1);
  tws.PopBlockingTask().f->f();
  EXPECT_EQ(result, 2);

  run_handler_thread_pool.AddWorkToQueue(&tws, /*is_blocking=*/false,
                                         TaskFunction(fn));
  EXPECT_EQ(tws.TaskQueueSize(/*is_blocking=*/false), 1);
  run_handler_thread_pool.AddWorkToQueue(&tws, /*is_blocking=*/false,
                                         TaskFunction(fn2));
  EXPECT_EQ(tws.TaskQueueSize(/*is_blocking=*/false), 2);
  tws.PopNonBlockingTask(0, true).f->f();
  EXPECT_EQ(result, 1);
  tws.PopNonBlockingTask(0, true).f->f();
  EXPECT_EQ(result, 2);
}

TEST_P(RunHandlerThreadPoolTest, FindTask) {
  Eigen::MaxSizeVector<tensorflow::mutex> waiters_mu(2);
  waiters_mu.resize(2);
  Eigen::MaxSizeVector<internal::Waiter> waiters(2);
  waiters.resize(2);
  internal::RunHandlerThreadPool run_handler_thread_pool(
      internal::RunHandlerThreadPool::Options(
          /*num_blocking_threads=*/1, /*num_non_blocking_threads=*/0,
          /*wait_if_no_active_request=*/true,
          /*non_blocking_threads_sleep_time_micro_sec=*/250,
          /*blocking_threads_max_sleep_time_micro_sec=*/250,
          /*use_adaptive_waiting_time=*/true, /*enable_wake_up=*/true,
          /*max_concurrent_handler=*/128,
          /*num_threads_in_sub_thread_pool=*/{1},
          /*sub_thread_request_percentage=*/{1}),
      tensorflow::Env::Default(), tensorflow::ThreadOptions(),
      "tf_run_handler_pool", &waiters_mu, &waiters);

  Eigen::MaxSizeVector<internal::ThreadWorkSource*> thread_work_sources(5);
  thread_work_sources.resize(5);
  for (int i = 0; i < 5; ++i) {
    thread_work_sources[i] = new internal::ThreadWorkSource();
    thread_work_sources[i]->SetWaiter(1, &waiters[0], &waiters_mu[0]);
  }

  {
    // The thread should search the task following round robin fashion.
    int result = -1;
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[3],
        /*is_blocking=*/true, TaskFunction([&result] { result = 3; }));
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[3],
        /*is_blocking=*/true, TaskFunction([&result] { result = 3; }));

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
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[3],
        /*is_blocking=*/true, TaskFunction([&result] { result = 3; }));

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
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[3],
        /*is_blocking=*/true, TaskFunction([&result] { result = 3; }));

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
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));

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
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[3],
        /*is_blocking=*/true, TaskFunction([&result] { result = 3; }));
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
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/false, TaskFunction([&result] { result = 2; }));
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));
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
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/false, TaskFunction([&result] { result = 2; }));
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));

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
    run_handler_thread_pool.AddWorkToQueue(
        thread_work_sources[2],
        /*is_blocking=*/true, TaskFunction([&result] { result = 2; }));

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

TEST_P(RunHandlerThreadPoolTest, RoundRobinExecution) {
  Eigen::MaxSizeVector<tensorflow::mutex> waiters_mu(1);
  waiters_mu.resize(1);
  Eigen::MaxSizeVector<internal::Waiter> waiters(1);
  waiters.resize(1);
  internal::RunHandlerThreadPool* run_handler_thread_pool =
      new internal::RunHandlerThreadPool(
          internal::RunHandlerThreadPool::Options(
              /*num_blocking_threads=*/1, /*num_non_blocking_threads=*/0,
              /*wait_if_no_active_request=*/true,
              /*non_blocking_threads_sleep_time_micro_sec=*/250,
              /*blocking_threads_max_sleep_time_micro_sec=*/250,
              /*use_adaptive_waiting_time=*/true, /*enable_wake_up=*/true,
              /*max_concurrent_handler=*/128,
              /*num_threads_in_sub_thread_pool=*/{1},
              /*sub_thread_request_percentage=*/{1}),
          tensorflow::Env::Default(), tensorflow::ThreadOptions(),
          "tf_run_handler_pool", &waiters_mu, &waiters);
  Eigen::MaxSizeVector<internal::ThreadWorkSource*> thread_work_sources(3);
  thread_work_sources.resize(3);
  internal::ThreadWorkSource tws[3];
  for (int i = 0; i < 3; ++i) {
    tws[i].SetWaiter(1, &waiters[0], &waiters_mu[0]);
    thread_work_sources[i] = &tws[i];
  }

  int result = 0;
  tensorflow::mutex mu;
  bool ok_to_execute = false;
  bool ok_to_validate = false;
  tensorflow::condition_variable function_start;
  tensorflow::condition_variable function_end;
  std::vector<std::function<void()>> fns;
  for (int i = 0; i < 3; ++i) {
    fns.push_back([&result, &mu, &function_start, &function_end, &ok_to_execute,
                   &ok_to_validate, i] {
      tensorflow::mutex_lock l(mu);
      while (!ok_to_execute) {
        function_start.wait(l);
      }
      result = i;
      ok_to_execute = false;
      ok_to_validate = true;
      function_end.notify_one();
    });
    run_handler_thread_pool->AddWorkToQueue(&tws[i], /*is_blocking=*/true,
                                            TaskFunction(fns[i]));
    run_handler_thread_pool->AddWorkToQueue(&tws[i], /*is_blocking=*/true,
                                            TaskFunction(fns[i]));
  }
  run_handler_thread_pool->Start();
  run_handler_thread_pool->SetThreadWorkSources(
      /*tid=*/0, /*version=*/1, thread_work_sources);

  // Validate the execution should be roundrobin.
  tensorflow::mutex_lock l(mu);
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

TEST_P(RunHandlerThreadPoolTest, MultipleSubThreadPool) {
  Eigen::MaxSizeVector<tensorflow::mutex> waiters_mu(2);
  waiters_mu.resize(2);
  Eigen::MaxSizeVector<internal::Waiter> waiters(2);
  waiters.resize(2);
  internal::RunHandlerThreadPool* run_handler_thread_pool =
      new internal::RunHandlerThreadPool(
          internal::RunHandlerThreadPool::Options(
              /*num_blocking_threads=*/2, /*num_non_blocking_threads=*/0,
              /*wait_if_no_active_request=*/true,
              /*non_blocking_threads_sleep_time_micro_sec=*/250,
              /*blocking_threads_max_sleep_time_micro_sec=*/250,
              /*use_adaptive_waiting_time=*/true, /*enable_wake_up=*/true,
              /*max_concurrent_handler=*/128,
              /*num_threads_in_sub_thread_pool=*/{1, 1},
              /*sub_thread_request_percentage=*/{0.5, 1}),
          tensorflow::Env::Default(), tensorflow::ThreadOptions(),
          "tf_run_handler_pool", &waiters_mu, &waiters);
  Eigen::MaxSizeVector<internal::ThreadWorkSource*> thread_work_sources(4);
  thread_work_sources.resize(4);
  internal::ThreadWorkSource tws[4];
  for (int i = 0; i < 4; ++i) {
    tws[i].SetWaiter(1, &waiters[i / 2], &waiters_mu[i / 2]);
    thread_work_sources[i] = &tws[i];
  }

  int result = 0;
  tensorflow::mutex mu;
  bool ok_to_execute = false;
  bool ok_to_validate = false;
  tensorflow::condition_variable function_start;
  tensorflow::condition_variable function_end;

  std::vector<std::function<void()>> fns;
  for (int i = 0; i < 4; ++i) {
    fns.push_back([&result, &mu, &function_start, &function_end, &ok_to_execute,
                   &ok_to_validate, i] {
      tensorflow::mutex_lock l(mu);
      while (!ok_to_execute) {
        function_start.wait(l);
      }
      result = i;
      ok_to_execute = false;
      ok_to_validate = true;
      function_end.notify_one();
    });
    run_handler_thread_pool->AddWorkToQueue(&tws[i], /*is_blocking=*/true,
                                            TaskFunction(fns[i]));
    run_handler_thread_pool->AddWorkToQueue(&tws[i], /*is_blocking=*/true,
                                            TaskFunction(fns[i]));
  }
  run_handler_thread_pool->StartOneThreadForTesting();
  run_handler_thread_pool->SetThreadWorkSources(
      /*tid=*/0, /*version=*/1, thread_work_sources);
  run_handler_thread_pool->SetThreadWorkSources(
      /*tid=*/1, /*version=*/1, thread_work_sources);

  // Pick task from the given sub thread pool requests in a round robin fashion.
  tensorflow::mutex_lock l(mu);
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

INSTANTIATE_TEST_SUITE_P(Parameter, RunHandlerThreadPoolTest,
                         testing::Combine(::testing::Bool(),
                                          ::testing::Bool()));

}  // namespace
}  // namespace tf
}  // namespace tfrt
