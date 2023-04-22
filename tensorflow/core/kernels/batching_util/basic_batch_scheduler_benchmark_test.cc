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

// Benchmarks for performance (throughput and latency) of BasicBatchScheduler
// under various rates of task injection.

#include "tensorflow/core/kernels/batching_util/basic_batch_scheduler.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace serving {
namespace {

using ::tensorflow::histogram::Histogram;

// An abstract class for injecting load into a system at a specific rate.
class LoadInjector {
 public:
  virtual ~LoadInjector() = default;

  // Run 'injector' 'num_injection' times, with average inter-injection spacing
  // as 'average_injection_interval_micros' (in microseconds).
  virtual void InjectLoad(std::function<void()> injector, int num_injections,
                          int64_t average_injection_interval_micros) const = 0;
};

// A load injector that uses uniform inter-injection spacing, i.e. each pair of
// injections is separated in time by 'average_injection_interval_micros' (as
// best as possible).
class UniformLoadInjector : public LoadInjector {
 public:
  UniformLoadInjector() = default;
  ~UniformLoadInjector() override = default;

  void InjectLoad(std::function<void()> injector, int num_injections,
                  int64_t average_injection_interval_micros) const override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(UniformLoadInjector);
};

void UniformLoadInjector::InjectLoad(
    std::function<void()> injector, const int num_injections,
    const int64 average_injection_interval_micros) const {
  int num_injections_performed = 0;
  const int64 start_time_micros = Env::Default()->NowMicros();
  while (num_injections_performed < num_injections) {
    // Inject.
    injector();
    ++num_injections_performed;

    // Wait until it's time for the next injection.
    const int64 next_injection_time_micros =
        start_time_micros +
        (num_injections_performed * average_injection_interval_micros);
    int64_t now_micros = Env::Default()->NowMicros();
    while (now_micros < next_injection_time_micros) {
      const int64 kSleepThresholdMicros = 1000;
      if (next_injection_time_micros - now_micros >= kSleepThresholdMicros) {
        Env::Default()->SleepForMicroseconds(1 /* minimum time */);
      }
      now_micros = Env::Default()->NowMicros();
    }
  }
}

class BenchmarkBatchTask : public BatchTask {
 public:
  BenchmarkBatchTask();

  BenchmarkBatchTask(const BenchmarkBatchTask&) = delete;
  BenchmarkBatchTask& operator=(const BenchmarkBatchTask&) = delete;

  ~BenchmarkBatchTask() override = default;

  size_t size() const override { return 1; }

  uint64 start_time_micros() const { return start_time_micros_; }

 private:
  // The time at which the task was created, in microseconds.
  const uint64 start_time_micros_;
};

BenchmarkBatchTask::BenchmarkBatchTask()
    : start_time_micros_(Env::Default()->NowMicros()) {}

// The state and logic associated with a throughput benchmark, which injects a
// large number of tasks into a batch scheduler and measures the total time to
// process all the tasks.
class ThroughputBenchmark {
 public:
  explicit ThroughputBenchmark(
      const BasicBatchScheduler<BenchmarkBatchTask>::Options&
          scheduler_options);

  ThroughputBenchmark(const ThroughputBenchmark&) = delete;
  ThroughputBenchmark& operator=(const ThroughputBenchmark&) = delete;

  // Perform the benchmark run, based on the parameters supplied to the ctor.
  void RunBenchmark(::testing::benchmark::State& state);

 private:
  // Resets all mutable state, including the scheduler.
  void ResetState();

  // Processes a batch of tasks. (Invoked by 'scheduler_' on one of its batch
  // threads.)
  void ProcessBatch(std::unique_ptr<Batch<BenchmarkBatchTask>> batch);

  // Parameters for the BasicBatchScheduler being benchmarked.
  const BasicBatchScheduler<BenchmarkBatchTask>::Options scheduler_options_;

  // The BasicBatchScheduler being benchmarked.
  std::unique_ptr<BasicBatchScheduler<BenchmarkBatchTask>> scheduler_;
};

ThroughputBenchmark::ThroughputBenchmark(
    const BasicBatchScheduler<BenchmarkBatchTask>::Options& scheduler_options)
    : scheduler_options_(scheduler_options) {}

void ThroughputBenchmark::RunBenchmark(::testing::benchmark::State& state) {
  CHECK_GE(state.max_iterations, 1);

  ResetState();

  // Have each iteration issue a reasonably large number of tasks, to ensure our
  // measurements reflect steady-state behavior.
  const int kNumTasksPerIteration = 100 * 1000;
  testing::UseRealTime();

  // Schedule 'num_iterations_*kNumTasksPerIteration' tasks.
  for (auto s : state) {
    for (int j = 0; j < kNumTasksPerIteration; ++j) {
      auto task = std::unique_ptr<BenchmarkBatchTask>(new BenchmarkBatchTask);
      TF_CHECK_OK(scheduler_->Schedule(&task));
    }
  }

  // Wait for the scheduler to process all tasks.
  scheduler_.reset();
  state.SetItemsProcessed(state.iterations() * kNumTasksPerIteration);
}

void ThroughputBenchmark::ResetState() {
  auto process_batch_callback =
      [this](std::unique_ptr<Batch<BenchmarkBatchTask>> batch) {
        ProcessBatch(std::move(batch));
      };
  TF_CHECK_OK(BasicBatchScheduler<BenchmarkBatchTask>::Create(
      scheduler_options_, process_batch_callback, &scheduler_));
}

void ThroughputBenchmark::ProcessBatch(
    std::unique_ptr<Batch<BenchmarkBatchTask>> batch) {
  // No-op.
}

// The state and logic associated with a latency benchmark, which injects tasks
// into a batch scheduler at a controlled rate and measures the distribution of
// task completion latencies.
//
// Reports the measurements to std::cout (not LOG(INFO)), like the throughput
// measurements.
class LatencyBenchmark {
 public:
  LatencyBenchmark(
      const BasicBatchScheduler<BenchmarkBatchTask>::Options& scheduler_options,
      int64_t task_injection_interval_micros, int batch_cpu_cost);

  LatencyBenchmark(const LatencyBenchmark&) = delete;
  LatencyBenchmark& operator=(const LatencyBenchmark&) = delete;

  // Perform the benchmark run, based on the parameters supplied to the ctor.
  void RunBenchmark();

 private:
  // Resets all mutable state, including the scheduler and latency measurements.
  void ResetState() TF_LOCKS_EXCLUDED(mu_);

  // Processes a batch of tasks. (Invoked by 'scheduler_' on one of its batch
  // threads.)
  void ProcessBatch(std::unique_ptr<Batch<BenchmarkBatchTask>> batch);

  // Performs one batch's dummy CPU work.
  void PerformBatchCpuWork() const;

  // Parameters for the BasicBatchScheduler being benchmarked.
  const BasicBatchScheduler<BenchmarkBatchTask>::Options scheduler_options_;

  // The time interval between successively injected tasks, in microseconds.
  // A large interval corresponds to a slow rate of task injection, and vice-
  // versa.
  const int64 task_injection_interval_micros_;

  // The amount of work to do while processing one batch of tasks. (The cost is
  // independent of the number of tasks in the batch.)
  const int batch_cpu_cost_;

  // The BasicBatchScheduler being benchmarked.
  std::unique_ptr<BasicBatchScheduler<BenchmarkBatchTask>> scheduler_;

  mutable mutex mu_;

  // A histogram of the task latencies, i.e. queue time plus processing time, in
  // milliseconds.
  Histogram task_latency_millis_histogram_ TF_GUARDED_BY(mu_);

  // A histogram of the batch sizes.
  Histogram batch_size_histogram_ TF_GUARDED_BY(mu_);
};

LatencyBenchmark::LatencyBenchmark(
    const BasicBatchScheduler<BenchmarkBatchTask>::Options& scheduler_options,
    int64_t task_injection_interval_micros, int batch_cpu_cost)
    : scheduler_options_(scheduler_options),
      task_injection_interval_micros_(task_injection_interval_micros),
      batch_cpu_cost_(batch_cpu_cost) {}

void LatencyBenchmark::RunBenchmark() {
  ResetState();

  // Arrange to inject tasks at the specified rate, for a fixed total time
  // duration.
  const int kTimeDurationMicros = 100 * 1000 * 1000 /* 100 seconds */;
  const int kNumTasks = kTimeDurationMicros / task_injection_interval_micros_;
  CHECK_GE(kNumTasks, 100000)
      << "Not enough tasks to report meaningful 99.9% latency";

  const int64 start_time_micros = Env::Default()->NowMicros();

  // Inject the tasks.
  UniformLoadInjector injector;
  injector.InjectLoad(
      [this] {
        auto task = std::unique_ptr<BenchmarkBatchTask>(new BenchmarkBatchTask);
        TF_CHECK_OK(scheduler_->Schedule(&task));
      },
      kNumTasks, task_injection_interval_micros_);

  // Be sure we were able to more-or-less match our target injection rate.
  const int64 target_injection_time_micros =
      kNumTasks * task_injection_interval_micros_;
  const int64 actual_injection_time_micros =
      Env::Default()->NowMicros() - start_time_micros;
  if (actual_injection_time_micros > 1.1 * target_injection_time_micros) {
    LOG(FATAL) << "Unable to inject tasks at the requested rate";
  }

  // Wait for the scheduler to process all injected tasks.
  scheduler_.reset();

  // Be sure the scheduler was able to process the tasks at close to the
  // injection rate. If not, our latency measurements will be dominated by queue
  // waiting time
  const int64 actual_processing_time_micros =
      Env::Default()->NowMicros() - start_time_micros;
  if (actual_processing_time_micros > 1.01 * actual_injection_time_micros) {
    LOG(FATAL) << "Unable to keep up with task injection rate";
  }

  // Report benchmark measurements.
  {
    mutex_lock l(mu_);
    std::cout << "\t"
              << "99.9% latency: "
              << task_latency_millis_histogram_.Percentile(99.9) << "ms"
              << "\t"
              << "99% batch size: " << batch_size_histogram_.Percentile(99)
              << std::endl;
  }
}

void LatencyBenchmark::ResetState() {
  auto process_batch_callback =
      [this](std::unique_ptr<Batch<BenchmarkBatchTask>> batch) {
        ProcessBatch(std::move(batch));
      };
  TF_CHECK_OK(BasicBatchScheduler<BenchmarkBatchTask>::Create(
      scheduler_options_, process_batch_callback, &scheduler_));

  {
    mutex_lock l(mu_);
    task_latency_millis_histogram_.Clear();
    batch_size_histogram_.Clear();
  }
}

void LatencyBenchmark::ProcessBatch(
    std::unique_ptr<Batch<BenchmarkBatchTask>> batch) {
  PerformBatchCpuWork();
  const uint64 batch_completion_time = Env::Default()->NowMicros();

  {
    mutex_lock l(mu_);
    batch_size_histogram_.Add(batch->num_tasks());
  }

  for (int i = 0; i < batch->num_tasks(); ++i) {
    const BenchmarkBatchTask& task = batch->task(i);

    const uint64 task_latency_micros =
        batch_completion_time - task.start_time_micros();

    {
      mutex_lock l(mu_);
      task_latency_millis_histogram_.Add(task_latency_micros / 1000.0);
    }
  }
}

void LatencyBenchmark::PerformBatchCpuWork() const {
  int dummy = 1;
  for (int i = 0; i < batch_cpu_cost_; ++i) {
    dummy += dummy * 2;
  }
  CHECK_NE(dummy, 0);
}

static void RunThroughputBenchmark(::testing::benchmark::State& state,
                                   int64_t batch_timeout_micros,
                                   int num_batch_threads) {
  BasicBatchScheduler<BenchmarkBatchTask>::Options scheduler_options;
  const int kMaxBatchSize = 100;
  scheduler_options.max_batch_size = kMaxBatchSize;
  scheduler_options.batch_timeout_micros = batch_timeout_micros;
  scheduler_options.num_batch_threads = num_batch_threads;
  scheduler_options.max_enqueued_batches = INT_MAX;  // Unbounded queue.
  ThroughputBenchmark benchmark(scheduler_options);
  benchmark.RunBenchmark(state);
}

static void ThroughputBM_ZeroTimeout(::testing::benchmark::State& state) {
  RunThroughputBenchmark(state, 0 /* 0 ms timeout */, state.range(0));
}
BENCHMARK(ThroughputBM_ZeroTimeout)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

static void ThroughputBM_SmallTimeout(::testing::benchmark::State& state) {
  RunThroughputBenchmark(state, 1 * 1000 /* 1 ms timeout */, state.range(0));
}
BENCHMARK(ThroughputBM_SmallTimeout)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

static void ThroughputBM_LargeTimeout(::testing::benchmark::State& state) {
  RunThroughputBenchmark(state, 50 * 1000 /* 50 ms timeout */, state.range(0));
}
BENCHMARK(ThroughputBM_LargeTimeout)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

static void RunLatencyBenchmark(int64_t task_injection_interval_micros,
                                int64_t batch_timeout_micros) {
  BasicBatchScheduler<BenchmarkBatchTask>::Options scheduler_options;
  const int kMaxBatchSize = 100;
  scheduler_options.max_batch_size = kMaxBatchSize;
  scheduler_options.batch_timeout_micros = batch_timeout_micros;
  const int kNumBatchThreads = 2;
  scheduler_options.num_batch_threads = kNumBatchThreads;
  scheduler_options.max_enqueued_batches = INT_MAX;  // Unbounded queue.
  const int kBatchCpuCost = 10 * 1000 * 1000;
  LatencyBenchmark benchmark(scheduler_options, task_injection_interval_micros,
                             kBatchCpuCost);
  benchmark.RunBenchmark();
}

static void RunLatencyBenchmarks() {
  for (const int64 batch_timeout_micros : {0, 1 * 1000, 2 * 1000, 5 * 1000}) {
    for (const int64 task_injection_interval_micros : {1000, 50, 20}) {
      std::cout << "Latency benchmark w/ batch timeout "
                << batch_timeout_micros / 1000.0 << "ms"
                << "; "
                << "task injection rate "
                << 1000000.0 / task_injection_interval_micros << "/sec"
                << "\t...";
      RunLatencyBenchmark(task_injection_interval_micros, batch_timeout_micros);
    }
    std::cout << std::endl;
  }
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  std::setprecision(5);

  // Run latency benchmarks (outside of tensorflow benchmark framework).
  tensorflow::serving::RunLatencyBenchmarks();

  // Run throughput benchmarks (via tensorflow benchmark framework).
  tensorflow::testing::RunBenchmarks();

  return 0;
}
