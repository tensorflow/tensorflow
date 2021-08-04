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
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace serving {
namespace {

using ::tensorflow::histogram::Histogram;

// Fixed duration to run latency benchmark.
static int latency_benchmark_duration_secs = 100;

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
    const int64_t average_injection_interval_micros) const {
  int num_injections_performed = 0;
  const int64_t start_time_micros = Env::Default()->NowMicros();
  while (num_injections_performed < num_injections) {
    // Inject.
    injector();
    ++num_injections_performed;

    // Wait until it's time for the next injection.
    const int64_t next_injection_time_micros =
        start_time_micros +
        (num_injections_performed * average_injection_interval_micros);
    int64_t now_micros = Env::Default()->NowMicros();
    while (now_micros < next_injection_time_micros) {
      const int64_t kSleepThresholdMicros = 1000;
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
  void RunBenchmark(::testing::benchmark::State& state);

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

void LatencyBenchmark::RunBenchmark(::testing::benchmark::State& state) {
  ResetState();

  // Arrange to inject tasks at the specified rate, for a total duration of
  // of kTimeDurationMicros.
  const int kTimeDurationMicros = latency_benchmark_duration_secs * 1000 * 1000;
  const int kNumTasks = kTimeDurationMicros / task_injection_interval_micros_;
  if (kNumTasks <= 10000) {
    LOG(WARNING) << "Not enough tasks (" << kNumTasks << ")"
                 << " to report meaningful 99.9% latency!"
                 << " duration: " << kTimeDurationMicros
                 << " interval: " << task_injection_interval_micros_;
  }

  const int64_t start_time_micros = Env::Default()->NowMicros();

  // Inject the tasks.
  UniformLoadInjector injector;
  injector.InjectLoad(
      [this] {
        auto task = std::unique_ptr<BenchmarkBatchTask>(new BenchmarkBatchTask);
        TF_CHECK_OK(scheduler_->Schedule(&task));
      },
      kNumTasks, task_injection_interval_micros_);

  // Be sure we were able to more-or-less match our target injection rate.
  const int64_t target_injection_time_micros =
      kNumTasks * task_injection_interval_micros_;
  const int64_t actual_injection_time_micros =
      Env::Default()->NowMicros() - start_time_micros;
  if (actual_injection_time_micros > 1.1 * target_injection_time_micros) {
    LOG(FATAL) << "Unable to inject tasks at the requested rate";
  }

  // Wait for the scheduler to process all injected tasks.
  scheduler_.reset();

  // Be sure the scheduler was able to process the tasks at close to the
  // injection rate. If not, our latency measurements will be dominated by queue
  // waiting time
  const int64_t actual_processing_time_micros =
      Env::Default()->NowMicros() - start_time_micros;
  if (actual_processing_time_micros > 1.01 * actual_injection_time_micros) {
    LOG(FATAL) << "Unable to keep up with task injection rate";
  }

  // Report benchmark measurements.
  {
    mutex_lock l(mu_);
    state.SetLabel(absl::StrCat(
        "lat_p99.9=", task_latency_millis_histogram_.Percentile(99.9),
        "ms,batchsz_p99=", batch_size_histogram_.Percentile(99)));
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

static void RunLatencyBenchmark(::testing::benchmark::State& state,
                                int64_t task_injection_interval_micros,
                                int64_t batch_threads,
                                int64_t batch_timeout_micros) {
  BasicBatchScheduler<BenchmarkBatchTask>::Options scheduler_options;
  const int kMaxBatchSize = 100;
  scheduler_options.max_batch_size = kMaxBatchSize;
  scheduler_options.batch_timeout_micros = batch_timeout_micros;
  scheduler_options.num_batch_threads = batch_threads;
  scheduler_options.max_enqueued_batches = INT_MAX;  // Unbounded queue.
  const int kBatchCpuCost = 10 * 1000 * 1000;
  LatencyBenchmark benchmark(scheduler_options, task_injection_interval_micros,
                             kBatchCpuCost);
  for (auto s : state) {
    benchmark.RunBenchmark(state);
  }
}

#define LATENCY_BM(type, timeout)                                             \
  static void LatencyBM_##type##Timeout(::testing::benchmark::State& state) { \
    RunLatencyBenchmark(state, state.range(0), state.range(1), (timeout));    \
  }                                                                           \
  /* Run benchmark for a pair of <inject_interval_micros, num_threads> */     \
  BENCHMARK(LatencyBM_##type##Timeout)                                        \
      ->UseRealTime()                                                         \
      ->ArgPair(20, 2)                                                        \
      ->ArgPair(20, 4)                                                        \
      ->ArgPair(20, 8)                                                        \
      ->ArgPair(20, 16)                                                       \
      ->ArgPair(50, 2)                                                        \
      ->ArgPair(50, 4)                                                        \
      ->ArgPair(50, 8)                                                        \
      ->ArgPair(50, 16)                                                       \
      ->ArgPair(1000, 2)                                                      \
      ->ArgPair(1000, 4)                                                      \
      ->ArgPair(1000, 8)                                                      \
      ->ArgPair(1000, 16)

LATENCY_BM(Zero, 0);
LATENCY_BM(Small, 2000 /* 2ms timeout */);
LATENCY_BM(Large, 5000 /* 5ms timeout */);

}  // namespace
}  // namespace serving
}  // namespace tensorflow

int main(int argc, char** argv) {
  const std::vector<tensorflow::Flag> flag_list = {tensorflow::Flag(
      "scheduler_latency_bm_fixed_duration_secs",
      &tensorflow::serving::latency_benchmark_duration_secs,
      "Fixed duration that the latency benchmark must be run.")};
  if (!tensorflow::Flags::Parse(&argc, argv, flag_list)) {
    std::cout << tensorflow::Flags::Usage(argv[0], flag_list);
    return -1;
  }

  tensorflow::port::InitMain(argv[0], &argc, &argv);
  std::setprecision(5);

#ifdef PLATFORM_GOOGLE
  // Latency benchmark is a long running fixed interval (by time) benchmark, and
  // should only be run once, as we measure and report latency over this fixed
  // interval. Running for more than once will take very long time to complete.
  const auto min_iters = absl::GetFlag(FLAGS_benchmark_min_iters);
  const auto max_iters = absl::GetFlag(FLAGS_benchmark_max_iters);
  absl::SetFlag(&FLAGS_benchmark_min_iters, 1);
  absl::SetFlag(&FLAGS_benchmark_max_iters, 1);
  absl::SetFlag(&FLAGS_benchmark_filter, ".*Latency.*");
  tensorflow::testing::RunBenchmarks();
  absl::SetFlag(&FLAGS_benchmark_min_iters, min_iters);
  absl::SetFlag(&FLAGS_benchmark_max_iters, max_iters);
  absl::SetFlag(&FLAGS_benchmark_filter, ".*Through.*");
  tensorflow::testing::RunBenchmarks();
#else
  tensorflow::testing::RunBenchmarks();
#endif

  return 0;
}
