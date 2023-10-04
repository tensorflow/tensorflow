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
  UniformLoadInjector(const UniformLoadInjector&) = delete;
  void operator=(const UniformLoadInjector&) = delete;
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

// The state associated with a throughput benchmark
class ThroughputBenchmark {
 public:
  explicit ThroughputBenchmark(
      const BasicBatchScheduler<BenchmarkBatchTask>::Options& scheduler_options)
      : scheduler_options_(scheduler_options) {
    auto process_batch_callback =
        [this](std::unique_ptr<Batch<BenchmarkBatchTask>> batch) {
          ProcessBatch(std::move(batch));
        };
    TF_CHECK_OK(BasicBatchScheduler<BenchmarkBatchTask>::Create(
        scheduler_options_, process_batch_callback, &scheduler_));
  }

  ThroughputBenchmark(const ThroughputBenchmark&) = delete;
  ThroughputBenchmark& operator=(const ThroughputBenchmark&) = delete;

  BasicBatchScheduler<BenchmarkBatchTask>* GetScheduler() const {
    return scheduler_.get();
  }

  // Reset scheduler. This has a side-effect of waiting for all work to be
  // completed prior to reset.
  void ResetScheduler() { return scheduler_.reset(); }

 private:
  // Processes a batch of tasks. (Invoked by 'scheduler_' on one of its batch
  // threads.)
  void ProcessBatch(std::unique_ptr<Batch<BenchmarkBatchTask>> batch) {
    // No-op
  }

  // Parameters for the BasicBatchScheduler being benchmarked.
  const BasicBatchScheduler<BenchmarkBatchTask>::Options scheduler_options_;

  // The BasicBatchScheduler being benchmarked.
  std::unique_ptr<BasicBatchScheduler<BenchmarkBatchTask>> scheduler_;
};

// The state associated with a latency benchmark, which injects tasks into a
// batch scheduler at a controlled rate and measures the distribution of task
// completion latencies.
class LatencyBenchmark {
 public:
  LatencyBenchmark(
      const BasicBatchScheduler<BenchmarkBatchTask>::Options& scheduler_options,
      int64_t task_injection_interval_micros, int batch_cpu_cost);

  LatencyBenchmark(const LatencyBenchmark&) = delete;
  LatencyBenchmark& operator=(const LatencyBenchmark&) = delete;

  // Inject tasks at specified rate for `latency_benchmark_duration_secs`.
  void InjectLoad();

  // Return latency and batch size stat.
  string ReportLatencyBatchSz();

  // Reset scheduler. This has a side-effect of waiting for all work to be
  // completed prior to reset.
  void ResetScheduler() { return scheduler_.reset(); }

 private:
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
  const int64_t task_injection_interval_micros_;

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
      batch_cpu_cost_(batch_cpu_cost) {
  auto process_batch_callback =
      [this](std::unique_ptr<Batch<BenchmarkBatchTask>> batch) {
        ProcessBatch(std::move(batch));
      };
  TF_CHECK_OK(BasicBatchScheduler<BenchmarkBatchTask>::Create(
      scheduler_options_, process_batch_callback, &scheduler_));
}

void LatencyBenchmark::InjectLoad() {
  // Arrange to inject tasks at the specified rate, for a total duration of
  // of kTimeDurationMicros.
  const int kTimeDurationMicros = latency_benchmark_duration_secs * 1000 * 1000;
  const int kNumTasks = kTimeDurationMicros / task_injection_interval_micros_;
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

  // Be sure the scheduler was able to process the tasks at close to the
  // injection rate. If not, our latency measurements will be dominated by queue
  // waiting time
  const int64_t actual_processing_time_micros =
      Env::Default()->NowMicros() - start_time_micros;
  if (actual_processing_time_micros > 1.01 * actual_injection_time_micros) {
    LOG(FATAL) << "Unable to keep up with task injection rate";
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
    const uint64 task_latency_micros =
        batch_completion_time - batch->task(i).start_time_micros();
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

string LatencyBenchmark::ReportLatencyBatchSz() {
  mutex_lock l(mu_);
  return absl::StrCat(
      "lat_p99.9=", task_latency_millis_histogram_.Percentile(99.9),
      "ms,batchsz_p99=", batch_size_histogram_.Percentile(99));
}

// Injects a large number of tasks into a batch scheduler and measures
// the total time to process all the tasks.
//
// Multi-threaded (thread > 1) version simulates N concurrent request streams.
void ThroughputBM(::testing::benchmark::State& state) {
  static std::unique_ptr<ThroughputBenchmark> bm;
  if (state.thread_index() == 0) {
    BasicBatchScheduler<BenchmarkBatchTask>::Options scheduler_options;
    const int kMaxBatchSize = 100;
    scheduler_options.max_batch_size = kMaxBatchSize;
    scheduler_options.batch_timeout_micros = state.range(0) * 1000;
    scheduler_options.num_batch_threads = state.range(1);
    scheduler_options.max_enqueued_batches = INT_MAX;  // Unbounded queue.
    bm.reset(new ThroughputBenchmark(scheduler_options));
  }

  // Have each iteration issue a reasonably large number of tasks, to ensure our
  // measurements reflect steady-state behavior.
  const int kNumTasksPerIteration = 100 * 1000;

  // Schedule 'num_iterations_*kNumTasksPerIteration' tasks.
  for (auto s : state) {
    for (int j = 0; j < kNumTasksPerIteration; ++j) {
      auto task = std::unique_ptr<BenchmarkBatchTask>(new BenchmarkBatchTask);
      TF_CHECK_OK(bm->GetScheduler()->Schedule(&task));
    }
  }

  if (state.thread_index() == 0) {
    state.ResumeTiming();
    // Wait for the scheduler to process all tasks.
    bm->ResetScheduler();
    state.PauseTiming();
    bm.reset();
  }
  state.SetItemsProcessed(state.iterations() * kNumTasksPerIteration);
}
BENCHMARK(ThroughputBM)
    ->UseRealTime()
    ->Threads(1)
    ->Threads(8)
    ->Threads(16)
    ->ArgNames({"timeout", "batch_threads"})
    ->ArgsProduct({{0, 2, 10}, {1, 4, 8, 16}});

// Latency benchmark is a long running fixed interval (by time) benchmark and is
// run once (see ->Iterations(1) below). We measure and report latency over this
// fixed interval.
//
// Multi-threaded (thread > 1) version simulates N concurrent request streams.
void LatencyBM(::testing::benchmark::State& state) {
  static std::unique_ptr<LatencyBenchmark> bm;
  if (state.thread_index() == 0) {
    BasicBatchScheduler<BenchmarkBatchTask>::Options scheduler_options;
    const int kMaxBatchSize = 100;
    scheduler_options.max_batch_size = kMaxBatchSize;
    scheduler_options.batch_timeout_micros = state.range(0);
    scheduler_options.num_batch_threads = state.range(1);
    scheduler_options.max_enqueued_batches = INT_MAX;  // Unbounded queue.
    const int kBatchCpuCost = 10 * 1000 * 1000;
    const int64 kQps = state.range(2);
    const int64 kInjectionIntervalMicros = 1000000 / (kQps / state.threads());
    const int64 kNumTasks = latency_benchmark_duration_secs * kQps;
    if (kNumTasks <= 10000) {
      LOG(WARNING) << "Not enough tasks (" << kNumTasks << ")"
                   << " to report meaningful 99.9% latency!"
                   << " duration: " << latency_benchmark_duration_secs
                   << " interval: " << kInjectionIntervalMicros;
    }
    bm.reset(new LatencyBenchmark(scheduler_options, kInjectionIntervalMicros,
                                  kBatchCpuCost));
  }

  for (auto s : state) {
    bm->InjectLoad();
  }

  if (state.thread_index() == 0) {
    state.ResumeTiming();
    // Wait for the scheduler to process all tasks.
    bm->ResetScheduler();
    state.PauseTiming();
    state.SetLabel(bm->ReportLatencyBatchSz());
    bm.reset();
  }
}
BENCHMARK(LatencyBM)
    ->UseRealTime()
    ->Iterations(1)
    ->Threads(1)
    ->Threads(8)
    ->Threads(16)
    ->ArgNames({"timeout", "batch_threads", "qps"})
    ->ArgsProduct({{0, 2, 10}, {1, 4, 8, 16}, {50000, 20000, 1000}});

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

  ::benchmark::Initialize(&argc, argv);
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
