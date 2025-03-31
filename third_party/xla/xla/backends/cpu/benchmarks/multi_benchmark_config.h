/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_BENCHMARKS_MULTI_BENCHMARK_CONFIG_H_
#define XLA_BACKENDS_CPU_BENCHMARKS_MULTI_BENCHMARK_CONFIG_H_

#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/backends/cpu/benchmarks/aot_benchmark_helper.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace xla::cpu {

// Forwards a config to multiple benchmarks.
class MultiBenchmarkConfig {
 public:
  MultiBenchmarkConfig(
      std::initializer_list<benchmark::internal::Benchmark*> ptrs)
      : benchmarks_(ptrs) {}

  MultiBenchmarkConfig* Arg(int64_t x) {
    for (auto b : benchmarks_) {
      b->Arg(x);
    }
    return this;
  }

  MultiBenchmarkConfig* Unit(benchmark::TimeUnit unit) {
    for (auto b : benchmarks_) {
      b->Unit(unit);
    }
    return this;
  }

  MultiBenchmarkConfig* Range(int64_t start, int64_t limit) {
    for (auto b : benchmarks_) {
      b->Range(start, limit);
    }
    return this;
  }

  MultiBenchmarkConfig* DenseRange(int64_t start, int64_t limit, int step = 1) {
    for (auto b : benchmarks_) {
      b->DenseRange(start, limit, step);
    }
    return this;
  }

  MultiBenchmarkConfig* Args(const std::vector<int64_t>& args) {
    for (auto b : benchmarks_) {
      b->Args(args);
    }
    return this;
  }

  MultiBenchmarkConfig* ArgPair(int64_t x, int64_t y) {
    std::vector<int64_t> args;
    args.push_back(x);
    args.push_back(y);
    return Args(args);
  }

  MultiBenchmarkConfig* Ranges(
      const std::vector<std::pair<int64_t, int64_t> >& ranges) {
    for (auto b : benchmarks_) {
      b->Ranges(ranges);
    }
    return this;
  }

  MultiBenchmarkConfig* ArgsProduct(
      const std::vector<std::vector<int64_t> >& arglists) {
    for (auto b : benchmarks_) {
      b->ArgsProduct(arglists);
    }
    return this;
  }

  MultiBenchmarkConfig* ArgName(const std::string& name) {
    for (auto b : benchmarks_) {
      b->ArgName(name);
    }
    return this;
  }

  MultiBenchmarkConfig* ArgNames(const std::vector<std::string>& names) {
    for (auto b : benchmarks_) {
      b->ArgNames(names);
    }
    return this;
  }

  MultiBenchmarkConfig* RangePair(int64_t lo1, int64_t hi1, int64_t lo2,
                                  int64_t hi2) {
    std::vector<std::pair<int64_t, int64_t> > ranges;
    ranges.push_back(std::make_pair(lo1, hi1));
    ranges.push_back(std::make_pair(lo2, hi2));
    return Ranges(ranges);
  }

  MultiBenchmarkConfig* Apply(
      void (*func)(benchmark::internal::Benchmark* benchmark)) {
    for (auto b : benchmarks_) {
      b->Apply(func);
    }
    return this;
  }

  MultiBenchmarkConfig* RangeMultiplier(int multiplier) {
    for (auto b : benchmarks_) {
      b->RangeMultiplier(multiplier);
    }
    return this;
  }

  MultiBenchmarkConfig* MinTime(double t) {
    for (auto b : benchmarks_) {
      b->MinTime(t);
    }
    return this;
  }

  MultiBenchmarkConfig* MinWarmUpTime(double t) {
    for (auto b : benchmarks_) {
      b->MinWarmUpTime(t);
    }
    return this;
  }

  MultiBenchmarkConfig* Iterations(benchmark::IterationCount n) {
    for (auto b : benchmarks_) {
      b->Iterations(n);
    }
    return this;
  }

  MultiBenchmarkConfig* Repetitions(int n) {
    for (auto b : benchmarks_) {
      b->Repetitions(n);
    }
    return this;
  }

  MultiBenchmarkConfig* ReportAggregatesOnly(bool value = true) {
    for (auto b : benchmarks_) {
      b->ReportAggregatesOnly(value);
    }
    return this;
  }

  MultiBenchmarkConfig* DisplayAggregatesOnly(bool value = true) {
    for (auto b : benchmarks_) {
      b->DisplayAggregatesOnly(value);
    }
    return this;
  }

  MultiBenchmarkConfig* MeasureProcessCPUTime() {
    for (auto b : benchmarks_) {
      b->MeasureProcessCPUTime();
    }
    return this;
  }

  MultiBenchmarkConfig* UseRealTime() {
    for (auto b : benchmarks_) {
      b->UseRealTime();
    }
    return this;
  }

  MultiBenchmarkConfig* UseManualTime() {
    for (auto b : benchmarks_) {
      b->UseManualTime();
    }
    return this;
  }

  MultiBenchmarkConfig* Complexity(
      benchmark::BigO complexity = benchmark::oAuto) {
    for (auto b : benchmarks_) {
      b->Complexity(complexity);
    }
    return this;
  }

  MultiBenchmarkConfig* Complexity(benchmark::BigOFunc* complexity) {
    for (auto b : benchmarks_) {
      b->Complexity(complexity);
    }
    return this;
  }

  MultiBenchmarkConfig* ComputeStatistics(
      const std::string& name, benchmark::StatisticsFunc* statistics,
      benchmark::StatisticUnit unit = benchmark::kTime) {
    for (auto b : benchmarks_) {
      b->ComputeStatistics(name, statistics, unit);
    }
    return this;
  }

  MultiBenchmarkConfig* Threads(int t) {
    for (auto b : benchmarks_) {
      b->Threads(t);
    }
    return this;
  }

  MultiBenchmarkConfig* ThreadRange(int min_threads, int max_threads) {
    for (auto b : benchmarks_) {
      b->ThreadRange(min_threads, max_threads);
    }
    return this;
  }

  MultiBenchmarkConfig* DenseThreadRange(int min_threads, int max_threads,
                                         int stride = 1) {
    for (auto b : benchmarks_) {
      b->DenseThreadRange(min_threads, max_threads, stride);
    }
    return this;
  }

  MultiBenchmarkConfig* ThreadPerCpu() {
    for (auto b : benchmarks_) {
      b->ThreadPerCpu();
    }
    return this;
  }

 private:
  const std::vector<benchmark::internal::Benchmark*> benchmarks_;
};

// Benchmarks 'fn' in JIT and AOT modes. The JIT benchmark
// keeps the given 'name'; AOT is suffixed with '_Aot'.
inline MultiBenchmarkConfig* RegisterJitAndAotBenchmarks(
    absl::string_view name,
    void(fn)(benchmark::State&, const HloBenchmarkOptions&)) {
  std::string jit_name(name);
  std::string aot_name = jit_name + "_Aot";
  auto jit_fn = [fn](benchmark::State& state) {
    HloBenchmarkOptions options;
    fn(state, options);
  };
  auto aot_fn = [fn](benchmark::State& state) {
    HloBenchmarkOptions options;
    options.aot_options = GetAotCompilationOptions();
    fn(state, options);
  };
  benchmark::internal::Benchmark* jit =
      benchmark::RegisterBenchmark(jit_name, jit_fn);
  benchmark::internal::Benchmark* aot =
      benchmark::RegisterBenchmark(aot_name, aot_fn);
  return new MultiBenchmarkConfig({jit, aot});
};

// Registers the given benchmark in both JIT and AOT modes.
// The benchmark's function signature must be as follows:
// `void BenchmarkFunc(benchmark::State&, const HloBenchmarkOptions&)`.
#define XLA_CPU_BENCHMARK(n) \
  static MultiBenchmarkConfig* n##_ptr = RegisterJitAndAotBenchmarks(#n, n)

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_BENCHMARKS_MULTI_BENCHMARK_CONFIG_H_
