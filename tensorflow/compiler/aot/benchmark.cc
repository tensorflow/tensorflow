/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// The purpose of the benchmark library is to support building an aot binary
// with minimal dependencies, to demonstrate small binary sizes.
//
// KEEP THE DEPENDENCIES MINIMAL.

#include "tensorflow/compiler/aot/benchmark.h"

#include <sys/time.h>

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tfcompile {
namespace benchmark {

// Returns current wall time in micros.
//
// TODO(b/33546473): Refactor tensorflow::Env::NowMicros() so that we can re-use
// the implementation without pulling in all of the Env dependencies.
static double NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<uint64>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

void DumpStatsToStdout(const Stats& stats) {
  // Compute stats.
  std::vector<int64> sorted_us(stats.per_iter_us);
  std::sort(sorted_us.begin(), sorted_us.end());
  const size_t count_us = sorted_us.size();
  double sum_us = 0;
  size_t count_us_trimmed = 0;
  double sum_us_trimmed = 0;
  size_t count_us_best = 0;
  double sum_us_best = 0;
  static constexpr float trim_ratio = 0.25;
  static constexpr float best_ratio = 0.1;
  const size_t count_trimmed = count_us * trim_ratio;
  const size_t count_best = count_us * best_ratio;
  for (size_t i = 0; i < sorted_us.size(); ++i) {
    const int64 us = sorted_us[i];
    sum_us += us;
    if (i >= count_trimmed && i < count_us - count_trimmed) {
      sum_us_trimmed += us;
      ++count_us_trimmed;
    }
    if (i < count_best) {
      sum_us_best += us;
      ++count_us_best;
    }
  }
  // Prepare nicely-formatted data.
  const int kBufSize = 1000;
  char buf[kBufSize];
  snprintf(buf, kBufSize, "Mean with %2.0f%% trimmed:", trim_ratio * 100);
  const string label_trimmed(buf);
  snprintf(buf, kBufSize, "Mean of %2.0f%% best:", best_ratio * 100);
  const string label_best(buf);
  std::vector<std::pair<string, double>> groups = {
      {"Best:", sorted_us.front()},
      {"Worst:", sorted_us.back()},
      {"Median:", sorted_us[count_us / 2]},
      {"Mean:", sum_us / count_us},
      {label_trimmed, sum_us_trimmed / count_us_trimmed},
      {label_best, sum_us_best / count_us_best},
  };
  int max_label_size = 0;
  double max_us = 0;
  for (const auto& g : groups) {
    if (g.first.size() > max_label_size) {
      max_label_size = g.first.size();
    }
    if (g.second > max_us) {
      max_us = g.second;
    }
  }
  int max_digits = 1;
  while (max_us >= 10.0) {
    max_us /= 10.0;
    ++max_digits;
  }
  // Dump stats out.
  printf("Benchmark ran %zu iterations over %lld us\n", count_us,
         stats.total_us);
  for (const auto& g : groups) {
    printf("  %-*s %*.3f us\n", max_label_size, g.first.c_str(), max_digits + 4,
           g.second);
  }
}

void Benchmark(const Options& options, const BenchmarkFn& fn, Stats* stats) {
  // If neither max_seconds or max_iters is set, stop at kDefaultMicros.
  const int64 max_us = (options.max_micros <= 0 && options.max_iters <= 0)
                           ? Options::kDefaultMicros
                           : options.max_micros;
  printf("Running benchmark for %lld us\n", max_us);
  const int64 start_us = NowMicros();
  int64 iters = 0;
  while (true) {
    const int64 iter_start_us = NowMicros();
    fn();
    const int64 end_us = NowMicros();
    // Collect stats and decide whether to stop.
    stats->per_iter_us.push_back(end_us - iter_start_us);
    const int64 total_us = end_us - start_us;
    ++iters;
    if ((max_us > 0 && total_us >= max_us) ||
        (options.max_iters > 0 && iters >= options.max_iters)) {
      stats->total_us = total_us;
      break;
    }
  }
}

}  // namespace benchmark
}  // namespace tfcompile
}  // namespace tensorflow
