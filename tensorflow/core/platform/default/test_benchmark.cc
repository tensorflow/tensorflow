/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/platform/test_benchmark.h"

#include <cstdio>
#include <cstdlib>

#include <vector>
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/util/reporter.h"

namespace tensorflow {
namespace testing {

static std::vector<Benchmark*>* all_benchmarks = nullptr;
static std::string label;
static int64 bytes_processed;
static int64 items_processed;
static int64 accum_time = 0;
static int64 start_time = 0;
static Env* env;

Benchmark::Benchmark(const char* name, void (*fn)(int))
    : name_(name), num_args_(0), fn0_(fn) {
  args_.push_back(-1);
  Register();
}

Benchmark::Benchmark(const char* name, void (*fn)(int, int))
    : name_(name), num_args_(1), fn1_(fn) {
  Register();
}

Benchmark* Benchmark::Arg(int x) {
  CHECK_EQ(num_args_, 1);
  args_.push_back(x);
  return this;
}

Benchmark* Benchmark::Range(int lo, int hi) {
  Arg(lo);
  for (int32 i = 1; i < kint32max / 8 && i < hi; i *= 8) {
    Arg(i);
  }
  if (lo != hi) Arg(hi);
  return this;
}

void Benchmark::Run(const char* pattern) {
  if (!all_benchmarks) return;

  if (StringPiece(pattern) == "all") {
    pattern = ".*";
  }

  // Compute name width.
  int width = 10;
  string name;
  for (auto b : *all_benchmarks) {
    name = b->name_;
    for (auto arg : b->args_) {
      name.resize(b->name_.size());
      if (arg >= 0) {
        strings::StrAppend(&name, "/", arg);
      }
      if (RE2::PartialMatch(name, pattern)) {
        width = std::max<int>(width, name.size());
      }
    }
  }

  printf("%-*s %10s %10s\n", width, "Benchmark", "Time(ns)", "Iterations");
  printf("%s\n", string(width + 22, '-').c_str());
  for (auto b : *all_benchmarks) {
    name = b->name_;
    for (auto arg : b->args_) {
      name.resize(b->name_.size());
      if (arg >= 0) {
        strings::StrAppend(&name, "/", arg);
      }
      if (!RE2::PartialMatch(name, pattern)) {
        continue;
      }

      int iters;
      double seconds;
      b->Run(arg, &iters, &seconds);

      char buf[100];
      std::string full_label = label;
      if (bytes_processed > 0) {
        snprintf(buf, sizeof(buf), " %.1fMB/s",
                 (bytes_processed * 1e-6) / seconds);
        full_label += buf;
      }
      if (items_processed > 0) {
        snprintf(buf, sizeof(buf), " %.1fM items/s",
                 (items_processed * 1e-6) / seconds);
        full_label += buf;
      }
      printf("%-*s %10.0f %10d\t%s\n", width, name.c_str(),
             seconds * 1e9 / iters, iters, full_label.c_str());

      TestReporter reporter(name);
      Status s = reporter.Initialize();
      if (!s.ok()) {
        LOG(ERROR) << s.ToString();
        exit(EXIT_FAILURE);
      }
      s = reporter.Benchmark(iters, 0.0, seconds,
                             items_processed * 1e-6 / seconds);
      if (!s.ok()) {
        LOG(ERROR) << s.ToString();
        exit(EXIT_FAILURE);
      }
      s = reporter.Close();
      if (!s.ok()) {
        LOG(ERROR) << s.ToString();
        exit(EXIT_FAILURE);
      }
    }
  }
}

void Benchmark::Register() {
  if (!all_benchmarks) all_benchmarks = new std::vector<Benchmark*>;
  all_benchmarks->push_back(this);
}

void Benchmark::Run(int arg, int* run_count, double* run_seconds) {
  env = Env::Default();
  static const int64 kMinIters = 100;
  static const int64 kMaxIters = 1000000000;
  static const double kMinTime = 0.5;
  int64 iters = kMinIters;
  while (true) {
    accum_time = 0;
    start_time = env->NowMicros();
    bytes_processed = -1;
    items_processed = -1;
    label.clear();
    if (fn0_) {
      (*fn0_)(iters);
    } else {
      (*fn1_)(iters, arg);
    }
    StopTiming();
    const double seconds = accum_time * 1e-6;
    if (seconds >= kMinTime || iters >= kMaxIters) {
      *run_count = iters;
      *run_seconds = seconds;
      return;
    }

    // Update number of iterations.  Overshoot by 40% in an attempt
    // to succeed the next time.
    double multiplier = 1.4 * kMinTime / std::max(seconds, 1e-9);
    multiplier = std::min(10.0, multiplier);
    if (multiplier <= 1.0) multiplier *= 2.0;
    iters = std::max<int64>(multiplier * iters, iters + 1);
    iters = std::min(iters, kMaxIters);
  }
}

// TODO(vrv): Add support for running a subset of benchmarks by having
// RunBenchmarks take in a spec (and maybe other options such as
// benchmark_min_time, etc).
void RunBenchmarks() { Benchmark::Run("all"); }
void SetLabel(const std::string& l) { label = l; }
void BytesProcessed(int64 n) { bytes_processed = n; }
void ItemsProcessed(int64 n) { items_processed = n; }
void StartTiming() {
  if (start_time == 0) start_time = env->NowMicros();
}
void StopTiming() {
  if (start_time != 0) {
    accum_time += (env->NowMicros() - start_time);
    start_time = 0;
  }
}
void UseRealTime() {}

}  // namespace testing
}  // namespace tensorflow
