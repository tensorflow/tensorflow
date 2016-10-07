/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <vector>
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
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
  args_.push_back(std::make_pair(-1, -1));
  Register();
}

Benchmark::Benchmark(const char* name, void (*fn)(int, int))
    : name_(name), num_args_(1), fn1_(fn) {
  Register();
}

Benchmark::Benchmark(const char* name, void (*fn)(int, int, int))
    : name_(name), num_args_(2), fn2_(fn) {
  Register();
}

Benchmark* Benchmark::Arg(int x) {
  CHECK_EQ(num_args_, 1);
  args_.push_back(std::make_pair(x, -1));
  return this;
}

Benchmark* Benchmark::ArgPair(int x, int y) {
  CHECK_EQ(num_args_, 2);
  args_.push_back(std::make_pair(x, y));
  return this;
}

namespace {

void AddRange(std::vector<int>* dst, int lo, int hi, int mult) {
  CHECK_GE(lo, 0);
  CHECK_GE(hi, lo);

  // Add "lo"
  dst->push_back(lo);

  // Now space out the benchmarks in multiples of "mult"
  for (int32 i = 1; i < kint32max / mult; i *= mult) {
    if (i >= hi) break;
    if (i > lo) {
      dst->push_back(i);
    }
  }
  // Add "hi" (if different from "lo")
  if (hi != lo) {
    dst->push_back(hi);
  }
}

}  // namespace

Benchmark* Benchmark::Range(int lo, int hi) {
  std::vector<int> args;
  AddRange(&args, lo, hi, 8);
  for (int arg : args) {
    Arg(arg);
  }
  return this;
}

Benchmark* Benchmark::RangePair(int lo1, int hi1, int lo2, int hi2) {
  std::vector<int> args1;
  std::vector<int> args2;
  AddRange(&args1, lo1, hi1, 8);
  AddRange(&args2, lo2, hi2, 8);
  for (int arg1 : args1) {
    for (int arg2 : args2) {
      ArgPair(arg1, arg2);
    }
  }
  return this;
}

void Benchmark::Run(const char* pattern) {
  if (!all_benchmarks) return;

  // Converts "all" into the wildcard '.*'.  Currently pattern isn't
  // specified by clients, but we keep this here to match the internal
  // Google implementation, should we ever enable user-specified
  // pattern specification.
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
      if (arg.first >= 0) {
        strings::StrAppend(&name, "/", arg.first);
        if (arg.second >= 0) {
          strings::StrAppend(&name, "/", arg.second);
        }
      }

      // TODO(vrv): Check against 'pattern' using a regex before
      // computing the width, if we start allowing clients to pass in
      // a custom pattern.
      width = std::max<int>(width, name.size());
    }
  }

  printf("%-*s %10s %10s\n", width, "Benchmark", "Time(ns)", "Iterations");
  printf("%s\n", string(width + 22, '-').c_str());
  for (auto b : *all_benchmarks) {
    name = b->name_;
    for (auto arg : b->args_) {
      name.resize(b->name_.size());
      if (arg.first >= 0) {
        strings::StrAppend(&name, "/", arg.first);
        if (arg.second >= 0) {
          strings::StrAppend(&name, "/", arg.second);
        }
      }

      // TODO(vrv): Match 'name' against 'pattern' using a regex
      // before continuing, if we start allowing clients to pass in a
      // custom pattern.

      int iters;
      double seconds;
      b->Run(arg.first, arg.second, &iters, &seconds);

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

void Benchmark::Run(int arg1, int arg2, int* run_count, double* run_seconds) {
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
    } else if (fn1_) {
      (*fn1_)(iters, arg1);
    } else {
      (*fn2_)(iters, arg1, arg2);
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
