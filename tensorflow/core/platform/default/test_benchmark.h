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

// Simple benchmarking facility.
#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_TEST_BENCHMARK_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_TEST_BENCHMARK_H_

#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>  // for _ReadWriteBarrier
#endif

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"

#define BENCHMARK(n)                                            \
  static ::tensorflow::testing::Benchmark* TF_BENCHMARK_CONCAT( \
      __benchmark_, n, __LINE__) TF_ATTRIBUTE_UNUSED =          \
      (new ::tensorflow::testing::Benchmark(#n, (n)))
#define TF_BENCHMARK_CONCAT(a, b, c) TF_BENCHMARK_CONCAT2(a, b, c)
#define TF_BENCHMARK_CONCAT2(a, b, c) a##b##c

namespace testing {
namespace benchmark {
class State;
}
}  // namespace testing

namespace tensorflow {
namespace testing {
namespace internal {
void UseCharPointer(char const volatile*);
}

// The DoNotOptimize(...) function can be used to prevent a value or
// expression from being optimized away by the compiler. This function is
// intended to add little to no overhead.
// See: http://stackoverflow.com/questions/28287064
//
// The specific guarantees of DoNotOptimize(x) are:
//  1) x, and any data it transitively points to, will exist (in a register or
//     in memory) at the current point in the program.
//  2) The optimizer will assume that DoNotOptimize(x) could mutate x or
//     anything it transitively points to (although it actually doesn't).
//
// To see this in action:
//
//   void BM_multiply(benchmark::State& state) {
//     int a = 2;
//     int b = 4;
//     for (auto _ : state) {
//       testing::DoNotOptimize(a);
//       testing::DoNotOptimize(b);
//       int c = a * b;
//       testing::DoNotOptimize(c);
//     }
//   }
//   BENCHMARK(BM_multiply);
//
// Guarantee (2) applied to 'a' and 'b' prevents the compiler lifting the
// multiplication outside of the loop. Guarantee (1) applied to 'c' prevents the
// compiler from optimizing away 'c' as dead code.
template <class T>
void DoNotOptimize(const T& var) {
#if defined(_MSC_VER)
  internal::UseCharPointer(reinterpret_cast<char const volatile*>(&var));
  _ReadWriteBarrier();
#else
  asm volatile("" : "+m"(const_cast<T&>(var)));
#endif
}

class Benchmark {
 public:
  [[deprecated("use `benchmark::State&` instead.")]] Benchmark(const char* name,
                                                               void (*fn)(int));

  [[deprecated("use `benchmark::State&` instead.")]] Benchmark(const char* name,
                                                               void (*fn)(int,
                                                                          int));

  [[deprecated("use `benchmark::State&` instead.")]] Benchmark(
      const char* name, void (*fn)(int, int, int));

  Benchmark(const char* name, void (*fn)(::testing::benchmark::State&));

  Benchmark* Arg(int x);
  Benchmark* ArgPair(int x, int y);
  Benchmark* Range(int lo, int hi);
  Benchmark* RangePair(int lo1, int hi1, int lo2, int hi2);

  Benchmark* UseRealTime();

  static void Run(const char* pattern);

 private:
  string name_;
  int num_args_;
  int instantiated_num_args_ = -1;
  std::vector<std::pair<int, int> > args_;
  void (*fn0_)(int) = nullptr;
  void (*fn1_)(int, int) = nullptr;
  void (*fn2_)(int, int, int) = nullptr;
  void (*fn_state_)(::testing::benchmark::State&) = nullptr;

  void Register();
  void Run(int arg1, int arg2, int* run_count, double* run_seconds);

  void CheckArgCount(int expected);
};

void RunBenchmarks();
void SetLabel(const std::string& label);
void BytesProcessed(int64);
void ItemsProcessed(int64);
void StartTiming();
void StopTiming();
void UseRealTime();

}  // namespace testing
}  // namespace tensorflow

// Support `void BM_Func(benchmark::State&)` interface so that the it is
// compatible with the internal version.
namespace testing {
namespace benchmark {
// State is passed as an argument to a benchmark function.
// Each thread in threaded benchmarks receives own object.
class State {
 public:
  // Incomplete iterator-like type with dummy value type so that
  // benchmark::State can support iteration with a range-based for loop.
  //
  // The only supported usage:
  //
  //   static void BM_Foo(benchmark::State& state) {
  //     for (auto s : state) {
  //       // perform single iteration
  //     }
  //   }
  //
  // This is meant to replace the deprecated API :
  //
  //   static void BM_Foo(int iters) {
  //     while (iters-- > 0) {
  //       // perform single iteration
  //     }
  //   }
  //
  // See go/benchmark#old-benchmark-interface for more details.
  class Iterator {
   public:
    struct Value {
      // Non-trivial destructor to avoid warning for unused dummy variable in
      // the range-based for loop.
      ~Value() {}
    };

    explicit Iterator(State* parent);

    Iterator& operator++();

    bool operator!=(const Iterator& other);

    Value operator*();

   private:
    State* const parent_;
  };

  Iterator begin();
  Iterator end();

  void PauseTiming();
  void ResumeTiming();

  // Set the number of bytes processed by the current benchmark
  // execution.  This routine is typically called once at the end of a
  // throughput oriented benchmark.  If this routine is called with a
  // value > 0, then bytes processed per second is also reported.
  void SetBytesProcessed(::tensorflow::int64 bytes);

  // If this routine is called with items > 0, then an items/s
  // label is printed on the benchmark report line for the currently
  // executing benchmark. It is typically called at the end of a processing
  // benchmark where a processing items/second output is desired.
  void SetItemsProcessed(::tensorflow::int64 items);

  // If this method is called, the specified label is printed at the
  // end of the benchmark report line for the currently executing
  // benchmark.  Example:
  //  static void BM_Compress(benchmark::State& state) {
  //    ...
  //    double compression = input_size / output_size;
  //    state.SetLabel(StringPrintf("compress:%.1f%%", 100.0*compression));
  //  }
  // Produces output that looks like:
  //  BM_Compress   50         50   14115038  compress:27.3%
  //
  // REQUIRES: a benchmark is currently executing
  void SetLabel(absl::string_view label);

  // For parameterized benchmarks, range(i) returns the value of the ith
  // parameter. Simple benchmarks are not parameterized and do not need to call
  // range().
  int range(size_t i) const;

  // Total number of iterations processed so far.
  size_t iterations() const;

  const size_t
      max_iterations;  // NOLINT: for compatibility with OSS benchmark library

  // Disallow copy and assign.
  State(const State&) = delete;
  State& operator=(const State&) = delete;

 protected:
  friend class tensorflow::testing::Benchmark;
  State(size_t max_iterations, int formal_arg_count, std::vector<int> args);

 private:
  size_t completed_iterations_;
  const int formal_arg_count_;
  const std::vector<int> args_;
};

inline State::Iterator::Iterator(State* parent) : parent_(parent) {}

inline size_t State::iterations() const { return completed_iterations_; }

inline bool State::Iterator::operator!=(const Iterator& other) {
  DCHECK_EQ(other.parent_, nullptr);
  DCHECK_NE(parent_, nullptr);

  if (parent_->completed_iterations_ < parent_->max_iterations) {
    return true;
  }

  ++parent_->completed_iterations_;
  // If this is the last iteration, stop the timer.
  parent_->PauseTiming();
  return false;
}

inline State::Iterator& State::Iterator::operator++() {
  DCHECK_LT(parent_->completed_iterations_, parent_->max_iterations);
  ++parent_->completed_iterations_;
  return *this;
}

inline State::Iterator::Value State::Iterator::operator*() { return Value(); }

inline State::Iterator State::begin() {
  // Starts the timer here because if the code uses this API, it expects
  // the timer to starts at the beginning of this loop.
  ResumeTiming();
  return Iterator(this);
}

inline State::Iterator State::end() { return Iterator(nullptr); }

void RunSpecifiedBenchmarks();

}  // namespace benchmark
}  // namespace testing

#endif  // TENSORFLOW_CORE_PLATFORM_DEFAULT_TEST_BENCHMARK_H_
