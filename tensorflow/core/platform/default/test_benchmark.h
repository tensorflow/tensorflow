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

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"

#define BENCHMARK(n)                                            \
  static ::tensorflow::testing::Benchmark* TF_BENCHMARK_CONCAT( \
      __benchmark_, n, __LINE__) TF_ATTRIBUTE_UNUSED =          \
      (new ::tensorflow::testing::Benchmark(#n, (n)))
#define TF_BENCHMARK_CONCAT(a, b, c) TF_BENCHMARK_CONCAT2(a, b, c)
#define TF_BENCHMARK_CONCAT2(a, b, c) a##b##c

namespace tensorflow {
namespace testing {

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
  asm volatile("" : "+m"(const_cast<T&>(var)));
}

class Benchmark {
 public:
  Benchmark(const char* name, void (*fn)(int));
  Benchmark(const char* name, void (*fn)(int, int));
  Benchmark(const char* name, void (*fn)(int, int, int));

  Benchmark* Arg(int x);
  Benchmark* ArgPair(int x, int y);
  Benchmark* Range(int lo, int hi);
  Benchmark* RangePair(int lo1, int hi1, int lo2, int hi2);
  static void Run(const char* pattern);

 private:
  string name_;
  int num_args_;
  std::vector<std::pair<int, int> > args_;
  void (*fn0_)(int) = nullptr;
  void (*fn1_)(int, int) = nullptr;
  void (*fn2_)(int, int, int) = nullptr;

  void Register();
  void Run(int arg1, int arg2, int* run_count, double* run_seconds);
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

#endif  // TENSORFLOW_CORE_PLATFORM_DEFAULT_TEST_BENCHMARK_H_
