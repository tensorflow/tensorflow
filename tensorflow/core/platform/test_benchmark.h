// Simple benchmarking facility.
#ifndef TENSORFLOW_PLATFORM_TEST_BENCHMARK_H_
#define TENSORFLOW_PLATFORM_TEST_BENCHMARK_H_

#include "tensorflow/core/platform/port.h"

#if defined(PLATFORM_GOOGLE)
#include "testing/base/public/benchmark.h"

#else
#define BENCHMARK(n)                                            \
  static ::tensorflow::testing::Benchmark* TF_BENCHMARK_CONCAT( \
      __benchmark_, n, __LINE__) TF_ATTRIBUTE_UNUSED =          \
      (new ::tensorflow::testing::Benchmark(#n, (n)))
#define TF_BENCHMARK_CONCAT(a, b, c) TF_BENCHMARK_CONCAT2(a, b, c)
#define TF_BENCHMARK_CONCAT2(a, b, c) a##b##c

#endif  // PLATFORM_GOOGLE

namespace tensorflow {
namespace testing {

#if defined(PLATFORM_GOOGLE)
using ::testing::Benchmark;
#else
class Benchmark {
 public:
  Benchmark(const char* name, void (*fn)(int));
  Benchmark(const char* name, void (*fn)(int, int));

  Benchmark* Arg(int x);
  Benchmark* Range(int lo, int hi);
  static void Run(const char* pattern);

 private:
  string name_;
  int num_args_;
  std::vector<int> args_;
  void (*fn0_)(int) = nullptr;
  void (*fn1_)(int, int) = nullptr;

  void Register();
  void Run(int arg, int* run_count, double* run_seconds);
};
#endif

void RunBenchmarks();
void SetLabel(const std::string& label);
void BytesProcessed(int64);
void ItemsProcessed(int64);
void StartTiming();
void StopTiming();
void UseRealTime();

}  // namespace testing
}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_TEST_BENCHMARK_H_
