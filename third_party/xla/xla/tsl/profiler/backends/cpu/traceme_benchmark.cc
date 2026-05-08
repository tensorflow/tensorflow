/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Microbenchmark for TraceMe instrumentation.
//
// We test the relative performance of three configurations:
//   - uninstrumented code (NoTraceMe)
//   - instrumented code, but tracing disabled (DisabledTraceMe)
//   - instrumented code, tracing enabled (TraceMe)

#include <sys/types.h>

#include <cstdint>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "benchmark/benchmark.h"
#include "xla/tsl/profiler/utils/trace_filter_utils.h"
#include "tsl/platform/init_main.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::tsl::profiler::TraceMe;

// Substitute TraceMe that will be optimized out to nothing by the compiler.
struct NoTraceMe {
  explicit NoTraceMe(const char*, int) {}
  explicit NoTraceMe(const char*, int, uint64_t) {}
  template <typename NameGeneratorT>
  explicit NoTraceMe(NameGeneratorT name_generator, int) {}
  template <typename NameGeneratorT>
  explicit NoTraceMe(NameGeneratorT name_generator, int, uint64_t) {}
};

#define FILTER_MASK_FAILING 1ull << 1
#define FILTER_MASK_PASSING 1ull << 0
#define TRACEME_LEVEL 1

// Benchmark that traces a sleep (or with NoTraceMe, doesn't trace it).
template <typename TraceMe, bool use_strcat = false>
void BM(benchmark::State& state) {
  int i = 1;
  for (auto _ : state) {
    if (use_strcat) {
      TraceMe trace(
          [i] {
            return absl::StrCat(
                "Some long string that actually requires some memory."
                "In practice, these strings are often 100 bytes or more. "
                "string "
                "nr:",
                i);
          },
          TRACEME_LEVEL);
      ++i;
    } else {
      TraceMe trace("work", TRACEME_LEVEL);
    }
  }
}

template <typename TraceMe, bool use_strcat = false, uint64_t filter = 0>
void BM_WithFilter(benchmark::State& state) {
  int i = 1;
  for (auto _ : state) {
    if (use_strcat) {
      TraceMe trace(
          [i] {
            return absl::StrCat(
                "Some long string that actually requires some memory."
                "In practice, these strings are often 100 bytes or more. "
                "string "
                "nr:",
                i);
          },
          TRACEME_LEVEL, filter);
      ++i;
    } else {
      TraceMe trace("work", TRACEME_LEVEL, filter);
    }
  }
}

static tsl::ProfilerSession* session = nullptr;

static void StartTracing(const benchmark::State& state) {
  tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
  options.set_device_type(tensorflow::ProfileOptions::CPU);
  options.set_host_tracer_level(1);  // user TraceMe's
  session = ABSL_DIE_IF_NULL(tsl::ProfilerSession::Create(options)).release();
  CHECK_OK(session->Status());
}

static void StartTracingWithFilter(const benchmark::State& state) {
  tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
  options.set_device_type(tensorflow::ProfileOptions::CPU);
  options.set_host_tracer_level(1);  // user TraceMe's
  options.mutable_trace_options()->set_host_traceme_filter_mask(
      TraceMeFiltersToMask({tsl::profiler::TraceMeFilter::kTraceMemory}));
  session = ABSL_DIE_IF_NULL(tsl::ProfilerSession::Create(options)).release();
  CHECK_OK(session->Status());
}

static void StopTracing(const benchmark::State& state) {
  tensorflow::profiler::XSpace space;
  session->CollectData(&space).IgnoreError();
  CHECK_OK(session->Status());
  delete session;
}

using DisabledTraceMe = TraceMe;  // To give DisabledTraceMe cases nice names.
#define EnableTracing() Setup(StartTracing)->Teardown(StopTracing)
#define EnableTracingWithFilter() \
  Setup(StartTracingWithFilter)->Teardown(StopTracing)

BENCHMARK(BM<NoTraceMe>)->ThreadRange(1, 8);
BENCHMARK(BM<DisabledTraceMe>)->ThreadRange(1, 8);
BENCHMARK(BM<TraceMe>)->EnableTracing()->ThreadRange(1, 8);
BENCHMARK(BM<TraceMe, true>)->EnableTracing()->ThreadRange(1, 1);

BENCHMARK(BM_WithFilter<NoTraceMe, false, FILTER_MASK_PASSING>)
    ->ThreadRange(1, 8);
BENCHMARK(BM_WithFilter<DisabledTraceMe, false, FILTER_MASK_PASSING>)
    ->ThreadRange(1, 8);
BENCHMARK(BM_WithFilter<TraceMe, false, FILTER_MASK_PASSING>)
    ->EnableTracingWithFilter()
    ->ThreadRange(1, 8);
BENCHMARK(BM_WithFilter<TraceMe, false, FILTER_MASK_FAILING>)
    ->EnableTracingWithFilter()
    ->ThreadRange(1, 8);
BENCHMARK(BM_WithFilter<TraceMe, true, FILTER_MASK_PASSING>)
    ->EnableTracingWithFilter()
    ->ThreadRange(1, 1);
BENCHMARK(BM_WithFilter<TraceMe, true, FILTER_MASK_FAILING>)
    ->EnableTracingWithFilter()
    ->ThreadRange(1, 1);

}  // namespace
}  // namespace profiler
}  // namespace tsl

int main(int argc, char** argv) {
  // Initialize benchmark library to parse flags.
  benchmark::Initialize(&argc, argv);
  // Initialize TensorFlow environment.
  tsl::port::InitMain(argv[0], &argc, &argv);
  if (benchmark::GetBenchmarkFilter() == "console") {
    absl::PrintF(
        "IMPORTANT NOTE: Reported walltime should be multiplied by #threads!\n"
        "The displayed 'inverse throughput' is not meaningful for TraceMe.\n"
        "See go/benchmark#multithreaded-benchmarks\n\n");
  }
  // Run benchmarks.
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
