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

#include <cstdint>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "benchmark/benchmark.h"
#include "tsl/platform/init_main.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl::profiler {
namespace {

using tsl::profiler::ContextType;
using ::tsl::profiler::TraceMe;
using ::tsl::profiler::TraceMeConsumer;
using ::tsl::profiler::TraceMeProducer;

template <int sleep_micros>
void BM_NoTraceMe(benchmark::State& state) {
  for (auto _ : state) {
  }
}

template <typename TraceMe>
void BM_TraceMeProducerManual(benchmark::State& state) {
  uint64_t id = 0;
  for (auto _ : state) {
    TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode(
          "producer", {{"_pt", ContextType::kTfExecutor}, {"_p", id}});
    });
    id += 1;
  }
}

template <typename TraceMe>
void BM_TraceMeConsumerManual(benchmark::State& state) {
  uint64_t id = 0;
  for (auto _ : state) {
    TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode(
          "consumer", {{"_ct", ContextType::kTfExecutor}, {"_c", id}});
    });
    id += 1;
  }
}

template <typename TraceMeProducer>
void BM_TraceMeProducer(benchmark::State& state) {
  uint64_t id = 0;
  for (auto _ : state) {
    TraceMeProducer trace([&] { return "producer"; }, ContextType::kTfExecutor,
                          id);
    id += 1;
  }
}

template <typename TraceMeProducer>
void BM_TraceMeProducerGenericContext(benchmark::State& state) {
  uint64_t id = 0;
  for (auto _ : state) {
    TraceMeProducer trace([&] { return "producer"; });
    id += 1;
  }
}

template <typename TraceMeConsumer>
void BM_TraceMeConsumer(benchmark::State& state) {
  uint64_t id = 0;
  for (auto _ : state) {
    TraceMeConsumer trace([&] { return "consumer"; }, ContextType::kTfExecutor,
                          id);
    id += 1;
  }
}

template <typename TraceMeConsumer>
void BM_TraceMeConsumerGenericContext(benchmark::State& state) {
  uint64_t id = 0;
  for (auto _ : state) {
    TraceMeConsumer trace([&] { return "consumer"; }, id);
    id += 1;
  }
}

static tsl::ProfilerSession* session = nullptr;

static void StartTracing(const benchmark::State& state) {
  tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
  options.set_device_type(tensorflow::ProfileOptions::CPU);
  options.set_host_tracer_level(2);  // user TraceMe's
  session = ABSL_DIE_IF_NULL(tsl::ProfilerSession::Create(options)).release();
  CHECK_OK(session->Status());
}

static void StopTracing(const benchmark::State& state) {
  tensorflow::profiler::XSpace space;
  session->CollectData(&space).IgnoreError();
  CHECK_OK(session->Status());
  delete session;
}

// To print the benchmark names accordingly.
using DisabledTraceMe = TraceMe;
using DisabledTraceMeProducer = TraceMeProducer;
using DisabledTraceMeConsumer = TraceMeConsumer;

#define EnableTracing() Setup(StartTracing)->Teardown(StopTracing)

BENCHMARK(BM_NoTraceMe<0>);

BENCHMARK(BM_TraceMeProducerManual<DisabledTraceMe>);
BENCHMARK(BM_TraceMeConsumerManual<DisabledTraceMe>);
BENCHMARK(BM_TraceMeProducer<DisabledTraceMeProducer>);
BENCHMARK(BM_TraceMeProducerGenericContext<DisabledTraceMeProducer>);
BENCHMARK(BM_TraceMeConsumer<DisabledTraceMeConsumer>);
BENCHMARK(BM_TraceMeConsumerGenericContext<DisabledTraceMeConsumer>);

BENCHMARK(BM_TraceMeProducerManual<TraceMe>)->EnableTracing();
BENCHMARK(BM_TraceMeConsumerManual<TraceMe>)->EnableTracing();
BENCHMARK(BM_TraceMeProducer<TraceMeProducer>)->EnableTracing();
BENCHMARK(BM_TraceMeProducerGenericContext<TraceMeProducer>)->EnableTracing();
BENCHMARK(BM_TraceMeConsumer<TraceMeConsumer>)->EnableTracing();
BENCHMARK(BM_TraceMeConsumerGenericContext<TraceMeConsumer>)->EnableTracing();

}  // namespace
}  // namespace tsl::profiler

int main(int argc, char** argv) {
  // Initialize benchmark library to parse flags.
  benchmark::Initialize(&argc, argv);
  // Initialize TensorFlow environment.
  tsl::port::InitMain(argv[0], &argc, &argv);
  // Run benchmarks.
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
