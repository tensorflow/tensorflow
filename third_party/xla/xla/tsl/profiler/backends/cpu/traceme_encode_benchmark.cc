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

#include <string>

#include "absl/strings/str_cat.h"
#include "benchmark/benchmark.h"
#include "tsl/platform/init_main.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xprof {
namespace {

void BM_EncodeUsingStrCat(benchmark::State& state) {
  int device_ordinal = 1;
  int request_id = 42;
  int queue_addr = 0xdeadbeef;
  for (auto _ : state) {
    std::string result =
        absl::StrCat("HostCallbackRequest#device_ordinal=", device_ordinal,
                     ",request_id=", request_id, ",queue_addr=", queue_addr,
                     ",cat=runtime#");
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_EncodeUsingStrCat);

void BM_EncodeUsingTraceMeEncode(benchmark::State& state) {
  int device_ordinal = 1;
  int request_id = 42;
  int queue_addr = 0xdeadbeef;
  for (auto _ : state) {
    std::string result = tsl::profiler::TraceMeEncode(
        "HostCallbackRequest", {{"device_ordinal", device_ordinal},
                                {"request_id", request_id},
                                {"queue_addr", queue_addr},
                                {"cat", "runtime"}});
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_EncodeUsingTraceMeEncode);

}  // namespace
}  // namespace xprof

int main(int argc, char** argv) {
  // Initialize benchmark library to parse flags.
  benchmark::Initialize(&argc, argv);
  // Initialize TensorFlow environment.
  tsl::port::InitMain(argv[0], &argc, &argv);
  // Run benchmarks.
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
