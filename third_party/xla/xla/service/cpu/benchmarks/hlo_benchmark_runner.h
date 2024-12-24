/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_BENCHMARKS_HLO_BENCHMARK_RUNNER_H_
#define XLA_SERVICE_CPU_BENCHMARKS_HLO_BENCHMARK_RUNNER_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {

// A string-to-string mapping that allows to parametrize HLO benchmarks.
using StrToStrMapping =
    std::initializer_list<std::pair<absl::string_view, absl::string_view>>;

// Runs the given HLO module as a benchmark.
//
// The HLO text can be interpolated using the given string replacements. Each
// replacement is a mapping that will be applied to the HLO module before
// running the benchmark.
//
// If `disable_parallel_task_assigner` is true, the parallel task assigner will
// not be run on the HLO module before running the benchmark. Therefore,
// parallel backend will not be executed.
absl::Status RunHloBenchmark(benchmark::State& state,
                             absl::string_view hlo_module,
                             absl::Span<const Literal* const> args,
                             StrToStrMapping replacements = {},
                             bool disable_parallel_task_assigner = false);

// Benchmarks the given HLO's compilation time.
//
// Takes the same options as RunHloBenchmark, except no arguments since the
// HLO is only compiled, not run.
absl::Status CompileHloBenchmark(benchmark::State& state,
                                 absl::string_view hlo_module,
                                 StrToStrMapping replacements = {},
                                 bool disable_parallel_task_assigner = false);

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_BENCHMARKS_HLO_BENCHMARK_RUNNER_H_
