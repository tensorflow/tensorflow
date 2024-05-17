/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/pjrt/metrics.h"

#include <cstdint>

#include "tsl/lib/monitoring/counter.h"
#include "tsl/lib/monitoring/gauge.h"

namespace xla {
namespace {

auto* pjrt_executable_executions = tsl::monitoring::Counter<0>::New(
    "/jax/pjrt/pjrt_executable_executions",
    "The number of PjRtExecutable::ExecuteHelper calls.");

auto* pjrt_executable_execution_time_usecs = tsl::monitoring::Counter<0>::New(
    "/jax/pjrt/pjrt_executable_execution_time_usecs",
    "The total time spent on PjRtExecutable::ExecuteHelper in "
    "microseconds.");

auto* pjrt_compiler_is_compiling_computation =
    tsl::monitoring::Gauge<bool, 0>::New(
        metrics::kPjrtCompilerCompileComputationMetricName,
        "Whether the PjRT compiler is compiling computations.");

auto* pjrt_compiler_is_compiling_module = tsl::monitoring::Gauge<bool, 0>::New(
    metrics::kPjrtCompilerCompileModuleMetricName,
    "Whether the PjRT compiler is compiling modules.");

}  // namespace

namespace metrics {

void ReportExecutableEnqueueTime(const uint64_t running_time_usecs) {
  if (running_time_usecs > 0) {
    static auto* pjrt_executable_executions_cell =
        pjrt_executable_executions->GetCell();
    static auto* pjrt_executable_execution_time_usecs_cell =
        pjrt_executable_execution_time_usecs->GetCell();
    pjrt_executable_executions_cell->IncrementBy(1);
    pjrt_executable_execution_time_usecs_cell->IncrementBy(running_time_usecs);
  }
}

void RecordPjrtCompilerCompileComputationStatus(bool is_compiling) {
  pjrt_compiler_is_compiling_computation->GetCell()->Set(is_compiling);
}

void RecordPjrtCompilerCompileModuleStatus(bool is_compiling) {
  pjrt_compiler_is_compiling_module->GetCell()->Set(is_compiling);
}

}  // namespace metrics
}  // namespace xla
