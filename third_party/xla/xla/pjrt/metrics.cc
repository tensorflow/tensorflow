/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
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

auto* free_gpu_system_memory = tsl::monitoring::Gauge<int64_t, 1>::New(
    metrics::kPjrtCompilerFreeGpuSystemMemoryMetricName,
    "Record the free GPU system memory.", "gpu_id");

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

void RecordFreeGpuSystemMemory() {
  tensorflow::se::Platform* gpu_manager = tensorflow::se::GPUMachineManager();
  int visible_device_count = gpu_manager->VisibleDeviceCount();
  if (gpu_manager == nullptr || visible_device_count <= 0) return;

  for (int i = 0; i < visible_device_count; ++i) {
    tensorflow::se::StreamExecutor* se =
        gpu_manager->ExecutorForDevice(i).value();
    int64_t free_memory = 0, total_memory = 0;
    if (se->DeviceMemoryUsage(&free_memory, &total_memory)) {
      free_gpu_system_memory->GetCell(absl::StrCat(i))->Set(free_memory);
    } else {
      LOG(ERROR) << "Failed to query available memory for GPU " << i;
    }
  }
}

int64_t GetFreeGpuSystemMemory(int gpu_id) {
  return free_gpu_system_memory->GetCell(absl::StrCat(gpu_id))->value();
}

}  // namespace metrics
}  // namespace xla
