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

#include "xla/backends/cpu/nanort/nanort_executable.h"

#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/executable.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/threadpool.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla::cpu {

using ::tsl::profiler::TraceMe;
using ::tsl::profiler::TraceMeEncode;

absl::StatusOr<std::unique_ptr<NanoRtExecutable>> NanoRtExecutable::Create(
    std::unique_ptr<Executable> executable,
    std::shared_ptr<tsl::thread::ThreadPool> thread_pool) {
  auto* cpu_executable = tsl::down_cast<cpu::CpuExecutable*>(executable.get());
  if (!cpu_executable->has_thunks()) {
    return Internal("NanoRtExecutable requires CPU executable to use thunks");
  }

  return absl::WrapUnique(
      new NanoRtExecutable(std::move(executable), std::move(thread_pool)));
}

NanoRtExecutable::NanoRtExecutable(
    std::unique_ptr<Executable> executable,
    std::shared_ptr<tsl::thread::ThreadPool> thread_pool)
    : executable_(std::move(executable)),
      thread_pool_(std::move(thread_pool)) {}

static se::DeviceMemoryBase ToDeviceMemory(
    const NanoRtExecutable::Argument& argument) {
  return stream_executor::DeviceMemoryBase(
      const_cast<void*>(reinterpret_cast<const void*>(argument.data().data())),
      argument.data().size());
}

static se::DeviceMemoryBase ToDeviceMemory(
    const NanoRtExecutable::Result& result) {
  return stream_executor::DeviceMemoryBase(
      reinterpret_cast<void*>(result.data().data()), result.data().size());
}

static se::DeviceMemoryBase ToDeviceMemory(
    const NanoRtExecutable::PreallocatedTemp& temp) {
  return stream_executor::DeviceMemoryBase(reinterpret_cast<void*>(temp.data()),
                                           temp.size());
}

tsl::AsyncValueRef<NanoRtExecutable::ExecuteEvent> NanoRtExecutable::Execute(
    absl::Span<const Argument> arguments, absl::Span<const Result> results,
    const PreallocatedTemp& temp) {
  TraceMe trace([&] {
    return TraceMeEncode("NanoRtExecutable::Execute",
                         {{"name", executable_->module().name()}});
  });

  auto* executable = tsl::down_cast<cpu::CpuExecutable*>(executable_.get());

  // Convert arguments, results, and temp to device memory.
  absl::InlinedVector<MaybeOwningDeviceMemory, 8> buffer_device_mem;
  buffer_device_mem.reserve(arguments.size() + results.size() + 1);

  for (const Result& result : results) {
    buffer_device_mem.emplace_back(ToDeviceMemory(result));
  }
  for (const Argument& argument : arguments) {
    buffer_device_mem.emplace_back(ToDeviceMemory(argument));
  }
  buffer_device_mem.emplace_back(ToDeviceMemory(temp));

  // Prepare buffer allocations for arguments, results, and temp.
  cpu::BufferAllocations allocations(buffer_device_mem);

  cpu::Thunk::ExecuteParams execute_params = {
      &executable->function_registry(),
      &allocations,
  };

  return executable->thunks().Execute(execute_params);
}

}  // namespace xla::cpu
