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

#ifndef XLA_STREAM_EXECUTOR_MOCK_STREAM_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_MOCK_STREAM_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/testlib/test.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/allocator_stats.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

// Implements StreamExecutor for testing.
class MockStreamExecutor : public StreamExecutor {
 public:
  MockStreamExecutor() = default;
  MOCK_METHOD(absl::Status, Init, (), (override));
  MOCK_METHOD(int, device_ordinal, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Kernel>>, LoadKernel,
              (const MultiKernelLoaderSpec& spec), (override));
  MOCK_METHOD(std::unique_ptr<ActivateContext>, Activate, (), (override));
  MOCK_METHOD(bool, UnloadModule, (ModuleHandle module_handle), (override));
  MOCK_METHOD(absl::StatusOr<ModuleHandle>, LoadModule,
              (const MultiModuleLoaderSpec& spec), (override));
  MOCK_METHOD(absl::StatusOr<std::shared_ptr<DeviceMemoryBase>>,
              CreateOrShareConstant,
              (Stream * stream, absl::Span<const uint8_t> content), (override));
  MOCK_METHOD(DeviceMemoryBase, Allocate, (uint64_t size, int64_t memory_space),
              (override));
  MOCK_METHOD(void, Deallocate, (DeviceMemoryBase * mem), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<MemoryAllocation>>,
              HostMemoryAllocate, (uint64_t size), (override));
  MOCK_METHOD(bool, SynchronizeAllActivity, (), (override));
  MOCK_METHOD(absl::Status, SynchronousMemZero,
              (DeviceMemoryBase * location, uint64_t size), (override));
  MOCK_METHOD(absl::Status, SynchronousMemcpy,
              (DeviceMemoryBase * device_dst, const void* host_src,
               uint64_t size),
              (override));
  MOCK_METHOD(absl::Status, SynchronousMemcpy,
              (void* host_dst, const DeviceMemoryBase& device_src,
               uint64_t size),
              (override));
  MOCK_METHOD(void, DeallocateStream, (Stream * stream), (override));
  MOCK_METHOD(absl::Status, EnablePeerAccessTo, (StreamExecutor * other),
              (override));
  MOCK_METHOD(bool, CanEnablePeerAccessTo, (StreamExecutor * other),
              (override));
  MOCK_METHOD(bool, DeviceMemoryUsage, (int64_t* free, int64_t* total),
              (const, override));
  MOCK_METHOD(absl::StatusOr<DeviceMemoryBase>, GetSymbol,
              (const std::string& symbol_name, ModuleHandle module_handle),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<DeviceDescription>>,
              CreateDeviceDescription, (), (const, override));
  MOCK_METHOD(blas::BlasSupport*, AsBlas, (), (override));
  MOCK_METHOD(fft::FftSupport*, AsFft, (), (override));
  MOCK_METHOD(dnn::DnnSupport*, AsDnn, (), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<CommandBuffer>>,
              CreateCommandBuffer, (CommandBuffer::Mode mode), (override));
  MOCK_METHOD(std::optional<AllocatorStats>, GetAllocatorStats, (), (override));
  MOCK_METHOD(bool, ClearAllocatorStats, (), (override));
  MOCK_METHOD(absl::Status, FlushCompilationCache, (), (override));
  MOCK_METHOD(Stream*, FindAllocatedStream, (void* device_stream), (override));
  MOCK_METHOD(const Platform*, GetPlatform, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Stream>>, CreateStream,
              ((std::optional<std::variant<StreamPriority, int>>)), (override));
  MOCK_METHOD(int64_t, GetMemoryLimitBytes, (), (const.override));
  MOCK_METHOD(const DeviceDescription&, GetDeviceDescription, (),
              (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Event>>, CreateEvent, (),
              (override));
  MOCK_METHOD(void, UnloadKernel, (const Kernel* kernel), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<EventBasedTimer>>,
              CreateEventBasedTimer, (Stream * stream, bool use_delay_kernel),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<MemoryAllocator>>,
              CreateMemoryAllocator, (MemoryType type), (override));
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MOCK_STREAM_EXECUTOR_H_
