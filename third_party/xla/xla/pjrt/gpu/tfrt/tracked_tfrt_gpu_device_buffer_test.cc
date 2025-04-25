/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/gpu/tfrt/tracked_tfrt_gpu_device_buffer.h"

#include <stdlib.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/literal.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/mem.h"

namespace xla {
namespace {

using ::tsl::BlockUntilReady;
using ::tsl::MakeConstructedAsyncValueRef;

class TestAllocator : public tsl::Allocator {
 public:
  std::string Name() override { return "test_allocator"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return tsl::port::AlignedMalloc(num_bytes, alignment);
  }
  void DeallocateRaw(void* ptr) override { return tsl::port::AlignedFree(ptr); }
};

class TestDevice : public PjRtDevice {
 public:
  TestDevice() = default;

  PjRtLocalHardwareId local_hardware_id() const override {
    return PjRtLocalHardwareId(0);
  }

  PjRtClient* client() const override {
    LOG(FATAL) << "Unimplemented for TestDevice.";
  }

  bool IsAddressable() const override {
    LOG(FATAL) << "Unimplemented for TestDevice.";
  }

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    LOG(FATAL) << "Unimplemented for TestDevice.";
  }

  absl::Status TransferToInfeed(const LiteralSlice& literal) override {
    return Unimplemented("Unimplemented for TestDeivce.");
  }

  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override {
    return Unimplemented("Unimplemented for TestDeivce.");
  }

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    LOG(FATAL) << "Unimplemented for TestDevice.";
  }

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override {
    LOG(FATAL) << "Unimplemented for TestDevice.";
  }
};

TEST(MaybeOwningGpuMemoryTest, MoveConstructorSetOriginalToNull) {
  TestAllocator allocator;
  size_t size = sizeof(int);
  void* data = allocator.AllocateRaw(tsl::Allocator::kAllocatorAlignment, size);
  se::DeviceMemoryBase device_memory(data, size);
  MaybeOwningGpuMemory memory(&allocator, device_memory);
  MaybeOwningGpuMemory another_memory = std::move(memory);
  EXPECT_TRUE(another_memory.owns_data());
  EXPECT_EQ(another_memory.buffer(), device_memory);
}

TEST(MaybeOwningGpuMemoryTest, OwningToNonOwning) {
  TestAllocator allocator;
  MaybeOwningGpuMemory memory(&allocator, se::DeviceMemoryBase());
  EXPECT_TRUE(memory.owns_data());
  memory.SetUnOwned();
  EXPECT_FALSE(memory.owns_data());
  EXPECT_EQ(memory.allocator(), nullptr);
}

TEST(MaybeOwningGpuMemoryTest, AsShapeBuffer) {
  LocalClient* client = ClientLibrary::LocalClientOrDie();
  TestDevice device;
  Shape shape = ShapeUtil::MakeShape(F32, {1, 2, 3});
  TestAllocator allocator;
  int64_t byte_size =
      client->backend().transfer_manager()->GetByteSizeRequirement(shape);
  TF_ASSERT_OK_AND_ASSIGN(
      auto memory, MaybeOwningGpuMemory::AllocateShared(&allocator, byte_size));
  ShapedBuffer result_shaped_buffer = memory.AsShapedBuffer(
      client->backend().transfer_manager()->HostShapeToDeviceShape(shape),
      &device);
  EXPECT_EQ(result_shaped_buffer.root_buffer().size(), byte_size);
}

TEST(TrackedTfrtGpuDeviceBufferTest, TrackedDeviceBufferUsageEndToEnd) {
  std::string expected = "tracked_tfrt_gpu_device_buffer_test";
  auto usage_event = MakeConstructedAsyncValueRef<GpuEvent>();

  TestAllocator allocator;
  auto test_buffer = MakeConstructedAsyncValueRef<MaybeOwningGpuMemory>(
      &allocator, se::DeviceMemoryBase(expected.data(), expected.size()));

  auto definition_event = MakeConstructedAsyncValueRef<GpuEvent>();

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(),
                                      "tracked_buffer_test",
                                      /*num_threads=*/4);

  TrackedTfrtGpuDeviceBuffer tracked_buffer(test_buffer, definition_event,
                                            /*on_delete_callback_=*/nullptr);
  tracked_buffer.SetUnOwned();
  {
    MarkGpuEventReadyOnExit ready_on_exit(usage_event);
    tracked_buffer.AddUsageEvents(absl::MakeSpan(&usage_event, 1));
    // Mimic transfer event in a thread pool.
    thread_pool.Schedule([&]() {
      std::memcpy(test_buffer->buffer().opaque(), expected.data(),
                  expected.size());
      definition_event.SetStateConcrete();
      test_buffer.SetStateConcrete();
    });
    BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());
    EXPECT_EQ(tracked_buffer.buffer()->size(), expected.size());
    auto result = tracked_buffer.buffer();
    ASSERT_TRUE(result.IsAvailable());
    EXPECT_FALSE(result->owns_data());
    EXPECT_EQ(std::memcmp(expected.data(), result->buffer().opaque(),
                          expected.size()),
              0);
  }
  BlockUntilReady(tracked_buffer.AfterAllUsageEvents());
  BlockUntilReady(tracked_buffer.LockUseAndTransferUsageEvents());
}

}  // namespace

}  // namespace xla
