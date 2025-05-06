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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
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
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::tsl::BlockUntilReady;
using ::tsl::MakeConstructedAsyncValueRef;

void* kOpaque = reinterpret_cast<void*>(1234567890);

class TestAllocator : public se::DeviceMemoryAllocator {
 public:
  TestAllocator() : DeviceMemoryAllocator(nullptr) {}

  using se::DeviceMemoryAllocator::Allocate;
  absl::StatusOr<stream_executor::OwningDeviceMemory> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override {
    const se::DeviceMemoryBase base(kOpaque, size);
    return stream_executor::OwningDeviceMemory(base, 0, this);
  }
  absl::Status Deallocate(int device_ordinal,
                          se::DeviceMemoryBase mem) override {
    return absl::OkStatus();
  }
  absl::StatusOr<se::Stream*> GetStream(int device_ordinal) override {
    LOG(FATAL) << "Unimplemented for TestAllocator.";
  }
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
  TF_ASSERT_OK_AND_ASSIGN(auto owning_memory, allocator.Allocate(0, 100));
  MaybeOwningGpuMemory memory(std::move(owning_memory));
  EXPECT_EQ(memory.buffer().opaque(), kOpaque);

  MaybeOwningGpuMemory another_memory = std::move(memory);
  EXPECT_TRUE(another_memory.owns_data());
  EXPECT_EQ(another_memory.buffer().opaque(), kOpaque);
}

TEST(MaybeOwningGpuMemoryTest, OwningToNonOwning) {
  TestAllocator allocator;
  TF_ASSERT_OK_AND_ASSIGN(auto owning_memory, allocator.Allocate(0, 100));
  MaybeOwningGpuMemory memory(std::move(owning_memory));
  EXPECT_TRUE(memory.owns_data());
  memory.SetUnOwned();
  EXPECT_FALSE(memory.owns_data());
}

TEST(MaybeOwningGpuMemoryTest, AsShapeBuffer) {
  LocalClient* client = ClientLibrary::LocalClientOrDie();
  TestDevice device;
  Shape shape = ShapeUtil::MakeShape(F32, {1, 2, 3});
  TestAllocator allocator;
  int64_t byte_size =
      client->backend().transfer_manager()->GetByteSizeRequirement(shape);
  TF_ASSERT_OK_AND_ASSIGN(auto memory, MaybeOwningGpuMemory::AllocateShared(
                                           &allocator, 0, byte_size));
  ShapedBuffer result_shaped_buffer = memory.AsShapedBuffer(
      client->backend().transfer_manager()->HostShapeToDeviceShape(shape),
      &device);
  EXPECT_EQ(result_shaped_buffer.root_buffer().size(), byte_size);
}

TEST(TrackedTfrtGpuDeviceBufferTest, TrackedDeviceBufferUsageEndToEnd) {
  auto usage_event = MakeConstructedAsyncValueRef<GpuEvent>();

  TestAllocator allocator;
  TF_ASSERT_OK_AND_ASSIGN(auto owning_memory, allocator.Allocate(0, 100));
  MaybeOwningGpuMemory memory(std::move(owning_memory));
  auto test_buffer =
      MakeConstructedAsyncValueRef<MaybeOwningGpuMemory>(std::move(memory));

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
      absl::SleepFor(absl::Milliseconds(50));
      definition_event.SetStateConcrete();
      test_buffer.SetStateConcrete();
    });
    BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());
    EXPECT_EQ(tracked_buffer.buffer()->size(), 100);
    auto result = tracked_buffer.buffer();
    ASSERT_TRUE(result.IsAvailable());
    EXPECT_FALSE(result->owns_data());
    EXPECT_EQ(result->buffer().opaque(), kOpaque);
  }
  BlockUntilReady(tracked_buffer.AfterAllUsageEvents());
  BlockUntilReady(tracked_buffer.LockUseAndTransferUsageEvents());
}

}  // namespace

}  // namespace xla
