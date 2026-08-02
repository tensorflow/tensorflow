/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/pjrt/se/tracked_device_buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/scoped_async_tracking_event.h"
#include "xla/runtime/chip_id.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class TestDevice : public PjRtDevice {
 public:
  TestDevice() = default;

  LocalChipId local_hardware_id() const override { return LocalChipId(0); }

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

absl::StatusOr<tsl::AsyncValueRef<RawSEDeviceMemory>> MakeArray(
    const Shape& shape, LocalClient* client) {
  std::vector<tsl::AsyncValueRef<RawSEDeviceMemory>> device_buffers;
  RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      client->backend().transfer_manager()->HostShapeToDeviceShape(shape),
      [&](const Shape& subshape, const ShapeIndex&) -> absl::Status {
        ASSIGN_OR_RETURN(
            se::ScopedDeviceAddress<uint8_t> device_memory,
            client->backend().memory_allocator()->Allocate(
                /*device_ordinal=*/0,
                client->backend().transfer_manager()->GetByteSizeRequirement(
                    subshape)));
        auto se_mem = *device_memory;
        device_buffers.push_back(RawSEDeviceMemory::CreateForeign(
            se_mem, [device_memory = std::move(device_memory)]() {}));
        return absl::OkStatus();
      }));
  return device_buffers[0];
}

TEST(TrackedDeviceBufferTest, AsShapedBuffer) {
  LocalClient* client = ClientLibrary::LocalClientOrDie();
  TestDevice device;

  Shape a_shape = ShapeUtil::MakeShape(F32, {3, 101, 4});
  Shape b_shape = ShapeUtil::MakeShape(S8, {77});
  Shape c_shape = ShapeUtil::MakeShape(S64, {});
  TF_ASSERT_OK_AND_ASSIGN(auto a_buffer, MakeArray(a_shape, client));
  TF_ASSERT_OK_AND_ASSIGN(auto b_buffer, MakeArray(b_shape, client));
  TF_ASSERT_OK_AND_ASSIGN(auto c_buffer, MakeArray(c_shape, client));

  std::vector<se::DeviceAddressBase> expected_buffer_sequence = {
      a_buffer->mem(), b_buffer->mem(), c_buffer->mem()};
  ShapedBuffer shaped_a = a_buffer->AsShapedBuffer(
      &device,
      client->backend().transfer_manager()->HostShapeToDeviceShape(a_shape));
  ShapedBuffer shaped_b = b_buffer->AsShapedBuffer(
      &device,
      client->backend().transfer_manager()->HostShapeToDeviceShape(b_shape));
  ShapedBuffer shaped_c = c_buffer->AsShapedBuffer(
      &device,
      client->backend().transfer_manager()->HostShapeToDeviceShape(c_shape));
  auto expected_it = expected_buffer_sequence.begin();
  for (auto it = shaped_a.buffers().begin(); it != shaped_a.buffers().end();
       ++it) {
    ASSERT_TRUE(expected_it != expected_buffer_sequence.end());
    EXPECT_TRUE(expected_it->IsSameAs(it->second));
    ++expected_it;
  }
  for (auto it = shaped_b.buffers().begin(); it != shaped_b.buffers().end();
       ++it) {
    ASSERT_TRUE(expected_it != expected_buffer_sequence.end());
    EXPECT_TRUE(expected_it->IsSameAs(it->second));
    ++expected_it;
  }
  for (auto it = shaped_c.buffers().begin(); it != shaped_c.buffers().end();
       ++it) {
    ASSERT_TRUE(expected_it != expected_buffer_sequence.end());
    EXPECT_TRUE(expected_it->IsSameAs(it->second));
    ++expected_it;
  }
  EXPECT_TRUE(expected_it == expected_buffer_sequence.end());
}

TEST(TrackedDeviceBufferTest, AsyncSliceSuccess) {
  // Create an unconstructed parent buffer holding ForeignRawSEDeviceMemory.
  auto parent_ref =
      tsl::MakeUnconstructedAsyncValueRef<ForeignRawSEDeviceMemory>();
  tsl::AsyncValueRef<RawSEDeviceMemory> parent(parent_ref);

  EXPECT_TRUE(parent.IsUnavailable());

  // Create a slice immediately (parent is still unconstructed).
  size_t offset = 16;
  size_t size = 32;
  tsl::AsyncValueRef<RawSEDeviceMemory> slice =
      RawSEDeviceMemory::CreateSlice(parent, offset, size);

  // The slice must be returned immediately in unconstructed (unavailable)
  // state.
  EXPECT_TRUE(slice.IsUnavailable());

  // Construct the parent buffer (simulate computation completing successfully).
  std::vector<char> dummy_data(128, 0xAB);
  se::DeviceAddressBase parent_address(dummy_data.data(), dummy_data.size());
  parent_ref.emplace(parent_address, [] {});

  // The parent is now concrete.
  EXPECT_TRUE(parent.IsConcrete());

  // Block until slice is ready.
  tsl::BlockUntilReady(slice);
  EXPECT_TRUE(slice.IsConcrete());

  // Verify slice address and size.
  EXPECT_EQ(slice->mem().opaque(), dummy_data.data() + offset);
  EXPECT_EQ(slice->mem().size(), size);
}

TEST(TrackedDeviceBufferTest, AsyncSliceFailure) {
  // Create an unconstructed parent buffer holding ForeignRawSEDeviceMemory.
  auto parent_ref =
      tsl::MakeUnconstructedAsyncValueRef<ForeignRawSEDeviceMemory>();
  tsl::AsyncValueRef<RawSEDeviceMemory> parent(parent_ref);

  EXPECT_TRUE(parent.IsUnavailable());

  // Create a slice immediately.
  size_t offset = 16;
  size_t size = 32;
  tsl::AsyncValueRef<RawSEDeviceMemory> slice =
      RawSEDeviceMemory::CreateSlice(parent, offset, size);

  EXPECT_TRUE(slice.IsUnavailable());

  // Transition the parent to an error state (simulating computation failure).
  absl::Status failure_status =
      absl::InternalError("Simulated GPU execution failure");
  parent_ref.SetError(failure_status);

  // Verify parent is in error state.
  EXPECT_TRUE(parent.IsError());
  EXPECT_EQ(parent.GetError(), failure_status);

  // Verify that the slice also transitioned to the same error state
  // gracefully.
  tsl::BlockUntilReady(slice);
  EXPECT_TRUE(slice.IsError());
  EXPECT_EQ(slice.GetError(), failure_status);
}

TEST(TrackedDeviceBufferTest, SyncSliceSuccess) {
  std::vector<char> dummy_data(128, 0xAB);
  se::DeviceAddressBase parent_address(dummy_data.data(), dummy_data.size());
  auto parent = tsl::MakeAvailableAsyncValueRef<ForeignRawSEDeviceMemory>(
      parent_address, [] {});

  EXPECT_TRUE(parent.IsConcrete());

  size_t offset = 16;
  size_t size = 32;
  tsl::AsyncValueRef<RawSEDeviceMemory> slice =
      RawSEDeviceMemory::CreateSlice(parent, offset, size);

  // Since parent is already available, the slice should also be immediately
  // available.
  EXPECT_TRUE(slice.IsConcrete());
  EXPECT_EQ(slice->mem().opaque(), dummy_data.data() + offset);
  EXPECT_EQ(slice->mem().size(), size);
}

}  // namespace
}  // namespace xla
