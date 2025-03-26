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

#include "xla/pjrt/tracked_device_buffer.h"

#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

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

absl::StatusOr<std::shared_ptr<TrackedDeviceBuffer>> MakeArray(
    const Shape& shape, LocalClient* client, PjRtDevice* device) {
  std::vector<tsl::RCReference<RawSEDeviceMemory>> device_buffers;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      client->backend().transfer_manager()->HostShapeToDeviceShape(shape),
      [&](const Shape& subshape, const ShapeIndex&) -> absl::Status {
        TF_ASSIGN_OR_RETURN(
            se::OwningDeviceMemory device_memory,
            client->backend().memory_allocator()->Allocate(
                /*device_ordinal=*/0,
                client->backend().transfer_manager()->GetByteSizeRequirement(
                    subshape)));
        device_buffers.push_back(RawSEDeviceMemory::Create(
            device_memory.Release(), device->local_device_id(),
            client->backend().memory_allocator()));
        return absl::OkStatus();
      }));
  return std::make_shared<TrackedDeviceBuffer>(
      device, device_buffers,
      absl::Span<const std::shared_ptr<BufferSequencingEvent>>());
}

TEST(TrackedDeviceBufferTest, AsShapedBuffer) {
  LocalClient* client = ClientLibrary::LocalClientOrDie();
  TestDevice device;

  Shape a_shape = ShapeUtil::MakeShape(F32, {3, 101, 4});
  Shape b_shape = ShapeUtil::MakeShape(S8, {77});
  Shape c_shape = ShapeUtil::MakeShape(S64, {});
  TF_ASSERT_OK_AND_ASSIGN(auto a_buffer, MakeArray(a_shape, client, &device));
  TF_ASSERT_OK_AND_ASSIGN(auto b_buffer, MakeArray(b_shape, client, &device));
  TF_ASSERT_OK_AND_ASSIGN(auto c_buffer, MakeArray(c_shape, client, &device));

  ASSERT_EQ(a_buffer->device_memory().size(), 1);
  ASSERT_EQ(b_buffer->device_memory().size(), 1);
  ASSERT_EQ(c_buffer->device_memory().size(), 1);
  std::vector<se::DeviceMemoryBase> expected_buffer_sequence = {
      a_buffer->device_memory()[0]->mem(), b_buffer->device_memory()[0]->mem(),
      c_buffer->device_memory()[0]->mem()};
  ShapedBuffer shaped_a = a_buffer->AsShapedBuffer(
      client->backend().transfer_manager()->HostShapeToDeviceShape(a_shape));
  ShapedBuffer shaped_b = b_buffer->AsShapedBuffer(
      client->backend().transfer_manager()->HostShapeToDeviceShape(b_shape));
  ShapedBuffer shaped_c = c_buffer->AsShapedBuffer(
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

TEST(TrackedDeviceBufferTest, FromScopedShapedBuffer) {
  TestDevice device;
  LocalClient* client = ClientLibrary::LocalClientOrDie();

  Literal literal = LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateFullWithDescendingLayout<float>({10, 3, 7}, 33.4f),
      LiteralUtil::One(S64));

  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer shaped_buffer,
      client->LiteralToShapedBuffer(literal, /*device_ordinal=*/0));
  std::shared_ptr<TrackedDeviceBuffer> device_buffer =
      TrackedDeviceBuffer::FromScopedShapedBuffer(&shaped_buffer, {}, &device);

  EXPECT_EQ(device_buffer->device_memory().size(),
            ShapeUtil::SubshapeCount(
                client->backend().transfer_manager()->HostShapeToDeviceShape(
                    literal.shape())));
}

}  // namespace
}  // namespace xla
