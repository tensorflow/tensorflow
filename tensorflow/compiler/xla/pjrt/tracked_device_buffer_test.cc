/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/tracked_device_buffer.h"

#include <memory>

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {
namespace {

StatusOr<std::shared_ptr<TrackedDeviceBuffer>> MakeArray(const Shape& shape,
                                                         LocalClient* client) {
  std::vector<stream_executor::DeviceMemoryBase> device_buffers;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      client->backend().transfer_manager()->HostShapeToDeviceShape(shape),
      [&](const Shape& subshape, const ShapeIndex&) -> Status {
        TF_ASSIGN_OR_RETURN(
            se::OwningDeviceMemory device_memory,
            client->backend().memory_allocator()->Allocate(
                /*device_ordinal=*/0,
                client->backend().transfer_manager()->GetByteSizeRequirement(
                    subshape)));
        device_buffers.push_back(device_memory.Release());
        return ::tensorflow::OkStatus();
      }));
  return std::make_shared<TrackedDeviceBuffer>(
      client->backend().memory_allocator(), /*device_ordinal=*/0,
      device_buffers,
      absl::Span<const std::shared_ptr<BufferSequencingEvent>>(), nullptr);
}

TEST(TrackedDeviceBufferTest, AsShapedBuffer) {
  LocalClient* client = ClientLibrary::LocalClientOrDie();

  Shape a_shape = ShapeUtil::MakeShape(F32, {3, 101, 4});
  Shape b_shape = ShapeUtil::MakeShape(S8, {77});
  Shape c_shape = ShapeUtil::MakeShape(S64, {});
  TF_ASSERT_OK_AND_ASSIGN(auto a_buffer, MakeArray(a_shape, client));
  TF_ASSERT_OK_AND_ASSIGN(auto b_buffer, MakeArray(b_shape, client));
  TF_ASSERT_OK_AND_ASSIGN(auto c_buffer, MakeArray(c_shape, client));

  ASSERT_EQ(a_buffer->device_memory().size(), 1);
  ASSERT_EQ(b_buffer->device_memory().size(), 1);
  ASSERT_EQ(c_buffer->device_memory().size(), 1);
  std::vector<se::DeviceMemoryBase> expected_buffer_sequence = {
      a_buffer->device_memory()[0], b_buffer->device_memory()[0],
      c_buffer->device_memory()[0]};
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
  LocalClient* client = ClientLibrary::LocalClientOrDie();

  Literal literal = LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateFullWithDescendingLayout<float>({10, 3, 7}, 33.4f),
      LiteralUtil::One(S64));

  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer shaped_buffer,
      client->LiteralToShapedBuffer(literal, /*device_ordinal=*/0));
  std::shared_ptr<TrackedDeviceBuffer> device_buffer =
      TrackedDeviceBuffer::FromScopedShapedBuffer(&shaped_buffer, {});

  EXPECT_EQ(device_buffer->device_memory().size(),
            ShapeUtil::SubshapeCount(
                client->backend().transfer_manager()->HostShapeToDeviceShape(
                    literal.shape())));
}

}  // namespace
}  // namespace xla
