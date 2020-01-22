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

#include "tensorflow/compiler/xla/python/shared_device_buffer.h"

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace {

TEST(SharedDeviceBufferTest, MakeArray) {
  LocalClient* client = ClientLibrary::LocalClientOrDie();

  Shape shape = ShapeUtil::MakeShape(F32, {3, 101, 4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer, SharedDeviceBuffer::MakeArray(
                       shape, client->backend().transfer_manager(),
                       client->backend().memory_allocator(), 0, nullptr));
  EXPECT_EQ(buffer->children().size(), 0);
  EXPECT_EQ(buffer->device_ordinal(), 0);
  EXPECT_EQ(buffer->allocator(), client->backend().memory_allocator());
  ASSERT_EQ(buffer->device_memory().size(), 1);
  EXPECT_FALSE(buffer->device_memory()[0].is_null());
}

TEST(SharedDeviceBufferTest, MakeTuple) {
  LocalClient* client = ClientLibrary::LocalClientOrDie();

  Shape a_shape = ShapeUtil::MakeShape(F32, {3, 101, 4});
  Shape b_shape = ShapeUtil::MakeShape(S8, {77});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({a_shape, b_shape});
  TF_ASSERT_OK_AND_ASSIGN(
      auto a_buffer, SharedDeviceBuffer::MakeArray(
                         a_shape, client->backend().transfer_manager(),
                         client->backend().memory_allocator(), 0, nullptr));
  TF_ASSERT_OK_AND_ASSIGN(
      auto b_buffer, SharedDeviceBuffer::MakeArray(
                         b_shape, client->backend().transfer_manager(),
                         client->backend().memory_allocator(), 0, nullptr));
  TF_ASSERT_OK_AND_ASSIGN(
      auto tuple_buffer, SharedDeviceBuffer::MakeTuple(
                             {a_buffer, b_buffer}, tuple_shape,
                             client->backend().transfer_manager(),
                             client->backend().memory_allocator(), 0, nullptr));
  ASSERT_EQ(tuple_buffer->children().size(), 2);
  EXPECT_EQ(tuple_buffer->children()[0], a_buffer);
  EXPECT_EQ(tuple_buffer->children()[1], b_buffer);
  ASSERT_EQ(tuple_buffer->device_memory().size(), 1);
  EXPECT_EQ(tuple_buffer->device_ordinal(), 0);
  EXPECT_EQ(tuple_buffer->allocator(), client->backend().memory_allocator());
  EXPECT_FALSE(tuple_buffer->device_memory()[0].is_null());
}

TEST(SharedDeviceBufferTest, AsShapedBuffer) {
  LocalClient* client = ClientLibrary::LocalClientOrDie();

  Shape a_shape = ShapeUtil::MakeShape(F32, {3, 101, 4});
  Shape b_shape = ShapeUtil::MakeShape(S8, {77});
  Shape ab_tuple_shape = ShapeUtil::MakeTupleShape({a_shape, b_shape});
  Shape c_shape = ShapeUtil::MakeShape(S64, {});
  Shape abc_tuple_shape = ShapeUtil::MakeTupleShape({c_shape, ab_tuple_shape});
  TF_ASSERT_OK_AND_ASSIGN(
      auto a_buffer, SharedDeviceBuffer::MakeArray(
                         a_shape, client->backend().transfer_manager(),
                         client->backend().memory_allocator(), 0, nullptr));
  TF_ASSERT_OK_AND_ASSIGN(
      auto b_buffer, SharedDeviceBuffer::MakeArray(
                         b_shape, client->backend().transfer_manager(),
                         client->backend().memory_allocator(), 0, nullptr));
  TF_ASSERT_OK_AND_ASSIGN(
      auto ab_tuple_buffer,
      SharedDeviceBuffer::MakeTuple({a_buffer, b_buffer}, ab_tuple_shape,
                                    client->backend().transfer_manager(),
                                    client->backend().memory_allocator(), 0,
                                    nullptr));
  TF_ASSERT_OK_AND_ASSIGN(
      auto c_buffer, SharedDeviceBuffer::MakeArray(
                         c_shape, client->backend().transfer_manager(),
                         client->backend().memory_allocator(), 0, nullptr));
  TF_ASSERT_OK_AND_ASSIGN(
      auto abc_tuple_buffer,
      SharedDeviceBuffer::MakeTuple(
          {c_buffer, ab_tuple_buffer}, abc_tuple_shape,
          client->backend().transfer_manager(),
          client->backend().memory_allocator(), 0, nullptr));
  Shape abc_tuple_device_shape =
      client->backend().transfer_manager()->HostShapeToDeviceShape(
          abc_tuple_shape);

  ShapedBuffer shaped_buffer = abc_tuple_buffer->AsShapedBuffer(
      abc_tuple_shape, abc_tuple_device_shape, client->platform());
  EXPECT_EQ(shaped_buffer.on_host_shape(), abc_tuple_shape);
  EXPECT_EQ(shaped_buffer.on_device_shape(), abc_tuple_device_shape);

  ASSERT_EQ(a_buffer->device_memory().size(), 1);
  ASSERT_EQ(b_buffer->device_memory().size(), 1);
  ASSERT_EQ(c_buffer->device_memory().size(), 1);
  ASSERT_EQ(ab_tuple_buffer->device_memory().size(), 1);
  ASSERT_EQ(abc_tuple_buffer->device_memory().size(), 1);
  std::vector<se::DeviceMemoryBase> expected_buffer_sequence = {
      abc_tuple_buffer->device_memory()[0], c_buffer->device_memory()[0],
      ab_tuple_buffer->device_memory()[0],  a_buffer->device_memory()[0],
      b_buffer->device_memory()[0],
  };
  auto it = shaped_buffer.buffers().begin();
  auto expected_it = expected_buffer_sequence.begin();
  while (it != shaped_buffer.buffers().end()) {
    ASSERT_TRUE(expected_it != expected_buffer_sequence.end());
    EXPECT_TRUE(expected_it->IsSameAs(it->second));
    ++it;
    ++expected_it;
  }
  EXPECT_TRUE(expected_it == expected_buffer_sequence.end());
}

TEST(SharedDeviceBufferTest, FromScopedShapedBuffer) {
  LocalClient* client = ClientLibrary::LocalClientOrDie();

  Literal literal = LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateFullWithDescendingLayout<float>({10, 3, 7}, 33.4f),
      LiteralUtil::One(S64));

  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer shaped_buffer,
      client->LiteralToShapedBuffer(literal, /*device_ordinal=*/0));
  std::shared_ptr<SharedDeviceBuffer> device_buffer =
      SharedDeviceBuffer::FromScopedShapedBuffer(&shaped_buffer, nullptr);

  ASSERT_EQ(device_buffer->device_memory().size(), 1);
  ASSERT_EQ(device_buffer->children().size(), 2);

  EXPECT_EQ(device_buffer->children()[0]->device_memory().size(),
            ShapeUtil::SubshapeCount(
                client->backend().transfer_manager()->HostShapeToDeviceShape(
                    ShapeUtil::MakeShape(F32, {10, 3, 7}))));
  EXPECT_EQ(device_buffer->children()[1]->device_memory().size(),
            ShapeUtil::SubshapeCount(
                client->backend().transfer_manager()->HostShapeToDeviceShape(
                    ShapeUtil::MakeShape(S64, {}))));
}

}  // namespace
}  // namespace xla
