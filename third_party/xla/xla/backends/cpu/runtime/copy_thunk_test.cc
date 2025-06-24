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

#include "xla/backends/cpu/runtime/copy_thunk.h"

#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

TEST(CopyThunkTest, CopyEmptyShape) {
  auto src = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto dst = LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});

  BufferAllocations allocations = CreateBufferAllocations(src, dst);
  auto [src_alloc, dst_alloc] = CreateBufferAllocation(src, dst);

  BufferAllocation::Slice src_slice =
      CreateBufferAllocationSlice(src_alloc, 0, 0);
  BufferAllocation::Slice dst_slice =
      CreateBufferAllocationSlice(src_alloc, 0, 0);

  Shape shape = ShapeUtil::MakeShape(F32, {0, 2});

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      CopyThunk::Create({"copy"}, src_slice, shape, dst_slice, shape));

  Thunk::ExecuteParams params = {nullptr, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());
}

TEST(CopyThunkTest, CopySameShape) {
  auto src = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto dst = LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});

  BufferAllocations allocations = CreateBufferAllocations(src, dst);

  auto [src_alloc, dst_alloc] = CreateBufferAllocation(src, dst);
  auto [src_slice, dst_slice] =
      CreateBufferAllocationSlice(src_alloc, dst_alloc);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CopyThunk::Create({"copy"}, src_slice, src.shape(), dst_slice,
                                    dst.shape()));

  Thunk::ExecuteParams params = {nullptr, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(src, dst);
}

TEST(CopyThunkTest, CopyTransposed) {
  auto src = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto dst = LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});

  BufferAllocations allocations = CreateBufferAllocations(src, dst);

  auto [src_alloc, dst_alloc] = CreateBufferAllocation(src, dst);
  auto [src_slice, dst_slice] =
      CreateBufferAllocationSlice(src_alloc, dst_alloc);

  Shape transposed_shape = src.shape();
  *transposed_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CopyThunk::Create({"copy"}, src_slice, transposed_shape,
                                    dst_slice, dst.shape()));

  Thunk::ExecuteParams params = {nullptr, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(dst, LiteralUtil::CreateR2<float>({{1.0, 3.0}, {2.0, 4.0}}));
}

TEST(CopyThunkTest, CopyTransposedEmptyShape) {
  auto src = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  auto dst = LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}});

  BufferAllocations allocations = CreateBufferAllocations(src, dst);
  auto [src_alloc, dst_alloc] = CreateBufferAllocation(src, dst);

  BufferAllocation::Slice src_slice =
      CreateBufferAllocationSlice(src_alloc, 0, 0);
  BufferAllocation::Slice dst_slice =
      CreateBufferAllocationSlice(src_alloc, 0, 0);

  Shape shape = ShapeUtil::MakeShape(F32, {0, 2});

  Shape transposed_shape = shape;
  *transposed_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CopyThunk::Create({"copy"}, src_slice, transposed_shape,
                                    dst_slice, shape));

  Thunk::ExecuteParams params = {nullptr, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());
}

}  // namespace
}  // namespace xla::cpu
