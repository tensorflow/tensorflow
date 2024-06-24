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

#include "xla/service/cpu/runtime/copy_thunk.h"

#include <cstddef>
#include <vector>

#include "xla/layout_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/buffer_allocations.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

TEST(CopyThunkTest, CopySameShape) {
  std::vector<MaybeOwningDeviceMemory> buffers;
  std::vector<float> src = {1.0, 2.0, 3.0, 4.0};
  std::vector<float> dst(4, 0.0);

  size_t size_in_bytes = src.size() * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(src.data(), size_in_bytes));
  buffers.emplace_back(se::DeviceMemoryBase(dst.data(), size_in_bytes));

  BufferAllocations allocations(buffers);

  BufferAllocation src_alloc(0, size_in_bytes, 0);
  BufferAllocation dst_alloc(1, size_in_bytes, 0);

  BufferAllocation::Slice src_slice(&src_alloc, 0, size_in_bytes);
  BufferAllocation::Slice dst_slice(&dst_alloc, 0, size_in_bytes);

  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      CopyThunk::Create({"copy"}, src_slice, shape, dst_slice, shape));

  Thunk::ExecuteParams params = {nullptr, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(src, dst);
}

TEST(CopyThunkTest, CopyTransposed) {
  std::vector<MaybeOwningDeviceMemory> buffers;
  std::vector<float> src = {1.0, 2.0, 3.0, 4.0};
  std::vector<float> dst(4, 0.0);

  size_t size_in_bytes = src.size() * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(src.data(), size_in_bytes));
  buffers.emplace_back(se::DeviceMemoryBase(dst.data(), size_in_bytes));

  BufferAllocations allocations(buffers);

  BufferAllocation src_alloc(0, size_in_bytes, 0);
  BufferAllocation dst_alloc(1, size_in_bytes, 0);

  BufferAllocation::Slice src_slice(&src_alloc, 0, size_in_bytes);
  BufferAllocation::Slice dst_slice(&dst_alloc, 0, size_in_bytes);

  Shape src_shape = ShapeUtil::MakeShape(F32, {2, 2});
  *src_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  Shape dst_shape = ShapeUtil::MakeShape(F32, {2, 2});

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      CopyThunk::Create({"copy"}, src_slice, src_shape, dst_slice, dst_shape));

  Thunk::ExecuteParams params = {nullptr, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  std::vector<float> expected = {1.0, 3.0, 2.0, 4.0};
  EXPECT_EQ(expected, dst);
}

}  // namespace
}  // namespace xla::cpu
