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

#include "xla/backends/cpu/runtime/xnnpack/xnn_dot_thunk.h"

#include <cstddef>
#include <vector>

#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

TEST(XnnDotThunkTest, SimpleDot) {
  std::vector<MaybeOwningDeviceMemory> buffers;

  std::vector<float> lhs = {1.0, 2.0, 3.0, 4.0};  // 2x2 matrix
  std::vector<float> rhs = {4.0, 3.0, 2.0, 1.0};  // 2x2 matrix
  std::vector<float> out(4, 0.0);                 // 2x2 matrix

  size_t size_in_bytes = lhs.size() * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(lhs.data(), size_in_bytes));
  buffers.emplace_back(se::DeviceMemoryBase(rhs.data(), size_in_bytes));
  buffers.emplace_back(se::DeviceMemoryBase(out.data(), size_in_bytes));

  BufferAllocations allocations(buffers);

  BufferAllocation lhs_alloc(0, size_in_bytes, 0);
  BufferAllocation rhs_alloc(1, size_in_bytes, 0);
  BufferAllocation out_alloc(2, size_in_bytes, 0);

  BufferAllocation::Slice lhs_slice(&lhs_alloc, 0, size_in_bytes);
  BufferAllocation::Slice rhs_slice(&rhs_alloc, 0, size_in_bytes);
  BufferAllocation::Slice out_slice(&out_alloc, 0, size_in_bytes);

  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});

  DotDimensionNumbers dot_dimensions;
  dot_dimensions.add_lhs_contracting_dimensions(1);
  dot_dimensions.add_rhs_contracting_dimensions(0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, XnnDotThunk::Create({"dot"}, dot_dimensions, lhs_slice, shape,
                                      rhs_slice, shape, out_slice, shape));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError()) << execute_event.GetError();

  std::vector<float> expected = {8.0, 5.0, 20.0, 13.0};
  EXPECT_EQ(out, expected);
}

}  // namespace
}  // namespace xla::cpu
