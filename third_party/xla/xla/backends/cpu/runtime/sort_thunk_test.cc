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

#include "xla/backends/cpu/runtime/sort_thunk.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

class SortThunkTest : public testing::TestWithParam<bool> {};

static bool LessThan(const void** data) {
  auto* lhs = reinterpret_cast<const float*>(data[0]);
  auto* rhs = reinterpret_cast<const float*>(data[1]);
  return *lhs < *rhs;
}

class LessThanComparator : public Thunk::FunctionRegistry {
 public:
  static void LessThanWrapper(bool* result, const void*, const void** data,
                              const void*, const void*, const void*) {
    *result = LessThan(data);
  }

  absl::StatusOr<Comparator> FindComparator(std::string_view name) final {
    DCHECK_EQ(name, "less_than");
    return LessThanWrapper;
  }
};

TEST_P(SortThunkTest, Sort1D) {
  bool is_stable = GetParam();

  std::vector<MaybeOwningDeviceMemory> buffers;
  std::vector<float> data = {2.0, 4.0, 1.0, 3.0};
  std::vector<int32_t> indices = {0, 1, 2, 3};

  size_t size_in_bytes = data.size() * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(data.data(), size_in_bytes));
  buffers.emplace_back(se::DeviceMemoryBase(indices.data(), size_in_bytes));

  BufferAllocations allocations(buffers);

  BufferAllocation alloc0(0, size_in_bytes, 0);
  BufferAllocation alloc1(1, size_in_bytes, 0);

  BufferAllocation::Slice slice0(&alloc0, 0, size_in_bytes);
  BufferAllocation::Slice slice1(&alloc1, 0, size_in_bytes);

  Shape data_shape = ShapeUtil::MakeShape(F32, {4});
  Shape indices_shape = ShapeUtil::MakeShape(S32, {4});

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, SortThunk::Create(
                      {"sort"}, {{slice0, data_shape}, {slice1, indices_shape}},
                      /*dimension=*/0, is_stable, LessThan));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  std::vector<float> expected_data = {1.0, 2.0, 3.0, 4.0};
  std::vector<int32_t> expected_indices = {2, 0, 3, 1};

  EXPECT_EQ(data, expected_data);
  EXPECT_EQ(indices, expected_indices);
}

TEST_P(SortThunkTest, Sort2D) {
  bool is_stable = GetParam();

  std::vector<MaybeOwningDeviceMemory> buffers;
  std::vector<float> data = {2.0, 4.0, 1.0, 3.0};
  std::vector<int32_t> indices = {0, 1, 2, 3};

  size_t size_in_bytes = data.size() * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(data.data(), size_in_bytes));
  buffers.emplace_back(se::DeviceMemoryBase(indices.data(), size_in_bytes));

  BufferAllocations allocations(buffers);

  BufferAllocation alloc0(0, size_in_bytes, 0);
  BufferAllocation alloc1(1, size_in_bytes, 0);

  BufferAllocation::Slice slice0(&alloc0, 0, size_in_bytes);
  BufferAllocation::Slice slice1(&alloc1, 0, size_in_bytes);

  Shape data_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape indices_shape = ShapeUtil::MakeShape(S32, {2, 2});

  // Sort along the dimension `0`.
  TF_ASSERT_OK_AND_ASSIGN(
      auto sort_dim0,
      SortThunk::Create({"sort"},
                        {{slice0, data_shape}, {slice1, indices_shape}},
                        /*dimension=*/0, is_stable, "less_than"));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  LessThanComparator less_than_comparator;
  params.function_registry = &less_than_comparator;

  auto execute_event0 = sort_dim0->Execute(params);
  tsl::BlockUntilReady(execute_event0);
  ASSERT_FALSE(execute_event0.IsError());

  std::vector<float> expected_data = {1.0, 3.0, 2.0, 4.0};
  std::vector<int32_t> expected_indices = {2, 3, 0, 1};

  EXPECT_EQ(data, expected_data);
  EXPECT_EQ(indices, expected_indices);

  // Reset data and indices to make it unsorted along the dimension `1`.
  data = {4.0, 3.0, 2.0, 1.0};
  indices = {0, 1, 2, 3};

  TF_ASSERT_OK_AND_ASSIGN(
      auto sort_dim1,
      SortThunk::Create({"sort"},
                        {{slice0, data_shape}, {slice1, indices_shape}},
                        /*dimension=*/1,
                        /*is_stable=*/false, "less_than"));

  auto execute_event1 = sort_dim1->Execute(params);
  tsl::BlockUntilReady(execute_event1);
  ASSERT_FALSE(execute_event1.IsError());

  expected_data = {3.0, 4.0, 1.0, 2.0};
  expected_indices = {1, 0, 3, 2};

  EXPECT_EQ(data, expected_data);
  EXPECT_EQ(indices, expected_indices);
}

TEST_P(SortThunkTest, Sort2DWithLayout) {
  bool is_stable = GetParam();

  std::vector<MaybeOwningDeviceMemory> buffers;
  std::vector<float> data = {4.0, 3.0, 2.0, 1.0};
  std::vector<int32_t> indices = {0, 1, 2, 3};

  size_t size_in_bytes = data.size() * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(data.data(), size_in_bytes));
  buffers.emplace_back(se::DeviceMemoryBase(indices.data(), size_in_bytes));

  BufferAllocations allocations(buffers);

  BufferAllocation alloc0(0, size_in_bytes, 0);
  BufferAllocation alloc1(1, size_in_bytes, 0);

  BufferAllocation::Slice slice0(&alloc0, 0, size_in_bytes);
  BufferAllocation::Slice slice1(&alloc1, 0, size_in_bytes);

  Shape data_shape = ShapeUtil::MakeShape(F32, {2, 2});
  *data_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  Shape indices_shape = ShapeUtil::MakeShape(S32, {2, 2});
  *indices_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  // Sort along the dimension `0`.
  TF_ASSERT_OK_AND_ASSIGN(
      auto sort_dim0,
      SortThunk::Create({"sort"},
                        {{slice0, data_shape}, {slice1, indices_shape}},
                        /*dimension=*/0, is_stable, "less_than"));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  LessThanComparator less_than_comparator;
  params.function_registry = &less_than_comparator;

  auto execute_event0 = sort_dim0->Execute(params);
  tsl::BlockUntilReady(execute_event0);
  ASSERT_FALSE(execute_event0.IsError());

  std::vector<float> expected_data = {3.0, 4.0, 1.0, 2.0};
  std::vector<int32_t> expected_indices = {1, 0, 3, 2};

  EXPECT_EQ(data, expected_data);
  EXPECT_EQ(indices, expected_indices);

  // Reset data and indices to make it unsorted along the dimension `1`.
  data = {2.0, 4.0, 1.0, 3.0};
  indices = {0, 1, 2, 3};

  TF_ASSERT_OK_AND_ASSIGN(
      auto sort_dim1,
      SortThunk::Create({"sort"},
                        {{slice0, data_shape}, {slice1, indices_shape}},
                        /*dimension=*/1,
                        /*is_stable=*/false, "less_than"));

  auto execute_event1 = sort_dim1->Execute(params);
  tsl::BlockUntilReady(execute_event1);
  ASSERT_FALSE(execute_event1.IsError());

  expected_data = {1.0, 3.0, 2.0, 4.0};
  expected_indices = {2, 3, 0, 1};

  EXPECT_EQ(data, expected_data);
  EXPECT_EQ(indices, expected_indices);
}

INSTANTIATE_TEST_SUITE_P(SortThunk, SortThunkTest, testing::Bool(),
                         testing::PrintToStringParamName());

}  // namespace
}  // namespace xla::cpu
