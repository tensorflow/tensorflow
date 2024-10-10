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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
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
#include "tsl/platform/test_benchmark.h"

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

TEST_P(SortThunkTest, SortPlainArray) {
  bool is_stable = GetParam();
  const int data_size = 10000;

  std::vector<MaybeOwningDeviceMemory> buffers;
  std::vector<float> data(data_size);

  std::default_random_engine gen;
  std::uniform_real_distribution<float> distribution(0.0, 1000.0);

  for (int i = 0; i < data_size; i++) {
    data[i] = distribution(gen);
  }

  const size_t size_in_bytes = data_size * sizeof(float);
  buffers.emplace_back(se::DeviceMemoryBase(data.data(), size_in_bytes));

  const BufferAllocations allocations(buffers);
  const BufferAllocation alloc(0, size_in_bytes, 0);
  const BufferAllocation::Slice slice0(&alloc, 0, size_in_bytes);
  const Shape data_shape = ShapeUtil::MakeShape(F32, {data_size});

  // 1D sort implementation has its own highly efficient comparator.
  const auto comparator = [](const void** data) { return false; };

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, SortThunk::Create({"sort"}, {{slice0, data_shape}},
                                    /*dimension=*/0, is_stable, comparator));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_TRUE(std::is_sorted(data.cbegin(), data.cend()));
}

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

TEST_P(SortThunkTest, DynamicSort1D) {
  bool is_stable = GetParam();

  // 33 empty slices + 2 slices with data = 35 slices
  // This amount of slices will call the dynamic sort implementation.
  constexpr int num_of_empty_slices = 33;
  constexpr int total_num_of_slices = num_of_empty_slices + 2;

  // size of each of 33 data buffers
  constexpr int data_size = 31;

  // values range will be [5.0, 35.0]
  constexpr float starting_value = 5.0f;

  std::array<float, data_size> data{
      17.0f, 16.0f, 5.0f,  10.0f, 30.0f, 8.0f,  9.0f,  21.0f,
      14.0f, 32.0f, 29.0f, 28.0f, 19.0f, 12.0f, 25.0f, 22.0f,
      18.0f, 35.0f, 34.0f, 23.0f, 7.0f,  13.0f, 26.0f, 33.0f,
      15.0f, 24.0f, 20.0f, 31.0f, 6.0f,  27.0f, 11.0f};
  std::array<int32_t, data_size> indices;
  std::iota(indices.begin(), indices.end(), 0);

  // This is a container for the rest of the buffers.
  std::array<uint32_t, data_size * num_of_empty_slices> empty;

  const size_t data_size_in_bytes = data.size() * sizeof(float);
  const size_t ind_size_in_bytes = indices.size() * sizeof(int32_t);
  const size_t empty_size_in_bytes = empty.size() * sizeof(uint32_t);

  const BufferAllocation alloc0(0, data_size_in_bytes, 0);
  const BufferAllocation alloc1(1, ind_size_in_bytes, 0);
  const BufferAllocation rest(2, empty_size_in_bytes, 0);

  const BufferAllocation::Slice slice0(&alloc0, 0, data_size_in_bytes);
  const BufferAllocation::Slice slice1(&alloc1, 0, ind_size_in_bytes);

  const Shape data_shape = ShapeUtil::MakeShape(F32, {data_size});
  const Shape indices_shape = ShapeUtil::MakeShape(S32, {data_size});
  const Shape rest_shape = ShapeUtil::MakeShape(U32, {data_size});

  std::vector<MaybeOwningDeviceMemory> buffers;
  buffers.emplace_back(se::DeviceMemoryBase(data.data(), data_size_in_bytes));
  buffers.emplace_back(se::DeviceMemoryBase(indices.data(), ind_size_in_bytes));
  buffers.emplace_back(se::DeviceMemoryBase(empty.data(), empty_size_in_bytes));

  BufferAllocations allocations(buffers);

  std::array<SortThunk::Input, total_num_of_slices> inputs{
      {{slice0, data_shape}, {slice1, indices_shape}}};
  for (int i = 0; i < num_of_empty_slices; ++i) {
    constexpr size_t empty_slice_in_bytes = data_size * sizeof(uint32_t);
    inputs[i + 2].slice = BufferAllocation::Slice(
        &rest, i * empty_slice_in_bytes, empty_slice_in_bytes);
    inputs[i + 2].shape = rest_shape;
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, SortThunk::Create({"sort"}, inputs,
                                    /*dimension=*/0, is_stable, LessThan));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  std::array<float, data_size> expected_data;
  std::iota(expected_data.begin(), expected_data.end(), starting_value);
  const std::array<int32_t, data_size> expected_indices{
      2, 28, 20, 5,  6,  3,  30, 13, 21, 8, 24, 1, 0,  16, 12, 26,
      7, 15, 19, 25, 14, 22, 29, 11, 10, 4, 27, 9, 23, 18, 17};

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

void BM_DynamicSort1D(::testing::benchmark::State& state, bool is_stable) {
  const int total_num_of_slices = state.range(0);
  const int num_of_empty_slices = total_num_of_slices - 2;

  // size of each of data buffers
  constexpr int data_size = 31;

  const std::array<float, data_size> data{
      17.0f, 16.0f, 5.0f,  10.0f, 30.0f, 8.0f,  9.0f,  21.0f,
      14.0f, 32.0f, 29.0f, 28.0f, 19.0f, 12.0f, 25.0f, 22.0f,
      18.0f, 35.0f, 34.0f, 23.0f, 7.0f,  13.0f, 26.0f, 33.0f,
      15.0f, 24.0f, 20.0f, 31.0f, 6.0f,  27.0f, 11.0f};
  std::array<int32_t, data_size> indices;
  std::iota(indices.begin(), indices.end(), 0);

  // This is the container for the rest of the buffers.
  std::vector<uint32_t> empty(data_size * num_of_empty_slices);

  const size_t data_size_in_bytes = data.size() * sizeof(float);
  const size_t ind_size_in_bytes = indices.size() * sizeof(int32_t);
  const size_t empty_size_in_bytes = empty.size() * sizeof(uint32_t);

  const BufferAllocation alloc0(0, data_size_in_bytes, 0);
  const BufferAllocation alloc1(1, ind_size_in_bytes, 0);
  const BufferAllocation rest(2, empty_size_in_bytes, 0);

  const BufferAllocation::Slice slice0(&alloc0, 0, data_size_in_bytes);
  const BufferAllocation::Slice slice1(&alloc1, 0, ind_size_in_bytes);

  const Shape data_shape = ShapeUtil::MakeShape(F32, {data_size});
  const Shape indices_shape = ShapeUtil::MakeShape(S32, {data_size});
  const Shape rest_shape = ShapeUtil::MakeShape(U32, {data_size});

  for (auto s : state) {
    // Pause timing to avoid counting the time spent in the setup.
    state.PauseTiming();
    auto data_clone(data);
    auto indices_clone(indices);

    std::vector<MaybeOwningDeviceMemory> buffers;
    buffers.emplace_back(
        se::DeviceMemoryBase(data_clone.data(), data_size_in_bytes));
    buffers.emplace_back(
        se::DeviceMemoryBase(indices_clone.data(), ind_size_in_bytes));
    buffers.emplace_back(
        se::DeviceMemoryBase(empty.data(), empty_size_in_bytes));

    BufferAllocations allocations(buffers);

    std::vector<SortThunk::Input> inputs(total_num_of_slices);
    inputs[0] = {slice0, data_shape};
    inputs[1] = {slice1, indices_shape};
    for (int i = 0; i < num_of_empty_slices; ++i) {
      constexpr size_t empty_slice_in_bytes = data_size * sizeof(uint32_t);
      inputs[i + 2].slice = BufferAllocation::Slice(
          &rest, i * empty_slice_in_bytes, empty_slice_in_bytes);
      inputs[i + 2].shape = rest_shape;
    }

    Thunk::ExecuteParams params;
    params.buffer_allocations = &allocations;

    state.ResumeTiming();
    TF_ASSERT_OK_AND_ASSIGN(
        auto thunk, SortThunk::Create({"sort"}, inputs,
                                      /*dimension=*/0, is_stable, LessThan));

    auto execute_event = thunk->Execute(params);
    tsl::BlockUntilReady(execute_event);
    ASSERT_FALSE(execute_event.IsError());
  }
}

void BM_SortPlainArray(::testing::benchmark::State& state, bool is_stable) {
  const int data_size = state.range(0);

  std::vector<float> data(data_size);

  std::default_random_engine gen;
  std::uniform_real_distribution<float> distribution(0.0, 1000.0);

  for (int i = 0; i < data_size; i++) {
    data[i] = distribution(gen);
  }

  const size_t size_in_bytes = data_size * sizeof(float);
  const BufferAllocation alloc(0, size_in_bytes, 0);
  const BufferAllocation::Slice slice0(&alloc, 0, size_in_bytes);
  const Shape data_shape = ShapeUtil::MakeShape(F32, {data_size});

  // 1D sort implementation has its own highly efficient comparator.
  const auto comparator = [](const void** data) { return false; };

  for (auto s : state) {
    state.PauseTiming();
    auto data_clone(data);
    std::vector<MaybeOwningDeviceMemory> buffer;
    buffer.emplace_back(se::DeviceMemoryBase(data_clone.data(), size_in_bytes));

    const BufferAllocations allocations(buffer);

    Thunk::ExecuteParams params;
    params.buffer_allocations = &allocations;

    state.ResumeTiming();
    TF_ASSERT_OK_AND_ASSIGN(
        auto thunk, SortThunk::Create({"sort"}, {{slice0, data_shape}},
                                      /*dimension=*/0, is_stable, comparator));

    auto execute_event = thunk->Execute(params);
    tsl::BlockUntilReady(execute_event);
    ASSERT_FALSE(execute_event.IsError());
  }
}

void BM_StableDynamicSort1D(::testing::benchmark::State& state) {
  BM_DynamicSort1D(state, /*is_stable=*/true);
}

void BM_UnstableDynamicSort1D(::testing::benchmark::State& state) {
  BM_DynamicSort1D(state, /*is_stable=*/false);
}

void BM_StableSortPlainArray(::testing::benchmark::State& state) {
  BM_SortPlainArray(state, /*is_stable=*/true);
}

void BM_UnstableSortPlainArray(::testing::benchmark::State& state) {
  BM_SortPlainArray(state, /*is_stable=*/false);
}

BENCHMARK(BM_StableDynamicSort1D)
    ->MeasureProcessCPUTime()
    ->Arg(35)
    ->Arg(50)
    ->Arg(100);

BENCHMARK(BM_UnstableDynamicSort1D)
    ->MeasureProcessCPUTime()
    ->Arg(35)
    ->Arg(50)
    ->Arg(100);

BENCHMARK(BM_StableSortPlainArray)
    ->MeasureProcessCPUTime()
    ->Arg(10000)
    ->Arg(100000);

BENCHMARK(BM_UnstableSortPlainArray)
    ->MeasureProcessCPUTime()
    ->Arg(10000)
    ->Arg(100000);

INSTANTIATE_TEST_SUITE_P(SortThunk, SortThunkTest, testing::Bool(),
                         testing::PrintToStringParamName());

}  // namespace
}  // namespace xla::cpu
