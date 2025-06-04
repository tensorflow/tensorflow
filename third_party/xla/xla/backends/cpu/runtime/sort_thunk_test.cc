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
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

class SortThunkTest : public testing::TestWithParam<bool> {};

// Sorts the data using only the first input (that must be float!).
static bool LessThan(const void** data) {
  auto* lhs = reinterpret_cast<const float*>(data[0]);
  auto* rhs = reinterpret_cast<const float*>(data[1]);
  return *lhs < *rhs;
}

class LessThanComparator : public FunctionLibrary {
 public:
  absl::StatusOr<void*> ResolveFunction(TypeId type_id,
                                        absl::string_view name) final {
    DCHECK_EQ(name, "less_than");
    return reinterpret_cast<void*>(LessThanWrapper);
  }

 private:
  static void LessThanWrapper(bool* result, const void*, const void** data,
                              const void*, const void*, const void*) {
    *result = LessThan(data);
  }
};

TEST_P(SortThunkTest, DescendingSortPlainArray) {
  bool is_stable = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(auto data,
                          LiteralUtil::CreateRandomLiteral<F32>(
                              ShapeUtil::MakeShape(F32, {10000}), 1.0f, 0.1f));

  BufferAllocations allocations = CreateBufferAllocations(data);
  BufferAllocation alloc = CreateBufferAllocation(0, data);
  BufferAllocation::Slice slice = CreateBufferAllocationSlice(alloc);

  // The comparator function is not used in the plain array sort when the sort
  // direction is specified and data types are supported.
  auto fake_less_than = [](const void** data) { return false; };

  // Use sort direction to activate the most efficient sorting function.
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, SortThunk::Create({"sort"}, {{slice, data.shape()}},
                                    /*dimension=*/0, is_stable, fake_less_than,
                                    SortThunk::SortDirection::kDescending));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_TRUE(std::is_sorted(data.data<float>().begin(),
                             data.data<float>().end(), std::greater<float>()));
}

TEST_P(SortThunkTest, Sort1D) {
  bool is_stable = GetParam();

  auto data = LiteralUtil::CreateR1<float>({2.0, 4.0, 1.0, 3.0});
  auto indices = LiteralUtil::CreateR1<int32_t>({0, 1, 2, 3});

  BufferAllocations allocations = CreateBufferAllocations(data, indices);

  auto [alloc0, alloc1] = CreateBufferAllocation(data, indices);
  auto [slice0, slice1] = CreateBufferAllocationSlice(alloc0, alloc1);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      SortThunk::Create({"sort"},
                        {{slice0, data.shape()}, {slice1, indices.shape()}},
                        /*dimension=*/0, is_stable, LessThan,
                        SortThunk::SortDirection::kAscending));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(data, LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, 4.0}));
  EXPECT_EQ(indices, LiteralUtil::CreateR1<int32_t>({2, 0, 3, 1}));
}

TEST_P(SortThunkTest, Sort1DDynamicNumInputs) {
  bool is_stable = GetParam();

  Literal data = LiteralUtil::CreateR1<float>(
      {17.0f, 16.0f, 5.0f,  10.0f, 30.0f, 8.0f,  9.0f,  21.0f,
       14.0f, 32.0f, 29.0f, 28.0f, 19.0f, 12.0f, 25.0f, 22.0f,
       18.0f, 35.0f, 34.0f, 23.0f, 7.0f,  13.0f, 26.0f, 33.0f,
       15.0f, 24.0f, 20.0f, 31.0f, 6.0f,  27.0f, 11.0f});

  Literal indices = LiteralUtil::CreateR1<int32_t>(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});

  // We use dummy data to create large number of input to trigger the dynamic
  // sort implementation, but we don't use it for sorting.
  TF_ASSERT_OK_AND_ASSIGN(
      Literal dummy_data,
      LiteralUtil::CreateRandomLiteral<F32>(data.shape(), 1.0f, 0.1f));

  BufferAllocations allocations =
      CreateBufferAllocations(data, indices, dummy_data);

  auto [data_alloc, indices_alloc, dummy_alloc] =
      CreateBufferAllocation(data, indices, dummy_data);
  auto [data_slice, indices_slice, dummy_slice] =
      CreateBufferAllocationSlice(data_alloc, indices_alloc, dummy_alloc);

  // We use only first input for sorting, the rest of the inputs are shuffled
  // according to the values in the `data` literal.
  std::vector<SortThunk::Input> inputs = {{data_slice, data.shape()},
                                          {indices_slice, indices.shape()}};
  inputs.resize(40, {dummy_slice, dummy_data.shape()});

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, SortThunk::Create({"sort"}, inputs,
                                    /*dimension=*/0, is_stable, LessThan,
                                    SortThunk::SortDirection::kAscending));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  auto expected_data = LiteralUtil::CreateR1<float>(
      {5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
       13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
       21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f,
       29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f});

  auto expected_indices = LiteralUtil::CreateR1<int32_t>(
      {2, 28, 20, 5,  6,  3,  30, 13, 21, 8, 24, 1, 0,  16, 12, 26,
       7, 15, 19, 25, 14, 22, 29, 11, 10, 4, 27, 9, 23, 18, 17});

  EXPECT_EQ(data, expected_data);
  EXPECT_EQ(indices, expected_indices);
}

TEST_P(SortThunkTest, Sort2D) {
  bool is_stable = GetParam();

  auto data = LiteralUtil::CreateR2<float>({{2.0, 4.0}, {1.0, 3.0}});
  auto indices = LiteralUtil::CreateR2<int32_t>({{0, 1}, {2, 3}});

  BufferAllocations allocations = CreateBufferAllocations(data, indices);

  auto [alloc0, alloc1] = CreateBufferAllocation(data, indices);
  auto [slice0, slice1] = CreateBufferAllocationSlice(alloc0, alloc1);

  // Sort along the dimension `0`.
  TF_ASSERT_OK_AND_ASSIGN(
      auto sort_dim0,
      SortThunk::Create({"sort"},
                        {{slice0, data.shape()}, {slice1, indices.shape()}},
                        /*dimension=*/0, is_stable, "less_than",
                        SortThunk::SortDirection::kAscending));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  LessThanComparator less_than_comparator;
  params.function_library = &less_than_comparator;

  auto execute_event0 = sort_dim0->Execute(params);
  tsl::BlockUntilReady(execute_event0);
  ASSERT_FALSE(execute_event0.IsError());

  EXPECT_EQ(data, LiteralUtil::CreateR2<float>({{1.0, 3.0}, {2.0, 4.0}}));
  EXPECT_EQ(indices, LiteralUtil::CreateR2<int32_t>({{2, 3}, {0, 1}}));

  // Reset data and indices to make it unsorted along the dimension `1`.
  data = LiteralUtil::CreateR2<float>({{4.0, 3.0}, {2.0, 1.0}});
  indices = LiteralUtil::CreateR2<int32_t>({{0, 1}, {2, 3}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto sort_dim1,
      SortThunk::Create({"sort"},
                        {{slice0, data.shape()}, {slice1, indices.shape()}},
                        /*dimension=*/1,
                        /*is_stable=*/false, "less_than",
                        SortThunk::SortDirection::kAscending));

  auto execute_event1 = sort_dim1->Execute(params);
  tsl::BlockUntilReady(execute_event1);
  ASSERT_FALSE(execute_event1.IsError());

  EXPECT_EQ(data, LiteralUtil::CreateR2<float>({{3.0, 4.0}, {1.0, 2.0}}));
  EXPECT_EQ(indices, LiteralUtil::CreateR2<int32_t>({{1, 0}, {3, 2}}));
}

TEST_P(SortThunkTest, Sort2DWithLayout) {
  bool is_stable = GetParam();

  auto data = LiteralUtil::CreateR2<float>({{4.0, 3.0}, {2.0, 1.0}});
  auto indices = LiteralUtil::CreateR2<int32_t>({{0, 1}, {2, 3}});

  BufferAllocations allocations = CreateBufferAllocations(data, indices);

  auto [alloc0, alloc1] = CreateBufferAllocation(data, indices);
  auto [slice0, slice1] = CreateBufferAllocationSlice(alloc0, alloc1);

  Shape data_shape = data.shape();
  *data_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  Shape indices_shape = indices.shape();
  *indices_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  // Sort along the dimension `0`.
  TF_ASSERT_OK_AND_ASSIGN(
      auto sort_dim0,
      SortThunk::Create({"sort"},
                        {{slice0, data_shape}, {slice1, indices_shape}},
                        /*dimension=*/0, is_stable, "less_than",
                        SortThunk::SortDirection::kAscending));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  LessThanComparator less_than_comparator;
  params.function_library = &less_than_comparator;

  auto execute_event0 = sort_dim0->Execute(params);
  tsl::BlockUntilReady(execute_event0);
  ASSERT_FALSE(execute_event0.IsError());

  EXPECT_EQ(data, LiteralUtil::CreateR2<float>({{3.0, 4.0}, {1.0, 2.0}}));
  EXPECT_EQ(indices, LiteralUtil::CreateR2<int32_t>({{1, 0}, {3, 2}}));

  // Reset data and indices to make it unsorted along the dimension `1`.
  data = LiteralUtil::CreateR2<float>({{2.0, 4.0}, {1.0, 3.0}});
  indices = LiteralUtil::CreateR2<int32_t>({{0, 1}, {2, 3}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto sort_dim1,
      SortThunk::Create({"sort"},
                        {{slice0, data_shape}, {slice1, indices_shape}},
                        /*dimension=*/1,
                        /*is_stable=*/false, "less_than",
                        SortThunk::SortDirection::kAscending));

  auto execute_event1 = sort_dim1->Execute(params);
  tsl::BlockUntilReady(execute_event1);
  ASSERT_FALSE(execute_event1.IsError());

  EXPECT_EQ(data, LiteralUtil::CreateR2<float>({{1.0, 3.0}, {2.0, 4.0}}));
  EXPECT_EQ(indices, LiteralUtil::CreateR2<int32_t>({{2, 3}, {0, 1}}));
}

INSTANTIATE_TEST_SUITE_P(SortThunk, SortThunkTest, testing::Bool(),
                         testing::PrintToStringParamName());

//===----------------------------------------------------------------------===//
// Performance benchmarks below.
//===----------------------------------------------------------------------===//

void BM_Sort1D(benchmark::State& state) {
  int64_t input_size = state.range(0);
  int64_t num_inputs = state.range(1);
  bool is_stable = state.range(2);
  bool sort_ascending = state.range(3);

  CHECK_GE(num_inputs, 1) << "Number of inputs must be at least 1";  // Crash OK

  auto data = LiteralUtil::CreateRandomLiteral<F32>(
      ShapeUtil::MakeShape(F32, {input_size}), 1.0f, 1.0f);
  CHECK_OK(data) << "Failed to create random literal";  // Crash OK

  // We use dummy data to create additional inputs, but we don't use it for
  // sorting and simply shuffle it according to the values in the first input.
  auto dummy_data =
      LiteralUtil::CreateRandomLiteral<F32>(data->shape(), 1.f, 1.f);
  CHECK_OK(dummy_data) << "Failed to create random literal";  // Crash OK

  // Use sort direction to activate the most efficient sorting function, or fall
  // back on the comparator functor.
  std::optional<SortThunk::SortDirection> direction;
  if (sort_ascending) direction = SortThunk::SortDirection::kAscending;

  auto [alloc, dummy_alloc] = CreateBufferAllocation(*data, *dummy_data);
  auto [slice, dummy_slice] = CreateBufferAllocationSlice(alloc, dummy_alloc);

  for (auto s : state) {
    // Clone the data to avoid sorting already sorted data.
    Literal data_copy = data->Clone();
    BufferAllocations allocations =
        CreateBufferAllocations(data_copy, *dummy_data);

    std::vector<SortThunk::Input> inputs = {{slice, data_copy.shape()}};
    inputs.resize(num_inputs, {dummy_slice, dummy_data->shape()});

    Thunk::ExecuteParams params;
    params.buffer_allocations = &allocations;

    auto thunk =
        SortThunk::Create({"sort"}, inputs,
                          /*dimension=*/0, is_stable, LessThan, direction);
    CHECK_OK(thunk) << "Failed to create sort thunk";  // Crash OK

    auto execute_event = (*thunk)->Execute(params);
    tsl::BlockUntilReady(execute_event);
    CHECK(execute_event.IsConcrete());
  }
}

BENCHMARK(BM_Sort1D)
    ->MeasureProcessCPUTime()
    ->ArgNames({"input_size", "num_inputs", "is_stable", "sort_ascending"})
    // Sort using ascending directions.
    ->Args({1000, 1, false, true})
    ->Args({1000, 2, false, true})
    ->Args({1000, 4, false, true})
    ->Args({1000, 8, false, true})
    ->Args({1000, 16, false, true})
    ->Args({1000, 32, false, true})
    // Sort using LessThan comparator.
    ->Args({1000, 1, false, false})
    ->Args({1000, 2, false, false})
    ->Args({1000, 4, false, false})
    ->Args({1000, 8, false, false})
    ->Args({1000, 16, false, false})
    ->Args({1000, 32, false, false});

}  // namespace
}  // namespace xla::cpu
