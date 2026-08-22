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
#include <memory>
#include <optional>
#include <utility>
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
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

class SortThunkTest : public testing::TestWithParam<bool> {};

template <typename T>
static bool TypedLessThan(const void** data) {
  auto* lhs = reinterpret_cast<const T*>(data[0]);
  auto* rhs = reinterpret_cast<const T*>(data[1]);
  return *lhs < *rhs;
}

static bool LessThan(const void** data) { return TypedLessThan<float>(data); }

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

  ASSERT_OK_AND_ASSIGN(auto data,
                       LiteralUtil::CreateRandomLiteral<F32>(
                           ShapeUtil::MakeShape(F32, {10000}), 1.0f, 0.1f));

  BufferAllocations allocations = CreateBufferAllocations(data);
  BufferAllocation alloc = CreateBufferAllocation(0, data);
  BufferAllocation::Slice slice = CreateBufferAllocationSlice(alloc);

  // The comparator function is not used in the plain array sort when the sort
  // direction is specified and data types are supported.
  auto fake_less_than = [](const void** data) { return false; };

  // Use sort direction to activate the most efficient sorting function.
  ASSERT_OK_AND_ASSIGN(
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

TEST_P(SortThunkTest, DescendingSortPlainArrayBF16) {
  bool is_stable = GetParam();

  ASSERT_OK_AND_ASSIGN(
      auto data, LiteralUtil::CreateRandomLiteral<BF16>(
                     ShapeUtil::MakeShape(BF16, {10000}),
                     static_cast<bfloat16>(1.0f), static_cast<bfloat16>(0.1f)));

  BufferAllocations allocations = CreateBufferAllocations(data);
  BufferAllocation alloc = CreateBufferAllocation(0, data);
  BufferAllocation::Slice slice = CreateBufferAllocationSlice(alloc);

  auto fake_less_than = [](const void** data) { return false; };

  ASSERT_OK_AND_ASSIGN(
      auto thunk, SortThunk::Create({"sort"}, {{slice, data.shape()}},
                                    /*dimension=*/0, is_stable, fake_less_than,
                                    SortThunk::SortDirection::kDescending));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_TRUE(std::is_sorted(data.data<bfloat16>().begin(),
                             data.data<bfloat16>().end(),
                             std::greater<bfloat16>()));
}

TEST_P(SortThunkTest, DescendingSortPlainArrayF16) {
  bool is_stable = GetParam();

  ASSERT_OK_AND_ASSIGN(auto data,
                       LiteralUtil::CreateRandomLiteral<F16>(
                           ShapeUtil::MakeShape(F16, {10000}),
                           static_cast<half>(1.0f), static_cast<half>(0.1f)));

  BufferAllocations allocations = CreateBufferAllocations(data);
  BufferAllocation alloc = CreateBufferAllocation(0, data);
  BufferAllocation::Slice slice = CreateBufferAllocationSlice(alloc);

  auto fake_less_than = [](const void** data) { return false; };

  ASSERT_OK_AND_ASSIGN(
      auto thunk, SortThunk::Create({"sort"}, {{slice, data.shape()}},
                                    /*dimension=*/0, is_stable, fake_less_than,
                                    SortThunk::SortDirection::kDescending));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_TRUE(std::is_sorted(data.data<half>().begin(), data.data<half>().end(),
                             std::greater<half>()));
}

TEST_P(SortThunkTest, Sort1D) {
  bool is_stable = GetParam();

  auto data = LiteralUtil::CreateR1<float>({2.0, 4.0, 1.0, 3.0});
  auto indices = LiteralUtil::CreateR1<int32_t>({0, 1, 2, 3});

  BufferAllocations allocations = CreateBufferAllocations(data, indices);

  auto [alloc0, alloc1] = CreateBufferAllocation(data, indices);
  auto [slice0, slice1] = CreateBufferAllocationSlice(alloc0, alloc1);

  ASSERT_OK_AND_ASSIGN(
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
  ASSERT_OK_AND_ASSIGN(
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

  ASSERT_OK_AND_ASSIGN(auto thunk,
                       SortThunk::Create({"sort"}, inputs,
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
  ASSERT_OK_AND_ASSIGN(
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

  ASSERT_OK_AND_ASSIGN(
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
  ASSERT_OK_AND_ASSIGN(
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

  ASSERT_OK_AND_ASSIGN(
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

TEST_P(SortThunkTest, SortKeyValueBfloat16Descending) {
  bool is_stable = GetParam();

  auto data = LiteralUtil::CreateR1<bfloat16>(
      {bfloat16(2.0f), bfloat16(4.0f), bfloat16(1.0f), bfloat16(3.0f)});
  auto indices = LiteralUtil::CreateR1<int32_t>({0, 1, 2, 3});

  BufferAllocations allocations = CreateBufferAllocations(data, indices);

  auto [alloc0, alloc1] = CreateBufferAllocation(data, indices);
  auto [slice0, slice1] = CreateBufferAllocationSlice(alloc0, alloc1);

  auto fake_less_than = [](const void**) { return false; };

  ASSERT_OK_AND_ASSIGN(
      auto thunk,
      SortThunk::Create({"sort"},
                        {{slice0, data.shape()}, {slice1, indices.shape()}},
                        /*dimension=*/0, is_stable, fake_less_than,
                        SortThunk::SortDirection::kDescending));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(data,
            LiteralUtil::CreateR1<bfloat16>({bfloat16(4.0f), bfloat16(3.0f),
                                             bfloat16(2.0f), bfloat16(1.0f)}));
  EXPECT_EQ(indices, LiteralUtil::CreateR1<int32_t>({1, 3, 0, 2}));
}

TEST_P(SortThunkTest, SortKeyValueStridedSlices) {
  bool is_stable = GetParam();

  // Shape [2, 3, 2], sort along dimension 1 (inner_dim_size = 2, sort_dim_size
  // = 3)
  auto data = LiteralUtil::CreateR3<float>(
      {{{3.0f, 6.0f}, {1.0f, 4.0f}, {2.0f, 5.0f}},
       {{9.0f, 12.0f}, {7.0f, 10.0f}, {8.0f, 11.0f}}});
  auto indices = LiteralUtil::CreateR3<int32_t>(
      {{{0, 1}, {2, 3}, {4, 5}}, {{6, 7}, {8, 9}, {10, 11}}});

  BufferAllocations allocations = CreateBufferAllocations(data, indices);

  auto [alloc0, alloc1] = CreateBufferAllocation(data, indices);
  auto [slice0, slice1] = CreateBufferAllocationSlice(alloc0, alloc1);

  auto fake_less_than = [](const void**) { return false; };

  ASSERT_OK_AND_ASSIGN(
      auto thunk,
      SortThunk::Create({"sort"},
                        {{slice0, data.shape()}, {slice1, indices.shape()}},
                        /*dimension=*/1, is_stable, fake_less_than,
                        SortThunk::SortDirection::kAscending));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(data, LiteralUtil::CreateR3<float>(
                      {{{1.0f, 4.0f}, {2.0f, 5.0f}, {3.0f, 6.0f}},
                       {{7.0f, 10.0f}, {8.0f, 11.0f}, {9.0f, 12.0f}}}));
  EXPECT_EQ(indices,
            LiteralUtil::CreateR3<int32_t>(
                {{{2, 3}, {4, 5}, {0, 1}}, {{8, 9}, {10, 11}, {6, 7}}}));
}

INSTANTIATE_TEST_SUITE_P(SortThunk, SortThunkTest, testing::Bool(),
                         testing::PrintToStringParamName());

class ParallelSortThunkTest : public testing::TestWithParam<bool> {
 protected:
  ParallelSortThunkTest()
      : thread_pool_(tsl::Env::Default(), "test", 4),
        device_(thread_pool_.AsEigenThreadPool(), thread_pool_.NumThreads()) {}

  template <PrimitiveType kType>
  void RunTest(const Shape& shape, int64_t dimension) {
    using NativeT = typename primitive_util::PrimitiveTypeToNative<kType>::type;
    ASSERT_OK_AND_ASSIGN(Literal data, LiteralUtil::CreateRandomLiteral<kType>(
                                           shape, static_cast<NativeT>(1.0f),
                                           static_cast<NativeT>(0.5f)));
    ExecuteSort(data, dimension, GetParam());
    VerifySlicesAreSorted<NativeT>(data, shape, dimension);
  }

  template <PrimitiveType kKeyType, PrimitiveType kValType>
  void RunKeyValueTest(const Shape& shape, int64_t dimension,
                       SortThunk::SortDirection direction =
                           SortThunk::SortDirection::kAscending) {
    using KeyT = typename primitive_util::PrimitiveTypeToNative<kKeyType>::type;
    using ValT = typename primitive_util::PrimitiveTypeToNative<kValType>::type;

    Shape val_shape = ShapeUtil::ChangeElementType(shape, kValType);
    ASSERT_OK_AND_ASSIGN(
        Literal keys,
        LiteralUtil::CreateRandomLiteral<kKeyType>(
            shape, static_cast<KeyT>(1.0f), static_cast<KeyT>(0.5f)));
    ASSERT_OK_AND_ASSIGN(
        Literal values,
        LiteralUtil::CreateRandomLiteral<kValType>(
            val_shape, static_cast<ValT>(1), static_cast<ValT>(1)));

    Literal orig_keys = keys.Clone();
    Literal orig_values = values.Clone();

    ExecuteSortKeyValue(keys, values, dimension, GetParam(), direction);
    VerifyKeyValueSlicesAreSorted<KeyT, ValT>(
        keys, values, orig_keys, orig_values, shape, dimension, direction);
  }

 private:
  void ExecuteSort(Literal& data, int64_t dimension, bool is_stable) {
    BufferAllocations allocations = CreateBufferAllocations(data);
    BufferAllocation alloc = CreateBufferAllocation(0, data);
    BufferAllocation::Slice slice = CreateBufferAllocationSlice(alloc);

    auto fake_less_than = [](const void**) { return false; };

    ASSERT_OK_AND_ASSIGN(
        auto thunk, SortThunk::Create({"sort"}, {{slice, data.shape()}},
                                      dimension, is_stable, fake_less_than,
                                      SortThunk::SortDirection::kAscending));

    Thunk::ExecuteParams params;
    params.buffer_allocations = &allocations;
    params.intra_op_threadpool = &device_;

    auto execute_event = thunk->Execute(params);
    tsl::BlockUntilReady(execute_event);
    ASSERT_FALSE(execute_event.IsError());
  }

  void ExecuteSortKeyValue(Literal& keys, Literal& values, int64_t dimension,
                           bool is_stable, SortThunk::SortDirection direction) {
    BufferAllocations allocations = CreateBufferAllocations(keys, values);
    auto [alloc0, alloc1] = CreateBufferAllocation(keys, values);
    auto [slice0, slice1] = CreateBufferAllocationSlice(alloc0, alloc1);

    auto fake_less_than = [](const void**) { return false; };

    ASSERT_OK_AND_ASSIGN(
        auto thunk,
        SortThunk::Create({"sort"},
                          {{slice0, keys.shape()}, {slice1, values.shape()}},
                          dimension, is_stable, fake_less_than, direction));

    Thunk::ExecuteParams params;
    params.buffer_allocations = &allocations;
    params.intra_op_threadpool = &device_;

    auto execute_event = thunk->Execute(params);
    tsl::BlockUntilReady(execute_event);
    ASSERT_FALSE(execute_event.IsError());
  }

  template <typename NativeT>
  void VerifySlicesAreSorted(const Literal& data, const Shape& shape,
                             int64_t dimension) {
    int64_t outer_dim_size = 1;
    for (int64_t i = 0; i < dimension; ++i) {
      outer_dim_size *= shape.dimensions(i);
    }
    int64_t sort_dim_size = shape.dimensions(dimension);
    int64_t inner_dim_size = 1;
    for (int64_t i = dimension + 1; i < shape.dimensions().size(); ++i) {
      inner_dim_size *= shape.dimensions(i);
    }

    auto span = data.data<NativeT>();
    for (int64_t outer = 0; outer < outer_dim_size; ++outer) {
      for (int64_t inner = 0; inner < inner_dim_size; ++inner) {
        std::vector<NativeT> slice_elements;
        slice_elements.reserve(sort_dim_size);
        for (int64_t sort_idx = 0; sort_idx < sort_dim_size; ++sort_idx) {
          slice_elements.push_back(
              span[(outer * sort_dim_size + sort_idx) * inner_dim_size +
                   inner]);
        }
        EXPECT_TRUE(std::is_sorted(slice_elements.begin(), slice_elements.end(),
                                   std::less<NativeT>()));
      }
    }
  }

  template <typename KeyT, typename ValT>
  void VerifyKeyValueSlicesAreSorted(const Literal& keys, const Literal& values,
                                     const Literal& orig_keys,
                                     const Literal& orig_values,
                                     const Shape& shape, int64_t dimension,
                                     SortThunk::SortDirection direction) {
    int64_t outer_dim_size = 1;
    for (int64_t i = 0; i < dimension; ++i) {
      outer_dim_size *= shape.dimensions(i);
    }
    int64_t sort_dim_size = shape.dimensions(dimension);
    int64_t inner_dim_size = 1;
    for (int64_t i = dimension + 1; i < shape.dimensions_size(); ++i) {
      inner_dim_size *= shape.dimensions(i);
    }

    auto keys_span = keys.data<KeyT>();
    auto vals_span = values.data<ValT>();
    auto orig_keys_span = orig_keys.data<KeyT>();
    auto orig_vals_span = orig_values.data<ValT>();

    for (int64_t outer = 0; outer < outer_dim_size; ++outer) {
      for (int64_t inner = 0; inner < inner_dim_size; ++inner) {
        std::vector<KeyT> sorted_keys;
        std::vector<ValT> sorted_vals;
        std::vector<std::pair<KeyT, ValT>> orig_pairs;
        sorted_keys.reserve(sort_dim_size);
        sorted_vals.reserve(sort_dim_size);
        orig_pairs.reserve(sort_dim_size);

        for (int64_t sort_idx = 0; sort_idx < sort_dim_size; ++sort_idx) {
          int64_t idx =
              (outer * sort_dim_size + sort_idx) * inner_dim_size + inner;
          sorted_keys.push_back(keys_span[idx]);
          sorted_vals.push_back(vals_span[idx]);
          orig_pairs.emplace_back(orig_keys_span[idx], orig_vals_span[idx]);
        }

        if (direction == SortThunk::SortDirection::kAscending) {
          EXPECT_TRUE(std::is_sorted(sorted_keys.begin(), sorted_keys.end(),
                                     std::less<KeyT>()));
        } else {
          EXPECT_TRUE(std::is_sorted(sorted_keys.begin(), sorted_keys.end(),
                                     std::greater<KeyT>()));
        }

        // Verify that (key, value) pairs are preserved as an intact multiset.
        // Because unstable sort may reorder elements with equal keys
        // arbitrarily, we sort both pair vectors by (key, value) to verify
        // multiset equality without relying on specific tie-breaking.
        std::vector<std::pair<KeyT, ValT>> sorted_pairs;
        sorted_pairs.reserve(sort_dim_size);
        for (int64_t i = 0; i < sort_dim_size; ++i) {
          sorted_pairs.emplace_back(sorted_keys[i], sorted_vals[i]);
        }
        std::sort(orig_pairs.begin(), orig_pairs.end());
        std::sort(sorted_pairs.begin(), sorted_pairs.end());
        EXPECT_EQ(sorted_pairs, orig_pairs);
      }
    }
  }

  tsl::thread::ThreadPool thread_pool_;
  Eigen::ThreadPoolDevice device_;
};

TEST_P(ParallelSortThunkTest, Sort2DF32) {
  RunTest<F32>(ShapeUtil::MakeShape(F32, {32, 64}), /*dimension=*/1);
}

TEST_P(ParallelSortThunkTest, Sort3DF32) {
  RunTest<F32>(ShapeUtil::MakeShape(F32, {4, 16, 8}), /*dimension=*/1);
}

TEST_P(ParallelSortThunkTest, Sort2DBF16) {
  RunTest<BF16>(ShapeUtil::MakeShape(BF16, {32, 64}), /*dimension=*/1);
}

TEST_P(ParallelSortThunkTest, Sort2DF16) {
  RunTest<F16>(ShapeUtil::MakeShape(F16, {32, 64}), /*dimension=*/1);
}

TEST_P(ParallelSortThunkTest, Sort2DKeyValueF32S32) {
  RunKeyValueTest<F32, S32>(ShapeUtil::MakeShape(F32, {8, 2048}),
                            /*dimension=*/1);
}

TEST_P(ParallelSortThunkTest, Sort2DKeyValueBF16S32) {
  RunKeyValueTest<BF16, S32>(ShapeUtil::MakeShape(BF16, {8, 2048}),
                             /*dimension=*/1);
}

TEST_P(ParallelSortThunkTest, Sort2DKeyValueLargeSortDim) {
  RunKeyValueTest<F32, S32>(ShapeUtil::MakeShape(F32, {4, 4096}),
                            /*dimension=*/1);
}

TEST_P(ParallelSortThunkTest, Sort2DKeyValueDescending) {
  RunKeyValueTest<F32, S32>(ShapeUtil::MakeShape(F32, {8, 2048}),
                            /*dimension=*/1,
                            SortThunk::SortDirection::kDescending);
}

INSTANTIATE_TEST_SUITE_P(ParallelSortThunk, ParallelSortThunkTest,
                         testing::Bool(), testing::PrintToStringParamName());

//===----------------------------------------------------------------------===//
// Performance benchmarks below.
//===----------------------------------------------------------------------===//

template <PrimitiveType kType>
void BM_Sort1D(benchmark::State& state) {
  int64_t input_size = state.range(0);
  int64_t num_inputs = state.range(1);
  bool is_stable = state.range(2);
  bool sort_ascending = state.range(3);

  CHECK_GE(num_inputs, 1) << "Number of inputs must be at least 1";  // Crash OK

  using NativeT = typename primitive_util::PrimitiveTypeToNative<kType>::type;
  auto data_or = LiteralUtil::CreateRandomLiteral<kType>(
      ShapeUtil::MakeShape(kType, {input_size}), static_cast<NativeT>(1.0f),
      static_cast<NativeT>(1.0f));
  CHECK_OK(data_or);
  Literal data = std::move(data_or).value();

  // We use dummy data to create additional inputs, but we don't use it for
  // sorting and simply shuffle it according to the values in the first input.
  auto dummy_data_or = LiteralUtil::CreateRandomLiteral<kType>(
      data.shape(), static_cast<NativeT>(1.0f), static_cast<NativeT>(1.0f));
  CHECK_OK(dummy_data_or);
  Literal dummy_data = std::move(dummy_data_or).value();

  // Use sort direction to activate the most efficient sorting function, or fall
  // back on the comparator functor.
  std::optional<SortThunk::SortDirection> direction;
  if (sort_ascending) {
    direction = SortThunk::SortDirection::kAscending;
  }

  auto [alloc, dummy_alloc] = CreateBufferAllocation(data, dummy_data);
  auto [slice, dummy_slice] = CreateBufferAllocationSlice(alloc, dummy_alloc);

  for (auto s : state) {
    // Clone the data to avoid sorting already sorted data.
    Literal data_copy = data.Clone();
    BufferAllocations allocations =
        CreateBufferAllocations(data_copy, dummy_data);

    std::vector<SortThunk::Input> inputs = {{slice, data_copy.shape()}};
    inputs.resize(num_inputs, {dummy_slice, dummy_data.shape()});

    Thunk::ExecuteParams params;
    params.buffer_allocations = &allocations;

    auto thunk_or = SortThunk::Create({"sort"}, inputs,
                                      /*dimension=*/0, is_stable,
                                      TypedLessThan<NativeT>, direction);
    CHECK_OK(thunk_or);
    std::unique_ptr<SortThunk> thunk = std::move(thunk_or).value();

    auto execute_event = thunk->Execute(params);
    tsl::BlockUntilReady(execute_event);
    CHECK(execute_event.IsConcrete());
  }
}

BENCHMARK_TEMPLATE(BM_Sort1D, F32)
    ->MeasureProcessCPUTime()
    ->ArgNames({"input_size", "num_inputs", "is_stable", "sort_ascending"})
    // Sort using ascending directions.
    ->Args({1000, 1, false, true})
    ->Args({10000, 1, false, true})
    ->Args({100000, 1, false, true})
    // Sort using LessThan comparator callback.
    ->Args({1000, 1, false, false})
    ->Args({10000, 1, false, false})
    ->Args({100000, 1, false, false});

BENCHMARK_TEMPLATE(BM_Sort1D, BF16)
    ->MeasureProcessCPUTime()
    ->ArgNames({"input_size", "num_inputs", "is_stable", "sort_ascending"})
    // Sort using ascending directions (inlined specialized).
    ->Args({1000, 1, false, true})
    ->Args({10000, 1, false, true})
    ->Args({100000, 1, false, true})
    // Sort using LessThan comparator callback (fallback baseline).
    ->Args({1000, 1, false, false})
    ->Args({10000, 1, false, false})
    ->Args({100000, 1, false, false});

BENCHMARK_TEMPLATE(BM_Sort1D, F16)
    ->MeasureProcessCPUTime()
    ->ArgNames({"input_size", "num_inputs", "is_stable", "sort_ascending"})
    // Sort using ascending directions (inlined specialized).
    ->Args({1000, 1, false, true})
    ->Args({10000, 1, false, true})
    ->Args({100000, 1, false, true})
    // Sort using LessThan comparator callback (fallback baseline).
    ->Args({1000, 1, false, false})
    ->Args({10000, 1, false, false})
    ->Args({100000, 1, false, false});

template <PrimitiveType kType>
void BM_Sort2D(benchmark::State& state) {
  int64_t outer_dim = state.range(0);
  int64_t sort_dim = state.range(1);
  int64_t num_threads = state.range(2);

  using NativeT = typename primitive_util::PrimitiveTypeToNative<kType>::type;
  auto data_or = LiteralUtil::CreateRandomLiteral<kType>(
      ShapeUtil::MakeShape(kType, {outer_dim, sort_dim}),
      static_cast<NativeT>(1.0f), static_cast<NativeT>(1.0f));
  CHECK_OK(data_or);
  Literal data = std::move(data_or).value();

  std::optional<tsl::thread::ThreadPool> threads;
  std::optional<Eigen::ThreadPoolDevice> device;
  if (num_threads > 0) {
    threads.emplace(tsl::Env::Default(), "benchmark", num_threads);
    device.emplace(threads->AsEigenThreadPool(), threads->NumThreads());
  }

  for (auto s : state) {
    Literal data_copy = data.Clone();
    BufferAllocations allocations = CreateBufferAllocations(data_copy);
    BufferAllocation alloc = CreateBufferAllocation(0, data_copy);
    BufferAllocation::Slice slice = CreateBufferAllocationSlice(alloc);

    Thunk::ExecuteParams params;
    params.buffer_allocations = &allocations;
    if (device.has_value()) {
      params.intra_op_threadpool = &*device;
    }

    auto fake_less_than = [](const void**) { return false; };
    auto thunk_or =
        SortThunk::Create({"sort"}, {{slice, data_copy.shape()}},
                          /*dimension=*/1, /*is_stable=*/false, fake_less_than,
                          SortThunk::SortDirection::kAscending);
    CHECK_OK(thunk_or);
    std::unique_ptr<SortThunk> thunk = std::move(thunk_or).value();

    auto execute_event = thunk->Execute(params);
    tsl::BlockUntilReady(execute_event);
    CHECK(execute_event.IsConcrete());
  }
}

BENCHMARK_TEMPLATE(BM_Sort2D, F32)
    ->MeasureProcessCPUTime()
    ->ArgNames({"outer_dim", "sort_dim", "num_threads"})
    // Single-threaded baseline (num_threads = 0) vs multi-threaded (num_threads
    // = 4, 8, 16)
    ->Args({16, 1024, 0})
    ->Args({16, 1024, 4})
    ->Args({16, 1024, 8})
    ->Args({64, 1024, 0})
    ->Args({64, 1024, 4})
    ->Args({64, 1024, 8})
    ->Args({64, 4096, 0})
    ->Args({64, 4096, 4})
    ->Args({64, 4096, 8})
    ->Args({64, 4096, 16})
    ->Args({256, 4096, 0})
    ->Args({256, 4096, 4})
    ->Args({256, 4096, 8})
    ->Args({256, 4096, 16});

BENCHMARK_TEMPLATE(BM_Sort2D, BF16)
    ->MeasureProcessCPUTime()
    ->ArgNames({"outer_dim", "sort_dim", "num_threads"})
    ->Args({16, 1024, 0})
    ->Args({16, 1024, 4})
    ->Args({16, 1024, 8})
    ->Args({64, 1024, 0})
    ->Args({64, 1024, 4})
    ->Args({64, 1024, 8})
    ->Args({64, 4096, 0})
    ->Args({64, 4096, 4})
    ->Args({64, 4096, 8})
    ->Args({64, 4096, 16})
    ->Args({256, 4096, 0})
    ->Args({256, 4096, 4})
    ->Args({256, 4096, 8})
    ->Args({256, 4096, 16});

template <PrimitiveType kKeyType, PrimitiveType kValType>
void BM_SortKeyValue2D(benchmark::State& state) {
  int64_t outer_dim = state.range(0);
  int64_t sort_dim = state.range(1);
  int64_t num_threads = state.range(2);
  bool sort_ascending = state.range(3);

  using KeyT = typename primitive_util::PrimitiveTypeToNative<kKeyType>::type;
  using ValT = typename primitive_util::PrimitiveTypeToNative<kValType>::type;

  // CreateRandomLiteral samples from a normal distribution with (mean, stddev).
  auto keys_or = LiteralUtil::CreateRandomLiteral<kKeyType>(
      ShapeUtil::MakeShape(kKeyType, {outer_dim, sort_dim}),
      static_cast<KeyT>(1.0f), static_cast<KeyT>(1.0f));
  CHECK_OK(keys_or);
  Literal keys = std::move(keys_or).value();

  auto vals_or = LiteralUtil::CreateRandomLiteral<kValType>(
      ShapeUtil::MakeShape(kValType, {outer_dim, sort_dim}),
      static_cast<ValT>(1), static_cast<ValT>(1));
  CHECK_OK(vals_or);
  Literal vals = std::move(vals_or).value();

  std::optional<tsl::thread::ThreadPool> threads;
  std::optional<Eigen::ThreadPoolDevice> device;
  if (num_threads > 0) {
    threads.emplace(tsl::Env::Default(), "benchmark", num_threads);
    device.emplace(threads->AsEigenThreadPool(), threads->NumThreads());
  }

  std::optional<SortThunk::SortDirection> direction;
  if (sort_ascending) {
    direction = SortThunk::SortDirection::kAscending;
  }

  for (auto s : state) {
    Literal keys_copy = keys.Clone();
    Literal vals_copy = vals.Clone();
    BufferAllocations allocations =
        CreateBufferAllocations(keys_copy, vals_copy);
    auto [alloc0, alloc1] = CreateBufferAllocation(keys_copy, vals_copy);
    auto [slice0, slice1] = CreateBufferAllocationSlice(alloc0, alloc1);

    Thunk::ExecuteParams params;
    params.buffer_allocations = &allocations;
    if (device.has_value()) {
      params.intra_op_threadpool = &*device;
    }

    auto thunk_or = SortThunk::Create(
        {"sort"}, {{slice0, keys_copy.shape()}, {slice1, vals_copy.shape()}},
        /*dimension=*/1, /*is_stable=*/false, TypedLessThan<KeyT>, direction);
    CHECK_OK(thunk_or);
    std::unique_ptr<SortThunk> thunk = std::move(thunk_or).value();

    auto execute_event = thunk->Execute(params);
    tsl::BlockUntilReady(execute_event);
    CHECK(execute_event.IsConcrete());
  }
}

template <PrimitiveType kKeyType, PrimitiveType kValType>
void BM_SortKeyValue3D(benchmark::State& state) {
  int64_t outer_dim = state.range(0);
  int64_t sort_dim = state.range(1);
  int64_t inner_dim = state.range(2);
  int64_t num_threads = state.range(3);
  bool sort_ascending = state.range(4);

  using KeyT = typename primitive_util::PrimitiveTypeToNative<kKeyType>::type;
  using ValT = typename primitive_util::PrimitiveTypeToNative<kValType>::type;

  // CreateRandomLiteral samples from a normal distribution with (mean, stddev).
  auto keys_or = LiteralUtil::CreateRandomLiteral<kKeyType>(
      ShapeUtil::MakeShape(kKeyType, {outer_dim, sort_dim, inner_dim}),
      static_cast<KeyT>(1.0f), static_cast<KeyT>(1.0f));
  CHECK_OK(keys_or);
  Literal keys = std::move(keys_or).value();

  auto vals_or = LiteralUtil::CreateRandomLiteral<kValType>(
      ShapeUtil::MakeShape(kValType, {outer_dim, sort_dim, inner_dim}),
      static_cast<ValT>(1), static_cast<ValT>(1));
  CHECK_OK(vals_or);
  Literal vals = std::move(vals_or).value();

  std::optional<tsl::thread::ThreadPool> threads;
  std::optional<Eigen::ThreadPoolDevice> device;
  if (num_threads > 0) {
    threads.emplace(tsl::Env::Default(), "benchmark", num_threads);
    device.emplace(threads->AsEigenThreadPool(), threads->NumThreads());
  }

  std::optional<SortThunk::SortDirection> direction;
  if (sort_ascending) {
    direction = SortThunk::SortDirection::kAscending;
  }

  for (auto s : state) {
    Literal keys_copy = keys.Clone();
    Literal vals_copy = vals.Clone();
    BufferAllocations allocations =
        CreateBufferAllocations(keys_copy, vals_copy);
    auto [alloc0, alloc1] = CreateBufferAllocation(keys_copy, vals_copy);
    auto [slice0, slice1] = CreateBufferAllocationSlice(alloc0, alloc1);

    Thunk::ExecuteParams params;
    params.buffer_allocations = &allocations;
    if (device.has_value()) {
      params.intra_op_threadpool = &*device;
    }

    auto thunk_or = SortThunk::Create(
        {"sort"}, {{slice0, keys_copy.shape()}, {slice1, vals_copy.shape()}},
        /*dimension=*/1, /*is_stable=*/false, TypedLessThan<KeyT>, direction);
    CHECK_OK(thunk_or);
    std::unique_ptr<SortThunk> thunk = std::move(thunk_or).value();

    auto execute_event = thunk->Execute(params);
    tsl::BlockUntilReady(execute_event);
    CHECK(execute_event.IsConcrete());
  }
}

BENCHMARK_TEMPLATE2(BM_SortKeyValue2D, BF16, S32)
    ->MeasureProcessCPUTime()
    ->ArgNames({"outer_dim", "sort_dim", "num_threads", "sort_ascending"})
    // Single-threaded: inlined vs fallback
    ->Args({1024, 512, 0, true})
    ->Args({1024, 512, 0, false})
    ->Args({64, 4096, 0, true})
    ->Args({64, 4096, 0, false})
    // Multi-threaded (16 threads): inlined vs fallback
    ->Args({1024, 512, 16, true})
    ->Args({1024, 512, 16, false})
    ->Args({64, 4096, 16, true})
    ->Args({64, 4096, 16, false});

BENCHMARK_TEMPLATE2(BM_SortKeyValue2D, F32, S32)
    ->MeasureProcessCPUTime()
    ->ArgNames({"outer_dim", "sort_dim", "num_threads", "sort_ascending"})
    // Single-threaded: inlined vs fallback
    ->Args({1024, 512, 0, true})
    ->Args({1024, 512, 0, false})
    ->Args({64, 4096, 0, true})
    ->Args({64, 4096, 0, false})
    // Multi-threaded (16 threads): inlined vs fallback
    ->Args({1024, 512, 16, true})
    ->Args({1024, 512, 16, false})
    ->Args({64, 4096, 16, true})
    ->Args({64, 4096, 16, false});

BENCHMARK_TEMPLATE2(BM_SortKeyValue3D, F32, S32)
    ->MeasureProcessCPUTime()
    ->ArgNames({"outer_dim", "sort_dim", "inner_dim", "num_threads",
                "sort_ascending"})
    // Strided multi-threaded (16 threads): inlined vs fallback
    ->Args({16, 512, 64, 16, true})
    ->Args({16, 512, 64, 16, false})
    ->Args({4, 2048, 128, 16, true})
    ->Args({4, 2048, 128, 16, false})
    ->Args({4, 8192, 128, 16, true})
    ->Args({4, 8192, 128, 16, false});

BENCHMARK_TEMPLATE2(BM_SortKeyValue3D, BF16, S32)
    ->MeasureProcessCPUTime()
    ->ArgNames({"outer_dim", "sort_dim", "inner_dim", "num_threads",
                "sort_ascending"})
    // Strided multi-threaded (16 threads): inlined vs fallback
    ->Args({16, 512, 64, 16, true})
    ->Args({16, 512, 64, 16, false})
    ->Args({4, 2048, 128, 16, true})
    ->Args({4, 2048, 128, 16, false})
    ->Args({4, 8192, 128, 16, true})
    ->Args({4, 8192, 128, 16, false});

}  // namespace
}  // namespace xla::cpu
