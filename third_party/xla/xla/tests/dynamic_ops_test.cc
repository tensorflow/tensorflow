/* Copyright 2017 The OpenXLA Authors.

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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/client/client_library.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/error_spec.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal_util.h"
#include "xla/service/platform_util.h"
#include "xla/service/service.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/types.h"

namespace xla {
namespace {

class DynamicSliceTest : public ClientLibraryTestRunnerMixin<HloTestBase> {
 protected:
  template <typename IndexT, typename DataT>
  void TestR1() {
    // Slice at dimension start.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {0}, {5}, {0, 1, 2, 3, 4});
    // Slice in the middle.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {2}, {3}, {2, 3, 4});
    // Slice at dimension boundaries.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {5}, {3}, {5, 6, 7});
    // Zero element slice.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {2}, {0}, {});
  }

  template <typename IndexT, typename DataT>
  void TestR1OOB() {
    // Slice at dimension boundaries, but with out of bounds indices.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {6}, {4}, {4, 5, 6, 7});
  }

  template <typename IndexT, typename DataT>
  void TestR2() {
    // Slice at dimension start.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {0, 0}, {2, 2},
                         {{1, 2}, {4, 5}});
    // Slice in the middle.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {1, 1}, {2, 1},
                         {{5}, {8}});
    // Slice at dimension boundaries.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {1, 1}, {2, 1},
                         {{5}, {8}});
    // Zero element slice: 2x0.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {0, 0}, {2, 0},
                         {{}, {}});
    // Zero element slice: 0x2.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {0, 0}, {0, 2},
                         Array2D<int>(0, 2));
  }

  template <typename IndexT, typename DataT>
  void TestR2OOB() {
    // Slice at dimension boundaries, but with out of bounds indices.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {1, 1}, {3, 3},
                         {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  }

  template <typename IndexT, typename DataT>
  void TestR3() {
    // R3 Shape: [2, 3, 2]
    // clang-format off

    // Slice at dimension start.
    RunR3<IndexT, DataT>(
      {{{1, 2}, {3, 4}, {5, 6}},
       {{7, 8}, {9, 10}, {11, 12}}},
      {0, 0, 0}, {2, 1, 2},
      {{{1, 2}}, {{7, 8}}});

    // Slice in the middle.
    RunR3<IndexT, DataT>(
      {{{1, 2}, {3, 4}, {5, 6}},
       {{7, 8}, {9, 10}, {11, 12}}},
      {0, 1, 1}, {2, 2, 1},
      {{{4}, {6}}, {{10}, {12}}});
    // clang-format on
  }

  template <typename IndexT, typename DataT>
  void TestR3OOB() {
    // Slice at dimension boundaries, but with out of bounds indices.
    RunR3<IndexT, DataT>(
        {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}}, {0, 2, 1},
        {2, 1, 2}, {{{5, 6}}, {{11, 12}}});
  }

  template <typename IndexT, typename DataT>
  void RunR1(absl::Span<const int> input_values_int,
             const std::vector<IndexT> slice_starts,
             const std::vector<int64_t>& slice_sizes,
             absl::Span<const int> expected_values_int) {
    // bfloat16 has explicit constructors, so it does not implicitly convert the
    // way built-in types do, which is why we can't take the parameter as an
    // Span<DataT>. We also can't convert it to a vector, because
    // vector<bool> is special so that it cannot be a Span<bool>, which
    // is what the code below wants. So instead we do this.
    Literal input_values =
        LiteralUtil::CreateR1(input_values_int)
            .Convert(primitive_util::NativeToPrimitiveType<DataT>())
            .value();
    Literal expected_values =
        std::move(LiteralUtil::CreateR1(expected_values_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());

    XlaBuilder builder(TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    XlaOp starts;
    const Literal start_data = CreateR0Parameter<IndexT>(
        slice_starts[0], 0, "slice_starts", &builder, &starts);
    // Build dynamic slice computation.
    auto input = ConstantLiteral(&builder, input_values);
    DynamicSlice(input, absl::Span<const XlaOp>({starts}), slice_sizes);
    // Run computation and compare against expected values.
    ComputeAndCompareLiteral(&builder, expected_values, {&start_data});
  }

  template <typename IndexT, typename DataT>
  void RunR2(const Array2D<int>& input_values_int,
             const std::vector<IndexT> slice_starts,
             const std::vector<int64_t>& slice_sizes,
             const Array2D<int>& expected_values_int) {
    Literal input_values =
        std::move(LiteralUtil::CreateR2FromArray2D(input_values_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());
    Literal expected_values =
        std::move(LiteralUtil::CreateR2FromArray2D(expected_values_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());

    XlaBuilder builder(TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    std::vector<XlaOp> starts(2);
    std::vector<Literal> start_data(2);
    for (int i = 0; i < 2; ++i) {
      start_data[i] = CreateR0Parameter<IndexT>(
          slice_starts[i], i, "slice_starts", &builder, &starts[i]);
    }

    // Build dynamic slice computation.
    auto input = ConstantLiteral(&builder, input_values);
    DynamicSlice(input, starts, slice_sizes);
    // Run computation and compare against expected values.
    std::vector<const Literal*> argument_ptrs;
    absl::c_transform(start_data, std::back_inserter(argument_ptrs),
                      [](const Literal& argument) { return &argument; });
    ComputeAndCompareLiteral(&builder, expected_values, argument_ptrs);
  }

  template <typename IndexT, typename DataT>
  void RunR3(const Array3D<int>& input_values_int,
             const std::vector<IndexT> slice_starts,
             const std::vector<int64_t>& slice_sizes,
             const Array3D<int>& expected_values_int) {
    Literal input_values =
        std::move(LiteralUtil::CreateR3FromArray3D(input_values_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());
    Literal expected_values =
        std::move(LiteralUtil::CreateR3FromArray3D(expected_values_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());

    XlaBuilder builder(TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    std::vector<XlaOp> starts(3);
    std::vector<Literal> start_data(3);
    for (int i = 0; i < 3; ++i) {
      start_data[i] = CreateR0Parameter<IndexT>(
          slice_starts[i], i, "slice_starts", &builder, &starts[i]);
    }
    // Build dynamic slice computation.
    auto input = ConstantLiteral(&builder, input_values);
    DynamicSlice(input, starts, slice_sizes);
    // Run computation and compare against expected values.
    std::vector<const Literal*> argument_ptrs;
    absl::c_transform(start_data, std::back_inserter(argument_ptrs),
                      [](const Literal& argument) { return &argument; });
    ComputeAndCompareLiteral(&builder, expected_values, argument_ptrs);
  }
};

TEST_F(DynamicSliceTest, Int32R1BF16) { TestR1<int32_t, bfloat16>(); }
TEST_F(DynamicSliceTest, Int32R1) { TestR1<int32_t, int32_t>(); }
TEST_F(DynamicSliceTest, Int32R1OOB) { TestR1OOB<int32_t, int32_t>(); }
TEST_F(DynamicSliceTest, Int64R1) { TestR1<int64_t, float>(); }
TEST_F(DynamicSliceTest, UInt64R1) { TestR1<uint64_t, float>(); }
TEST_F(DynamicSliceTest, UInt32R1OOB) {
  RunR1<uint32_t, int32_t>({0, 1, 2, 3, 4}, {2147483648u}, {2}, {3, 4});
}
TEST_F(DynamicSliceTest, UInt8R1) {
  std::vector<int32_t> data(129);
  absl::c_iota(data, 0);
  RunR1<uint8_t, int32_t>(data, {128}, {1}, {128});
}
TEST_F(DynamicSliceTest, Int32R2BF16) { TestR2<int32_t, bfloat16>(); }
TEST_F(DynamicSliceTest, Int32R2) { TestR2<int32_t, int32_t>(); }
TEST_F(DynamicSliceTest, Int32R2OOB) { TestR2OOB<int32_t, int32_t>(); }
TEST_F(DynamicSliceTest, Int64R2) { TestR2<int64_t, float>(); }
TEST_F(DynamicSliceTest, UInt64R2) { TestR2<uint64_t, int32_t>(); }
TEST_F(DynamicSliceTest, UInt32R2OOB) {
  RunR2<uint32_t, int32_t>({{0, 1}, {2, 3}}, {2147483648u, 0}, {1, 1}, {{2}});
}

TEST_F(DynamicSliceTest, Int32R3BF16) { TestR3<int32_t, bfloat16>(); }
TEST_F(DynamicSliceTest, Int32R3) { TestR3<int32_t, float>(); }
TEST_F(DynamicSliceTest, Int32R3OOB) { TestR3OOB<int32_t, float>(); }
TEST_F(DynamicSliceTest, Int64R3) { TestR3<int64_t, float>(); }
TEST_F(DynamicSliceTest, UInt64R3) { TestR3<uint64_t, float>(); }
TEST_F(DynamicSliceTest, UInt32R3OOB) {
  RunR3<uint32_t, int32_t>({{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}},
                           {2147483648u, 0, 2147483648u}, {1, 1, 1}, {{{5}}});
}

TEST_F(DynamicSliceTest, Int32R1Pred) {
  // Slice at dimension start.
  RunR1<int32_t, bool>({true, false, false, true, false, true, true, false},
                       {0}, {5}, {true, false, false, true, false});
  // Slice in the middle.
  RunR1<int32_t, bool>({true, false, false, true, false, true, true, false},
                       {2}, {3}, {false, true, false});
  // Slice at dimension boundaries.
  RunR1<int32_t, bool>({true, false, false, true, false, true, true, false},
                       {5}, {3}, {true, true, false});
  // Zero element slice.
  RunR1<int32_t, bool>({true, false, false, true, false, true, true, false},
                       {2}, {0}, {});
}

TEST_F(DynamicSliceTest, Int32R2Pred) {
  // Slice at dimension start.
  RunR2<int32_t, bool>(
      {{true, false, true}, {false, false, true}, {true, true, false}}, {0, 0},
      {2, 2}, {{true, false}, {false, false}});
  // Slice in the middle.
  RunR2<int32_t, bool>(
      {{true, false, true}, {false, false, true}, {true, true, false}}, {1, 1},
      {2, 1}, {{false}, {true}});
  // Slice at dimension boundaries.
  RunR2<int32_t, bool>(
      {{true, false, true}, {false, false, true}, {true, true, false}}, {1, 1},
      {2, 1}, {{false}, {true}});
  // Zero element slice: 2x0.
  RunR2<int32_t, bool>(
      {{true, false, true}, {false, false, true}, {true, true, false}}, {0, 0},
      {2, 0}, {{}, {}});
  // Zero element slice: 0x2.
  RunR2<int32_t, bool>(
      {{true, false, true}, {false, false, true}, {true, true, false}}, {0, 0},
      {0, 2}, Array2D<int>(0, 2));
}

TEST_F(DynamicSliceTest, Int32R3Pred) {
  // R3 Shape: [2, 3, 2]
  // clang-format off

  // Slice at dimension start.
  RunR3<int32_t, bool>(
    {{{true, false}, {false, true}, {true, true}},
     {{false, true}, {true, false}, {false, false}}},
    {0, 0, 0}, {2, 1, 2},
    {{{true, false}}, {{false, true}}});

  // Slice in the middle.
  RunR3<int32_t, bool>(
    {{{true, false}, {false, true}, {true, true}},
     {{false, true}, {true, false}, {false, false}}},
    {0, 1, 1}, {2, 2, 1},
    {{{true}, {true}}, {{false}, {false}}});

  // clang-format on
}

class DynamicUpdateSliceTest
    : public ClientLibraryTestRunnerMixin<HloTestBase> {
 protected:
  template <typename IndexT, typename DataT>
  void TestR0() {
    // Disable algebraic simplifier, otherwise the op will be replaced by a
    // constant.
    mutable_debug_options()->add_xla_disable_hlo_passes("algsimp");
    RunR0<IndexT, DataT>(0, 123, {}, 123);
  }

  template <typename IndexT, typename DataT>
  void TestR1() {
    // Slice at dimension start.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10}, {0},
                         {8, 9, 10, 3, 4, 5, 6, 7});
    // Slice in the middle.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10}, {2},
                         {0, 1, 8, 9, 10, 5, 6, 7});
    // Slice at dimension boundaries.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10}, {5},
                         {0, 1, 2, 3, 4, 8, 9, 10});
    // Zero-sized update.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {}, {2},
                         {0, 1, 2, 3, 4, 5, 6, 7});
  }

  template <typename IndexT, typename DataT>
  void TestR2() {
    // Slice at dimension start.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{10, 11}}, {0, 0},
                         {{10, 11, 3}, {4, 5, 6}, {7, 8, 9}});
    // Slice in the middle.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{10, 11}}, {1, 1},
                         {{1, 2, 3}, {4, 10, 11}, {7, 8, 9}});
    // Slice at dimension boundaries.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{10, 11}}, {2, 1},
                         {{1, 2, 3}, {4, 5, 6}, {7, 10, 11}});
    // Zero-sized update.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{}}, {2, 1},
                         {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  }

  template <typename IndexT, typename DataT>
  void TestR3() {
    // R3 Shape: [2, 3, 2]
    // Slice at dimension start.
    RunR3<IndexT, DataT>(
        {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}},
        {{{13, 14}, {15, 16}}, {{17, 18}, {19, 20}}}, {0, 0, 0},
        {{{13, 14}, {15, 16}, {5, 6}}, {{17, 18}, {19, 20}, {11, 12}}});
    // Slice in the middle.
    RunR3<IndexT, DataT>(
        {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}}, {{{13}, {15}}},
        {1, 1, 1}, {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 13}, {11, 15}}});
  }

  template <typename IndexT, typename DataT>
  void TestOOB() {
    // // Slice at dimension boundaries, but with out of bounds indices.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10}, {6},
                         {0, 1, 2, 3, 4, 8, 9, 10});
    // R2 Shape: [3, 3]
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{10, 11}}, {2, 2},
                         {{1, 2, 3}, {4, 5, 6}, {7, 10, 11}});
    // R3 Shape: [2, 3, 2]
    RunR3<IndexT, DataT>(
        {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}}, {{{13}, {15}}},
        {1, 2, 1}, {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 13}, {11, 15}}});
  }

  template <typename IndexT, typename DataT>
  void RunR0(int input_value_int, int update_value_int,
             const std::vector<IndexT> slice_starts, int expected_value_int) {
    Literal input_value =
        std::move(LiteralUtil::CreateR0(input_value_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());
    Literal update_value =
        std::move(LiteralUtil::CreateR0(update_value_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());
    Literal expected_value =
        std::move(LiteralUtil::CreateR0(expected_value_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());

    XlaBuilder builder(TestName());
    // Build dynamic slice computation.
    auto input = ConstantLiteral(&builder, input_value);
    auto update = ConstantLiteral(&builder, update_value);
    DynamicUpdateSlice(input, update, absl::Span<const XlaOp>({}));
    // Run computation and compare against expected values.
    ComputeAndCompareLiteral(&builder, expected_value, {});
  }

  template <typename IndexT, typename DataT>
  void RunR1(absl::Span<const int> input_values_int,
             absl::Span<const int> update_values_int,
             const std::vector<IndexT> slice_starts,
             absl::Span<const int> expected_values_int) {
    Literal input_values =
        std::move(LiteralUtil::CreateR1(input_values_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());
    Literal update_values =
        std::move(LiteralUtil::CreateR1(update_values_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());
    Literal expected_values =
        std::move(LiteralUtil::CreateR1(expected_values_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());

    XlaBuilder builder(TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    XlaOp starts;
    const Literal start_data = CreateR0Parameter<IndexT>(
        slice_starts[0], 0, "slice_starts", &builder, &starts);
    // Build dynamic slice computation.
    auto input = ConstantLiteral(&builder, input_values);
    auto update = ConstantLiteral(&builder, update_values);
    DynamicUpdateSlice(input, update, absl::Span<const XlaOp>({starts}));
    // Run computation and compare against expected values.
    ComputeAndCompareLiteral(&builder, expected_values, {&start_data});
  }

  template <typename IndexT, typename DataT>
  void RunR2(const Array2D<int>& input_values_int,
             const Array2D<int>& update_values_int,
             const std::vector<IndexT> slice_starts,
             const Array2D<int>& expected_values_int) {
    Literal input_values =
        std::move(LiteralUtil::CreateR2FromArray2D(input_values_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());
    Literal update_values =
        std::move(LiteralUtil::CreateR2FromArray2D(update_values_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());
    Literal expected_values =
        std::move(LiteralUtil::CreateR2FromArray2D(expected_values_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());

    XlaBuilder builder(TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    std::vector<XlaOp> starts(2);
    std::vector<Literal> start_data(2);
    for (int i = 0; i < 2; ++i) {
      start_data[i] = CreateR0Parameter<IndexT>(
          slice_starts[i], i, "slice_starts", &builder, &starts[i]);
    }
    // Build dynamic slice computation.
    auto input = ConstantLiteral(&builder, input_values);
    auto update = ConstantLiteral(&builder, update_values);
    DynamicUpdateSlice(input, update, starts);
    // Run computation and compare against expected values.
    std::vector<const Literal*> argument_ptrs;
    absl::c_transform(start_data, std::back_inserter(argument_ptrs),
                      [](const Literal& argument) { return &argument; });
    ComputeAndCompareLiteral(&builder, expected_values, argument_ptrs);
  }

  template <typename IndexT, typename DataT>
  void RunR3(const Array3D<int>& input_values_int,
             const Array3D<int>& update_values_int,
             const std::vector<IndexT> slice_starts,
             const Array3D<int>& expected_values_int) {
    Literal input_values =
        std::move(LiteralUtil::CreateR3FromArray3D(input_values_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());
    Literal update_values =
        std::move(LiteralUtil::CreateR3FromArray3D(update_values_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());
    Literal expected_values =
        std::move(LiteralUtil::CreateR3FromArray3D(expected_values_int)
                      .Convert(primitive_util::NativeToPrimitiveType<DataT>())
                      .value());

    XlaBuilder builder(TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    std::vector<XlaOp> starts(3);
    std::vector<Literal> start_data(3);
    for (int i = 0; i < 3; ++i) {
      start_data[i] = CreateR0Parameter<IndexT>(
          slice_starts[i], i, "slice_starts", &builder, &starts[i]);
    }

    // Build dynamic slice computation.
    auto input = ConstantLiteral(&builder, input_values);
    auto update = ConstantLiteral(&builder, update_values);
    DynamicUpdateSlice(input, update, starts);
    // Run computation and compare against expected values.
    std::vector<const Literal*> argument_ptrs;
    absl::c_transform(start_data, std::back_inserter(argument_ptrs),
                      [](const Literal& argument) { return &argument; });
    ComputeAndCompareLiteral(&builder, expected_values, argument_ptrs);
  }

  template <class T>
  void RunR3Contiguous(std::vector<int32_t> operand_shape, int32_t index,
                       int32_t size) {
    const int32_t kSeq = operand_shape[0];
    const int32_t kBatch = operand_shape[1];
    const int32_t kDim = operand_shape[2];
    Array3D<T> input_values(kSeq, kBatch, kDim);
    Array3D<T> update_values(size, kBatch, kDim);
    Array3D<T> expected_values(kSeq, kBatch, kDim);
    index = std::min(std::max(0, index), kSeq - size);

    input_values.FillIota(static_cast<T>(0));
    T value = static_cast<T>(10);
    update_values.FillIota(static_cast<T>(value));

    // TODO(b/34128753) Expected values may vary depending on backend when
    // the indices are out of bounds.
    expected_values.FillIota(static_cast<T>(0));
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < kBatch; j++) {
        for (int k = 0; k < kDim; k++) {
          expected_values(index + i, j, k) = value++;
        }
      }
    }
    if (VLOG_IS_ON(1)) {
      DumpArray<T>("input", input_values);
      DumpArray<T>("update", update_values);
      DumpArray<T>("expected", expected_values);
    }

    // Build dynamic slice computation.
    XlaBuilder builder(TestName());
    // Initialize and transfer input parameter.
    XlaOp input;
    const Literal input_data =
        CreateR3Parameter<T>(input_values, 0, "input_values", &builder, &input);
    // Initialize and transfer update parameter.
    XlaOp update;
    const Literal update_data = CreateR3Parameter<T>(
        update_values, 1, "update_values", &builder, &update);
    auto constant_index = ConstantR0<int32_t>(&builder, index);
    auto zero = ConstantR0<int32_t>(&builder, 0);
    DynamicUpdateSlice(input, update, {constant_index, zero, zero});

    // Run computation and compare against expected values.
    ComputeAndCompareR3<T>(&builder, expected_values,
                           {&input_data, &update_data}, ErrorSpec(0.000001));
  }

  template <typename NativeT>
  void DumpArray(const std::string& name, const Array3D<NativeT> values) {
    Literal literal = LiteralUtil::CreateR3FromArray3D<NativeT>(values);
    LOG(INFO) << name << ":" << literal.ToString();
  }
};

TEST_F(DynamicUpdateSliceTest, Int32R0BF16) { TestR0<int32_t, bfloat16>(); }
TEST_F(DynamicUpdateSliceTest, Int32R0) { TestR0<int32_t, float>(); }
TEST_F(DynamicUpdateSliceTest, Int64R0) { TestR0<int64_t, float>(); }
TEST_F(DynamicUpdateSliceTest, UInt64R0) { TestR0<uint64_t, float>(); }

TEST_F(DynamicUpdateSliceTest, Int32R1BF16) { TestR1<int32_t, bfloat16>(); }
TEST_F(DynamicUpdateSliceTest, Int32R1) { TestR1<int32_t, float>(); }
TEST_F(DynamicUpdateSliceTest, Int64R1) { TestR1<int64_t, float>(); }
TEST_F(DynamicUpdateSliceTest, UInt64R1) { TestR1<uint64_t, float>(); }
TEST_F(DynamicUpdateSliceTest, UInt32R1OOB) {
  RunR1<uint32_t, int32_t>({0, 1, 2, 3, 4}, {5, 6}, {2147483648u},
                           {0, 1, 2, 5, 6});
}
TEST_F(DynamicUpdateSliceTest, UInt8R1) {
  std::vector<int32_t> data(129);
  absl::c_iota(data, 0);
  std::vector<int32_t> expected = data;
  expected[128] = -1;
  RunR1<uint8_t, int32_t>(data, {-1}, {128}, expected);
}

TEST_F(DynamicUpdateSliceTest, Int32R2BF16) { TestR2<int32_t, bfloat16>(); }
TEST_F(DynamicUpdateSliceTest, Int32R2) { TestR2<int32_t, float>(); }
TEST_F(DynamicUpdateSliceTest, Int64R2) { TestR2<int64_t, int64_t>(); }
TEST_F(DynamicUpdateSliceTest, UInt64R2) { TestR2<uint64_t, int32_t>(); }
TEST_F(DynamicUpdateSliceTest, UInt32R2OOB) {
  RunR2<uint32_t, int32_t>({{0, 1}, {2, 3}}, {{4}}, {2147483648u, 0},
                           {{0, 1}, {4, 3}});
}

TEST_F(DynamicUpdateSliceTest, Int32R3BF16) { TestR3<int32_t, bfloat16>(); }
TEST_F(DynamicUpdateSliceTest, Int32R3) { TestR3<int32_t, float>(); }
TEST_F(DynamicUpdateSliceTest, Int64R3) { TestR3<int64_t, int64_t>(); }
TEST_F(DynamicUpdateSliceTest, UInt64R3) { TestR3<uint64_t, uint64_t>(); }
TEST_F(DynamicUpdateSliceTest, UInt32R3OOB) {
  RunR3<uint32_t, int32_t>({{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}}, {{{8}}},
                           {2147483648u, 0, 2147483648u},
                           {{{0, 1}, {2, 3}}, {{4, 8}, {6, 7}}});
}

TEST_F(DynamicUpdateSliceTest, Int32OOBBF16) { TestOOB<int32_t, bfloat16>(); }
TEST_F(DynamicUpdateSliceTest, Int32OOB) { TestOOB<int32_t, float>(); }
TEST_F(DynamicUpdateSliceTest, Int64OOB) { TestOOB<int64_t, int64_t>(); }
TEST_F(DynamicUpdateSliceTest, UInt64OOB) { TestOOB<uint64_t, uint64_t>(); }

TEST_F(DynamicUpdateSliceTest, Int32R1Pred) {
  // Slice at dimension start.
  RunR1<int32_t, bool>({false, false, true, true, false, true, true, false},
                       {true, true, false}, {0},
                       {true, true, false, true, false, true, true, false});
  // Slice in the middle.
  RunR1<int32_t, bool>({false, false, true, true, false, true, true, false},
                       {false, true, true}, {2},
                       {false, false, false, true, true, true, true, false});
  // Slice at dimension boundaries.
  RunR1<int32_t, bool>({false, false, true, true, false, true, true, false},
                       {false, true, true}, {5},
                       {false, false, true, true, false, false, true, true});
  // Zero-sized update.
  RunR1<int32_t, bool>({false, false, true, true, false, true, true, false}, {},
                       {2},
                       {false, false, true, true, false, true, true, false});
}

TEST_F(DynamicUpdateSliceTest, Int32R2Pred) {
  // Slice at dimension start.
  RunR2<int32_t, bool>(
      {{false, true, false}, {true, false, true}, {false, true, true}},
      {{true, false}}, {0, 0},
      {{true, false, false}, {true, false, true}, {false, true, true}});
  // Slice in the middle.
  RunR2<int32_t, bool>(
      {{false, true, false}, {true, false, true}, {false, true, true}},
      {{true, false}}, {1, 1},
      {{false, true, false}, {true, true, false}, {false, true, true}});
  // Slice at dimension boundaries.
  RunR2<int32_t, bool>(
      {{false, true, false}, {true, false, true}, {false, true, true}},
      {{true, false}}, {2, 1},
      {{false, true, false}, {true, false, true}, {false, true, false}});
  // Zero-sized update.
  RunR2<int32_t, bool>(
      {{false, true, false}, {true, false, true}, {false, true, true}}, {{}},
      {2, 1}, {{false, true, false}, {true, false, true}, {false, true, true}});
}

TEST_F(DynamicUpdateSliceTest, Int32R3Pred) {
  // R3 Shape: [2, 3, 2]
  // Slice at dimension start.
  RunR3<int32_t, bool>(
      {{{true, false}, {false, true}, {true, true}},
       {{false, false}, {false, true}, {true, false}}},
      {{{false, true}, {true, false}}, {{true, true}, {false, true}}},
      {0, 0, 0},
      {{{false, true}, {true, false}, {true, true}},
       {{true, true}, {false, true}, {true, false}}});
  // Slice in the middle.
  RunR3<int32_t, bool>({{{true, false}, {false, true}, {true, true}},
                        {{false, false}, {false, true}, {true, false}}},
                       {{{false}, {true}}}, {1, 1, 1},
                       {{{true, false}, {false, true}, {true, true}},
                        {{false, false}, {false, false}, {true, true}}});
}

// Tests for simple R3 case where the update is contiguous (i.e. the minor
// two dimensions are not sliced).
TEST_F(DynamicUpdateSliceTest, R3ContiguousSingleElement) {
  // Single element, index in-bounds
  std::vector<int32_t> operand_shape({4, 5, 2});
  RunR3Contiguous<float>(operand_shape, /*index=*/1, /*size=*/1);
}

TEST_F(DynamicUpdateSliceTest, R3ContiguousSingleElementBF16) {
  // Single element, index in-bounds
  std::vector<int32_t> operand_shape({4, 5, 2});
  RunR3Contiguous<bfloat16>(operand_shape, /*index=*/1, /*size=*/1);
}

TEST_F(DynamicUpdateSliceTest, R3ContiguousMultipleElements) {
  // Multiples element, index in-bounds.
  std::vector<int32_t> operand_shape({4, 5, 2});
  RunR3Contiguous<float>(operand_shape, /*index=*/1, /*size=*/2);
}

TEST_F(DynamicUpdateSliceTest, R3ContiguousMultipleElementsBF16) {
  // Multiples element, index in-bounds.
  std::vector<int32_t> operand_shape({4, 5, 2});
  RunR3Contiguous<bfloat16>(operand_shape, /*index=*/1, /*size=*/2);
}

TEST_F(DynamicUpdateSliceTest, R3ContiguousMultipleOOB) {
  // Multiple element, index out of bounds.
  std::vector<int32_t> operand_shape({4, 5, 2});
  RunR3Contiguous<float>(operand_shape, /*index=*/3, /*size=*/2);
}

TEST_F(DynamicUpdateSliceTest, R3ContiguousMultipleOOBBF16) {
  // Multiple element, index out of bounds.
  std::vector<int32_t> operand_shape({4, 5, 2});
  RunR3Contiguous<bfloat16>(operand_shape, /*index=*/3, /*size=*/2);
}

TEST_F(DynamicUpdateSliceTest, R3ContiguousTooLarge) {
  // Multiple element, update size larger than operand.
  std::vector<int32_t> operand_shape({4, 5, 2});
  RunR3Contiguous<float>(operand_shape, /*index=*/5, /*size=*/2);
}

TEST_F(DynamicUpdateSliceTest, R3ContiguousTooLargeBF16) {
  // Multiple element, update size larger than operand.
  std::vector<int32_t> operand_shape({4, 5, 2});
  RunR3Contiguous<bfloat16>(operand_shape, /*index=*/5, /*size=*/2);
}

TEST_F(DynamicUpdateSliceTest, R3ContiguousUnaligned) {
  std::vector<int32_t> operand_shape({3, 123, 247});
  RunR3Contiguous<float>(operand_shape, /*index=*/1, /*size=*/1);
}

TEST_F(DynamicUpdateSliceTest, R3ContiguousUnalignedBF16) {
  std::vector<int32_t> operand_shape({3, 123, 247});
  RunR3Contiguous<bfloat16>(operand_shape, /*index=*/1, /*size=*/1);
}

TEST_F(DynamicUpdateSliceTest, R3ContiguousLarger) {
  std::vector<int32_t> operand_shape({32, 128, 1024});
  RunR3Contiguous<float>(operand_shape, /*index=*/7, /*size=*/1);
}

TEST_F(DynamicUpdateSliceTest, R3ContiguousLargerBF16) {
  std::vector<int32_t> operand_shape({32, 128, 1024});
  RunR3Contiguous<bfloat16>(operand_shape, /*index=*/7, /*size=*/1);
}

// This test that buffer assignment does not alias constants with the output of
// dynamic update slice.
TEST_F(HloTestBase, AddOfDUS) {
  const char* hlo_string = R"(
  HloModule m
  test {
    o = s32[6] constant({2,3,4,5,6,7})
    i = s32[] parameter(0)
    u = s32[2] parameter(1)
    dus = s32[6] dynamic-update-slice(o,u,i)
    a = s32[6] add(dus, dus)
    j = s32[] parameter(2)
    ROOT ds = s32[2] dynamic-slice(a, j), dynamic_slice_sizes={2}
  }
  )";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{0, 0}));
}

// These tests are known to fail for backends other than GPU, so we are
// disabling them when the backend is not a GPU. These tests verify that single
// and multiple output fusions of dynamic update slices produce the right
// results. On some backends (e.g. GPU), this is done inplace.
#ifdef XLA_TEST_BACKEND_GPU
TEST_F(HloTestBase, MultipleOutputFusedDynamicUpdateSlices) {
  const char* hlo_string = R"(
HloModule MultipleInplaceDus, input_output_alias={ {0}: (0, {}), {1}: (2, {}) }

fused_computation {
  p0 = bf16[10,11,12] parameter(0)
  p1 = bf16[1,11,12] parameter(1)
  p2 = bf16[8,11,12] parameter(2)
  p3 = bf16[1,11,12] parameter(3)
  p4 = s32[] parameter(4)
  c0 = s32[] constant(0)
  cmp = pred[] compare(p4, c0), direction=EQ
  broadcast = pred[1,11,12] broadcast(cmp), dimensions={}
  select = bf16[1,11,12] select(broadcast, p1, p3)
  dus0 = bf16[10,11,12] dynamic-update-slice(p0, select, c0, c0, c0)
  dus1 = bf16[8,11,12] dynamic-update-slice(p2, select, c0, c0, c0)
  ROOT tuple = (bf16[10,11,12], bf16[8,11,12]) tuple(dus0, dus1)
}

ENTRY main {
  p0 = bf16[10,11,12] parameter(0)
  p1 = bf16[1,11,12] parameter(1)
  p2 = bf16[8,11,12] parameter(2)
  p3 = bf16[1,11,12] parameter(3)
  p4 = s32[] parameter(4)
  ROOT fusion_root_multiple = (bf16[10,11,12], bf16[8,11,12]) fusion(p0, p1, p2, p3, p4), kind=kLoop, calls=fused_computation
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_string, ErrorSpec{0, 0}));
}

TEST_F(HloTestBase,
       MultipleOutputFusedDynamicUpdateSlicesWithTransposeBitcastedRoot) {
  const char* hlo_string = R"(
HloModule MultipleInplaceDusWithTransposeBitcastToTheRoot, input_output_alias={ {0}: (0, {}), {1}: (2, {}) }

fused_computation {
  p0 = bf16[10,11,12] parameter(0)
  p1 = bf16[1,11,12] parameter(1)
  p2 = bf16[8,11,12] parameter(2)
  p3 = bf16[1,11,12] parameter(3)
  p4 = s32[] parameter(4)
  c0 = s32[] constant(0)
  cmp = pred[] compare(p4, c0), direction=EQ
  broadcast = pred[1,11,12] broadcast(cmp), dimensions={}
  select = bf16[1,11,12] select(broadcast, p1, p3)
  dus0 = bf16[10,11,12] dynamic-update-slice(p0, select, c0, c0, c0)
  bitcasted_dus0 = bf16[11,10,12] bitcast(dus0)
  dus1 = bf16[8,11,12] dynamic-update-slice(p2, select, c0, c0, c0)
  ROOT tuple = (bf16[11,10,12], bf16[8,11,12]) tuple(bitcasted_dus0, dus1)
}

ENTRY main {
  p0 = bf16[10,11,12] parameter(0)
  p1 = bf16[1,11,12] parameter(1)
  p2 = bf16[8,11,12] parameter(2)
  p3 = bf16[1,11,12] parameter(3)
  p4 = s32[] parameter(4)
  ROOT fusion_root_multiple_transpose_bitcast = (bf16[11,10,12], bf16[8,11,12]) fusion(p0, p1, p2, p3, p4), kind=kLoop, calls=fused_computation
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_string, ErrorSpec{0, 0}));
}

TEST_F(HloTestBase, SingleFusedDynamicUpdateSliceWithTransposeBitcastedRoot) {
  const char* hlo_string = R"(
HloModule SingleInplaceDusWithTransposeBitcastToTheRoot, input_output_alias={ {}: (0, {}) }

single_inplace_dus_with_transpose_bitcast {
  p0 = bf16[10,11,12] parameter(0)
  p1 = bf16[1,11,12] parameter(1)
  p2 = bf16[1,11,12] parameter(2)
  p3 = s32[] parameter(3)
  c0 = s32[] constant(0)
  cmp = pred[] compare(p3, c0), direction=EQ
  broadcast = pred[1,11,12] broadcast(cmp), dimensions={}
  select = bf16[1,11,12] select(broadcast, p1, p2)
  dus0 = bf16[10,11,12] dynamic-update-slice(p0, select, c0, c0, c0)
  ROOT bitcasted_dus0 = bf16[11,10,12] bitcast(dus0)
}

ENTRY main {
  p0 = bf16[10,11,12] parameter(0)
  p1 = bf16[1,11,12] parameter(1)
  p2 = bf16[1,11,12] parameter(2)
  p3 = s32[] parameter(3)
  ROOT fusion_root_transpose_bitcast = bf16[11,10,12] fusion(p0, p1, p2, p3), kind=kLoop, calls=single_inplace_dus_with_transpose_bitcast
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_string, ErrorSpec{0, 0}));
}

TEST_F(HloTestBase, SingleFusedDynamicUpdateSliceWithReshapeBitcastedRoot) {
  const char* hlo_string = R"(
HloModule SingleInplaceDusWithReshapeBitcastToTheRoot, input_output_alias={ {}: (0, {}) }

single_inplace_dus_with_reshape_bitcast {
  p0 = bf16[10,11,12] parameter(0)
  p1 = bf16[1,11,12] parameter(1)
  p2 = bf16[1,11,12] parameter(2)
  p3 = s32[] parameter(3)
  c0 = s32[] constant(0)
  cmp = pred[] compare(p3, c0), direction=EQ
  broadcast = pred[1,11,12] broadcast(cmp), dimensions={}
  select = bf16[1,11,12] select(broadcast, p1, p2)
  dus0 = bf16[10,11,12] dynamic-update-slice(p0, select, c0, c0, c0)
  ROOT bitcasted_dus0 = bf16[10,11,6,2] bitcast(dus0)
}

ENTRY main {
  p0 = bf16[10,11,12] parameter(0)
  p1 = bf16[1,11,12] parameter(1)
  p2 = bf16[1,11,12] parameter(2)
  p3 = s32[] parameter(3)
  ROOT fusion_root_reshape_bitcast = bf16[10,11,6,2] fusion(p0, p1, p2, p3), kind=kLoop, calls=single_inplace_dus_with_reshape_bitcast
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_string, ErrorSpec{0, 0}));
}

TEST_F(HloTestBase,
       SingleFusedDynamicUpdateSliceWithBitcastedRootAndParameter) {
  const char* hlo_string = R"(
HloModule SingleInplaceDusWithBitcastToTheRootAndFromTheParameter, input_output_alias={ {}: (0, {}) }

single_inplace_dus_with_bitcast_to_the_root_and_from_the_parameter {
  p0 = bf16[10,11,12] parameter(0)
  p1 = bf16[1,11,12] parameter(1)
  p2 = bf16[1,11,12] parameter(2)
  p3 = s32[] parameter(3)
  c0 = s32[] constant(0)
  cmp = pred[] compare(p3, c0), direction=EQ
  broadcast = pred[1,11,12] broadcast(cmp), dimensions={}
  select = bf16[1,11,12] select(broadcast, p1, p2)
  bitcasted_p0 = bf16[10,6,2,11] bitcast(p0)
  bitcasted_select = bf16[1,6,2,11] bitcast(select)
  dus0 = bf16[10,6,2,11] dynamic-update-slice(bitcasted_p0, bitcasted_select, c0, c0, c0, c0)
  ROOT bitcasted_dus0 = bf16[10,11,6,2] bitcast(dus0)
}

ENTRY main {
  p0 = bf16[10,11,12] parameter(0)
  p1 = bf16[1,11,12] parameter(1)
  p2 = bf16[1,11,12] parameter(2)
  p3 = s32[] parameter(3)
  ROOT fusion_root_bitcast_both_ways = bf16[10,11,6,2] fusion(p0, p1, p2, p3), kind=kLoop, calls=single_inplace_dus_with_bitcast_to_the_root_and_from_the_parameter
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_string, ErrorSpec{0, 0}));
}

TEST_F(HloTestBase, SingleFusedDynamicUpdateSliceWithSameDynamicSliceAccess) {
  const char* hlo_string = R"(
HloModule fusion, input_output_alias={ {}: (0, {}) }

fused_computation {
  param_0.1 = s32[6]{0} parameter(0)
  bitcast = s32[2,3]{1,0} bitcast(param_0.1)
  zero = s32[] constant(0)
  param_1.1 = s32[] parameter(1)
  dynamic-slice = s32[1,1]{1,0} dynamic-slice(bitcast, param_1.1, zero), dynamic_slice_sizes={1,1}
  one = s32[] constant(1)
  bitcasted_one = s32[1,1]{1,0} bitcast(one)
  add = s32[1,1] add(dynamic-slice, bitcasted_one)
  dynamic-update-slice = s32[2,3]{1,0} dynamic-update-slice(bitcast, add, param_1.1, zero)
  ROOT bitcast.1 = s32[6]{0} bitcast(dynamic-update-slice)
}

ENTRY main {
  param_0 = s32[6]{0} parameter(0)
  param_1 = s32[] parameter(1)
  ROOT fusion = s32[6]{0} fusion(param_0, param_1), kind=kInput, calls=fused_computation
}
)";
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_string, ErrorSpec{0, 0}));
}

TEST_F(HloTestBase,
       SingleFusedDynamicUpdateSliceWithDynamicSliceAccessSlicesOfSizeOne) {
  const char* hlo_string = R"(
HloModule fusion, input_output_alias={ {}: (0, {}) }

fused_computation {
  param_0.1 = s32[6]{0} parameter(0)
  bitcast = s32[2,3]{1,0} bitcast(param_0.1)
  zero = s32[] constant(0)
  param_1.1 = s32[] parameter(1)
  dynamic-slice = s32[1,1]{1,0} dynamic-slice(bitcast, zero, param_1.1), dynamic_slice_sizes={1,1}
  one = s32[] constant(1)
  bitcasted_one = s32[1,1]{1,0} bitcast(one)
  add = s32[1,1] add(dynamic-slice, bitcasted_one)
  dynamic-update-slice = s32[2,3]{1,0} dynamic-update-slice(bitcast, add, param_1.1, zero)
  ROOT bitcast.1 = s32[6]{0} bitcast(dynamic-update-slice)
}

ENTRY main {
  param_0 = s32[6]{0} parameter(0)
  param_1 = s32[] parameter(1)
  ROOT fusion = s32[6]{0} fusion(param_0, param_1), kind=kInput, calls=fused_computation
}
)";
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_string, ErrorSpec{0, 0}));
}
#endif

void BM_DynamicSlice(::testing::benchmark::State& state) {
  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  auto executors = PlatformUtil::GetStreamExecutors(platform).value();
  se::StreamExecutorMemoryAllocator allocator(platform, executors);
  LocalClient* client = ClientLibrary::GetOrCreateLocalClient(platform).value();
  auto* transfer_manager = TransferManager::GetForPlatform(platform).value();
  int device_ordinal = client->default_device_ordinal();

  XlaBuilder builder("DynamicSlice");

  // Create input as a constant: shape [1, 2, 3, 4]
  auto input_literal = LiteralUtil::CreateR4(
      {{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
        {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}});
  auto input = ConstantLiteral(&builder, input_literal);

  auto stream = client->mutable_backend()->BorrowStream(device_ordinal).value();

  // Create dynamic slice start indices as a parameter: shape [4]
  auto start_indices_shape = ShapeUtil::MakeShape(S32, {});
  std::vector<XlaOp> start_indices(4);
  std::vector<ScopedShapedBuffer> shaped_buffers;
  std::vector<const Shape*> host_shapes(4);
  for (int i = 0; i < 4; ++i) {
    start_indices[i] =
        Parameter(&builder, i, start_indices_shape, "start_indices");
    auto start_index_literal = LiteralUtil::CreateR0<int32_t>(i + 1);
    // Initialize and transfer parameter buffer.
    shaped_buffers.emplace_back(
        client->backend()
            .transfer_manager()
            ->AllocateScopedShapedBuffer(start_indices_shape, &allocator,
                                         /*device_ordinal=*/0)
            .value());
    host_shapes[i] = &shaped_buffers[i].on_host_shape();
    ASSERT_IS_OK(transfer_manager->TransferLiteralToDevice(
        stream.get(), start_index_literal, shaped_buffers[i]));
  }

  // Add DynamicSlice op to the computation.
  DynamicSlice(input, start_indices, {1, 1, 1, 1});
  auto computation = builder.Build().value();

  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      client->Compile(computation, host_shapes, ExecutableBuildOptions()));
  auto executable = std::move(executables[0]);

  // Run some warm-up executions.
  ExecutableRunOptions options;
  options.set_allocator(&allocator);
  const int kWarmups = 2;
  std::vector<const ShapedBuffer*> shaped_buffer_ptrs;
  absl::c_transform(shaped_buffers, std::back_inserter(shaped_buffer_ptrs),
                    [](const ScopedShapedBuffer& buffer) { return &buffer; });

  for (int i = 0; i < kWarmups; ++i) {
    auto result = executable->Run(shaped_buffer_ptrs, options);
    ASSERT_TRUE(result.ok());
  }

  // Run benchmark.
  for (auto s : state) {
    auto result = executable->Run(shaped_buffer_ptrs, options);
    ASSERT_TRUE(result.ok());
  }
}
BENCHMARK(BM_DynamicSlice);

}  // namespace
}  // namespace xla
