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

// Tests that multi-dimensional arrays can be reduced among various
// user-provided dimensions.
//
// Note that comments for these tests are white-box in that they talk about the
// default data layout.
//
// The test space for reductions is the cartesian product of:
//
//    <possible ranks> x
//    <possible layouts for chosen rank> x
//    <possible subsets of dimensions in chosen rank>

#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/array4d.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/reference_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using FuncGeneratorForType = XlaComputation (*)(PrimitiveType, XlaBuilder*);

using FuncGenerator = XlaComputation (*)(XlaBuilder*);

class ReduceTest : public ClientLibraryTestBase {
 protected:
  ReduceTest() {
    // Implementation note: laid out z >> y >> x by default.
    // clang-format off
    literal_2d_ = LiteralUtil::CreateR2<float>({
      // x0   x1   x2
      { 1.f, 2.f, 3.f},  // y0
      { 4.f, 5.f, 6.f},  // y1
    });
    literal_3d_ = LiteralUtil::CreateR3Projected<float>({
      // x0   x1   x2
      { 1.f, 2.f, 3.f},  // y0
      { 4.f, 5.f, 6.f},  // y1
    }, 4);
    // clang-format on
    CHECK(ShapeUtil::Equal(
        literal_3d_.shape(),
        ShapeUtil::MakeShape(F32, {/*z=*/4, /*y=*/2, /*x=*/3})))
        << literal_3d_.shape().ToString();
  }

  // Runs an R1 => R0 reduction test with the given number of elements.
  void RunR1ToR0Test(int64_t element_count) {
    XlaBuilder builder(TestName());
    XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
    const Shape input_shape = ShapeUtil::MakeShape(F32, {element_count});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto zero = ConstantR0<float>(&builder, 0.0);
    Reduce(input, zero, add_f32, /*dimensions_to_reduce=*/{0});
    std::minstd_rand rng(seed_);

    std::vector<float> input_data(element_count);
    for (int64_t i = 0; i < element_count; ++i) {
      input_data[i] = rng() % 3;
      if (rng() % 2 == 0) {
        input_data[i] *= -1;
      }
    }
    Literal input_literal =
        LiteralUtil::CreateR1(absl::MakeConstSpan(input_data));
    std::unique_ptr<GlobalData> input_global_data =
        client_->TransferToServer(input_literal).value();

    float expected = absl::c_accumulate(input_data, 0.0f);
    ComputeAndCompareR0<float>(&builder, expected, {input_global_data.get()},
                               ErrorSpec(0.001));
  }

  void RunR1ToR0PredTest(bool and_reduce, absl::Span<const int> input_data) {
    const int element_count = input_data.size();
    XlaBuilder builder(TestName());
    const Shape input_shape = ShapeUtil::MakeShape(S32, {element_count});
    auto input_par = Parameter(&builder, 0, input_shape, "input");
    auto pred_values =
        Eq(input_par, ConstantR1<int>(&builder, element_count, 1));
    XlaOp init_value;
    XlaComputation reduce;
    if (and_reduce) {
      init_value = ConstantR0<bool>(&builder, true);
      reduce = CreateScalarAndComputation(PRED, &builder);
    } else {
      init_value = ConstantR0<bool>(&builder, false);
      reduce = CreateScalarOrComputation(PRED, &builder);
    }
    Reduce(pred_values, init_value, reduce,
           /*dimensions_to_reduce=*/{0});

    Literal input_literal = LiteralUtil::CreateR1(input_data);
    std::unique_ptr<GlobalData> input_global_data =
        client_->TransferToServer(input_literal).value();

    bool expected = and_reduce;
    for (bool item : input_data) {
      if (and_reduce) {
        expected = expected && item;
      } else {
        expected = expected || item;
      }
    }
    ComputeAndCompareR0<bool>(&builder, expected, {input_global_data.get()});
  }

  // Reduce predicate tensor with dimension rows * cols to dimension cols, to
  // test the implementation of atomic operations on misaligned small data
  // types.
  template <int64_t cols>
  void RunR2ToR1PredTest(bool and_reduce, int64_t rows, int64_t minor = 1,
                         int64_t major = 0) {
    XlaBuilder builder(TestName());
    const Shape input_shape = ShapeUtil::MakeShape(U8, {rows, cols});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto input_pred = Eq(input, ConstantR0<uint8_t>(&builder, 1));

    XlaOp init_value;
    XlaComputation reduce_op;
    if (and_reduce) {
      init_value = ConstantR0<bool>(&builder, true);
      reduce_op = CreateScalarAndComputation(PRED, &builder);
    } else {
      init_value = ConstantR0<bool>(&builder, false);
      reduce_op = CreateScalarOrComputation(PRED, &builder);
    }

    Reduce(input_pred, init_value, reduce_op,
           /*dimensions_to_reduce=*/{0});

    Array2D<uint8_t> input_data(rows, cols);
    input_data.FillRandom(0, 1);
    Literal input_literal = LiteralUtil::CreateR2FromArray2D(input_data);
    input_literal =
        input_literal.Relayout(LayoutUtil::MakeLayout({minor, major}));
    std::unique_ptr<GlobalData> input_global_data =
        client_->TransferToServer(input_literal).value();

    std::array<bool, cols> expected;
    for (int64_t colno = 0; colno < cols; ++colno) {
      bool column_sum = and_reduce ? true : false;
      for (int64_t rowno = 0; rowno < rows; ++rowno) {
        if (and_reduce) {
          column_sum = column_sum && input_data(rowno, colno);
        } else {
          column_sum = column_sum || input_data(rowno, colno);
        }
      }
      expected[colno] = column_sum;
    }

    ComputeAndCompareR1<bool>(&builder, expected, {input_global_data.get()});
  }

  // Runs an R2 => R0 reduction test with the given number of (rows, cols).
  void RunR2ToR0Test(int64_t rows, int64_t cols, int64_t minor = 1,
                     int64_t major = 0) {
    XlaBuilder builder(TestName());
    XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
    const Shape input_shape = ShapeUtil::MakeShape(F32, {rows, cols});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto zero = ConstantR0<float>(&builder, 0.0);
    Reduce(input, zero, add_f32, /*dimensions_to_reduce=*/{0, 1});

    Array2D<float> input_data(rows, cols);
    input_data.FillRandom(3.14f, 0.04);
    Literal input_literal = LiteralUtil::CreateR2FromArray2D(input_data);
    input_literal =
        input_literal.Relayout(LayoutUtil::MakeLayout({minor, major}));
    std::unique_ptr<GlobalData> input_global_data =
        client_->TransferToServer(input_literal).value();

    float expected = 0.0;
    for (int64_t rowno = 0; rowno < rows; ++rowno) {
      for (int64_t colno = 0; colno < cols; ++colno) {
        expected += input_data(rowno, colno);
      }
    }
    ComputeAndCompareR0<float>(&builder, expected, {input_global_data.get()},
                               ErrorSpec(0.01, 1e-4));
  }

  // Runs an R2 => R1 reduction test with the given number of (rows, cols).
  void RunR2ToR1Test(int64_t rows, int64_t cols, int64_t minor = 1,
                     int64_t major = 0) {
    XlaBuilder builder(TestName());
    XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
    const Shape input_shape = ShapeUtil::MakeShape(F32, {rows, cols});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto zero = ConstantR0<float>(&builder, 0.0);
    Reduce(input, zero, add_f32, /*dimensions_to_reduce=*/{0});

    Array2D<float> input_data(rows, cols);
    input_data.FillRandom(3.14f, 0.04);
    Literal input_literal = LiteralUtil::CreateR2FromArray2D(input_data);
    input_literal =
        input_literal.Relayout(LayoutUtil::MakeLayout({minor, major}));
    std::unique_ptr<GlobalData> input_global_data =
        client_->TransferToServer(input_literal).value();

    std::vector<float> expected;
    expected.reserve(cols);
    for (int64_t colno = 0; colno < cols; ++colno) {
      float column_sum = 0;
      for (int64_t rowno = 0; rowno < rows; ++rowno) {
        column_sum += input_data(rowno, colno);
      }
      expected.push_back(column_sum);
    }
    ComputeAndCompareR1<float>(&builder, expected, {input_global_data.get()},
                               ErrorSpec(0.01, 1e-4));
  }

  template <typename NativeT>
  void ComputeAndCompareGeneric(
      typename std::enable_if<std::is_floating_point<NativeT>::value,
                              XlaBuilder>::type* builder,
      absl::Span<const NativeT> expected,
      absl::Span<GlobalData* const> arguments) {
    ComputeAndCompareR1<NativeT>(builder, expected, arguments,
                                 ErrorSpec(0.01, 1e-4));
  }

  template <typename NativeT>
  void ComputeAndCompareGeneric(
      typename std::enable_if<std::is_integral<NativeT>::value,
                              XlaBuilder>::type* builder,
      absl::Span<const NativeT> expected,
      absl::Span<GlobalData* const> arguments) {
    ComputeAndCompareR1<NativeT>(builder, expected, arguments);
  }

  template <typename NativeT>
  void RunVectorizedReduceTestForType(
      const std::function<XlaComputation(XlaBuilder*)>&
          reduction_function_generator,
      const std::function<NativeT(NativeT, NativeT)>&
          reference_reduction_function,
      const NativeT& initial_value) {
    const int rows = 64, cols = 128;
    const int minor = 1, major = 0;
    XlaBuilder builder(TestName());
    XlaComputation reduction_function = reduction_function_generator(&builder);
    const Shape input_shape = ShapeUtil::MakeShape(
        xla::primitive_util::NativeToPrimitiveType<NativeT>(), {rows, cols});
    auto input = Parameter(&builder, 0, input_shape, "input");
    auto zero = ConstantR0<NativeT>(&builder, initial_value);
    Reduce(input, zero, reduction_function,
           /*dimensions_to_reduce=*/{0});

    Array2D<NativeT> input_data(rows, cols);
    input_data.FillUnique(initial_value);
    Literal input_literal = LiteralUtil::CreateR2FromArray2D(input_data);
    input_literal =
        input_literal.Relayout(LayoutUtil::MakeLayout({minor, major}));
    std::unique_ptr<GlobalData> input_global_data =
        client_->TransferToServer(input_literal).value();

    // NativeT can be bool, and std::vector<bool> does not convert to
    // Span.
    std::unique_ptr<NativeT[]> expected(new NativeT[cols]);
    for (int64_t colno = 0; colno < cols; ++colno) {
      NativeT column_result = initial_value;
      for (int64_t rowno = 0; rowno < rows; ++rowno) {
        column_result = reference_reduction_function(column_result,
                                                     input_data(rowno, colno));
      }
      expected[colno] = column_result;
    }

    ComputeAndCompareGeneric<NativeT>(
        &builder, absl::Span<const NativeT>(expected.get(), cols),
        {input_global_data.get()});
  }

  void RunVectorizedReduceTest(
      const std::function<XlaComputation(PrimitiveType, XlaBuilder*)>&
          reduction_function_generator_for_type,
      const std::function<float(float, float)>&
          reference_reduction_function_for_floats,
      const std::function<int32_t(int32_t, int32_t)>&
          reference_reduction_function_for_ints,
      const std::function<uint32_t(uint32_t, uint32_t)>&
          reference_reduction_function_for_uints,
      float floating_point_identity, int32_t signed_int_identity,
      uint32_t unsigned_int_identity) {
    // Float version
    RunVectorizedReduceTestForType<float>(
        [&](XlaBuilder* builder) {
          return reduction_function_generator_for_type(F32, builder);
        },
        reference_reduction_function_for_floats, floating_point_identity);

    // Signed int version
    RunVectorizedReduceTestForType<int32_t>(
        [&](XlaBuilder* builder) {
          return reduction_function_generator_for_type(S32, builder);
        },
        reference_reduction_function_for_ints, signed_int_identity);

    // Unsigned int version
    RunVectorizedReduceTestForType<uint32_t>(
        [&](XlaBuilder* builder) {
          return reduction_function_generator_for_type(U32, builder);
        },
        reference_reduction_function_for_uints, unsigned_int_identity);
  }

  Literal literal_2d_;
  Literal literal_3d_;
  uint32_t seed_ = 0xdeadbeef;
};

TEST_F(ReduceTest, ReduceR1_0_F32_To_R0) { RunR1ToR0Test(0); }
TEST_F(ReduceTest, ReduceR1_1_F32_To_R0) { RunR1ToR0Test(1); }
TEST_F(ReduceTest, ReduceR1_2_F32_To_R0) { RunR1ToR0Test(2); }
TEST_F(ReduceTest, ReduceR1_16_F32_To_R0) { RunR1ToR0Test(16); }
TEST_F(ReduceTest, ReduceR1_128_F32_To_R0) { RunR1ToR0Test(128); }
TEST_F(ReduceTest, ReduceR1_129_F32_To_R0) { RunR1ToR0Test(129); }
TEST_F(ReduceTest, ReduceR1_240_F32_To_R0) { RunR1ToR0Test(240); }
TEST_F(ReduceTest, ReduceR1_256_F32_To_R0) { RunR1ToR0Test(256); }
TEST_F(ReduceTest, ReduceR1_1024_F32_To_R0) { RunR1ToR0Test(1024); }
TEST_F(ReduceTest, ReduceR1_2048_F32_To_R0) { RunR1ToR0Test(2048); }
TEST_F(ReduceTest, ReduceR1_16K_F32_To_R0) { RunR1ToR0Test(16 * 1024); }
TEST_F(ReduceTest, ReduceR1_16KP1_F32_To_R0) { RunR1ToR0Test(16 * 1024 + 1); }
TEST_F(ReduceTest, ReduceR1_64K_F32_To_R0) { RunR1ToR0Test(64 * 1024); }
TEST_F(ReduceTest, ReduceR1_1M_F32_To_R0) { RunR1ToR0Test(1024 * 1024); }
TEST_F(ReduceTest, ReduceR1_16M_F32_To_R0) { RunR1ToR0Test(4096 * 4096); }

TEST_F(ReduceTest, ReduceR2_0x0_To_R0) { RunR2ToR0Test(0, 0); }
TEST_F(ReduceTest, ReduceR2_0x2_To_R0) { RunR2ToR0Test(0, 2); }
TEST_F(ReduceTest, ReduceR2_1x1_To_R0) { RunR2ToR0Test(1, 1); }
TEST_F(ReduceTest, ReduceR2_2x0_To_R0) { RunR2ToR0Test(2, 0); }
TEST_F(ReduceTest, ReduceR2_2x2_To_R0) { RunR2ToR0Test(2, 2); }
TEST_F(ReduceTest, ReduceR2_8x8_To_R0) { RunR2ToR0Test(8, 8); }
TEST_F(ReduceTest, ReduceR2_9x9_To_R0) { RunR2ToR0Test(9, 9); }
TEST_F(ReduceTest, ReduceR2_50x111_To_R0) { RunR2ToR0Test(50, 111); }
TEST_F(ReduceTest, ReduceR2_111x50_To_R0) { RunR2ToR0Test(111, 50); }
TEST_F(ReduceTest, ReduceR2_111x50_01_To_R0) { RunR2ToR0Test(111, 50, 0, 1); }
TEST_F(ReduceTest, ReduceR2_1024x1024_To_R0) { RunR2ToR0Test(1024, 1024); }
TEST_F(ReduceTest, ReduceR2_1000x1500_To_R0) { RunR2ToR0Test(1000, 1500); }

// Disabled due to b/33245142. Failed on 2016-11-30.
// TEST_F(ReduceTest, ReduceR2_0x0_To_R1) { RunR2ToR1Test(0, 0); }
TEST_F(ReduceTest, ReduceR2_0x2_To_R1) { RunR2ToR1Test(0, 2); }
TEST_F(ReduceTest, ReduceR2_1x1_To_R1) { RunR2ToR1Test(1, 1); }
// Disabled due to b/33245142. Failed on 2016-11-30.
// TEST_F(ReduceTest, ReduceR2_2x0_To_R1) { RunR2ToR1Test(2, 0); }
TEST_F(ReduceTest, ReduceR2_2x2_To_R1) { RunR2ToR1Test(2, 2); }
TEST_F(ReduceTest, ReduceR2_8x8_To_R1) { RunR2ToR1Test(8, 8); }
TEST_F(ReduceTest, ReduceR2_9x9_To_R1) { RunR2ToR1Test(9, 9); }
TEST_F(ReduceTest, ReduceR2_50x111_To_R1) { RunR2ToR1Test(50, 111); }
TEST_F(ReduceTest, ReduceR2_111x50_To_R1) { RunR2ToR1Test(111, 50); }
TEST_F(ReduceTest, ReduceR2_111x50_01_To_R1) { RunR2ToR1Test(111, 50, 0, 1); }
TEST_F(ReduceTest, ReduceR2_1024x1024_To_R1) { RunR2ToR1Test(1024, 1024); }
TEST_F(ReduceTest, ReduceR2_1000x1500_To_R1) { RunR2ToR1Test(1000, 1500); }

TEST_F(ReduceTest, AndReduceAllOnesR1_10_Pred) {
  constexpr int element_count = 10;
  std::vector<int> input(element_count, 1);
  RunR1ToR0PredTest(/*and_reduce=*/true, input);
}

TEST_F(ReduceTest, AndReduceOnesAndZerosR1_10_Pred) {
  constexpr int element_count = 10;
  std::vector<int> input(element_count);
  for (int i = 0; i < element_count; ++i) {
    input[i] = i % 2;
  }
  RunR1ToR0PredTest(/*and_reduce=*/true, input);
}

TEST_F(ReduceTest, OrReduceAllOnesR1_10_Pred) {
  constexpr int element_count = 10;
  std::vector<int> input(element_count, 1);
  RunR1ToR0PredTest(/*and_reduce=*/false, input);
}

TEST_F(ReduceTest, OrReduceOnesAndZerosR1_10_Pred) {
  constexpr int element_count = 10;
  std::vector<int> input(element_count);
  for (int i = 0; i < element_count; ++i) {
    input[i] = i % 2;
  }
  RunR1ToR0PredTest(/*and_reduce=*/false, input);
}

TEST_F(ReduceTest, ReduceElementwiseR2_111x50_To_R1) {
  const int64_t rows = 111, cols = 50;

  XlaBuilder builder(TestName());
  XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
  const Shape input_shape = ShapeUtil::MakeShape(F32, {rows, cols});
  auto input = Parameter(&builder, 0, input_shape, "input");
  auto zero = ConstantR0<float>(&builder, 0.0);
  auto log_ = Log(input);
  Reduce(log_, zero, add_f32, /*dimensions_to_reduce=*/{0});

  Array2D<float> input_data(rows, cols);
  input_data.FillRandom(3.14f, 0.04);
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(input_data);
  input_literal = input_literal.Relayout(LayoutUtil::MakeLayout({0, 1}));
  std::unique_ptr<GlobalData> input_global_data =
      client_->TransferToServer(input_literal).value();

  std::vector<float> expected;
  expected.reserve(cols);
  for (int64_t colno = 0; colno < cols; ++colno) {
    float column_sum = 0;
    for (int64_t rowno = 0; rowno < rows; ++rowno) {
      column_sum += std::log(input_data(rowno, colno));
    }
    expected.push_back(column_sum);
  }
  ComputeAndCompareR1<float>(&builder, expected, {input_global_data.get()},
                             ErrorSpec(0.01, 1e-4));
}

TEST_F(ReduceTest, TransposeAndReduceElementwiseR2_111x50_To_R1) {
  const int64_t rows = 111, cols = 50;

  XlaBuilder builder(TestName());
  XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
  const Shape input_shape = ShapeUtil::MakeShape(F32, {rows, cols});
  auto input = Parameter(&builder, 0, input_shape, "input");
  auto zero = ConstantR0<float>(&builder, 0.0);
  auto log_ = Log(input);
  auto transpose = Transpose(log_, {1, 0});
  Reduce(transpose, zero, add_f32, /*dimensions_to_reduce=*/{1});

  Array2D<float> input_data(rows, cols);
  input_data.FillRandom(3.14f, 0.04);
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(input_data);
  input_literal = input_literal.Relayout(LayoutUtil::MakeLayout({0, 1}));
  std::unique_ptr<GlobalData> input_global_data =
      client_->TransferToServer(input_literal).value();

  std::vector<float> expected;
  expected.reserve(cols);
  for (int64_t colno = 0; colno < cols; ++colno) {
    float column_sum = 0;
    for (int64_t rowno = 0; rowno < rows; ++rowno) {
      column_sum += std::log(input_data(rowno, colno));
    }
    expected.push_back(column_sum);
  }
  ComputeAndCompareR1<float>(&builder, expected, {input_global_data.get()},
                             ErrorSpec(0.01, 1e-4));
}

// Test that algebraic simplifier does not incorrectly fold a transpose into a
// reduction operation.
TEST_F(ReduceTest, TransposeAndReduceR3_12x111x50_To_R2) {
  XlaBuilder builder(TestName());
  XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
  const Shape input_shape = ShapeUtil::MakeShape(F32, {12, 111, 50});
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp zero = ConstantR0<float>(&builder, 0.0);
  XlaOp transpose = Transpose(input, /*permutation=*/{1, 0, 2});
  Reduce(transpose, zero, add_f32, /*dimensions_to_reduce=*/{0});

  TF_ASSERT_OK_AND_ASSIGN(Literal input_data, MakeFakeLiteral(input_shape));

  ComputeAndCompare(&builder, {std::move(input_data)}, ErrorSpec(0.01, 1e-4));
}

TEST_F(ReduceTest, Reshape_111x2x25Reduce_111x50_To_R1) {
  const int64_t rows = 111, cols = 50;

  XlaBuilder builder(TestName());
  XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
  const Shape input_shape = ShapeUtil::MakeShape(F32, {rows, 2, cols / 2});
  auto input = Parameter(&builder, 0, input_shape, "input");
  auto zero = ConstantR0<float>(&builder, 0.0);
  auto log_ = Tanh(input);
  auto reshape = Reshape(log_, {rows, cols});
  Reduce(reshape, zero, add_f32, /*dimensions_to_reduce=*/{0});

  Array3D<float> input_data(rows, 2, cols / 2);
  input_data.FillRandom(3.14f, 0.04);
  Literal input_literal = LiteralUtil::CreateR3FromArray3D(input_data);
  std::unique_ptr<GlobalData> input_global_data =
      client_->TransferToServer(input_literal).value();

  std::vector<float> expected;
  expected.reserve(cols);
  for (int64_t major = 0; major < 2; ++major) {
    for (int64_t colno = 0; colno < cols / 2; ++colno) {
      float column_sum = 0;
      for (int64_t rowno = 0; rowno < rows; ++rowno) {
        column_sum += std::tanh(input_data(rowno, major, colno));
      }
      expected.push_back(column_sum);
    }
  }
  ComputeAndCompareR1<float>(&builder, expected, {input_global_data.get()},
                             ErrorSpec(0.01, 1e-4));
}

struct BoundsLayout {
  std::vector<int64_t> bounds;
  std::vector<int64_t> layout;
  std::vector<int64_t> reduce_dims;
};

void PrintTo(const BoundsLayout& spec, std::ostream* os) {
  *os << absl::StrFormat("R%uToR%u%s_%s_Reduce%s", spec.bounds.size(),
                         spec.bounds.size() - spec.reduce_dims.size(),
                         absl::StrJoin(spec.bounds, "x"),
                         absl::StrJoin(spec.layout, ""),
                         absl::StrJoin(spec.reduce_dims, ""));
}

// Add-reduces a broadcasted scalar matrix among dimension 1 and 0.
TEST_F(ReduceTest, AddReduce2DScalarToR0) {
  XlaBuilder builder(TestName());
  auto add = CreateScalarAddComputation(F32, &builder);
  auto scalar = ConstantR0<float>(&builder, 42.0);
  auto broadcasted = Broadcast(scalar, {500, 500});
  Reduce(broadcasted, ConstantR0<float>(&builder, 0.0f), add, {0, 1});

  float expected = 42.0f * static_cast<float>(500 * 500);
  ComputeAndCompareR0<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

// Max-reduces a broadcasted scalar matrix among dimension 1 and 0.
TEST_F(ReduceTest, MaxReduce2DScalarToR0) {
  XlaBuilder builder(TestName());
  auto max = CreateScalarMaxComputation(F32, &builder);
  auto scalar = ConstantR0<float>(&builder, 42.0);
  auto broadcasted = Broadcast(scalar, {500, 500});
  Reduce(broadcasted, ConstantR0<float>(&builder, 0.0f), max, {0, 1});

  float expected = 42.0f;
  ComputeAndCompareR0<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

// Max-reduces a matrix among dimension 1 and 0.
TEST_F(ReduceTest, MaxReduce2DToR0) {
  XlaBuilder builder(TestName());
  auto max = CreateScalarMaxComputation(F32, &builder);
  Array2D<float> input(300, 250);
  input.FillRandom(214.0f);
  auto input_literal = LiteralUtil::CreateR2FromArray2D(input);
  Reduce(ConstantLiteral(&builder, input_literal),
         ConstantR0<float>(&builder, FLT_MIN), max, {0, 1});
  auto input_max = FLT_MIN;
  input.Each(
      [&](int64_t, int64_t, float* v) { input_max = std::max(input_max, *v); });
  ComputeAndCompareR0<float>(&builder, input_max, {}, ErrorSpec(0.0001));
}

// Min-reduces matrix among dimension 1 and 0.
TEST_F(ReduceTest, MinReduce2DToR0) {
  XlaBuilder builder(TestName());
  auto min = CreateScalarMinComputation(F32, &builder);
  Array2D<float> input(150, 130);
  input.FillRandom(214.0f);
  auto input_literal = LiteralUtil::CreateR2FromArray2D(input);
  Reduce(ConstantLiteral(&builder, input_literal),
         ConstantR0<float>(&builder, FLT_MAX), min, {0, 1});

  auto input_min = FLT_MAX;
  input.Each(
      [&](int64_t, int64_t, float* v) { input_min = std::min(input_min, *v); });
  ComputeAndCompareR0<float>(&builder, input_min, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceTest, UnsignedInt_MinReduce) {
  XlaBuilder builder(TestName());
  Array2D<uint32_t> input({{1}, {2}});
  auto min = CreateScalarMinComputation(U32, &builder);
  auto input_literal = LiteralUtil::CreateR2FromArray2D(input);
  auto initial_value =
      ConstantR0<uint32_t>(&builder, std::numeric_limits<uint32_t>::max());

  Reduce(ConstantLiteral(&builder, input_literal), initial_value, min, {0, 1});
  ComputeAndCompareR0<uint32_t>(&builder, 1, {});
}

TEST_F(ReduceTest, UnsignedInt_MaxReduce) {
  XlaBuilder builder(TestName());
  Array2D<uint32_t> input({{1}, {2}});
  auto max = CreateScalarMaxComputation(U32, &builder);
  auto input_literal = LiteralUtil::CreateR2FromArray2D(input);
  auto initial_value =
      ConstantR0<uint32_t>(&builder, std::numeric_limits<uint32_t>::min());

  Reduce(ConstantLiteral(&builder, input_literal), initial_value, max, {0, 1});
  ComputeAndCompareR0<uint32_t>(&builder, 2, {});
}

// Reduces a matrix among dimension 1.
TEST_F(ReduceTest, Reduce2DAmong1) {
  XlaBuilder builder(TestName());
  auto m = ConstantLiteral(&builder, literal_2d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  Reduce(m, ConstantR0<float>(&builder, 0.0f), add, {1});

  std::vector<float> expected = {6.f, 15.f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceTest, Reduce2DAmong0and1) {
  // Reduce a matrix among dimensions 0 and 1 (sum it up to a scalar).
  XlaBuilder builder(TestName());
  auto m = ConstantLiteral(&builder, literal_2d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  Reduce(m, ConstantR0<float>(&builder, 0.0f), add, {0, 1});

  ComputeAndCompareR0<float>(&builder, 21.0f, {}, ErrorSpec(0.0001, 1e-4));
}

// Tests 2D matrix ReduceToRow operation.
TEST_F(ReduceTest, Reduce2DAmongY) {
  XlaBuilder builder("reduce_among_y");
  auto m = ConstantLiteral(&builder, literal_2d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  Reduce(m, ConstantR0<float>(&builder, 0.0f), add, {0});

  std::vector<float> expected = {5.f, 7.f, 9.f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceTest, ReduceR3AmongDims_1_2) {
  XlaBuilder builder(TestName());
  auto m = ConstantLiteral(&builder, literal_3d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  Reduce(m, ConstantR0<float>(&builder, 0.0f), add, {1, 2});

  std::vector<float> expected = {21.f, 21.f, 21.f, 21.f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceTest, ReduceR3AmongDims_0_1) {
  XlaBuilder builder(TestName());
  auto m = ConstantLiteral(&builder, literal_3d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  Reduce(m, ConstantR0<float>(&builder, 0.0f), add, {0, 1});

  std::vector<float> expected = {20.f, 28.f, 36.f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceTest, ReduceR3ToR0) {
  XlaBuilder builder(TestName());
  auto m = ConstantLiteral(&builder, literal_3d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  Reduce(m, ConstantR0<float>(&builder, 0.0f), add, {0, 1, 2});

  float expected = 21.0f * 4.0;
  ComputeAndCompareR0<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceTest, ReduceR3AmongDim0) {
  XlaBuilder builder(TestName());
  auto m = ConstantLiteral(&builder, literal_3d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  Reduce(m, ConstantR0<float>(&builder, 0.0f), add, {0});

  // clang-format off
  Array2D<float> expected({
      {4.f, 8.f, 12.f},
      {16.f, 20.f, 24.f},
  });
  // clang-format on
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceTest, ReduceR3AmongDim1) {
  XlaBuilder builder(TestName());
  auto m = ConstantLiteral(&builder, literal_3d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  Reduce(m, ConstantR0<float>(&builder, 0.0f), add, {1});

  // clang-format off
  Array2D<float> expected({
      {5.f, 7.f, 9.f},
      {5.f, 7.f, 9.f},
      {5.f, 7.f, 9.f},
      {5.f, 7.f, 9.f},
  });
  // clang-format on
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceTest, ReduceR3AmongDim2) {
  XlaBuilder builder(TestName());
  auto m = ConstantLiteral(&builder, literal_3d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  Reduce(m, ConstantR0<float>(&builder, 0.0f), add, {2});

  // clang-format off
  Array2D<float> expected({
      {6.f, 15.f},
      {6.f, 15.f},
      {6.f, 15.f},
      {6.f, 15.f},
  });
  // clang-format on
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

TEST_F(ReduceTest, VectorizedReduce_Add) {
  RunVectorizedReduceTest(
      static_cast<FuncGeneratorForType>(CreateScalarAddComputation),
      [](float a, float b) { return a + b; },
      [](int32_t a, int32_t b) {
        return static_cast<int32_t>(static_cast<uint32_t>(a) +
                                    static_cast<uint32_t>(b));
      },
      [](uint32_t a, uint32_t b) { return a + b; }, 0.0, 0, 0);
}

TEST_F(ReduceTest, VectorizedReduce_Multiply) {
  RunVectorizedReduceTest(
      static_cast<FuncGeneratorForType>(CreateScalarMultiplyComputation),
      [](float a, float b) { return a * b; },
      [](int32_t a, int32_t b) {
        return static_cast<int32_t>(static_cast<uint32_t>(a) *
                                    static_cast<uint32_t>(b));
      },
      [](uint32_t a, uint32_t b) { return a * b; }, 1.0, 1, 1);
}

TEST_F(ReduceTest, VectorizedReduce_Max) {
  RunVectorizedReduceTest(
      static_cast<FuncGeneratorForType>(CreateScalarMaxComputation),
      [](float a, float b) { return std::max(a, b); },
      [](int32_t a, int32_t b) { return std::max(a, b); },
      [](uint32_t a, uint32_t b) { return std::max(a, b); },
      std::numeric_limits<float>::min(), std::numeric_limits<int32_t>::min(),
      std::numeric_limits<uint32_t>::min());
}

TEST_F(ReduceTest, VectorizedReduce_Min) {
  RunVectorizedReduceTest(
      static_cast<FuncGeneratorForType>(CreateScalarMinComputation),
      [](float a, float b) { return std::min(a, b); },
      [](int32_t a, int32_t b) { return std::min(a, b); },
      [](uint32_t a, uint32_t b) { return std::min(a, b); },
      std::numeric_limits<float>::max(), std::numeric_limits<int32_t>::max(),
      std::numeric_limits<uint32_t>::max());
}

TEST_F(ReduceTest, VectorizedReduce_BooleanAnd) {
  RunVectorizedReduceTestForType<bool>(
      static_cast<FuncGenerator>([](XlaBuilder* builder) {
        return CreateScalarAndComputation(PRED, builder);
      }),
      [](bool a, bool b) { return a && b; }, true);
}

TEST_F(ReduceTest, VectorizedReduce_BooleanOr) {
  RunVectorizedReduceTestForType<bool>(
      static_cast<FuncGenerator>([](XlaBuilder* builder) {
        return CreateScalarOrComputation(PRED, builder);
      }),
      [](bool a, bool b) { return a || b; }, false);
}

class ReduceR3ToR2Test : public ReduceTest,
                         public ::testing::WithParamInterface<BoundsLayout> {};

TEST_P(ReduceR3ToR2Test, ReduceR3ToR2) {
  XlaBuilder builder(TestName());
  const auto& bounds = GetParam().bounds;
  Array3D<float> input_array(bounds[0], bounds[1], bounds[2]);
  //  input_array.FillRandom(3.14f, 0.05);
  input_array.Fill(1.0f);

  auto input_literal = LiteralUtil::CreateR3FromArray3D(input_array);
  input_literal =
      input_literal.Relayout(LayoutUtil::MakeLayout(GetParam().layout));
  std::unique_ptr<GlobalData> input_data =
      client_->TransferToServer(input_literal).value();

  auto input_activations =
      Parameter(&builder, 0, input_literal.shape(), "input");
  XlaComputation add = CreateScalarAddComputation(F32, &builder);
  Reduce(input_activations, ConstantR0<float>(&builder, 0.0f), add,
         GetParam().reduce_dims);

  auto expected =
      ReferenceUtil::Reduce3DTo2D(input_array, 0.0f, GetParam().reduce_dims,
                                  [](float a, float b) { return a + b; });

  ComputeAndCompareR2<float>(&builder, *expected, {input_data.get()},
                             ErrorSpec(1e-3, 1e-3));
}

INSTANTIATE_TEST_CASE_P(
    ReduceR3ToR2Test_Instantiation, ReduceR3ToR2Test,
    // Specifies (shape, layout, reduction dimensions).
    ::testing::Values(BoundsLayout{{4, 8, 128}, {2, 1, 0}, {0}},
                      BoundsLayout{{4, 8, 128}, {2, 1, 0}, {1}},
                      BoundsLayout{{4, 8, 128}, {2, 1, 0}, {2}},
                      // These should be simplified into a reshape.
                      BoundsLayout{{1, 21, 43}, {2, 1, 0}, {0}},
                      BoundsLayout{{1, 1, 1}, {2, 1, 0}, {0}},
                      BoundsLayout{{1, 1, 1}, {2, 1, 0}, {1}},
                      BoundsLayout{{1, 1, 1}, {2, 1, 0}, {2}},
                      BoundsLayout{{8, 16, 24}, {0, 1, 2}, {0}},
                      BoundsLayout{{8, 16, 24}, {0, 1, 2}, {1}},
                      BoundsLayout{{8, 16, 24}, {0, 1, 2}, {2}},
                      BoundsLayout{{5, 10, 250}, {2, 1, 0}, {0}},
                      BoundsLayout{{5, 10, 250}, {2, 1, 0}, {1}},
                      BoundsLayout{{5, 10, 250}, {2, 1, 0}, {2}},
                      BoundsLayout{{8, 16, 256}, {2, 1, 0}, {0}},
                      BoundsLayout{{8, 16, 256}, {2, 1, 0}, {1}},
                      BoundsLayout{{8, 16, 256}, {2, 1, 0}, {2}},
                      BoundsLayout{{2, 300, 784}, {2, 1, 0}, {2}},
                      BoundsLayout{{2, 300, 784}, {2, 1, 0}, {1}},
                      BoundsLayout{{2, 300, 784}, {2, 1, 0}, {0}}));

TEST_F(ReduceTest, OperationOnConstantAsInitValue) {
  XlaBuilder builder(TestName());
  XlaComputation max_f32 = CreateScalarMaxComputation(F32, &builder);

  auto a = ConstantR0<float>(&builder, 2.0f);
  auto a2 = Abs(a);

  Literal b_literal = LiteralUtil::CreateR1<float>({1.0f, 4.0f});
  std::unique_ptr<GlobalData> b_data =
      client_->TransferToServer(b_literal).value();
  auto b = Parameter(&builder, 0, b_literal.shape(), "b");
  Reduce(b, a2, max_f32, {0});

  ComputeAndCompareR0<float>(&builder, 4.0f, {b_data.get()});
}

TEST_F(ReduceTest, ReduceAndPredR2_128x64_To_R1) {
  RunR2ToR1PredTest</*cols=64*/ 64>(/*and_reduce=true*/ true, /*rows=128*/ 128);
}
TEST_F(ReduceTest, ReduceOrPredR2_64x32_To_R1) {
  RunR2ToR1PredTest</*cols=32*/ 32>(/*and_reduce=false*/ false, /*rows=64*/ 64);
}

// Tests reductions with different initial values.  There's no test macro that
// combines TYPED_TEST and TYPED_P, so we have to do it manually.
class ReduceInitializerTest : public ReduceTest {
 protected:
  template <typename T>
  void DoTest(T initializer, int num_elems) {
    XlaBuilder builder(TestName());
    XlaComputation max_fn = CreateScalarMaxComputation(
        primitive_util::NativeToPrimitiveType<T>(), &builder);

    auto init = ConstantR0<T>(&builder, initializer);
    std::vector<T> input_arr(num_elems, std::numeric_limits<T>::lowest());
    auto input_literal = LiteralUtil::CreateR1<T>(input_arr);
    auto input_data = client_->TransferToServer(input_literal).value();
    Reduce(Parameter(&builder, 0, input_literal.shape(), "input"), init, max_fn,
           {0});

    ComputeAndCompareR0<T>(&builder, initializer, {input_data.get()});
  }
};

TEST_F(ReduceInitializerTest, U8Small) { DoTest<uint8_t>(42, 2); }

TEST_F(ReduceInitializerTest, U8BigPowerOf2) { DoTest<uint8_t>(42, 4096); }

TEST_F(ReduceInitializerTest, U8InitializerBigNonPowerOf2) {
  DoTest<uint8_t>(42, 4095);
}

TEST_F(ReduceInitializerTest, U64InitializerZero) { DoTest<uint64_t>(0, 1024); }

TEST_F(ReduceInitializerTest, U64InitializerOne) { DoTest<uint64_t>(1, 1024); }

TEST_F(ReduceInitializerTest, U64InitializerBigValue) {
  DoTest<uint64_t>(1234556789123, 1024);
}

// Test the operational semantic that the init value is passed on the lhs for
// reduces. Can be tested by performing an "identity" reduce (that simply
// returns one of the parameters). In this case, we return the rhs, which for
// a 1D array with one element, should not be the init value.
TEST_F(ReduceTest, ReduceIdentity) {
  XlaBuilder builder(TestName());
  Shape single_float = ShapeUtil::MakeShape(F32, {});
  Parameter(&builder, 0, single_float, "lhs-unused");
  Parameter(&builder, 1, single_float, "rhs-used");
  auto computation_status = builder.Build();
  TF_ASSERT_OK(computation_status.status());

  Shape operand_shape = ShapeUtil::MakeShape(F32, {1});
  Reduce(Parameter(&builder, 0, operand_shape, "operand"),
         Parameter(&builder, 1, single_float, "init"),
         computation_status.value(), {0});

  float operand[] = {42.0f};
  float init = 58.5f;
  float expected = 42.0f;
  Literal input_literal = LiteralUtil::CreateR1<float>(operand);
  std::unique_ptr<GlobalData> input_global_data =
      client_->TransferToServer(input_literal).value();
  Literal input_literal2 = LiteralUtil::CreateR0<float>(init);
  std::unique_ptr<GlobalData> input_global_data2 =
      client_->TransferToServer(input_literal2).value();
  ComputeAndCompareR0<float>(
      &builder, expected, {input_global_data.get(), input_global_data2.get()},
      ErrorSpec(0.0001));
}

TEST_F(ReduceTest, AndReduceU64) {
  XlaBuilder builder(TestName());
  Array2D<uint64_t> initializer = {
      {0x123456789ABCDEF0ULL, 0x3BCDEF12A4567890ULL},
      {0XFFFFFFFFFFFFFFD6ULL, 101},
      {1, 0XFFFFFFFFFFFFFFFFULL}};
  auto reducer = CreateScalarAndComputation(U64, &builder);
  auto m = ConstantR2FromArray2D(&builder, initializer);
  Reduce(m, ConstantR0<uint64_t>(&builder, 0xFFFFFFFFFFFFFFFFLL), reducer, {1});

  std::vector<uint64_t> expected = {0x1204461080145890LL, 68, 1};
  ComputeAndCompareR1<uint64_t>(&builder, expected, {});
}

TEST_F(ReduceTest, OrReduceU64) {
  XlaBuilder builder(TestName());
  Array2D<uint64_t> initializer = {
      {0x123456789ABCDEF0ULL, 0x3BCDEF12A4567890ULL},
      {0xFFFFFFFFFFFFFFD6ULL, 101},
      {1, 0xCAFEBEEFABABABABULL}};
  auto reducer = CreateScalarOrComputation(U64, &builder);
  auto m = ConstantR2FromArray2D(&builder, initializer);
  Reduce(m, ConstantR0<uint64_t>(&builder, 0), reducer, {1});

  std::vector<uint64_t> expected = {
      0X3BFDFF7ABEFEFEF0ULL, 0XFFFFFFFFFFFFFFF7ULL, 0xCAFEBEEFABABABABULL};
  ComputeAndCompareR1<uint64_t>(&builder, expected, {});
}

TEST_F(ReduceTest, R0ReduceInDisguise) {
  XlaBuilder builder(TestName());
  XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
  constexpr int element_count = 127;
  const Shape input_shape = ShapeUtil::MakeShape(F32, {element_count, 1});
  auto input = Parameter(&builder, 0, input_shape, "input");
  auto zero = ConstantR0<float>(&builder, 0.0);
  Reduce(input, zero, add_f32, /*dimensions_to_reduce=*/{0});

  Array2D<float> input_data(element_count, 1);
  input_data.FillRandom(3.0f);
  Literal input_literal = LiteralUtil::CreateR2FromArray2D(input_data);
  std::unique_ptr<GlobalData> input_global_data =
      client_->TransferToServer(input_literal).value();

  float expected = absl::c_accumulate(input_data, 0.0f);
  ComputeAndCompareR1<float>(&builder, {expected}, {input_global_data.get()},
                             ErrorSpec(0.001));
}

class ReduceHloTest : public HloTestBase {};

TEST_F(ReduceHloTest, HandleReductionToVectorAndOtherReduction) {
  absl::string_view hlo_string = R"(
  HloModule HandleReductionToVectorAndOtherReduction

  add {
    acc = f32[] parameter(1)
    op = f32[] parameter(0)
    ROOT out = f32[] add(acc, op)
  }

  ENTRY main {
    iota.3 = s32[2,2]{1,0} iota(), iota_dimension=0
    iota.2 = s32[2,2]{1,0} iota(), iota_dimension=1
    compare.0 = pred[2,2]{1,0} compare(iota.3, iota.2), direction=EQ
    broadcast = pred[2,2,2,2]{3,2,1,0} broadcast(compare.0), dimensions={2,3}
    param_0.16 = f32[2,2,2,2]{3,2,1,0} parameter(0)
    constant_4 = f32[] constant(0)
    broadcast.9 = f32[2,2,2,2]{3,2,1,0} broadcast(constant_4), dimensions={}
    select.0 = f32[2,2,2,2]{3,2,1,0} select(broadcast, param_0.16, broadcast.9)
    reduce.1 = f32[2,2,2]{2,1,0} reduce(select.0, constant_4), dimensions={2},
               to_apply=add
    abs.0 = f32[2,2,2]{2,1,0} abs(reduce.1)
    log.0 = f32[2,2,2]{2,1,0} log(abs.0)
    reduce.0 = f32[2,2]{1,0} reduce(log.0, constant_4), dimensions={2},
               to_apply=add
    ROOT tuple = (f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) tuple(reduce.0, reduce.1)
  }
  )";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ReduceHloTest, ReduceAtomicF16) {
  absl::string_view hlo_string = R"(
HloModule jit_reduce_axes12

region_0.3 {
  Arg_0.4 = f16[] parameter(0)
  Arg_1.5 = f16[] parameter(1)
  ROOT minimum.6 = f16[] minimum(Arg_0.4, Arg_1.5)
}

ENTRY main.8 {
  constant.1 = f16[] constant(1)
  Arg_0.1 = f16[2,16385,1]{2,1,0} broadcast(constant.1), dimensions={}
  constant.2 = f16[] constant(inf)
  ROOT reduce.7 = f16[2]{0} reduce(Arg_0.1, constant.2), dimensions={1,2}, to_apply=region_0.3
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ReduceHloTest, ReduceWithEpilogueMultiOutputFusion) {
  absl::string_view hlo_string = R"(
    HloModule test_module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }


    ENTRY main {
      %p0 = f32[1024] parameter(0)
      %p1 = f32[] parameter(1)
      %reduce = f32[] reduce(%p0, %p1), dimensions={0}, to_apply=add
      %p2 = f32[1024] parameter(2)
      %reduce2 = f32[] reduce(%p2, %p1), dimensions={0}, to_apply=add
      %negate = f32[] negate(%reduce)
      %log = f32[] log(%reduce)
      ROOT %tuple = (f32[], f32[], f32[]) tuple(%negate, %reduce2, %log)
    })";
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

class VariadicReduceTest : public HloTestBase {};

TEST_F(VariadicReduceTest, Reduce_R3x2_to_R2x2_simple) {
  absl::string_view hlo_string = R"(
  HloModule Reduce_R3x2_to_R1x2_simple

  add {
    op1 = f32[] parameter(0)
    op2 = f32[] parameter(1)
    acc1 = f32[] parameter(2)
    acc2 = f32[] parameter(3)
    out1 = f32[] add(acc1, op1)
    out2 = f32[] add(acc2, op2)
    ROOT result = (f32[], f32[]) tuple(out1, out2)
  }

  ENTRY main {
    inp1 = f32[3,4,5] parameter(0)
    inp2 = f32[3,4,5] parameter(1)
    zero = f32[] constant(0)

    ROOT out = (f32[3,5], f32[3,5]) reduce(inp1, inp2, zero, zero),
      dimensions={1},
      to_apply=add
  }
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(VariadicReduceTest, Reduce_R3x2_to_R1x2_simple) {
  absl::string_view hlo_string = R"(
  HloModule Reduce_R3x2_to_R1x2_simple

  add {
    op1 = f32[] parameter(0)
    op2 = f32[] parameter(1)
    acc1 = f32[] parameter(2)
    acc2 = f32[] parameter(3)
    out1 = f32[] add(acc1, op1)
    out2 = f32[] add(acc2, op2)
    ROOT result = (f32[], f32[]) tuple(out1, out2)
  }

  ENTRY main {
    inp1 = f32[10,20,3] parameter(0)
    inp2 = f32[10,20,3] parameter(1)
    zero = f32[] constant(0)

    ROOT out = (f32[10], f32[10]) reduce(inp1, inp2, zero, zero),
      dimensions={1,2},
      to_apply=add
  }
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(VariadicReduceTest, Reduce_R1x2_to_R0x2_simple) {
  absl::string_view hlo_string = R"(
  HloModule Reduce_R1x2_to_R0x2_simple

  add {
    op1 = f32[] parameter(0)
    op2 = f32[] parameter(1)
    acc1 = f32[] parameter(2)
    acc2 = f32[] parameter(3)
    out1 = f32[] add(acc1, op1)
    out2 = f32[] add(acc2, op2)
    ROOT result = (f32[], f32[]) tuple(out1, out2)
  }

  ENTRY main {
    inp1 = f32[100] parameter(0)
    inp2 = f32[100] parameter(1)
    zero = f32[] constant(0)

    ROOT out = (f32[], f32[]) reduce(inp1, inp2, zero, zero),
      dimensions={0},
      to_apply=add
  }
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(VariadicReduceTest, Reduce_R1x2_to_R0x2_argmax) {
  absl::string_view hlo_string = R"(
    HloModule Reduce_R1x2_to_R0x2_argmax

    argmax {
      running_max = f32[] parameter(0)
      running_max_idx = u32[] parameter(1)
      current_value = f32[] parameter(2)
      current_value_idx = u32[] parameter(3)

      current = (f32[], u32[]) tuple(running_max, running_max_idx)
      potential = (f32[], u32[]) tuple(current_value, current_value_idx)

      cmp_code = pred[] compare(current_value, running_max), direction=GT

      new_max = f32[] select(cmp_code, current_value, running_max)
      new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

      ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
    }

    ENTRY main {
      input = f32[100] parameter(0)
      idxs = u32[100]{0} iota(), iota_dimension=0
      zero = f32[] constant(0)
      zero_idx = u32[] constant(0)

      ROOT out = (f32[], u32[]) reduce(
        input, idxs, zero, zero_idx),
        dimensions={0},
        to_apply=%argmax
    }
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(VariadicReduceTest, Reduce_R1x2_to_R0x2_argmax_column) {
  absl::string_view hlo_string = R"(
    HloModule Reduce_R1x2_to_R0x2_argmax

    add {
      acc = f32[] parameter(1)
      op = f32[] parameter(0)
      ROOT out = f32[] add(acc, op)
    }

    argmax {
      running_max = f32[] parameter(0)
      running_max_idx = u32[] parameter(1)
      current_value = f32[] parameter(2)
      current_value_idx = u32[] parameter(3)

      current = (f32[], u32[]) tuple(running_max, running_max_idx)
      potential = (f32[], u32[]) tuple(current_value, current_value_idx)

      cmp_code = pred[] compare(current_value, running_max), direction=GT

      new_max = f32[] select(cmp_code, current_value, running_max)
      new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

      ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
    }

    ENTRY main {
      input = f32[32,128] parameter(0)
      idxs = u32[32,128] iota(), iota_dimension=0
      zero = f32[] constant(0)
      zero_idx = u32[] constant(0)

      ROOT argmax_result = (f32[128], u32[128]) reduce(
        input, idxs, zero, zero_idx),
        dimensions={0},
        to_apply=%argmax
    }
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(VariadicReduceTest, ReduceMultiOutputVariadicAnd) {
  absl::string_view hlo_string = R"(
    HloModule VariadicReduceMultiOutput

    VariadicAnd {
      value = pred[] parameter(0)
      value_idx = u32[] parameter(1)
      current_value = pred[] parameter(2)
      current_value_idx = u32[] parameter(3)
      ROOT out = (pred[], u32[]) tuple(value, value_idx)
    }

    ENTRY CheckBuffer {
      test_value = f32[] parameter(0)
      buffer = f32[100] parameter(1)
      value_broadcast = f32[100] broadcast(test_value), dimensions={}
      comparison_result = pred[100] compare(buffer, value_broadcast), direction=EQ
      true_constant = pred[] constant(true)

      zero_idx = u32[] constant(0)
      idxs = u32[100]{0} iota(), iota_dimension=0
      out = (pred[], u32[]) reduce(
         comparison_result, idxs, true_constant, zero_idx
      ), dimensions={0}, to_apply=VariadicAnd

      ROOT returned = u32[] get-tuple-element(out), index=1
    }
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(VariadicReduceTest, ReduceMultiOutputVariadicDifferentLayout) {
  absl::string_view hlo_string = R"(
HloModule ReduceWithLayoutChangeVariadicDifferent

argmax {
  running_max = f32[] parameter(0)
  running_max_idx = u32[] parameter(1)
  current_value = f32[] parameter(2)
  current_value_idx = u32[] parameter(3)

  current = (f32[], u32[]) tuple(running_max, running_max_idx)
  potential = (f32[], u32[]) tuple(current_value, current_value_idx)

  cmp_code = pred[] compare(current_value, running_max), direction=GT

  new_max = f32[] select(cmp_code, current_value, running_max)
  new_idx = u32[] select(cmp_code, current_value_idx, running_max_idx)

  ROOT out = (f32[], u32[]) tuple(new_max, new_idx)
}

ENTRY main {
  arg0 = f32[2,3,4,1024]{2,1,0,3}  parameter(0)
  idxs = u32[2,3,4,1024]{3,2,1,0}  parameter(1)
  constant0 = f32[] constant(0)
  constant1 = u32[] constant(0)
  ROOT reduce0 = (
      f32[2,3,4]{2,1,0},
      u32[2,3,4]{1,0,2}
    ) reduce(arg0, idxs, constant0,constant1), dimensions={3}, to_apply=argmax
}
)";

  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace xla
