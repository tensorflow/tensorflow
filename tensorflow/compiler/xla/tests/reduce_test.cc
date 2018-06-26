/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/client/global_data.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using FuncGeneratorForType = XlaComputation (*)(PrimitiveType, XlaBuilder*);

using FuncGenerator = XlaComputation (*)(XlaBuilder*);

class ReduceTest : public ClientLibraryTestBase {
 protected:
  ReduceTest() {
    // Implementation note: laid out z >> y >> x by default.
    // clang-format off
    literal_2d_ = Literal::CreateR2<float>({
      // x0   x1   x2
      { 1.f, 2.f, 3.f},  // y0
      { 4.f, 5.f, 6.f},  // y1
    });
    literal_3d_ = Literal::CreateR3Projected<float>({
      // x0   x1   x2
      { 1.f, 2.f, 3.f},  // y0
      { 4.f, 5.f, 6.f},  // y1
    }, 4);
    // clang-format on
    CHECK(ShapeUtil::Equal(
        literal_3d_->shape(),
        ShapeUtil::MakeShape(F32, {/*z=*/4, /*y=*/2, /*x=*/3})))
        << literal_3d_->shape().ShortDebugString();
  }

  // Runs an R1 => R0 reduction test with the given number of elements.
  void RunR1ToR0Test(int64 element_count) {
    XlaBuilder builder(TestName());
    XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
    const Shape input_shape = ShapeUtil::MakeShape(F32, {element_count});
    auto input = builder.Parameter(0, input_shape, "input");
    auto zero = builder.ConstantR0<float>(0.0);
    builder.Reduce(input, zero, add_f32, /*dimensions_to_reduce=*/{0});

    std::vector<float> input_data(element_count);
    for (int64 i = 0; i < element_count; ++i) {
      input_data[i] = rand_r(&seed_) % 3;
      if (rand_r(&seed_) % 2 == 0) {
        input_data[i] *= -1;
      }
    }
    std::unique_ptr<Literal> input_literal =
        Literal::CreateR1(AsSlice(input_data));
    std::unique_ptr<GlobalData> input_global_data =
        client_->TransferToServer(*input_literal).ConsumeValueOrDie();

    float expected = 0.0;
    for (float item : input_data) {
      expected += item;
    }
    ComputeAndCompareR0<float>(&builder, expected, {input_global_data.get()},
                               ErrorSpec(0.001));
  }

  void RunR1ToR0PredTest(bool and_reduce,
                         tensorflow::gtl::ArraySlice<int> input_data) {
    const int element_count = input_data.size();
    XlaBuilder builder(TestName());
    const Shape input_shape = ShapeUtil::MakeShape(S32, {element_count});
    auto input_par = builder.Parameter(0, input_shape, "input");
    auto pred_values =
        builder.Eq(input_par, builder.ConstantR1<int>(element_count, 1));
    XlaOp init_value;
    XlaComputation reduce;
    if (and_reduce) {
      init_value = builder.ConstantR0<bool>(true);
      reduce = CreateScalarAndComputation(&builder);
    } else {
      init_value = builder.ConstantR0<bool>(false);
      reduce = CreateScalarOrComputation(&builder);
    }
    builder.Reduce(pred_values, init_value, reduce,
                   /*dimensions_to_reduce=*/{0});

    std::unique_ptr<Literal> input_literal = Literal::CreateR1(input_data);
    std::unique_ptr<GlobalData> input_global_data =
        client_->TransferToServer(*input_literal).ConsumeValueOrDie();

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
  template <int64 cols>
  void RunR2ToR1PredTest(bool and_reduce, int64 rows, int64 minor = 1,
                         int64 major = 0) {
    XlaBuilder builder(TestName());
    const Shape input_shape = ShapeUtil::MakeShape(U8, {rows, cols});
    auto input = builder.Parameter(0, input_shape, "input");
    auto input_pred = builder.Eq(input, builder.ConstantR0<uint8>(1));

    XlaOp init_value;
    XlaComputation reduce_op;
    if (and_reduce) {
      init_value = builder.ConstantR0<bool>(true);
      reduce_op = CreateScalarAndComputation(&builder);
    } else {
      init_value = builder.ConstantR0<bool>(false);
      reduce_op = CreateScalarOrComputation(&builder);
    }

    builder.Reduce(input_pred, init_value, reduce_op,
                   /*dimensions_to_reduce=*/{0});

    Array2D<uint8> input_data(rows, cols);
    input_data.FillRandom(0, 1);
    std::unique_ptr<Literal> input_literal =
        Literal::CreateR2FromArray2D(input_data);
    input_literal =
        input_literal->Relayout(LayoutUtil::MakeLayout({minor, major}));
    std::unique_ptr<GlobalData> input_global_data =
        client_->TransferToServer(*input_literal).ConsumeValueOrDie();

    std::array<bool, cols> expected;
    for (int64 colno = 0; colno < cols; ++colno) {
      bool column_sum = and_reduce ? true : false;
      for (int64 rowno = 0; rowno < rows; ++rowno) {
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
  void RunR2ToR0Test(int64 rows, int64 cols, int64 minor = 1, int64 major = 0) {
    XlaBuilder builder(TestName());
    XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
    const Shape input_shape = ShapeUtil::MakeShape(F32, {rows, cols});
    auto input = builder.Parameter(0, input_shape, "input");
    auto zero = builder.ConstantR0<float>(0.0);
    builder.Reduce(input, zero, add_f32, /*dimensions_to_reduce=*/{0, 1});

    Array2D<float> input_data(rows, cols);
    input_data.FillRandom(3.14f, 0.04);
    std::unique_ptr<Literal> input_literal =
        Literal::CreateR2FromArray2D(input_data);
    input_literal =
        input_literal->Relayout(LayoutUtil::MakeLayout({minor, major}));
    std::unique_ptr<GlobalData> input_global_data =
        client_->TransferToServer(*input_literal).ConsumeValueOrDie();

    float expected = 0.0;
    for (int64 rowno = 0; rowno < rows; ++rowno) {
      for (int64 colno = 0; colno < cols; ++colno) {
        expected += input_data(rowno, colno);
      }
    }
    ComputeAndCompareR0<float>(&builder, expected, {input_global_data.get()},
                               ErrorSpec(0.01, 1e-4));
  }

  // Runs an R2 => R1 reduction test with the given number of (rows, cols).
  void RunR2ToR1Test(int64 rows, int64 cols, int64 minor = 1, int64 major = 0) {
    XlaBuilder builder(TestName());
    XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
    const Shape input_shape = ShapeUtil::MakeShape(F32, {rows, cols});
    auto input = builder.Parameter(0, input_shape, "input");
    auto zero = builder.ConstantR0<float>(0.0);
    builder.Reduce(input, zero, add_f32, /*dimensions_to_reduce=*/{0});

    Array2D<float> input_data(rows, cols);
    input_data.FillRandom(3.14f, 0.04);
    std::unique_ptr<Literal> input_literal =
        Literal::CreateR2FromArray2D(input_data);
    input_literal =
        input_literal->Relayout(LayoutUtil::MakeLayout({minor, major}));
    std::unique_ptr<GlobalData> input_global_data =
        client_->TransferToServer(*input_literal).ConsumeValueOrDie();

    std::vector<float> expected;
    for (int64 colno = 0; colno < cols; ++colno) {
      float column_sum = 0;
      for (int64 rowno = 0; rowno < rows; ++rowno) {
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
      tensorflow::gtl::ArraySlice<NativeT> expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
    ComputeAndCompareR1<NativeT>(builder, expected, arguments,
                                 ErrorSpec(0.01, 1e-4));
  }

  template <typename NativeT>
  void ComputeAndCompareGeneric(
      typename std::enable_if<std::is_integral<NativeT>::value,
                              XlaBuilder>::type* builder,
      tensorflow::gtl::ArraySlice<NativeT> expected,
      tensorflow::gtl::ArraySlice<GlobalData*> arguments) {
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
    auto input = builder.Parameter(0, input_shape, "input");
    auto zero = builder.ConstantR0<NativeT>(initial_value);
    builder.Reduce(input, zero, reduction_function,
                   /*dimensions_to_reduce=*/{0});

    Array2D<NativeT> input_data(rows, cols);
    input_data.FillUnique(initial_value);
    std::unique_ptr<Literal> input_literal =
        Literal::CreateR2FromArray2D(input_data);
    input_literal =
        input_literal->Relayout(LayoutUtil::MakeLayout({minor, major}));
    std::unique_ptr<GlobalData> input_global_data =
        client_->TransferToServer(*input_literal).ConsumeValueOrDie();

    // NativeT can be bool, and std::vector<bool> does not convert to
    // ArraySlice.
    std::unique_ptr<NativeT[]> expected(new NativeT[cols]);
    for (int64 colno = 0; colno < cols; ++colno) {
      NativeT column_result = initial_value;
      for (int64 rowno = 0; rowno < rows; ++rowno) {
        column_result = reference_reduction_function(column_result,
                                                     input_data(rowno, colno));
      }
      expected[colno] = column_result;
    }

    ComputeAndCompareGeneric<NativeT>(
        &builder, tensorflow::gtl::ArraySlice<NativeT>(expected.get(), cols),
        {input_global_data.get()});
  }

  void RunVectorizedReduceTest(
      const std::function<XlaComputation(PrimitiveType, XlaBuilder*)>&
          reduction_function_generator_for_type,
      const std::function<float(float, float)>&
          reference_reduction_function_for_floats,
      const std::function<int32(int32, int32)>&
          reference_reduction_function_for_ints,
      const std::function<uint32(uint32, uint32)>&
          reference_reduction_function_for_uints,
      float floating_point_identity, int32 signed_int_identity,
      uint32 unsigned_int_identity) {
    // Float version
    RunVectorizedReduceTestForType<float>(
        [&](XlaBuilder* builder) {
          return reduction_function_generator_for_type(F32, builder);
        },
        reference_reduction_function_for_floats, floating_point_identity);

    // Signed int version
    RunVectorizedReduceTestForType<int32>(
        [&](XlaBuilder* builder) {
          return reduction_function_generator_for_type(S32, builder);
        },
        reference_reduction_function_for_ints, signed_int_identity);

    // Unsigned int version
    RunVectorizedReduceTestForType<uint32>(
        [&](XlaBuilder* builder) {
          return reduction_function_generator_for_type(U32, builder);
        },
        reference_reduction_function_for_uints, unsigned_int_identity);
  }

  std::unique_ptr<Literal> literal_2d_;
  std::unique_ptr<Literal> literal_3d_;
  uint32 seed_ = 0xdeadbeef;
};

XLA_TEST_F(ReduceTest, ReduceR1_0_F32_To_R0) { RunR1ToR0Test(0); }
XLA_TEST_F(ReduceTest, ReduceR1_1_F32_To_R0) { RunR1ToR0Test(1); }
XLA_TEST_F(ReduceTest, ReduceR1_2_F32_To_R0) { RunR1ToR0Test(2); }
XLA_TEST_F(ReduceTest, ReduceR1_16_F32_To_R0) { RunR1ToR0Test(16); }
XLA_TEST_F(ReduceTest, ReduceR1_128_F32_To_R0) { RunR1ToR0Test(128); }
XLA_TEST_F(ReduceTest, ReduceR1_129_F32_To_R0) { RunR1ToR0Test(129); }
XLA_TEST_F(ReduceTest, ReduceR1_240_F32_To_R0) { RunR1ToR0Test(240); }
XLA_TEST_F(ReduceTest, ReduceR1_256_F32_To_R0) { RunR1ToR0Test(256); }
XLA_TEST_F(ReduceTest, ReduceR1_1024_F32_To_R0) { RunR1ToR0Test(1024); }
XLA_TEST_F(ReduceTest, ReduceR1_2048_F32_To_R0) { RunR1ToR0Test(2048); }
XLA_TEST_F(ReduceTest, ReduceR1_16K_F32_To_R0) { RunR1ToR0Test(16 * 1024); }
XLA_TEST_F(ReduceTest, ReduceR1_16KP1_F32_To_R0) {
  RunR1ToR0Test(16 * 1024 + 1);
}
XLA_TEST_F(ReduceTest, ReduceR1_64K_F32_To_R0) { RunR1ToR0Test(64 * 1024); }
XLA_TEST_F(ReduceTest, ReduceR1_1M_F32_To_R0) { RunR1ToR0Test(1024 * 1024); }
XLA_TEST_F(ReduceTest, ReduceR1_16M_F32_To_R0) { RunR1ToR0Test(4096 * 4096); }

XLA_TEST_F(ReduceTest, ReduceR2_0x0_To_R0) { RunR2ToR0Test(0, 0); }
XLA_TEST_F(ReduceTest, ReduceR2_0x2_To_R0) { RunR2ToR0Test(0, 2); }
XLA_TEST_F(ReduceTest, ReduceR2_1x1_To_R0) { RunR2ToR0Test(1, 1); }
XLA_TEST_F(ReduceTest, ReduceR2_2x0_To_R0) { RunR2ToR0Test(2, 0); }
XLA_TEST_F(ReduceTest, ReduceR2_2x2_To_R0) { RunR2ToR0Test(2, 2); }
XLA_TEST_F(ReduceTest, ReduceR2_8x8_To_R0) { RunR2ToR0Test(8, 8); }
XLA_TEST_F(ReduceTest, ReduceR2_9x9_To_R0) { RunR2ToR0Test(9, 9); }
XLA_TEST_F(ReduceTest, ReduceR2_50x111_To_R0) { RunR2ToR0Test(50, 111); }
XLA_TEST_F(ReduceTest, ReduceR2_111x50_To_R0) { RunR2ToR0Test(111, 50); }
XLA_TEST_F(ReduceTest, ReduceR2_111x50_01_To_R0) {
  RunR2ToR0Test(111, 50, 0, 1);
}
XLA_TEST_F(ReduceTest, ReduceR2_1024x1024_To_R0) { RunR2ToR0Test(1024, 1024); }
XLA_TEST_F(ReduceTest, ReduceR2_1000x1500_To_R0) { RunR2ToR0Test(1000, 1500); }

// Disabled due to b/33245142. Failed on 2016-11-30.
// XLA_TEST_F(ReduceTest, ReduceR2_0x0_To_R1) { RunR2ToR1Test(0, 0); }
XLA_TEST_F(ReduceTest, ReduceR2_0x2_To_R1) { RunR2ToR1Test(0, 2); }
XLA_TEST_F(ReduceTest, ReduceR2_1x1_To_R1) { RunR2ToR1Test(1, 1); }
// Disabled due to b/33245142. Failed on 2016-11-30.
// XLA_TEST_F(ReduceTest, ReduceR2_2x0_To_R1) { RunR2ToR1Test(2, 0); }
XLA_TEST_F(ReduceTest, ReduceR2_2x2_To_R1) { RunR2ToR1Test(2, 2); }
XLA_TEST_F(ReduceTest, ReduceR2_8x8_To_R1) { RunR2ToR1Test(8, 8); }
XLA_TEST_F(ReduceTest, ReduceR2_9x9_To_R1) { RunR2ToR1Test(9, 9); }
XLA_TEST_F(ReduceTest, ReduceR2_50x111_To_R1) { RunR2ToR1Test(50, 111); }
XLA_TEST_F(ReduceTest, ReduceR2_111x50_To_R1) { RunR2ToR1Test(111, 50); }
XLA_TEST_F(ReduceTest, ReduceR2_111x50_01_To_R1) {
  RunR2ToR1Test(111, 50, 0, 1);
}
XLA_TEST_F(ReduceTest, ReduceR2_1024x1024_To_R1) { RunR2ToR1Test(1024, 1024); }
XLA_TEST_F(ReduceTest, ReduceR2_1000x1500_To_R1) { RunR2ToR1Test(1000, 1500); }

XLA_TEST_F(ReduceTest, AndReduceAllOnesR1_10_Pred) {
  constexpr int element_count = 10;
  std::vector<int> input(element_count, 1);
  RunR1ToR0PredTest(/*and_reduce=*/true, input);
}

XLA_TEST_F(ReduceTest, AndReduceOnesAndZerosR1_10_Pred) {
  constexpr int element_count = 10;
  std::vector<int> input(element_count);
  for (int i = 0; i < element_count; ++i) {
    input[i] = i % 2;
  }
  RunR1ToR0PredTest(/*and_reduce=*/true, input);
}

XLA_TEST_F(ReduceTest, OrReduceAllOnesR1_10_Pred) {
  constexpr int element_count = 10;
  std::vector<int> input(element_count, 1);
  RunR1ToR0PredTest(/*and_reduce=*/false, input);
}

XLA_TEST_F(ReduceTest, OrReduceOnesAndZerosR1_10_Pred) {
  constexpr int element_count = 10;
  std::vector<int> input(element_count);
  for (int i = 0; i < element_count; ++i) {
    input[i] = i % 2;
  }
  RunR1ToR0PredTest(/*and_reduce=*/false, input);
}

XLA_TEST_F(ReduceTest, ReduceElementwiseR2_111x50_To_R1) {
  const int64 rows = 111, cols = 50;

  XlaBuilder builder(TestName());
  XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
  const Shape input_shape = ShapeUtil::MakeShape(F32, {rows, cols});
  auto input = builder.Parameter(0, input_shape, "input");
  auto zero = builder.ConstantR0<float>(0.0);
  auto log_ = builder.Log(input);
  builder.Reduce(log_, zero, add_f32, /*dimensions_to_reduce=*/{0});

  Array2D<float> input_data(rows, cols);
  input_data.FillRandom(3.14f, 0.04);
  std::unique_ptr<Literal> input_literal =
      Literal::CreateR2FromArray2D(input_data);
  input_literal = input_literal->Relayout(LayoutUtil::MakeLayout({0, 1}));
  std::unique_ptr<GlobalData> input_global_data =
      client_->TransferToServer(*input_literal).ConsumeValueOrDie();

  std::vector<float> expected;
  for (int64 colno = 0; colno < cols; ++colno) {
    float column_sum = 0;
    for (int64 rowno = 0; rowno < rows; ++rowno) {
      column_sum += log(input_data(rowno, colno));
    }
    expected.push_back(column_sum);
  }
  ComputeAndCompareR1<float>(&builder, expected, {input_global_data.get()},
                             ErrorSpec(0.01, 1e-4));
}

XLA_TEST_F(ReduceTest, TransposeAndReduceElementwiseR2_111x50_To_R1) {
  const int64 rows = 111, cols = 50;

  XlaBuilder builder(TestName());
  XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
  const Shape input_shape = ShapeUtil::MakeShape(F32, {rows, cols});
  auto input = builder.Parameter(0, input_shape, "input");
  auto zero = builder.ConstantR0<float>(0.0);
  auto log_ = builder.Log(input);
  auto transpose = builder.Transpose(log_, {1, 0});
  builder.Reduce(transpose, zero, add_f32, /*dimensions_to_reduce=*/{1});

  Array2D<float> input_data(rows, cols);
  input_data.FillRandom(3.14f, 0.04);
  std::unique_ptr<Literal> input_literal =
      Literal::CreateR2FromArray2D(input_data);
  input_literal = input_literal->Relayout(LayoutUtil::MakeLayout({0, 1}));
  std::unique_ptr<GlobalData> input_global_data =
      client_->TransferToServer(*input_literal).ConsumeValueOrDie();

  std::vector<float> expected;
  for (int64 colno = 0; colno < cols; ++colno) {
    float column_sum = 0;
    for (int64 rowno = 0; rowno < rows; ++rowno) {
      column_sum += log(input_data(rowno, colno));
    }
    expected.push_back(column_sum);
  }
  ComputeAndCompareR1<float>(&builder, expected, {input_global_data.get()},
                             ErrorSpec(0.01, 1e-4));
}

// Test that algebraic simplifier does not incorrectly fold a transpose into a
// reduction operation.
XLA_TEST_F(ReduceTest, TransposeAndReduceR3_12x111x50_To_R2) {
  XlaBuilder builder(TestName());
  XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
  const Shape input_shape = ShapeUtil::MakeShape(F32, {12, 111, 50});
  XlaOp input = builder.Parameter(0, input_shape, "input");
  XlaOp zero = builder.ConstantR0<float>(0.0);
  XlaOp transpose = builder.Transpose(input, /*permutation=*/{1, 0, 2});
  builder.Reduce(transpose, zero, add_f32, /*dimensions_to_reduce=*/{0});

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Literal> input_data,
                          MakeFakeLiteral(input_shape));

  ComputeAndCompare(&builder, {std::move(*input_data)}, ErrorSpec(0.01, 1e-4));
}

XLA_TEST_F(ReduceTest, Reshape_111x2x25Reduce_111x50_To_R1) {
  const int64 rows = 111, cols = 50;

  XlaBuilder builder(TestName());
  XlaComputation add_f32 = CreateScalarAddComputation(F32, &builder);
  const Shape input_shape = ShapeUtil::MakeShape(F32, {rows, 2, cols / 2});
  auto input = builder.Parameter(0, input_shape, "input");
  auto zero = builder.ConstantR0<float>(0.0);
  auto log_ = builder.Tanh(input);
  auto reshape = builder.Reshape(log_, {rows, cols});
  builder.Reduce(reshape, zero, add_f32, /*dimensions_to_reduce=*/{0});

  Array3D<float> input_data(rows, 2, cols / 2);
  input_data.FillRandom(3.14f, 0.04);
  std::unique_ptr<Literal> input_literal =
      Literal::CreateR3FromArray3D(input_data);
  std::unique_ptr<GlobalData> input_global_data =
      client_->TransferToServer(*input_literal).ConsumeValueOrDie();

  std::vector<float> expected;
  for (int64 major = 0; major < 2; ++major) {
    for (int64 colno = 0; colno < cols / 2; ++colno) {
      float column_sum = 0;
      for (int64 rowno = 0; rowno < rows; ++rowno) {
        column_sum += tanh(input_data(rowno, major, colno));
      }
      expected.push_back(column_sum);
    }
  }
  ComputeAndCompareR1<float>(&builder, expected, {input_global_data.get()},
                             ErrorSpec(0.01, 1e-4));
}

struct BoundsLayout {
  std::vector<int64> bounds;
  std::vector<int64> layout;
  std::vector<int64> reduce_dims;
};

void PrintTo(const BoundsLayout& spec, std::ostream* os) {
  *os << tensorflow::strings::Printf(
      "R%luToR%lu%s_%s_Reduce%s", spec.bounds.size(),
      spec.bounds.size() - spec.reduce_dims.size(),
      tensorflow::str_util::Join(spec.bounds, "x").c_str(),
      tensorflow::str_util::Join(spec.layout, "").c_str(),
      tensorflow::str_util::Join(spec.reduce_dims, "").c_str());
}

// Add-reduces a broadcasted scalar matrix among dimension 1 and 0.
XLA_TEST_F(ReduceTest, AddReduce2DScalarToR0) {
  XlaBuilder builder(TestName());
  auto add = CreateScalarAddComputation(F32, &builder);
  auto scalar = builder.ConstantR0<float>(42.0);
  auto broadcasted = builder.Broadcast(scalar, {500, 500});
  builder.Reduce(broadcasted, builder.ConstantR0<float>(0.0f), add, {0, 1});

  float expected = 42.0f * static_cast<float>(500 * 500);
  ComputeAndCompareR0<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

// Max-reduces a broadcasted scalar matrix among dimension 1 and 0.
XLA_TEST_F(ReduceTest, MaxReduce2DScalarToR0) {
  XlaBuilder builder(TestName());
  auto max = CreateScalarMaxComputation(F32, &builder);
  auto scalar = builder.ConstantR0<float>(42.0);
  auto broadcasted = builder.Broadcast(scalar, {500, 500});
  builder.Reduce(broadcasted, builder.ConstantR0<float>(0.0f), max, {0, 1});

  float expected = 42.0f;
  ComputeAndCompareR0<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

// Max-reduces a matrix among dimension 1 and 0.
XLA_TEST_F(ReduceTest, MaxReduce2DToR0) {
  XlaBuilder builder(TestName());
  auto max = CreateScalarMaxComputation(F32, &builder);
  Array2D<float> input(300, 250);
  input.FillRandom(214.0f);
  auto input_literal = Literal::CreateR2FromArray2D(input);
  builder.Reduce(builder.ConstantLiteral(*input_literal),
                 builder.ConstantR0<float>(FLT_MIN), max, {0, 1});
  auto input_max = FLT_MIN;
  input.Each(
      [&](int64, int64, float* v) { input_max = std::max(input_max, *v); });
  ComputeAndCompareR0<float>(&builder, input_max, {}, ErrorSpec(0.0001));
}

// Min-reduces matrix among dimension 1 and 0.
XLA_TEST_F(ReduceTest, MinReduce2DToR0) {
  XlaBuilder builder(TestName());
  auto min = CreateScalarMinComputation(F32, &builder);
  Array2D<float> input(150, 130);
  input.FillRandom(214.0f);
  auto input_literal = Literal::CreateR2FromArray2D(input);
  builder.Reduce(builder.ConstantLiteral(*input_literal),
                 builder.ConstantR0<float>(FLT_MAX), min, {0, 1});

  auto input_min = FLT_MAX;
  input.Each(
      [&](int64, int64, float* v) { input_min = std::min(input_min, *v); });
  ComputeAndCompareR0<float>(&builder, input_min, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ReduceTest, UnsignedInt_MinReduce) {
  XlaBuilder builder(TestName());
  Array2D<uint32> input({{1}, {2}});
  auto min = CreateScalarMinComputation(U32, &builder);
  auto input_literal = Literal::CreateR2FromArray2D(input);
  auto initial_value =
      builder.ConstantR0<uint32>(std::numeric_limits<uint32>::max());

  builder.Reduce(builder.ConstantLiteral(*input_literal), initial_value, min,
                 {0, 1});
  ComputeAndCompareR0<uint32>(&builder, 1, {});
}

XLA_TEST_F(ReduceTest, UnsignedInt_MaxReduce) {
  XlaBuilder builder(TestName());
  Array2D<uint32> input({{1}, {2}});
  auto max = CreateScalarMaxComputation(U32, &builder);
  auto input_literal = Literal::CreateR2FromArray2D(input);
  auto initial_value =
      builder.ConstantR0<uint32>(std::numeric_limits<uint32>::min());

  builder.Reduce(builder.ConstantLiteral(*input_literal), initial_value, max,
                 {0, 1});
  ComputeAndCompareR0<uint32>(&builder, 2, {});
}

// Reduces a matrix among dimension 1.
XLA_TEST_F(ReduceTest, Reduce2DAmong1) {
  XlaBuilder builder(TestName());
  auto m = builder.ConstantLiteral(*literal_2d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  builder.Reduce(m, builder.ConstantR0<float>(0.0f), add, {1});

  std::vector<float> expected = {6.f, 15.f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ReduceTest, Reduce2DAmong0and1) {
  // Reduce a matrix among dimensions 0 and 1 (sum it up to a scalar).
  XlaBuilder builder(TestName());
  auto m = builder.ConstantLiteral(*literal_2d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  builder.Reduce(m, builder.ConstantR0<float>(0.0f), add, {0, 1});

  ComputeAndCompareR0<float>(&builder, 21.0f, {}, ErrorSpec(0.0001, 1e-4));
}

// Tests 2D matrix ReduceToRow operation.
XLA_TEST_F(ReduceTest, Reduce2DAmongY) {
  XlaBuilder builder("reduce_among_y");
  auto m = builder.ConstantLiteral(*literal_2d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  builder.Reduce(m, builder.ConstantR0<float>(0.0f), add, {0});

  std::vector<float> expected = {5.f, 7.f, 9.f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ReduceTest, ReduceR3AmongDims_1_2) {
  XlaBuilder builder(TestName());
  auto m = builder.ConstantLiteral(*literal_3d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  builder.Reduce(m, builder.ConstantR0<float>(0.0f), add, {1, 2});

  std::vector<float> expected = {21.f, 21.f, 21.f, 21.f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ReduceTest, ReduceR3AmongDims_0_1) {
  XlaBuilder builder(TestName());
  auto m = builder.ConstantLiteral(*literal_3d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  builder.Reduce(m, builder.ConstantR0<float>(0.0f), add, {0, 1});

  std::vector<float> expected = {20.f, 28.f, 36.f};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ReduceTest, ReduceR3ToR0) {
  XlaBuilder builder(TestName());
  auto m = builder.ConstantLiteral(*literal_3d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  builder.Reduce(m, builder.ConstantR0<float>(0.0f), add, {0, 1, 2});

  float expected = 21.0f * 4.0;
  ComputeAndCompareR0<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ReduceTest, ReduceR3AmongDim0) {
  XlaBuilder builder(TestName());
  auto m = builder.ConstantLiteral(*literal_3d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  builder.Reduce(m, builder.ConstantR0<float>(0.0f), add, {0});

  // clang-format off
  Array2D<float> expected({
      {4.f, 8.f, 12.f},
      {16.f, 20.f, 24.f},
  });
  // clang-format on
  ComputeAndCompareR2<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

XLA_TEST_F(ReduceTest, ReduceR3AmongDim1) {
  XlaBuilder builder(TestName());
  auto m = builder.ConstantLiteral(*literal_3d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  builder.Reduce(m, builder.ConstantR0<float>(0.0f), add, {1});

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

XLA_TEST_F(ReduceTest, ReduceR3AmongDim2) {
  XlaBuilder builder(TestName());
  auto m = builder.ConstantLiteral(*literal_3d_);
  auto add = CreateScalarAddComputation(F32, &builder);
  builder.Reduce(m, builder.ConstantR0<float>(0.0f), add, {2});

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

XLA_TEST_F(ReduceTest, VectorizedReduce_Add) {
  RunVectorizedReduceTest(
      static_cast<FuncGeneratorForType>(CreateScalarAddComputation),
      [](float a, float b) { return a + b; },
      [](int32 a, int32 b) {
        return static_cast<int32>(static_cast<uint32>(a) +
                                  static_cast<uint32>(b));
      },
      [](uint32 a, uint32 b) { return a + b; }, 0.0, 0, 0);
}

XLA_TEST_F(ReduceTest, VectorizedReduce_Multiply) {
  RunVectorizedReduceTest(
      static_cast<FuncGeneratorForType>(CreateScalarMultiplyComputation),
      [](float a, float b) { return a * b; },
      [](int32 a, int32 b) {
        return static_cast<int32>(static_cast<uint32>(a) *
                                  static_cast<uint32>(b));
      },
      [](uint32 a, uint32 b) { return a * b; }, 1.0, 1, 1);
}

XLA_TEST_F(ReduceTest, VectorizedReduce_Max) {
  RunVectorizedReduceTest(
      static_cast<FuncGeneratorForType>(CreateScalarMaxComputation),
      [](float a, float b) { return std::max(a, b); },
      [](int32 a, int32 b) { return std::max(a, b); },
      [](uint32 a, uint32 b) { return std::max(a, b); },
      std::numeric_limits<float>::min(), std::numeric_limits<int32>::min(),
      std::numeric_limits<uint32>::min());
}

XLA_TEST_F(ReduceTest, VectorizedReduce_Min) {
  RunVectorizedReduceTest(
      static_cast<FuncGeneratorForType>(CreateScalarMinComputation),
      [](float a, float b) { return std::min(a, b); },
      [](int32 a, int32 b) { return std::min(a, b); },
      [](uint32 a, uint32 b) { return std::min(a, b); },
      std::numeric_limits<float>::max(), std::numeric_limits<int32>::max(),
      std::numeric_limits<uint32>::max());
}

XLA_TEST_F(ReduceTest, VectorizedReduce_BooleanAnd) {
  RunVectorizedReduceTestForType<bool>(
      static_cast<FuncGenerator>(CreateScalarAndComputation),
      [](bool a, bool b) { return a && b; }, true);
}

XLA_TEST_F(ReduceTest, VectorizedReduce_BooleanOr) {
  RunVectorizedReduceTestForType<bool>(
      static_cast<FuncGenerator>(CreateScalarOrComputation),
      [](bool a, bool b) { return a || b; }, false);
}

class ReduceR3ToR2Test : public ReduceTest,
                         public ::testing::WithParamInterface<BoundsLayout> {};

XLA_TEST_P(ReduceR3ToR2Test, ReduceR3ToR2) {
  XlaBuilder builder(TestName());
  const auto& bounds = GetParam().bounds;
  Array3D<float> input_array(bounds[0], bounds[1], bounds[2]);
  //  input_array.FillRandom(3.14f, 0.05);
  input_array.Fill(1.0f);

  auto input_literal = Literal::CreateR3FromArray3D(input_array);
  input_literal =
      input_literal->Relayout(LayoutUtil::MakeLayout(GetParam().layout));
  std::unique_ptr<GlobalData> input_data =
      client_->TransferToServer(*input_literal).ConsumeValueOrDie();

  auto input_activations =
      builder.Parameter(0, input_literal->shape(), "input");
  XlaComputation add = CreateScalarAddComputation(F32, &builder);
  builder.Reduce(input_activations, builder.ConstantR0<float>(0.0f), add,
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

// TODO(b/64093391) Disabled on GPU due to an assertion failure when running
// IrEmitterUnnested::EmitInitializer() for the Reduce operator.  Failed on
// 2017-07-26.
XLA_TEST_F(ReduceTest, DISABLED_ON_GPU(OperationOnConstantAsInitValue)) {
  XlaBuilder builder(TestName());
  XlaComputation max_f32 = CreateScalarMaxComputation(F32, &builder);

  auto a = builder.ConstantR0<float>(2.0f);
  auto a2 = builder.Abs(a);

  std::unique_ptr<Literal> b_literal = Literal::CreateR1<float>({1.0f, 4.0f});
  std::unique_ptr<GlobalData> b_data =
      client_->TransferToServer(*b_literal).ConsumeValueOrDie();
  auto b = builder.Parameter(0, b_literal->shape(), "b");
  builder.Reduce(b, a2, max_f32, {0});

  ComputeAndCompareR0<float>(&builder, 4.0f, {b_data.get()});
}

XLA_TEST_F(ReduceTest, ReduceAndPredR2_128x64_To_R1) {
  RunR2ToR1PredTest</*cols=64*/ 64>(/*and_reduce=true*/ true, /*rows=128*/ 128);
}
XLA_TEST_F(ReduceTest, ReduceOrPredR2_64x32_To_R1) {
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

    auto init = builder.ConstantR0<T>(initializer);
    std::vector<T> input_arr(num_elems, std::numeric_limits<T>::lowest());
    auto input_literal = Literal::CreateR1<T>(input_arr);
    auto input_data =
        client_->TransferToServer(*input_literal).ConsumeValueOrDie();
    builder.Reduce(builder.Parameter(0, input_literal->shape(), "input"), init,
                   max_fn, {0});

    ComputeAndCompareR0<T>(&builder, initializer, {input_data.get()});
  }
};

XLA_TEST_F(ReduceInitializerTest, U8Small) { DoTest<uint8>(42, 2); }

XLA_TEST_F(ReduceInitializerTest, U8BigPowerOf2) { DoTest<uint8>(42, 4096); }

XLA_TEST_F(ReduceInitializerTest, U8InitializerBigNonPowerOf2) {
  DoTest<uint8>(42, 4095);
}

XLA_TEST_F(ReduceInitializerTest, U64InitializerZero) {
  DoTest<uint64>(0, 1024);
}

XLA_TEST_F(ReduceInitializerTest, U64InitializerOne) {
  DoTest<uint64>(1, 1024);
}

XLA_TEST_F(ReduceInitializerTest, U64InitializerBigValue) {
  DoTest<uint64>(1234556789123, 1024);
}

// Test the operational semantic that the init value is passed on the lhs for
// reduces. Can be tested by performing an "identity" reduce (that simply
// returns one of the parameters). In this case, we return the rhs, which for
// a 1D array with one element, should not be the init value.
XLA_TEST_F(ReduceTest, ReduceIdentity) {
  XlaBuilder builder(TestName());
  Shape single_float = ShapeUtil::MakeShape(F32, {});
  builder.Parameter(0, single_float, "lhs-unused");
  builder.Parameter(1, single_float, "rhs-used");
  auto computation_status = builder.Build();
  TF_ASSERT_OK(computation_status.status());

  Shape operand_shape = ShapeUtil::MakeShape(F32, {1});
  builder.Reduce(builder.Parameter(0, operand_shape, "operand"),
                 builder.Parameter(1, single_float, "init"),
                 computation_status.ValueOrDie(), {0});

  float operand[] = {42.0f};
  float init = 58.5f;
  float expected = 42.0f;
  std::unique_ptr<Literal> input_literal = Literal::CreateR1<float>(operand);
  std::unique_ptr<GlobalData> input_global_data =
      client_->TransferToServer(*input_literal).ConsumeValueOrDie();
  std::unique_ptr<Literal> input_literal2 = Literal::CreateR0<float>(init);
  std::unique_ptr<GlobalData> input_global_data2 =
      client_->TransferToServer(*input_literal2).ConsumeValueOrDie();
  ComputeAndCompareR0<float>(
      &builder, expected, {input_global_data.get(), input_global_data2.get()},
      ErrorSpec(0.0001));
}

}  // namespace
}  // namespace xla
