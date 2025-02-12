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

#include <math.h>

#include <algorithm>
#include <memory>
#include <new>
#include <random>
#include <utility>

#define EIGEN_USE_THREADS

#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/array2d.h"
#include "xla/client/client_library.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/test_benchmark.h"

namespace xla {
namespace {

const int test_width = 2, test_height = 3;

const float test_float_vals[3][test_width][test_height] = {
    {{-1.0, -1.0, 1.0}, {-3.0, 0.0, -1.0}},
    {{-3.0, 2.0, 1.0}, {0.0, -3.0, 1.0}},
    {{-3.0, 0.0, -3.0}, {-1.0, -2.0, 1.0}}};

// Test whether fusion operations are emitted with no errors and compute
// accurate outputs.
class CpuGpuFusionTest : public HloTestBase {
 protected:
  template <typename T, int Arity>
  void TestElementwise2D(
      HloOpcode opcode,
      std::optional<ComparisonDirection> direction = std::nullopt) {
    // Create a variable for comparisons since they require the direction.
    bool is_compare = std::is_same<T, bool>::value;
    Array2D<float> operand_data[Arity];
    std::fill(std::begin(operand_data), std::end(operand_data),
              Array2D<float>(test_width, test_height));
    Array2D<T> answer_data(test_width, test_height);
    for (int i = 0; i < test_width; ++i) {
      for (int j = 0; j < test_height; ++j) {
        float xs[Arity];
        for (int k = 0; k < Arity; ++k) {
          xs[k] = test_float_vals[k][i][j];
          operand_data[k](i, j) = xs[k];
        }
        if (is_compare) {
          answer_data(i, j) = ComputeElementwiseAnswerCompare(*direction, xs);
        } else {
          answer_data(i, j) = ComputeElementwiseAnswerFloat(opcode, xs);
        }
      }
    }

    auto builder = HloComputation::Builder(TestName());
    auto hlo_module = CreateNewVerifiedModule();

    auto prim_type = primitive_util::NativeToPrimitiveType<T>();

    HloInstruction* hlos[4];
    for (int i = 0; i < Arity; ++i) {
      hlos[i + 1] = builder.AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR2FromArray2D(operand_data[i])));
    }
    auto answer_shape =
        ShapeUtil::MakeShape(prim_type, {test_width, test_height});
    std::unique_ptr<HloInstruction> root_hlo;
    switch (Arity) {
      case 1:
        root_hlo = HloInstruction::CreateUnary(answer_shape, opcode, hlos[1]);
        break;
      case 2:
        if (is_compare) {
          root_hlo = HloInstruction::CreateCompare(answer_shape, hlos[1],
                                                   hlos[2], *direction);
        } else {
          root_hlo = HloInstruction::CreateBinary(answer_shape, opcode, hlos[1],
                                                  hlos[2]);
        }
        break;
      case 3:
        root_hlo = HloInstruction::CreateTernary(answer_shape, opcode, hlos[1],
                                                 hlos[2], hlos[3]);
        break;
      default:
        LOG(FATAL) << "Bad arity: " << Arity;
    }
    hlos[0] = builder.AddInstruction(std::move(root_hlo));
    hlo_module->AddEntryComputation(builder.Build())
        ->CreateFusionInstruction(
            absl::Span<HloInstruction* const>(hlos).subspan(0, Arity + 1),
            HloInstruction::FusionKind::kLoop);

    auto expected = LiteralUtil::CreateR2FromArray2D(answer_data);
    auto actual = ExecuteAndTransfer(std::move(hlo_module), {});
    if (primitive_util::IsFloatingPointType(prim_type)) {
      EXPECT_TRUE(LiteralTestUtil::Near(expected, actual, ErrorSpec(1e-4)));
    } else {
      EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
    }
  }

 private:
  float ComputeElementwiseAnswerFloat(HloOpcode opcode,
                                      absl::Span<const float> xs);
  bool ComputeElementwiseAnswerCompare(ComparisonDirection direction,
                                       absl::Span<const float> xs);
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.add_xla_disable_hlo_passes("layout-assignment");
    return debug_options;
  }
};

float CpuGpuFusionTest::ComputeElementwiseAnswerFloat(
    HloOpcode opcode, absl::Span<const float> xs) {
  switch (opcode) {
    case HloOpcode::kAdd:
      return xs[0] + xs[1];
    case HloOpcode::kSubtract:
      return xs[0] - xs[1];
    case HloOpcode::kMultiply:
      return xs[0] * xs[1];
    case HloOpcode::kDivide:
      return xs[0] / xs[1];
    case HloOpcode::kPower:
      return powf(xs[0], xs[1]);
    case HloOpcode::kMinimum:
      return std::min(xs[0], xs[1]);
    case HloOpcode::kMaximum:
      return std::max(xs[0], xs[1]);
    case HloOpcode::kClamp:
      return std::min(xs[2], std::max(xs[1], xs[0]));
    default:
      LOG(FATAL) << "No elementwise opcode: " << opcode;
  }
}

bool CpuGpuFusionTest::ComputeElementwiseAnswerCompare(
    ComparisonDirection direction, absl::Span<const float> xs) {
  switch (direction) {
    case ComparisonDirection::kEq:
      return xs[0] == xs[1];
    case ComparisonDirection::kNe:
      return xs[0] != xs[1];
    case ComparisonDirection::kGt:
      return xs[0] > xs[1];
    case ComparisonDirection::kLt:
      return xs[0] < xs[1];
    case ComparisonDirection::kGe:
      return xs[0] >= xs[1];
    case ComparisonDirection::kLe:
      return xs[0] <= xs[1];
  }
}

XLA_TEST_F(CpuGpuFusionTest, Test) {
  // test expression:
  // slice(select({{T, F, T}, {F, T, F}},
  //              concat(transpose({{1.0}, {2.0}, {3.0}} +
  //                               {{-1.0}, {-1.0}, {-1.0}}),
  //                     {{1.62, 2.72, 3.14}}) +
  //                     (-{{1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}}),
  //              {{0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}})) = {{0.5}, {2.72}}
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0}, {2.0}, {3.0}})));
  auto const1 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{-1.0}, {-1.0}, {-1.0}})));
  auto add2 = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {3, 1}), HloOpcode::kAdd, const0, const1));
  auto reshape3 = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {1, 3}), add2, {1, 0}));
  auto const4 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.62, 2.72, 3.14}})));
  auto concat5 = builder.AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(F32, {2, 3}), {reshape3, const4}, 0));
  auto const6 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}})));
  auto negate7 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {2, 3}), HloOpcode::kNegate, const6));
  auto add8 = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {2, 3}), HloOpcode::kAdd, concat5, negate7));
  auto const9 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}})));
  auto const10 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2<bool>(
          {{true, false, true}, {false, true, false}})));
  auto select11 = builder.AddInstruction(
      HloInstruction::CreateTernary(ShapeUtil::MakeShape(F32, {2, 3}),
                                    HloOpcode::kSelect, const10, add8, const9));
  auto slice12 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(F32, {2, 1}), select11, {0, 1}, {2, 2}, {1, 1}));
  // CreateFusionInstruction needs the `instructions_to_fuse` argument in
  // reverse topological order, so the first element in `instructions_to_fuse`
  // must be the root.
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(
          {slice12, select11, const10, const9, add8, negate7, const6, concat5,
           const4, reshape3, add2, const1, const0},
          HloInstruction::FusionKind::kLoop);

  EXPECT_TRUE(LiteralTestUtil::Near(
      LiteralUtil::CreateR2<float>({{0.5}, {2.72}}),
      ExecuteAndTransfer(std::move(hlo_module), {}), ErrorSpec(1e-4)));
}

// Test whether we emit appropriate code for parameters of fusion instructions.
XLA_TEST_F(CpuGpuFusionTest, Parameter) {
  // Build a computation and fuse part of it so the fusion instruction has an
  // operand parameter.
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1.0, 2.0, 3.0}})));
  auto copy1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {1, 3}), HloOpcode::kCopy, const0));
  auto const2 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{-2.0, -2.0, -2.0}})));
  // add3 = copy1 + const2 = const0 + const2 = {1,2,3} + {-2,-2,-2} = {-1,0,+1}
  auto add3 = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {1, 3}), HloOpcode::kAdd, copy1, const2));
  // CreateFusionInstruction needs `instructions_to_fuse` in reverse topological
  // order.
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{add3, const2},
                                HloInstruction::FusionKind::kLoop);

  EXPECT_TRUE(LiteralTestUtil::Near(
      LiteralUtil::CreateR2<float>({{-1.0, 0.0, 1.0}}),
      ExecuteAndTransfer(std::move(hlo_module), {}), ErrorSpec(1e-4)));
}

XLA_TEST_F(CpuGpuFusionTest, RandomizedParallelPartition) {
  // Tests parallel partitioning of a fusion instruction.
  // Create shape with random outer dimension size to generate random parallel
  // partition counts for each test run.
  const int seed = tsl::testing::RandomSeed();
  LOG(INFO) << "RandomizedParallelPartition seed: " << seed;
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> distribution(128, 1024);
  const int64_t rand_dim0_size = distribution(generator);
  const int64_t dim1_size = 1024;
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(
      F32, {rand_dim0_size, dim1_size}, {1, 0});
  // Build simple fusion computation: y = x^2 (elementwise).
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();

  auto two = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0)));
  auto x =
      builder.AddInstruction(HloInstruction::CreateBroadcast(shape, two, {}));
  auto y = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, x, x));

  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{y, x, two},
                                HloInstruction::FusionKind::kLoop);
  // Compute result.
  auto result = ExecuteAndTransfer(std::move(hlo_module), {});
  // Every element of result should be y = x^2 = 4.0.
  for (int i = 0; i < rand_dim0_size; ++i) {
    for (int j = 0; j < dim1_size; ++j) {
      EXPECT_EQ(4.0, result.Get<float>({i, j}));
    }
  }
}

XLA_TEST_F(CpuGpuFusionTest, BroadcastIntoBinaryOp) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const_vector = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0})));
  auto const_array = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{-1.0, -2.0, -4.0}, {10.0, 20.0, 30.0}})));
  auto broadcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(const_array->shape(), const_vector, {1}));
  // add2 = broadcast(const_vector) + const_array
  //      = broadcast({1,2,3}) + {{-1.0, -2.0, -4.0}, {10.0, 20.0, 30.0}}
  //      = {{1, 2, 3}, {1, 2, 3}} + {{-1.0, -2.0, -4.0}, {10.0, 20.0, 30.0}}
  auto add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::MakeShape(F32, {2, 3}),
                                   HloOpcode::kAdd, broadcast, const_array));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{add2, broadcast},
                                HloInstruction::FusionKind::kLoop);

  EXPECT_TRUE(LiteralTestUtil::Near(
      LiteralUtil::CreateR2<float>({{0.0, 0.0, -1.0}, {11.0, 22.0, 33.0}}),
      ExecuteAndTransfer(std::move(hlo_module), {}), ErrorSpec(1e-4)));
}

XLA_TEST_F(CpuGpuFusionTest, ReshapeToScalar) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto single_element_array = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2<int32_t>({{5}})));
  auto reshape = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(S32, {}), single_element_array));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape},
                                HloInstruction::FusionKind::kLoop);
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR0<int32_t>(5),
                             ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, Reshape_3by2_1by2by3) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}, {5, 6}})));
  auto reshape1 = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(S32, {1, 2, 3}), const0));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR3<int32_t>({{{1, 2, 3}, {4, 5, 6}}}),
      ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, Reshape_1by2by3_3by2) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR3<int32_t>({{{1, 2, 3}, {4, 5, 6}}})));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {3, 2}), const0));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}, {5, 6}}),
      ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, Reshape_1by1by1_) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR3<int32_t>({{{7}}})));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {}), const0));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR0<int32_t>(7),
                             ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, Reshape__1by1by1) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(7)));
  auto reshape1 = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(S32, {1, 1, 1}), const0));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR3<int32_t>({{{7}}}),
                             ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, Reshape__) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(7)));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {}), const0));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR0<int32_t>(7),
                             ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, Reshape_3by3_3by3) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {3, 3}), const0));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}),
      ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, Transpose_2by3) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}})));
  auto reshape1 = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(S32, {3, 2}), const0, {1, 0}));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{1, 4}, {2, 5}, {3, 6}}),
      ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, Transpose_3by3) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32_t>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})));
  auto reshape1 = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(S32, {3, 3}), const0, {1, 0}));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reshape1},
                                HloInstruction::FusionKind::kLoop);
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{1, 4, 7}, {2, 5, 8}, {3, 6, 9}}),
      ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, Reverse) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int32_t>({1, 2, 3})));
  auto reverse1 = builder.AddInstruction(HloInstruction::CreateReverse(
      ShapeUtil::MakeShape(S32, {3}), const0, {0}));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reverse1},
                                HloInstruction::FusionKind::kLoop);

  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>({3, 2, 1}),
                             ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, ReverseNegate) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int32_t>({1, 2, 3})));
  auto reverse1 = builder.AddInstruction(HloInstruction::CreateReverse(
      ShapeUtil::MakeShape(S32, {3}), const0, {0}));
  auto negate2 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {3}), HloOpcode::kNegate, reverse1));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{negate2, reverse1},
                                HloInstruction::FusionKind::kLoop);

  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>({-3, -2, -1}),
                             ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, BroadcastNegate) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));
  auto broadcast1 = builder.AddInstruction(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(S32, {2}), const0, {}));
  auto negate2 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {2}), HloOpcode::kNegate, broadcast1));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{negate2, broadcast1},
                                HloInstruction::FusionKind::kLoop);

  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>({-1, -1}),
                             ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, SliceNegate) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4})));
  auto slice1 = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(S32, {2}), const0, {0}, {4}, {2}));
  auto negate2 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {2}), HloOpcode::kNegate, slice1));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{negate2, slice1},
                                HloInstruction::FusionKind::kLoop);

  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>({-1, -3}),
                             ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, DynamicSliceNegate) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4})));
  auto const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));
  auto dynamic_slice2 =
      builder.AddInstruction(HloInstruction::CreateDynamicSlice(
          ShapeUtil::MakeShape(S32, {2}), const0, {const1}, {2}));
  auto negate3 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {2}), HloOpcode::kNegate, dynamic_slice2));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(
          /*instructions_to_fuse=*/{negate3, dynamic_slice2},
          HloInstruction::FusionKind::kLoop);

  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>({-2, -3}),
                             ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, ReshapeNegate) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4})));
  auto reshape1 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {2, 2}), const0));
  auto negate2 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {2, 2}), HloOpcode::kNegate, reshape1));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{negate2, reshape1},
                                HloInstruction::FusionKind::kLoop);

  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{-1, -2}, {-3, -4}}),
      ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, TransposeNegate) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}})));
  auto transpose1 = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(S32, {2, 2}), const0, {1, 0}));
  auto negate2 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {2, 2}), HloOpcode::kNegate, transpose1));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{negate2, transpose1},
                                HloInstruction::FusionKind::kLoop);

  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{-1, -3}, {-2, -4}}),
      ExecuteAndTransfer(std::move(hlo_module), {})));
}

std::unique_ptr<HloComputation> MakeReduceTestComputation() {
  auto builder = HloComputation::Builder("add");
  auto lhs = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(S32, {}), "lhs"));
  auto rhs = builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(S32, {}), "rhs"));
  builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kAdd, lhs, rhs));
  return builder.Build();
}

XLA_TEST_F(CpuGpuFusionTest, DISABLED_ON_CPU(Reduce)) {
  auto hlo_module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto const0 = builder.AddInstruction(
      HloInstruction::CreateIota(ShapeUtil::MakeShape(S32, {32}), 0));
  auto const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  auto reduce2 = builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(S32, {}), const0, const1, {0},
      hlo_module->AddEmbeddedComputation(MakeReduceTestComputation())));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reduce2},
                                HloInstruction::FusionKind::kInput);

  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR0<int32_t>(496),
                             ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, ReduceImplicitBroadcast) {
  auto hlo_module = CreateNewVerifiedModule();

  auto builder = HloComputation::Builder(TestName());
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<int32_t>({1, 2, 4, 8})));
  auto const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));
  auto reduce2 = builder.AddInstruction(HloInstruction::CreateReduce(
      ShapeUtil::MakeShape(S32, {}), const0, const1, {0},
      hlo_module->AddEmbeddedComputation(MakeReduceTestComputation())));
  auto negate3 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kNegate, reduce2));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{negate3, reduce2},
                                HloInstruction::FusionKind::kLoop);

  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR0<int32_t>(-15),
                             ExecuteAndTransfer(std::move(hlo_module), {})));
}

XLA_TEST_F(CpuGpuFusionTest, DISABLED_ON_CPU(ReduceWindow)) {
  auto builder = HloComputation::Builder(TestName());
  auto hlo_module = CreateNewVerifiedModule();
  auto const0 = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<int32_t>({{2, 3, 5}, {7, 11, 13}, {17, 19, 23}})));
  auto const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));
  Window window;
  ASSERT_TRUE(
      tsl::protobuf::TextFormat::ParseFromString("dimensions:{\n"
                                                 "size:2\n"
                                                 "stride:1\n"
                                                 "padding_low:0\n"
                                                 "padding_high:0\n"
                                                 "window_dilation:1\n"
                                                 "base_dilation:1\n"
                                                 "}\n"
                                                 "dimensions:{\n"
                                                 "size:2\n"
                                                 "stride:1\n"
                                                 "padding_low:0\n"
                                                 "padding_high:0\n"
                                                 "window_dilation:1\n"
                                                 "base_dilation:1\n"
                                                 "}\n",
                                                 &window));
  auto nested_builder = HloComputation::Builder("mul");
  {
    auto x = nested_builder.AddInstruction(
        HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(S32, {}), "x"));
    auto y = nested_builder.AddInstruction(
        HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(S32, {}), "y"));
    nested_builder.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(S32, {}), HloOpcode::kMultiply, x, y));
  }
  auto nested_computation =
      hlo_module->AddEmbeddedComputation(nested_builder.Build());
  auto reduce_window2 =
      builder.AddInstruction(HloInstruction::CreateReduceWindow(
          ShapeUtil::MakeShape(S32, {2, 2}), const0, const1, window,
          nested_computation));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction(/*instructions_to_fuse=*/{reduce_window2},
                                HloInstruction::FusionKind::kLoop);

  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<int32_t>({{462, 2145}, {24871, 62491}}),
      ExecuteAndTransfer(std::move(hlo_module), {})));
}

// When a constant (or other op) which has multiple users is imported
// into a fusion, it should remain shared, rather than being duplicated
// within the fusion.
XLA_TEST_F(CpuGpuFusionTest, SharedConstant) {
  auto hlo_module = CreateNewVerifiedModule();

  auto builder = HloComputation::Builder(TestName());
  auto const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32_t>({0})));
  auto const1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32_t>({2})));
  auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(S32, {1}), HloOpcode::kAdd, const1, const0));
  auto add2 = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(S32, {1}), HloOpcode::kAdd, const1, add1));
  auto add3 = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(S32, {1}), HloOpcode::kAdd, const1, add2));
  auto add4 = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(S32, {1}), HloOpcode::kAdd, const1, add3));
  hlo_module->AddEntryComputation(builder.Build())
      ->CreateFusionInstruction({add4, add3, add2, add1, const1},
                                HloInstruction::FusionKind::kLoop);

  HloComputation* entry_comp = hlo_module->entry_computation();

  // entry computation contains the constant(0) and the fusion
  EXPECT_EQ(entry_comp->instruction_count(), 2);

  // fused instruction contains the constant(2), the parameter, and 4 adds
  EXPECT_EQ(entry_comp->root_instruction()->fused_instruction_count(), 6);

  EXPECT_TRUE(
      LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>({8}),
                             ExecuteAndTransfer(std::move(hlo_module), {})));
}

// Test that fusion can handle elementwise ops with more than one user. This
// test case needs deduplication to avoid exponential compile time.
XLA_TEST_F(CpuGpuFusionTest, Fibonacci) {
  const char* const kModuleStr = R"(
  HloModule fibonacci

  ENTRY main (f0: f32[5], f1: f32[5]) -> f32[5] {
    %fib0 = f32[5] parameter(0)
    %fib1 = f32[5] parameter(1)
    %fib2 = f32[5] add(f32[5] %fib0, f32[5] %fib1)
    %fib3 = f32[5] add(f32[5] %fib2, f32[5] %fib1)
    %fib4 = f32[5] add(f32[5] %fib3, f32[5] %fib2)
    %fib5 = f32[5] add(f32[5] %fib4, f32[5] %fib3)
    %fib6 = f32[5] add(f32[5] %fib5, f32[5] %fib4)
    %fib7 = f32[5] add(f32[5] %fib6, f32[5] %fib5)
    %fib8 = f32[5] add(f32[5] %fib7, f32[5] %fib6)
    %fib9 = f32[5] add(f32[5] %fib8, f32[5] %fib7)
    %fib10 = f32[5] add(f32[5] %fib9, f32[5] %fib8)
    %fib11 = f32[5] add(f32[5] %fib10, f32[5] %fib9)
    %fib12 = f32[5] add(f32[5] %fib11, f32[5] %fib10)
    %fib13 = f32[5] add(f32[5] %fib12, f32[5] %fib11)
    %fib14 = f32[5] add(f32[5] %fib13, f32[5] %fib12)
    %fib15 = f32[5] add(f32[5] %fib14, f32[5] %fib13)
    %fib16 = f32[5] add(f32[5] %fib15, f32[5] %fib14)
    %fib17 = f32[5] add(f32[5] %fib16, f32[5] %fib15)
    %fib18 = f32[5] add(f32[5] %fib17, f32[5] %fib16)
    %fib19 = f32[5] add(f32[5] %fib18, f32[5] %fib17)
    %fib20 = f32[5] add(f32[5] %fib19, f32[5] %fib18)
    %fib21 = f32[5] add(f32[5] %fib20, f32[5] %fib19)
    %fib22 = f32[5] add(f32[5] %fib21, f32[5] %fib20)
    %fib23 = f32[5] add(f32[5] %fib22, f32[5] %fib21)
    %fib24 = f32[5] add(f32[5] %fib23, f32[5] %fib22)
    %fib25 = f32[5] add(f32[5] %fib24, f32[5] %fib23)
    %fib26 = f32[5] add(f32[5] %fib25, f32[5] %fib24)
    %fib27 = f32[5] add(f32[5] %fib26, f32[5] %fib25)
    %fib28 = f32[5] add(f32[5] %fib27, f32[5] %fib26)
    %fib29 = f32[5] add(f32[5] %fib28, f32[5] %fib27)
    %fib30 = f32[5] add(f32[5] %fib29, f32[5] %fib28)
    %fib31 = f32[5] add(f32[5] %fib30, f32[5] %fib29)
    %fib32 = f32[5] add(f32[5] %fib31, f32[5] %fib30)
    %fib33 = f32[5] add(f32[5] %fib32, f32[5] %fib31)
    %fib34 = f32[5] add(f32[5] %fib33, f32[5] %fib32)
    ROOT %fib35 = f32[5] add(f32[5] %fib34, f32[5] %fib33)
  })";
  auto module = ParseAndReturnVerifiedModule(kModuleStr).value();
  auto literal0 = LiteralUtil::CreateR1<float>({1, 2, 3, 4, 5});
  auto literal1 = LiteralUtil::CreateR1<float>({1, 2, 3, 4, 5});
  EXPECT_TRUE(
      RunAndCompare(std::move(module), {&literal0, &literal1}, std::nullopt));
}

XLA_TEST_F(CpuGpuFusionTest, Add2D) {
  TestElementwise2D<float, 2>(HloOpcode::kAdd);
}

XLA_TEST_F(CpuGpuFusionTest, Subtract2D) {
  TestElementwise2D<float, 2>(HloOpcode::kSubtract);
}

XLA_TEST_F(CpuGpuFusionTest, Multiply2D) {
  TestElementwise2D<float, 2>(HloOpcode::kMultiply);
}

XLA_TEST_F(CpuGpuFusionTest, Divide2D) {
  TestElementwise2D<float, 2>(HloOpcode::kDivide);
}

XLA_TEST_F(CpuGpuFusionTest, Power2D) {
  TestElementwise2D<float, 2>(HloOpcode::kPower);
}

XLA_TEST_F(CpuGpuFusionTest, Minimum2D) {
  TestElementwise2D<float, 2>(HloOpcode::kMinimum);
}

XLA_TEST_F(CpuGpuFusionTest, Maximum2D) {
  TestElementwise2D<float, 2>(HloOpcode::kMaximum);
}

XLA_TEST_F(CpuGpuFusionTest, Equal2D) {
  TestElementwise2D<bool, 2>(HloOpcode::kCompare, ComparisonDirection::kEq);
}

XLA_TEST_F(CpuGpuFusionTest, Inequal2D) {
  TestElementwise2D<bool, 2>(HloOpcode::kCompare, ComparisonDirection::kNe);
}

XLA_TEST_F(CpuGpuFusionTest, Greater2D) {
  TestElementwise2D<bool, 2>(HloOpcode::kCompare, ComparisonDirection::kGt);
}

XLA_TEST_F(CpuGpuFusionTest, Lesser2D) {
  TestElementwise2D<bool, 2>(HloOpcode::kCompare, ComparisonDirection::kLt);
}

XLA_TEST_F(CpuGpuFusionTest, GreaterOrEqual2D) {
  TestElementwise2D<bool, 2>(HloOpcode::kCompare, ComparisonDirection::kGe);
}

XLA_TEST_F(CpuGpuFusionTest, LesserOrEqual2D) {
  TestElementwise2D<bool, 2>(HloOpcode::kCompare, ComparisonDirection::kLe);
}

XLA_TEST_F(CpuGpuFusionTest, Clamp2D) {
  TestElementwise2D<float, 3>(HloOpcode::kClamp);
}

class FusionClientLibraryTest : public ClientLibraryTestBase {};

XLA_TEST_F(FusionClientLibraryTest, ManyLayoutTransformations) {
  // On the GPU backend, it's possible to have too many transposes within one
  // fusion, causing the kernel to run out shared memory and thus not compile.
  // We want to check that doesn't happen.
  //
  // To do this, we create a computation that computes
  //
  //   P0 + P0*P1*P1 + P0*P2*P2 ...
  //
  // where even parameters have layout 1 and odd parameters have layout 2.
  //
  // Our goal is to tempt the backend into creating one giant multi-output
  // fusion for the whole computation, including the transposes.  Currently
  // multi-output fusion only fuses fusions, so each of the terms in the sum
  // needs to be a fusion itself, thus the contortions above.
  constexpr int kNumParams = 25;
  XlaBuilder b("ManyLayoutTransformations");

  // This test produces values that overflow int32_t, which is UB, so use
  // uint32_t, where overflow is OK.
  Array2D<uint32_t> arr(32, 32);
  arr.FillUnique();
  Literal l1 = LiteralUtil::CreateR2FromArray2D(arr).Relayout(
      LayoutUtil::MakeLayout({0, 1}));

  Literal l2 = LiteralUtil::CreateR2FromArray2D(arr).Relayout(
      LayoutUtil::MakeLayout({1, 0}));

  XlaOp p0 = AddParam(l1, &b);
  XlaOp sum = p0;
  for (int i = 1; i < kNumParams; ++i) {
    auto pN = AddParam((i % 2 == 0 ? l1 : l2), &b);
    sum = sum + p0 * pN * pN;
  }

  ComputeAndCompare(&b, {});
}

XLA_TEST_F(CpuGpuFusionTest, TransposeDiamondWithNonTrivialBranch) {
  const char* hlo = R"(
HloModule module

ENTRY entry {
  p = f64[16,16]{1,0} parameter(0)
  trans = f64[16,16]{1,0} transpose(p), dimensions={1,0}
  rev = f64[16,16]{1,0} reverse(trans), dimensions={0,1}
  sub = f64[16,16]{1,0} subtract(trans, trans)
  ROOT add = f64[16,16]{1,0} add(rev, sub)
}
)";
  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{1e-5, 1e-5}));
}

void BM_ParallelFusion(::testing::benchmark::State& state) {
  // Simple element-wise computation to benchmark parallel task partitioning.

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  auto executors = PlatformUtil::GetStreamExecutors(platform).value();
  se::StreamExecutorMemoryAllocator allocator(platform, executors);

  const int64_t intra_op_parallelism_threads = 24;
  xla::LocalClientOptions client_options;
  client_options.set_platform(platform);
  client_options.set_intra_op_parallelism_threads(intra_op_parallelism_threads);
  auto client = ClientLibrary::GetOrCreateLocalClient(client_options).value();

  int device_ordinal = client->default_device_ordinal();

  // Computation shape parameters.
  const int64_t param0_dim0 = 1024;
  const int64_t param0_dim1 = 1024;
  const int64_t param1_dim0 = 1024;
  const int64_t param1_dim1 = 1024;
  const int64_t param2_dim0 = 1024;
  const int64_t param2_dim1 = 1024;

  // Create computation.
  XlaBuilder builder("ParallelFusion");
  Shape shape0 = ShapeUtil::MakeShape(F32, {param0_dim0, param0_dim1});
  auto param0 = Parameter(&builder, 0, shape0, "param0");
  Shape shape1 = ShapeUtil::MakeShape(F32, {param1_dim0, param1_dim1});
  auto param1 = Parameter(&builder, 1, shape1, "param1");
  Shape shape2 = ShapeUtil::MakeShape(F32, {param2_dim0, param2_dim1});
  auto param2 = Parameter(&builder, 2, shape2, "param2");

  auto x = Mul(param0, param1);
  Add(x, param2);
  auto computation = builder.Build().value();

  // Transfer literals to device.
  auto param0_literal =
      LiteralUtil::CreateR2F32Linspace(1.0, 2.0, param0_dim0, param0_dim1);
  ScopedShapedBuffer buffer0 =
      client->LiteralToShapedBuffer(param0_literal, device_ordinal).value();

  auto param1_literal =
      LiteralUtil::CreateR2F32Linspace(1.0, 2.0, param1_dim0, param1_dim1);
  ScopedShapedBuffer buffer1 =
      client->LiteralToShapedBuffer(param1_literal, device_ordinal).value();

  auto param2_literal =
      LiteralUtil::CreateR2F32Linspace(1.0, 2.0, param2_dim0, param2_dim1);
  ScopedShapedBuffer buffer2 =
      client->LiteralToShapedBuffer(param2_literal, device_ordinal).value();

  // Build executable.
  auto executables =
      client
          ->Compile(computation,
                    {&buffer0.on_host_shape(), &buffer1.on_host_shape(),
                     &buffer2.on_host_shape()},
                    ExecutableBuildOptions())
          .value();
  auto executable = std::move(executables[0]);

  auto stream = executors[device_ordinal]->CreateStream().value();
  // Initialize thread pool.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "XLAEigen",
                               intra_op_parallelism_threads);
  Eigen::ThreadPoolDevice device(pool.AsEigenThreadPool(), pool.NumThreads());

  // Initialize ExecutableRunOptions.
  ExecutableRunOptions options;
  options.set_allocator(&allocator).set_stream(stream.get());
  options.set_intra_op_thread_pool(&device);

  // Run some warm-up executions.
  const int kWarmups = 2;
  for (int i = 0; i < kWarmups; ++i) {
    auto result = executable->Run({&buffer0, &buffer1, &buffer2}, options);
    ASSERT_TRUE(result.ok());
  }

  // Run benchmark.
  const int64_t total_bytes = param0_dim0 * param0_dim0 +
                              param1_dim0 * param1_dim0 +
                              param2_dim0 * param2_dim0;

  for (auto s : state) {
    auto result = executable->Run({&buffer0, &buffer1, &buffer2}, options);
    ASSERT_TRUE(result.ok());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          total_bytes * sizeof(float));
}

BENCHMARK(BM_ParallelFusion)->UseRealTime();

}  // namespace
}  // namespace xla
