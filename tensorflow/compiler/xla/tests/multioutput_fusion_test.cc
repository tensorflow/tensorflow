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

#include <math.h>

#include <algorithm>
#include <memory>
#include <new>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class MultiOutputFusionTest : public HloTestBase {
 protected:
  MultiOutputFusionTest() { error_spec_ = ErrorSpec{0.0001, 1e-2}; }

  // Layout assignment assumes that there are no fusions in the input graph.
  // Since the purpose of this test is to send pre-fused graphs to XLA, we have
  // to do layout assignment ourselves.
  DebugOptions GetDebugOptionsForTest() override {
    auto opts = HloTestBase::GetDebugOptionsForTest();
    opts.add_xla_disable_hlo_passes("layout-assignment");
    return opts;
  }

  void RunTest2D(bool manual_fusion, int64_t size) {
    auto builder = HloComputation::Builder(TestName());
    auto hlo_module = CreateNewVerifiedModule();

    const Shape elem_shape0 = ShapeUtil::MakeShapeWithLayout(F32, {}, {});
    const Shape elem_shape2 =
        ShapeUtil::MakeShapeWithLayout(F32, {size, size}, {1, 0});

    auto const0 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(8.0f)));
    auto param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, elem_shape0, "0"));

    auto add1 = builder.AddInstruction(HloInstruction::CreateBinary(
        elem_shape0, HloOpcode::kAdd, param0, const0));

    HloInstruction* broadcast = builder.AddInstruction(
        HloInstruction::CreateBroadcast(elem_shape2, add1, {}));

    auto param1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, elem_shape2, "1"));

    HloInstruction* add2 = builder.AddInstruction(HloInstruction::CreateBinary(
        elem_shape2, HloOpcode::kAdd, broadcast, param1));
    HloInstruction* sub = builder.AddInstruction(HloInstruction::CreateBinary(
        elem_shape2, HloOpcode::kSubtract, param1, broadcast));
    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(1);
    dot_dnums.add_rhs_contracting_dimensions(0);
    HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
        elem_shape2, sub, add2, dot_dnums, DefaultPrecisionConfig(2)));
    auto computation = hlo_module->AddEntryComputation(builder.Build(dot));

    if (manual_fusion) {
      auto tuple =
          computation->AddInstruction(HloInstruction::CreateTuple({sub, add2}));
      auto gte0 = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(elem_shape2, tuple, 0));
      auto gte1 = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(elem_shape2, tuple, 1));
      TF_CHECK_OK(dot->ReplaceOperandWith(0, gte0));
      TF_CHECK_OK(dot->ReplaceOperandWith(1, gte1));

      CHECK_NE(
          computation->CreateFusionInstruction(
              {tuple, sub, add2, broadcast}, HloInstruction::FusionKind::kLoop),
          nullptr);
    }

    Literal arg1(ShapeUtil::MakeShapeWithDescendingLayout(F32, {size, size}));
    arg1.PopulateWithValue<float>(2.5f);

    Literal expect(ShapeUtil::MakeShapeWithDescendingLayout(F32, {size, size}));
    expect.PopulateWithValue<float>(size * 1.5f * 3.5f);
    Literal literal_r0 = LiteralUtil::CreateR0<float>(-9.0f);
    auto actual =
        ExecuteAndTransfer(std::move(hlo_module), {&literal_r0, &arg1});
    EXPECT_TRUE(LiteralTestUtil::Near(expect, actual, error_spec_));
  }

  void RunTest1D(bool manual_fusion, int size) {
    auto builder = HloComputation::Builder(TestName());
    auto hlo_module = CreateNewVerifiedModule();

    const Shape elem_shape_F32 =
        ShapeUtil::MakeShapeWithDescendingLayout(F32, {size});
    const Shape elem_shape_U8 =
        ShapeUtil::MakeShapeWithDescendingLayout(F64, {size});
    auto param0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, elem_shape_F32, "0"));
    auto param1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, elem_shape_U8, "1"));

    HloInstruction* param0_U8 = builder.AddInstruction(
        HloInstruction::CreateConvert(elem_shape_U8, param0));
    HloInstruction* param1_F32 = builder.AddInstruction(
        HloInstruction::CreateConvert(elem_shape_F32, param1));
    HloInstruction* add = builder.AddInstruction(HloInstruction::CreateBinary(
        elem_shape_F32, HloOpcode::kAdd, param0, param1_F32));
    HloInstruction* sub_U8 =
        builder.AddInstruction(HloInstruction::CreateBinary(
            elem_shape_U8, HloOpcode::kSubtract, param0_U8, param1));
    HloInstruction* sub = builder.AddInstruction(
        HloInstruction::CreateConvert(elem_shape_F32, sub_U8));

    HloInstruction* reshape =
        builder.AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShapeWithDescendingLayout(F32, {size, 1}), add));
    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(0);
    dot_dnums.add_rhs_contracting_dimensions(0);
    HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
        ShapeUtil::MakeShapeWithDescendingLayout(F32, {1}), sub, reshape,
        dot_dnums, DefaultPrecisionConfig(2)));
    auto computation = hlo_module->AddEntryComputation(builder.Build(dot));

    if (manual_fusion) {
      auto tuple = computation->AddInstruction(
          HloInstruction::CreateTuple({sub_U8, add}));

      auto gte0 = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(elem_shape_U8, tuple, 0));
      auto gte1 = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(elem_shape_F32, tuple, 1));
      TF_CHECK_OK(sub->ReplaceOperandWith(0, gte0));
      TF_CHECK_OK(reshape->ReplaceOperandWith(0, gte1));

      CHECK_NE(computation->CreateFusionInstruction(
                   {tuple, sub_U8, add, param0_U8, param1_F32},
                   HloInstruction::FusionKind::kLoop),
               nullptr);
    }

    Literal input0(ShapeUtil::MakeShapeWithDescendingLayout(F32, {size}));
    input0.PopulateWithValue(2.5f);
    Literal input1(ShapeUtil::MakeShapeWithDescendingLayout(F64, {size}));
    input1.PopulateWithValue(1.);

    Literal expect = LiteralUtil::CreateR1<float>({size * 1.5f * 3.5f});
    auto actual = ExecuteAndTransfer(std::move(hlo_module), {&input0, &input1});
    EXPECT_TRUE(LiteralTestUtil::Near(expect, actual, error_spec_));
  }
};

XLA_TEST_F(MultiOutputFusionTest, 2DNofusion) { RunTest2D(false, 5); }
XLA_TEST_F(MultiOutputFusionTest, 2DFusion) { RunTest2D(true, 5); }
XLA_TEST_F(MultiOutputFusionTest, 2DFusionSize129) { RunTest2D(true, 129); }
XLA_TEST_F(MultiOutputFusionTest, DifferentTypesNoFusion) {
  RunTest1D(false, 8);
}
XLA_TEST_F(MultiOutputFusionTest, DifferentTypesFusion) { RunTest1D(true, 8); }

XLA_TEST_F(MultiOutputFusionTest, FusionNodeIsRoot) {
  const char* testcase = R"(
    HloModule m, is_scheduled=true

    fused_computation {
      x.param_0 = (((s32[]), f32[]), (f32[], s32[])) parameter(0)
      gte.3 = ((s32[]), f32[]) get-tuple-element(x.param_0), index=0
      gte.2 = (s32[]) get-tuple-element(gte.3), index=0
      gte.4 = s32[] get-tuple-element(gte.2), index=0
      copy = s32[] copy(gte.4)
      ROOT tuple = (s32[]) tuple(copy)
    }

    ENTRY thing.v3 {
      x = (((s32[]), f32[]), (f32[], s32[])) parameter(0)
      ROOT fusion = (s32[]) fusion(x), kind=kLoop, calls=fused_computation
    }
  )";
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  auto param = LiteralUtil::MakeTupleOwned(
      LiteralUtil::MakeTupleOwned(
          LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR0<int32_t>(42)),
          LiteralUtil::CreateR0<float>(1.0)),
      LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR0<float>(3.0),
                                  LiteralUtil::CreateR0<int32_t>(4)));
  Literal result = ExecuteNoHloPasses(std::move(module), {&param});
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR0<int32_t>(42)), result));
}

XLA_TEST_F(MultiOutputFusionTest, MultiOutputLoopFusion) {
  const char* testcase = R"(
    HloModule m, is_scheduled=true

    fused_computation {
      p = f32[4] parameter(0)
      multiply = f32[4] multiply(p, p)
      less-than = pred[4] compare(p, multiply), direction=LT
      ROOT tuple = (pred[4], f32[4]) tuple(less-than, multiply)
    }

    ENTRY PredFloatMOF {
      p0 = f32[4] parameter(0)
      fusion = (pred[4], f32[4]) fusion(p0), kind=kLoop, calls=fused_computation
      gte0 = pred[4] get-tuple-element(fusion), index=0
      gte1 = f32[4] get-tuple-element(fusion), index=1
      const = f32[4] constant({0, 0, 0, 0})
      ROOT select = f32[4] select(gte0, gte1, const)
    })";
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  auto param = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0, -1.0});
  Literal result = ExecuteNoHloPasses(std::move(module), {&param});
  LiteralTestUtil::ExpectR1Equal<float>({0.0, 4.0, 9.0, 1.0}, result);
}

XLA_TEST_F(MultiOutputFusionTest, MultiOutputLoopFeedingMap) {
  const char* testcase = R"(
    HloModule m, is_scheduled=true

    fused_computation {
      p = f32[] parameter(0)
      multiply = f32[] multiply(p, p)
      less-than = pred[] compare(p, multiply), direction=LT
      ROOT tuple = (pred[], f32[]) tuple(less-than, multiply)
    }

    map_computation {
      p0 = f32[] parameter(0)
      fusion = (pred[], f32[]) fusion(p0), kind=kLoop, calls=fused_computation
      gte0 = pred[] get-tuple-element(fusion), index=0
      gte1 = f32[] get-tuple-element(fusion), index=1
      const = f32[] constant(0)
      ROOT select = f32[] select(gte0, gte1, const)
    }

    ENTRY MapMOF {
      p1 = f32[3] parameter(0)
      ROOT map = f32[3] map(p1), to_apply=map_computation
    })";
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  auto param = LiteralUtil::CreateR1<float>({1.0, 2.0, 3.0});
  Literal result = ExecuteNoHloPasses(std::move(module), {&param});
  LiteralTestUtil::ExpectR1Equal<float>({0.0, 4.0, 9.0}, result);
}

const char* const kScalarOps = R"(
    HloModule m, is_scheduled=true

    Add {
      lhsadd = f32[] parameter(0)
      rhsadd = f32[] parameter(1)
      ROOT add = f32[] add(lhsadd, rhsadd)
    }

    Max {
      lhsmax = f32[] parameter(0)
      rhsmax = f32[] parameter(1)
      ROOT max = f32[] maximum(lhsmax, rhsmax)
    }
)";

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionMinor)) {
  const std::string testcase = absl::StrCat(kScalarOps, R"(
    fused_reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r1 = f32[32,32]{1,0} reduce(p0, c0), dimensions={2}, to_apply=Add
      mul = f32[32,32,32]{2,1,0} multiply(p0, p0)
      c1 = f32[] constant(5)
      r2 = f32[32,32]{1,0} reduce(mul, c1), dimensions={2}, to_apply=Max
      ROOT tuple = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(r1, r2)
    }

    ENTRY reduce {
      p = f32[32,32,32]{2,1,0} parameter(0)
      ROOT fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(p), kind=kInput,
        calls=fused_reduce
    })");
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionMajor)) {
  const std::string testcase = absl::StrCat(kScalarOps, R"(
    fused_reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r1 = f32[32,32]{1,0} reduce(p0, c0), dimensions={0}, to_apply=Add
      mul = f32[32,32,32]{2,1,0} multiply(p0, p0)
      c1 = f32[] constant(5)
      r2 = f32[32,32]{1,0} reduce(mul, c1), dimensions={0}, to_apply=Max
      ROOT tuple = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(r1, r2)
    }

    ENTRY reduce {
      p = f32[32,32,32]{2,1,0} parameter(0)
      ROOT fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(p), kind=kInput,
        calls=fused_reduce
    })");
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionScalar)) {
  const std::string testcase = absl::StrCat(kScalarOps, R"(
    fused_reduce {
      p0 = f32[2,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r1 = f32[32]{0} reduce(p0, c0), dimensions={0,2}, to_apply=Add
      mul = f32[2,32,32]{2,1,0} multiply(p0, p0)
      c1 = f32[] constant(1.17549e-38)
      r2 = f32[32]{0} reduce(mul, c1), dimensions={0,2}, to_apply=Max
      r3 = f32[32]{0} reduce(mul, c0), dimensions={0,2}, to_apply=Add
      ROOT tuple = (f32[32]{0}, f32[32]{0}, f32[32]{0}) tuple(r1, r2, r3)
    }

    ENTRY reduce {
      p = f32[2,32,32]{2,1,0} parameter(0)
      ROOT fusion = (f32[32]{0}, f32[32]{0}, f32[32]{0}) fusion(p), kind=kInput,
        calls=fused_reduce
    })");
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionMinorWithExtraOutput)) {
  const std::string testcase = absl::StrCat(kScalarOps, R"(
    fused_reduce {
      p0 = f32[2,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r1 = f32[2,32]{1,0} reduce(p0, c0), dimensions={2}, to_apply=Add
      mul = f32[2,32,32]{2,1,0} multiply(p0, p0)
      c1 = f32[] constant(5)
      r2 = f32[2,32]{1,0} reduce(mul, c1), dimensions={2}, to_apply=Max
      ROOT tuple = (f32[2,32,32]{2,1,0}, f32[2,32]{1,0}, f32[2,32]{1,0})
                     tuple(p0, r1, r2)
    }

    ENTRY reduce {
      p = f32[2,32,32]{2,1,0} parameter(0)
      ROOT fusion = (f32[2,32,32]{2,1,0}, f32[2,32]{1,0}, f32[2,32]{1,0})
        fusion(p), kind=kInput, calls=fused_reduce
    })");
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionMajorWithExtraOutput)) {
  const std::string testcase = absl::StrCat(kScalarOps, R"(
    fused_reduce {
      p0 = f32[32,32,2]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r1 = f32[32,2]{1,0} reduce(p0, c0), dimensions={0}, to_apply=Add
      mul = f32[32,32,2]{2,1,0} multiply(p0, p0)
      c1 = f32[] constant(5)
      r2 = f32[32,2]{1,0} reduce(mul, c1), dimensions={0}, to_apply=Max
      ROOT tuple = (f32[32,2]{1,0}, f32[32,32,2]{2,1,0}, f32[32,2]{1,0})
                     tuple(r1, mul, r2)
    }

    ENTRY reduce {
      p = f32[32,32,2]{2,1,0} parameter(0)
      ROOT fusion = (f32[32,2]{1,0}, f32[32,32,2]{2,1,0}, f32[32,2]{1,0})
        fusion(p), kind=kInput, calls=fused_reduce
    })");
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionScalarWithExtraOutput)) {
  const std::string testcase = R"(
    HloModule m, is_scheduled=true

    Add {
      lhsadd = f32[] parameter(0)
      rhsadd = f32[] parameter(1)
      ROOT add = f32[] add(lhsadd, rhsadd)
    }
    fused_reduce {
      p0 = f32[2,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      r1 = f32[32]{0} reduce(p0, c0), dimensions={0,2}, to_apply=Add
      mul = f32[2,32,32]{2,1,0} multiply(p0, p0)
      c1 = f32[] constant(5)
      b1 = f32[2,32,32]{2,1,0} broadcast(c1), dimensions={}
      mul2 = f32[2,32,32]{2,1,0} multiply(p0, b1)
      ROOT tuple = (f32[32]{0}, f32[2,32,32]{2,1,0}, f32[2,32,32]{2,1,0})
        tuple(r1, mul, mul2)
    }

    ENTRY reduce {
      p = f32[2,32,32]{2,1,0} parameter(0)
      ROOT fusion = (f32[32]{0}, f32[2,32,32]{2,1,0}, f32[2,32,32]{2,1,0})
        fusion(p), kind=kInput, calls=fused_reduce
    })";
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionNonConstInit)) {
  const std::string testcase = absl::StrCat(kScalarOps, R"(
    fused_reduce {
      p0 = f32[2,32,32]{2,1,0} parameter(0)
      init1 = f32[] parameter(1)
      init2 = f32[] parameter(2)
      r1 = f32[2,32]{1,0} reduce(p0, init1), dimensions={2}, to_apply=Add
      r2 = f32[2,32]{1,0} reduce(p0, init2), dimensions={2}, to_apply=Max
      ROOT tuple = (f32[2,32]{1,0}, f32[2,32]{1,0}) tuple(r1, r2)
    }

    ENTRY reduce {
      p = f32[2,32,32]{2,1,0} parameter(0)
      i = f32[] parameter(1)
      j = f32[] parameter(2)
      ROOT fusion = (f32[2,32]{1,0}, f32[2,32]{1,0}) fusion(p, i, j),
       kind=kInput, calls=fused_reduce
    })");
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

XLA_TEST_F(MultiOutputFusionTest,
           DISABLED_ON_CPU(MultiOutputReduceFusionDifferentElementTypes)) {
  const std::string testcase = absl::StrCat(kScalarOps, R"(
    fused_reduce (p0: f16[2,32,32]) -> (f32[2,32], f32[2,32], f16[2,32,32]) {
      p0 = f16[2,32,32]{2,1,0} parameter(0)
      convert = f32[2,32,32]{2,1,0} convert(p0)
      c0 = f32[] constant(0)
      r1 = f32[2,32]{1,0} reduce(convert, c0), dimensions={2}, to_apply=Add
      mul = f32[2,32,32]{2,1,0} multiply(convert, convert)
      c1 = f32[] constant(5)
      r2 = f32[2,32]{1,0} reduce(mul, c1), dimensions={2}, to_apply=Max
      ROOT tuple = (f32[2,32]{1,0}, f32[2,32]{1,0}, f16[2,32,32]{2,1,0})
                   tuple(r1, r2, p0)
    }

    ENTRY reduce {
      p = f16[2,32,32]{2,1,0} parameter(0)
      ROOT fusion = (f32[2,32]{1,0}, f32[2,32]{1,0}, f16[2,32,32]{2,1,0}) fusion(p),
                    kind=kInput, calls=fused_reduce
    })");
  auto module = ParseAndReturnVerifiedModule(testcase).ValueOrDie();
  EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), ErrorSpec(1e-5)));
}

}  // namespace
}  // namespace xla
