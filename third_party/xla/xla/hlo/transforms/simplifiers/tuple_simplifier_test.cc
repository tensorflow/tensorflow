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

#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"

#include <memory>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class TupleSimplifierTest : public HloHardwareIndependentTestBase {
 protected:
  void Run(HloModule* module, bool change_expected) {
    auto changed_status = RunHloPass(TupleSimplifier(), module);
    TF_ASSERT_OK(changed_status.status());
    EXPECT_EQ(change_expected, changed_status.value());
  }
  void Run(HloModule* module, bool change_expected, bool exclude_entry) {
    auto changed_status = RunHloPass(TupleSimplifier(exclude_entry), module);
    TF_ASSERT_OK(changed_status.status());
    EXPECT_EQ(change_expected, changed_status.value());
  }

  const Shape scalar_shape_ = ShapeUtil::MakeShape(F32, {});
  const Shape tuple_shape_ = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {}),
       ShapeUtil::MakeShape(F32, {})});
};

TEST_F(TupleSimplifierTest, TupleOfParameters) {
  // A Tuple constructed of a bunch of parameters should not be changed.
  constexpr absl::string_view kModuleStr = R"(
    HloModule TupleOfParameters, entry_computation_layout={(f32[], f32[], f32[])->(f32[], f32[], f32[])}

    ENTRY %TupleOfParameters (param0: f32[], param1: f32[], param2: f32[]) -> (f32[], f32[], f32[]) {
      %param0 = f32[] parameter(0)
      %param1 = f32[] parameter(1)
      %param2 = f32[] parameter(2)
      ROOT %tuple = (f32[], f32[], f32[]) tuple(f32[] %param0, f32[] %param1, f32[] %param2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Run(module.get(), /*change_expected=*/false);
}

TEST_F(TupleSimplifierTest, GteOfTupleOfParameter) {
  // A GTE of a tuple parameter should not be changed.
  constexpr absl::string_view kModuleStr = R"(
    HloModule GteOfTupleOfParameter, entry_computation_layout={((f32[], f32[], f32[]))->f32[]}

    ENTRY %GteOfTupleOfParameter (param: (f32[], f32[], f32[])) -> f32[] {
      %param = (f32[], f32[], f32[]) parameter(0)
      ROOT %get-tuple-element = f32[] get-tuple-element((f32[], f32[], f32[]) %param), index=1
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Run(module.get(), /*change_expected=*/false);
}

TEST_F(TupleSimplifierTest, GteOfTuple) {
  // A GTE of a Tuple should be short-circuited.
  constexpr absl::string_view kModuleStr = R"(
    HloModule GteOfTuple, entry_computation_layout={(f32[], f32[], f32[])->f32[]}

    ENTRY %GteOfTuple (param0: f32[], param1: f32[], param2: f32[]) -> f32[] {
      %param0 = f32[] parameter(0)
      %param1 = f32[] parameter(1)
      %param2 = f32[] parameter(2)
      %tuple = (f32[], f32[], f32[]) tuple(f32[] %param0, f32[] %param1, f32[] %param2)
      ROOT %get-tuple-element = f32[] get-tuple-element((f32[], f32[], f32[]) %tuple), index=1
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::GetTupleElement(op::Tuple()));

  Run(module.get(), /*change_expected=*/true);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Parameter(1));
}

TEST_F(TupleSimplifierTest, GteOfTupleChain) {
  // Verify a chain of GTE/Tuple instructions is collapsed.
  constexpr absl::string_view kModuleStr = R"(
    HloModule GteOfTupleChain, entry_computation_layout={(f32[])->f32[]}

    ENTRY %GteOfTupleChain (param: f32[]) -> f32[] {
      %param = f32[] parameter(0)
      %tuple = (f32[], f32[], f32[]) tuple(f32[] %param, f32[] %param, f32[] %param)
      %get-tuple-element = f32[] get-tuple-element((f32[], f32[], f32[]) %tuple), index=1
      %tuple.1 = (f32[], f32[], f32[]) tuple(f32[] %get-tuple-element, f32[] %get-tuple-element, f32[] %get-tuple-element)
      %get-tuple-element.1 = f32[] get-tuple-element((f32[], f32[], f32[]) %tuple.1), index=1
      %tuple.2 = (f32[], f32[], f32[]) tuple(f32[] %get-tuple-element.1, f32[] %get-tuple-element.1, f32[] %get-tuple-element.1)
      %get-tuple-element.2 = f32[] get-tuple-element((f32[], f32[], f32[]) %tuple.2), index=1
      %tuple.3 = (f32[], f32[], f32[]) tuple(f32[] %get-tuple-element.2, f32[] %get-tuple-element.2, f32[] %get-tuple-element.2)
      %get-tuple-element.3 = f32[] get-tuple-element((f32[], f32[], f32[]) %tuple.3), index=1
      %tuple.4 = (f32[], f32[], f32[]) tuple(f32[] %get-tuple-element.3, f32[] %get-tuple-element.3, f32[] %get-tuple-element.3)
      %get-tuple-element.4 = f32[] get-tuple-element((f32[], f32[], f32[]) %tuple.4), index=1
      %tuple.5 = (f32[], f32[], f32[]) tuple(f32[] %get-tuple-element.4, f32[] %get-tuple-element.4, f32[] %get-tuple-element.4)
      %get-tuple-element.5 = f32[] get-tuple-element((f32[], f32[], f32[]) %tuple.5), index=1
      %tuple.6 = (f32[], f32[], f32[]) tuple(f32[] %get-tuple-element.5, f32[] %get-tuple-element.5, f32[] %get-tuple-element.5)
      %get-tuple-element.6 = f32[] get-tuple-element((f32[], f32[], f32[]) %tuple.6), index=1
      %tuple.7 = (f32[], f32[], f32[]) tuple(f32[] %get-tuple-element.6, f32[] %get-tuple-element.6, f32[] %get-tuple-element.6)
      %get-tuple-element.7 = f32[] get-tuple-element((f32[], f32[], f32[]) %tuple.7), index=1
      %tuple.8 = (f32[], f32[], f32[]) tuple(f32[] %get-tuple-element.7, f32[] %get-tuple-element.7, f32[] %get-tuple-element.7)
      %get-tuple-element.8 = f32[] get-tuple-element((f32[], f32[], f32[]) %tuple.8), index=1
      %tuple.9 = (f32[], f32[], f32[]) tuple(f32[] %get-tuple-element.8, f32[] %get-tuple-element.8, f32[] %get-tuple-element.8)
      %get-tuple-element.9 = f32[] get-tuple-element((f32[], f32[], f32[]) %tuple.9), index=1
      ROOT %negate = f32[] negate(f32[] %get-tuple-element.9)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Negate(op::GetTupleElement(op::Tuple())));

  Run(module.get(), /*change_expected=*/true);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Negate(op::Parameter()));
}

TEST_F(TupleSimplifierTest, NestedGteOfTuples) {
  // Verify a nesting of GTE/Tuple instructions is collapsed. Tuples are nested
  // to some depth with a chain of Tuple instructions, then extracted with a
  // chain of GTE instructions.
  constexpr absl::string_view kModuleStr = R"(
    HloModule NestedGteOfTuples, entry_computation_layout={(f32[])->f32[]}

    ENTRY %NestedGteOfTuples (param: f32[]) -> f32[] {
      %param = f32[] parameter(0)
      %tuple = (f32[], f32[]) tuple(f32[] %param, f32[] %param)
      %tuple.1 = ((f32[], f32[]), (f32[], f32[])) tuple((f32[], f32[]) %tuple, (f32[], f32[]) %tuple)
      %tuple.2 = (((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[])))
                   tuple(
                     ((f32[], f32[]), (f32[], f32[])) %tuple.1,
                     ((f32[], f32[]), (f32[], f32[])) %tuple.1
                   )
      %tuple.3 = ((((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))),
                  (((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))))
                   tuple(
                     (((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))) %tuple.2,
                     (((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))) %tuple.2
                   )
      %tuple.4 = (((((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))),
                   (((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[])))),
                  ((((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))),
                   (((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[])))))
                   tuple(
                     ((((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))),
                      (((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[])))) %tuple.3,
                     ((((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))),
                      (((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[])))) %tuple.3
                   )

      %get-tuple-element = ((((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))),
                            (((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))))
                             get-tuple-element(
                               (((((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))),
                                (((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[])))),
                                ((((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))),
                                (((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))))) %tuple.4
                             ), index=0
      %get-tuple-element.1 = (((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[])))
                               get-tuple-element(
                                 ((((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))),
                                  (((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[])))) %get-tuple-element
                               ), index=0
      %get-tuple-element.2 = ((f32[], f32[]), (f32[], f32[]))
                               get-tuple-element(
                                 (((f32[], f32[]), (f32[], f32[])), ((f32[], f32[]), (f32[], f32[]))) %get-tuple-element.1
                               ), index=0
      %get-tuple-element.3 = (f32[], f32[])
                                get-tuple-element(
                                  ((f32[], f32[]), (f32[], f32[])) %get-tuple-element.2
                                ), index=0
      ROOT %get-tuple-element.4 = f32[] get-tuple-element((f32[], f32[]) %get-tuple-element.3), index=0
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::GetTupleElement());

  Run(module.get(), /*change_expected=*/true);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Parameter(0));
}

TEST_F(TupleSimplifierTest, TupleOfGteInstructions) {
  // Verify that a tuple reconstructed from GTE instructions extracting from the
  // same tuple is collapsed.
  constexpr absl::string_view kModuleStr = R"(
    HloModule TupleOfGteInstructions, entry_computation_layout={((f32[], f32[], f32[]))->(f32[], f32[], f32[])}

    ENTRY %TupleOfGteInstructions (param: (f32[], f32[], f32[])) -> (f32[], f32[], f32[]) {
      %param = (f32[], f32[], f32[]) parameter(0)
      %get-tuple-element = f32[] get-tuple-element((f32[], f32[], f32[]) %param), index=0
      %get-tuple-element.1 = f32[] get-tuple-element((f32[], f32[], f32[]) %param), index=1
      %get-tuple-element.2 = f32[] get-tuple-element((f32[], f32[], f32[]) %param), index=2
      ROOT %tuple = (f32[], f32[], f32[]) tuple(f32[] %get-tuple-element, f32[] %get-tuple-element.1, f32[] %get-tuple-element.2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                        op::GetTupleElement()));

  Run(module.get(), /*change_expected=*/true);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Parameter(0));
}

TEST_F(TupleSimplifierTest, TupleOfGteNotRemovedIfOrderIsNotPreserved) {
  // Verify that a tuple constructed of GTE instructions operating on the same
  // tuple is NOT collapsed if the original tuple is not constructed.
  //
  // Note that, in the HLO program below, the final tuple is rearranged
  // following the permutation (0, 2, 1).
  constexpr absl::string_view kModuleStr = R"(
    HloModule TupleOfGteInstructions, entry_computation_layout={((f32[], f32[], f32[]))->(f32[], f32[], f32[])}

    ENTRY %TupleOfGteInstructions (param: (f32[], f32[], f32[])) -> (f32[], f32[], f32[]) {
      %param = (f32[], f32[], f32[]) parameter(0)
      %get-tuple-element = f32[] get-tuple-element((f32[], f32[], f32[]) %param), index=0
      %get-tuple-element.1 = f32[] get-tuple-element((f32[], f32[], f32[]) %param), index=1
      %get-tuple-element.2 = f32[] get-tuple-element((f32[], f32[], f32[]) %param), index=2
      ROOT %tuple = (f32[], f32[], f32[]) tuple(f32[] %get-tuple-element, f32[] %get-tuple-element.2, f32[] %get-tuple-element.1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Run(module.get(), /*change_expected=*/false);
}

TEST_F(TupleSimplifierTest, IncompatibleTuples) {
  // Verify that a tuple->GTE->tuple construct is not simplified if the input
  // and output tuple are not compatible shapes.
  constexpr absl::string_view kModuleStr = R"(
    HloModule IncompatibleTuples, entry_computation_layout={((f32[], f32[], f32[]))->(f32[], f32[])}

    ENTRY %IncompatibleTuples (param: (f32[], f32[], f32[])) -> (f32[], f32[]) {
      %param = (f32[], f32[], f32[]) parameter(0)
      %get-tuple-element = f32[] get-tuple-element((f32[], f32[], f32[]) %param), index=0
      %get-tuple-element.1 = f32[] get-tuple-element((f32[], f32[], f32[]) %param), index=1
      ROOT %tuple = (f32[], f32[]) tuple(f32[] %get-tuple-element, f32[] %get-tuple-element.1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Run(module.get(), /*change_expected=*/false);
}

TEST_F(TupleSimplifierTest, CanExcludeEntryComputation) {
  //  Verify that the root computation can be excluded
  constexpr absl::string_view kModuleStr = R"(
    HloModule CanExcludeEntryComputation, entry_computation_layout={((f32[], f32[], f32[]))->(f32[], f32[])}

    %c1 (param: (f32[], f32[], f32[])) -> (f32[], f32[], f32[]) {
      %param = (f32[], f32[], f32[]) parameter(0)
      %get-tuple-element = f32[] get-tuple-element((f32[], f32[], f32[]) %param), index=0
      %get-tuple-element.1 = f32[] get-tuple-element((f32[], f32[], f32[]) %param), index=1
      %get-tuple-element.2 = f32[] get-tuple-element((f32[], f32[], f32[]) %param), index=2
      ROOT %tuple = (f32[], f32[], f32[]) tuple(f32[] %get-tuple-element, f32[] %get-tuple-element.1, f32[] %get-tuple-element.2)
    }

    %c2 (param.1: (f32[], f32[], f32[])) -> (f32[], f32[], f32[]) {
      %param.1 = (f32[], f32[], f32[]) parameter(0)
      %get-tuple-element.3 = f32[] get-tuple-element((f32[], f32[], f32[]) %param.1), index=0
      %get-tuple-element.4 = f32[] get-tuple-element((f32[], f32[], f32[]) %param.1), index=1
      %get-tuple-element.5 = f32[] get-tuple-element((f32[], f32[], f32[]) %param.1), index=2
      ROOT %tuple.1 = (f32[], f32[], f32[]) tuple(f32[] %get-tuple-element.3, f32[] %get-tuple-element.4, f32[] %get-tuple-element.5)
    }

    ENTRY %e (param.2: (f32[], f32[], f32[])) -> (f32[], f32[]) {
      %param.2 = (f32[], f32[], f32[]) parameter(0)
      %call = (f32[], f32[], f32[]) call((f32[], f32[], f32[]) %param.2), to_apply=%c1
      %get-tuple-element.6 = f32[] get-tuple-element((f32[], f32[], f32[]) %call), index=0
      %call.1 = (f32[], f32[], f32[]) call((f32[], f32[], f32[]) %param.2), to_apply=%c2
      %get-tuple-element.7 = f32[] get-tuple-element((f32[], f32[], f32[]) %call.1), index=1
      %tuple.2 = (f32[], f32[]) tuple(f32[] %get-tuple-element.6, f32[] %get-tuple-element.7)
      %get-tuple-element.8 = f32[] get-tuple-element((f32[], f32[]) %tuple.2), index=0
      %get-tuple-element.9 = f32[] get-tuple-element((f32[], f32[]) %tuple.2), index=1
      ROOT %tuple.3 = (f32[], f32[]) tuple(f32[] %get-tuple-element.8, f32[] %get-tuple-element.9)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Run(module.get(), /*change_expected=*/true, /*exclude_entry=*/true);

  EXPECT_THAT(FindComputation(module.get(), "c1")->root_instruction(),
              op::Parameter(0));
  EXPECT_THAT(FindComputation(module.get(), "c2")->root_instruction(),
              op::Parameter(0));
  EXPECT_EQ(module->entry_computation()->instruction_count(), 9);
}

TEST_F(TupleSimplifierTest, ShardingInfoIsNotBeLost) {
  // Guards against simplifications that would incorrectly drop or change
  // sharding information.
  constexpr absl::string_view kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = s32[10] parameter(0), sharding={devices=[2]0,1}
      t = (s32[10]) tuple(p0)
      ROOT %gte = s32[10] get-tuple-element(t), index=0, sharding={replicated}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  // Expect no change because the sharding in the root instruction is not the
  // same as that of the parameter instruction.
  Run(module.get(), /*change_expected=*/false);
}

TEST_F(TupleSimplifierTest, NestedTuple) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = s32[10] parameter(0), sharding={devices=[2]0,1}
      p1 = s32[10] parameter(1), sharding={devices=[2]0,1}
      p2 = s32[10] parameter(2), sharding={devices=[2]0,1}
      p3 = s32[10] parameter(3), sharding={devices=[2]0,1}
      t = (s32[10], s32[10]) tuple(p0, p1), sharding={{devices=[2]0,1}, {devices=[2]0,1}}
      t2 = ((s32[10], s32[10]), s32[10]) tuple(t, p2), sharding={{devices=[2]0,1}, {devices=[2]0,1}, {devices=[2]0,1}}
      t3 = (((s32[10], s32[10]), s32[10]), s32[10]) tuple(t2, p3), sharding={{devices=[2]0,1}, {devices=[2]0,1}, {devices=[2]0,1}, {devices=[2]0,1}}
      gte0 = ((s32[10], s32[10]), s32[10]) get-tuple-element(t3), index=0, sharding={{replicated}, {replicated}, {replicated}}
      gte1 = (s32[10], s32[10]) get-tuple-element(gte0), index=0, sharding={{replicated}, {replicated}}
      gte2 = s32[10] get-tuple-element(gte1), index=1, sharding={devices=[2]0,1}
      gte3 = s32[10] get-tuple-element(gte1), index=0, sharding={replicated}
      ROOT to = (s32[10], s32[10]) tuple(gte2, gte3)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  Run(module.get(), /*change_expected=*/true);
  auto* p1 = FindInstruction(module.get(), "p1");
  auto* gte3 = FindInstruction(module.get(), "gte3");
  EXPECT_EQ(module->entry_computation()->root_instruction()->operand(0), p1);
  EXPECT_EQ(module->entry_computation()->root_instruction()->operand(1), gte3);
}

}  // namespace
}  // namespace xla
