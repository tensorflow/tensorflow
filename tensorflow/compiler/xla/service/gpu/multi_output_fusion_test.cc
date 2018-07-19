/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/multi_output_fusion.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace gpu {

using MultiOutputFusionTest = HloTestBase;

const char kModulePrefix[] = R"(
    HloModule test_module

    scalar_add_computation {
      scalar_lhs.0 = f32[] parameter(0)
      scalar_rhs.0 = f32[] parameter(1)
      ROOT add.0 = f32[] add(scalar_lhs.0, scalar_rhs.0)
    }
    scalar_mul_computation {
      scalar_lhs.1 = f32[] parameter(0)
      scalar_rhs.1 = f32[] parameter(1)
      ROOT mul.1 = f32[] multiply(scalar_lhs.1, scalar_rhs.1)
    })";

TEST_F(MultiOutputFusionTest, MultiOutputFusionSiblingReduceAndReduceFusion) {
  // Fusion with reduce instruction root and a sibling reduce instruction
  // sharing the same input param.
  auto module = ParseHloString(tensorflow::strings::StrCat(kModulePrefix, R"(
    fused_computation {
      p1.1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      ROOT reduce.1 = f32[512]{0} reduce(mul, const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      const.2 = f32[] constant(1)
      fusion = f32[512] fusion(p0, p1), kind=kInput, calls=fused_computation
      reduce.2 = f32[512]{0} reduce(p1, const.2), dimensions={0,2,3}, to_apply=scalar_add_computation
      ROOT root = (f32[512]{0}, f32[512]{0}) tuple(fusion, reduce.2)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Reduce(), op::Reduce()));
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionDifferentReduceInputShapes) {
  auto module = ParseHloString(tensorflow::strings::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = f32[6400]{0} parameter(1)
      mul = f32[6400]{0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      ROOT reduce.1 = f32[] reduce(mul, const.1), dimensions={0}, to_apply=scalar_add_computation
    }

    fused_computation_2 {
      p1.2 = f32[6400]{0} parameter(1)
      r1 = f32[64,100]{0,1} reshape(p1.2)
      const.2 = f32[] parameter(0)
      ROOT reduce.2 = f32[] reduce(r1, const.2), dimensions={1,0}, to_apply=scalar_mul_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[6400]{0} parameter(1)
      fusion.1 = f32[] fusion(p0, p1), kind=kInput, calls=fused_computation_1
      fusion.2 = f32[] fusion(p0, p1), kind=kInput, calls=fused_computation_2
      ROOT root = (f32[], f32[]) tuple(fusion.1, fusion.2)
    })"))
                    .ValueOrDie();
  ASSERT_FALSE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionDifferentReduceOutputShapes) {
  auto module = ParseHloString(tensorflow::strings::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = f32[10,10]{1,0} parameter(1)
      mul = f32[10,10]{1,0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      ROOT reduce.1 = f32[] reduce(mul, const.1), dimensions={0,1}, to_apply=scalar_add_computation
    }

    fused_computation_2 {
      p1.2 = f32[10,10]{1,0} parameter(1)
      const.2 = f32[10]{0} parameter(0)
      ROOT reduce.2 = f32[10]{0} reduce(p1.2, const.2), dimensions={0}, to_apply=scalar_mul_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1.3 = f32[10,10]{1,0} parameter(1)
      fusion.1 = f32[] fusion(p0, p1.3), kind=kInput, calls=fused_computation_1
      p2 = f32[] parameter(2)
      fusion.2 = f32[10]{0} fusion(p2, p1.3), kind=kInput, calls=fused_computation_2
      ROOT root = (f32[], f32[10]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .ValueOrDie();
  ASSERT_FALSE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionSiblingReduceFusions) {
  // Two sibling fusions with reduce instruction roots sharing the same input
  // param.
  auto module = ParseHloString(tensorflow::strings::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      ROOT reduce.1 = f32[512]{0} reduce(mul, const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
    }

    fused_computation_2 {
      p1.2 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      const.2 = f32[] parameter(0)
      ROOT reduce.2 = f32[512]{0} reduce(p1.2, const.2), dimensions={0,2,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      fusion.1 = f32[512] fusion(p0, p1), kind=kInput, calls=fused_computation_1
      fusion.2 = f32[512] fusion(p0, p1), kind=kInput, calls=fused_computation_2
      ROOT root = (f32[512]{0}, f32[512]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Reduce(), op::Reduce()));
}

TEST_F(MultiOutputFusionTest,
       MultiOutputFusionSiblingReduceAndReduceMultiOutputFusion) {
  // Multi-output fusion with two reduce instructions root and a sibling reduce
  // instruction sharing the same input param.
  auto module = ParseHloString(tensorflow::strings::StrCat(kModulePrefix, R"(
    fused_computation (p0: f32[128,512,28,28]) -> (f32[512], f32[512]) {
      const.1 = f32[] constant(1)
      p0.1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(f32[128,512,28,28]{3,2,1,0} p0.1, f32[128,512,28,28]{3,2,1,0} p0.1)
      reduce.1 = f32[512]{0} reduce(f32[128,512,28,28]{3,2,1,0} mul, f32[] const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
      reduce.2 = f32[512]{0} reduce(f32[128,512,28,28]{3,2,1,0} p0.1, f32[] const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
      ROOT tuple = (f32[512]{0}, f32[512]{0}) tuple(f32[512]{0} reduce.1, f32[512]{0} reduce.2)
    }

    ENTRY entry (p0: f32[128,512,28,28]) -> (f32[512], f32[512], f32[512]) {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      const = f32[] constant(1)
      fusion = (f32[512]{0}, f32[512]{0}) fusion(f32[128,512,28,28]{3,2,1,0} p0), kind=kInput, calls=fused_computation
      get-tuple-element = f32[512]{0} get-tuple-element((f32[512]{0}, f32[512]{0}) fusion), index=0
      get-tuple-element.1 = f32[512]{0} get-tuple-element((f32[512]{0}, f32[512]{0}) fusion), index=1
      reduce.3 = f32[512]{0} reduce(p0, const), dimensions={0,2,3}, to_apply=scalar_add_computation
      ROOT root = (f32[512]{0}, f32[512]{0}, f32[512]{0}) tuple(f32[512]{0} get-tuple-element, f32[512]{0} get-tuple-element.1, f32[512]{0} reduce.3)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Reduce(), op::Reduce(), op::Reduce()));
}

TEST_F(MultiOutputFusionTest,
       MultiOutputFusionSiblingFusionCheckAgainstReduceOperand) {
  // Verify that if we already have a multi-output fusion that we prefer to pick
  // a reduce op from its operands for checking shape compatibility.
  auto module = ParseHloString(tensorflow::strings::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = f32[10,10]{1,0} parameter(1)
      mul = f32[10,10]{1,0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      reduce.1 = f32[] reduce(p1.1, const.1), dimensions={0,1}, to_apply=scalar_add_computation
      ROOT tuple = (f32[10,10], f32[]) tuple(mul, reduce.1)
    }

    fused_computation_2 {
      p1.2 = f32[10,10]{1,0} parameter(1)
      const.2 = f32[10] parameter(0)
      ROOT reduce.2 = f32[10] reduce(p1.2, const.2), dimensions={0}, to_apply=scalar_mul_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[10,10]{1,0} parameter(1)
      p2 = f32[10]{0} parameter(2)
      fusion.1 = (f32[10,10], f32[10]) fusion(p0, p1), kind=kInput, calls=fused_computation_1
      get-tuple-element.1 = f32[10,10] get-tuple-element((f32[10,10], f32[10]) fusion.1), index=0
      get-tuple-element.2 = f32[] get-tuple-element((f32[10,10], f32[10]) fusion.1), index=1
      fusion.2 = f32[10] fusion(p2, p1), kind=kInput, calls=fused_computation_2
      ROOT root = (f32[10,10], f32[], f32[10]) tuple(get-tuple-element.1, get-tuple-element.2, fusion.2)
    })"))
                    .ValueOrDie();
  ASSERT_FALSE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionTwoLoops) {
  auto module = ParseHloString(tensorflow::strings::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[6400]{0} parameter(0)
      ROOT mul = f32[6400]{0} multiply(p0.1, p0.1)
    }

    fused_computation_2 {
      p0.2 = f32[6400]{0} parameter(0)
      const.2 = f32[] constant(1)
      ROOT div = f32[6400]{0} divide(p0.2, const.2)
    }

    ENTRY entry {
      p0 = f32[6400]{0} parameter(0)
      fusion.1 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_2
      ROOT root = (f32[6400]{0}, f32[6400]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Multiply(), op::Divide()));
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionLoopFusionAndReduce) {
  auto module = ParseHloString(tensorflow::strings::StrCat(kModulePrefix, R"(
    fused_add {
      p0.1 = f32[2,2,2]{2,1,0} parameter(0)
      p1.1 = f32[2,2,2]{2,1,0} parameter(1)
      ROOT add = f32[2,2,2]{2,1,0} add(p0.1, p1.1)
    }

    ENTRY reduce {
      p0 = f32[2,2,2]{2,1,0} parameter(0)
      p1 = f32[2,2,2]{2,1,0} parameter(1)
      c0 = f32[] constant(0)
      add = f32[2,2,2]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_add
      reduce = f32[2,2]{1,0} reduce(add, c0), dimensions={2}, to_apply=scalar_add_computation
      ROOT root = (f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) tuple(reduce, add)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(), op::GetTupleElement()));
  const HloInstruction* fusion = root->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Reduce(), op::Add()));
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionLoopFusionAndReduceFusion) {
  auto module = ParseHloString(tensorflow::strings::StrCat(kModulePrefix, R"(
    fused_select {
      p1.1 = f32[2,2,2]{2,1,0} parameter(1)
      c0 = f32[] constant(0)
      broadcast = f32[2,2,2]{2,1,0} broadcast(f32[] c0), dimensions={}
      greater-than = pred[2,2,2]{2,1,0} greater-than(f32[2,2,2]{2,1,0} p1.1, f32[2,2,2]{2,1,0} broadcast)
      p0.1 = f32[2,2,2]{2,1,0} parameter(0)
      ROOT select = f32[2,2,2]{2,1,0} select(pred[2,2,2]{2,1,0} greater-than, f32[2,2,2]{2,1,0} p0.1, f32[2,2,2]{2,1,0} broadcast)
    }

    fused_reduce {
      p0.2 = f32[2,2,2]{2,1,0} parameter(0)
      c1 = f32[] constant(0)
      r1 = f32[2,2]{1,0} reduce(p0.2, c1), dimensions={2}, to_apply=scalar_add_computation
      mul = f32[2,2,2]{2,1,0} multiply(p0.2, p0.2)
      r2 = f32[2,2]{1,0} reduce(mul, c1), dimensions={2}, to_apply=scalar_add_computation
      ROOT tuple = (f32[2,2]{1,0}, f32[2,2]{1,0}) tuple(r1, r2)
    }

    ENTRY reduce {
      p0 = f32[2,2,2]{2,1,0} parameter(0)
      p1 = f32[2,2,2]{2,1,0} parameter(1)
      select = f32[2,2,2]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_select
      fusion = (f32[2,2]{1,0}, f32[2,2]{1,0}) fusion(select), kind=kInput, calls=fused_reduce
      gte0 = f32[2,2]{1,0} get-tuple-element(fusion), index=0
      gte1 = f32[2,2]{1,0} get-tuple-element(fusion), index=1
      ROOT root = (f32[2,2]{1,0}, f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) tuple(gte1, gte1, select)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                              op::GetTupleElement()));
  const HloInstruction* fusion = root->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Reduce(), op::Reduce(), op::Select()));
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionDoNotFuseLoopReduceFusion) {
  auto module = ParseHloString(tensorflow::strings::StrCat(kModulePrefix, R"(
    fused_element_wise {
      p0.1 = f32[2,2,2]{2,1,0} parameter(0)
      p1.1 = f32[2,2,2]{2,1,0} parameter(1)
      ROOT root = f32[2,2,2]{2,1,0} add(p0.1, p1.1)
    }

    fused_reduce {
      p0.2 = f32[2,2,2]{2,1,0} parameter(0)
      mul = f32[2,2,2]{2,1,0} multiply(f32[2,2,2]{2,1,0} p0.2, f32[2,2,2]{2,1,0} p0.2)
      c1 = f32[] constant(0)
      ROOT reduce = f32[2,2]{1,0} reduce(f32[2,2,2]{2,1,0} mul, f32[] c1), dimensions={1}, to_apply=scalar_add_computation
    }

    ENTRY reduce {
      p0 = f32[2,2,2]{2,1,0} parameter(0)
      p1 = f32[2,2,2]{2,1,0} parameter(1)
      element_wise = f32[2,2,2]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_element_wise
      fusion = (f32[2,2]{1,0}, f32[2,2]{1,0}) fusion(element_wise), kind=kLoop, calls=fused_reduce
      ROOT root = (f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) tuple(fusion, element_wise)
    })"))
                    .ValueOrDie();
  ASSERT_FALSE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
}

}  // namespace gpu
}  // namespace xla
