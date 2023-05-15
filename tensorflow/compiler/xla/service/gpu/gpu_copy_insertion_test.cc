/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <optional>

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

int64_t CountCopies(const HloComputation& computation) {
  int64_t count = 0;
  for (const auto& instruction : computation.instructions()) {
    if (instruction->opcode() == HloOpcode::kCopy) {
      count++;
    }
  }
  return count;
}

int64_t CountCopies(const HloModule& module) {
  int64_t count = 0;
  for (const auto& computation : module.computations()) {
    count += CountCopies(*computation);
  }
  return count;
}

void ExpectOptionalTrue(std::optional<bool> value) {
  EXPECT_TRUE(value.has_value());
  CHECK(value.has_value());
  EXPECT_TRUE(*value);
}

void ExpectOptionalFalse(std::optional<bool> value) {
  EXPECT_TRUE(value.has_value());
  CHECK(value.has_value());
  EXPECT_FALSE(*value);
}

using GpuCopyInsertionTest = HloTestBase;

// This is some kind of end-to-end test for FusionCanShareBufferHint.
TEST_F(GpuCopyInsertionTest, DUSBitcastNoCopy) {
  const char* const kModuleString = R"(
HloModule bitcast_fusion

fused_computation.549 {
  param_1.8511 = bf16[15,1,2,2048,48,128]{3,5,4,2,1,0} parameter(1)
  bitcast.52601 = bf16[15,1,2,48,128,2048]{5,4,3,2,1,0} bitcast(param_1.8511)
  param_0.6313 = bf16[2,48,128,2048]{3,2,1,0} parameter(0)
  bitcast.52600 = bf16[1,1,2,48,128,2048]{5,4,3,2,1,0} bitcast(param_0.6313)
  param_2.5901 = s32[] parameter(2)
  constant_7564 = s32[] constant(0)
  compare.3477 = pred[] compare(param_2.5901, constant_7564), direction=LT
  constant_11524 = s32[] constant(15)
  add.6580 = s32[] add(param_2.5901, constant_11524)
  select.5360 = s32[] select(compare.3477, add.6580, param_2.5901)
  ROOT dynamic-update-slice.325 = bf16[15,1,2,48,128,2048]{5,4,3,2,1,0} dynamic-update-slice(bitcast.52601, bitcast.52600, select.5360, constant_7564, constant_7564, constant_7564, constant_7564, constant_7564)
}

condition {
  constant_6915 = s32[] constant(15)
  param.218 = (bf16[2,48,128,2048]{3,2,1,0}, bf16[15,1,2,2048,48,128]{3,5,4,2,1,0}, s32[]) parameter(0)
  get-tuple-element.3714 = s32[] get-tuple-element(param.218), index=2
  ROOT compare.1738 = pred[] compare(get-tuple-element.3714, constant_6915), direction=LT
}

body {
  tuple_param = (bf16[2,48,128,2048]{3,2,1,0}, bf16[15,1,2,2048,48,128]{3,5,4,2,1,0}, s32[]) parameter(0)
  param_0 = bf16[2,48,128,2048]{3,2,1,0} get-tuple-element(tuple_param), index=0
  param_1 = bf16[15,1,2,2048,48,128]{3,5,4,2,1,0} get-tuple-element(tuple_param), index=1
  param_2 = s32[] get-tuple-element(tuple_param), index=2
  fusion.549 = bf16[15,1,2,48,128,2048]{5,4,3,2,1,0} fusion(param_0, param_1, param_2), kind=kLoop, calls=fused_computation.549
  bitcast = bf16[15,1,2,2048,48,128]{3,5,4,2,1,0} bitcast(fusion.549)
  constant_one = s32[] constant(1)
  add = s32[] add(param_2, constant_one)
  ROOT tuple = (bf16[2,48,128,2048]{3,2,1,0}, bf16[15,1,2,2048,48,128]{3,5,4,2,1,0}, s32[]) tuple(param_0, bitcast, add)
}

ENTRY main {
  param_0 = bf16[2,48,128,2048]{3,2,1,0} parameter(0)
  param_1 = bf16[15,1,2,2048,48,128]{3,5,4,2,1,0} parameter(1)
  zero = s32[] constant(0)
  tuple = tuple(param_0, param_1, zero)
  ROOT while = (bf16[2,48,128,2048]{3,2,1,0}, bf16[15,1,2,2048,48,128]{3,5,4,2,1,0}, s32[]) while(tuple), condition=condition, body=body, backend_config="{\"known_trip_count\":{\"n\":\"15\"}}"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  CopyInsertion copy_insertion(GpuCompiler::FusionCanShareBufferHint,
                               /*use_region_based_live_range_analysis=*/0);
  ASSERT_IS_OK(copy_insertion.Run(module.get(), {"foobar"}).status());
  VLOG(2) << module->ToString();
  // Copy insertion adds two copies inside the entry computation.
  EXPECT_EQ(CountCopies(*module->entry_computation()), 2);
  // We expect that for fusion.549, no further copy needs to be added to the
  // module.
  EXPECT_EQ(CountCopies(*module), 2);
}

using FusionCanShareBufferHintTest = HloTestBase;

TEST_F(FusionCanShareBufferHintTest, BufferCanBeSharedSameShape) {
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  param_0.1 = f32[2,3]{1,0} parameter(0)
  neg = f32[2,3]{1,0} negate(param_0.1)
  ROOT mul = f32[2,3]{1,0} multiply(param_0.1, neg)
}

ENTRY main {
  param_0 = f32[2,3]{1,0} parameter(0)
  ROOT fusion = f32[2,3]{1,0} fusion(param_0), kind=kLoop, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(
      GpuCompiler::FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
}

TEST_F(FusionCanShareBufferHintTest, BufferCanBeSharedBitcastedShape) {
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  param_0.1 = f32[2,3]{1,0} parameter(0)
  neg = f32[2,3]{1,0} negate(param_0.1)
  mul = f32[2,3]{1,0} multiply(param_0.1, neg)
  ROOT bitcast = f32[6]{0} bitcast(mul)
}

ENTRY main {
  param_0 = f32[2,3]{1,0} parameter(0)
  ROOT fusion = f32[6]{0} fusion(param_0), kind=kLoop, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(
      GpuCompiler::FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
}

TEST_F(FusionCanShareBufferHintTest,
       BufferCanBeSharedConvertedShapeSameByteWidth) {
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  param_0.1 = f32[2,3]{1,0} parameter(0)
  neg = f32[2,3]{1,0} negate(param_0.1)
  mul = f32[2,3]{1,0} multiply(param_0.1, neg)
  ROOT convert = s32[2,3]{1,0} convert(mul)
}

ENTRY main {
  param_0 = f32[2,3]{1,0} parameter(0)
  ROOT fusion = s32[2,3]{1,0} fusion(param_0), kind=kLoop, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(
      GpuCompiler::FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
}

TEST_F(FusionCanShareBufferHintTest, BufferCanBeSharedMultiOutputFusion) {
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  param_0.1 = f32[2,3]{1,0} parameter(0)
  param_1.1 = f32[2,3]{1,0} parameter(1)
  neg = f32[2,3]{1,0} negate(param_1.1)
  mul = f32[2,3]{1,0} multiply(param_0.1, neg)
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3]{1,0}) tuple(mul, neg)
}

ENTRY main {
  param_0 = f32[2,3]{1,0} parameter(0)
  param_1 = f32[2,3]{1,0} parameter(1)
  ROOT fusion = (f32[2,3]{1,0}, f32[2,3]{1,0}) fusion(param_0, param_1), kind=kLoop, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(
      GpuCompiler::FusionCanShareBufferHint(fusion, fusion->operand(0), {0}));
  // The second operand cannot share the buffer with the second fusion output,
  // because the 'neg' op is also used on the path to the first fusion output.
  ExpectOptionalFalse(
      GpuCompiler::FusionCanShareBufferHint(fusion, fusion->operand(1), {1}));
  // The first operand cannot share the buffer with the second fusion output,
  // because there is no path between them.
  ExpectOptionalFalse(
      GpuCompiler::FusionCanShareBufferHint(fusion, fusion->operand(0), {1}));
}

TEST_F(FusionCanShareBufferHintTest,
       BufferCannotBeSharedConvertedShapeDifferentByteWidth) {
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  param_0.1 = f32[2,3]{1,0} parameter(0)
  neg = f32[2,3]{1,0} negate(param_0.1)
  mul = f32[2,3]{1,0} multiply(param_0.1, neg)
  ROOT convert = f16[2,3]{1,0} convert(mul)
}

ENTRY main {
  param_0 = f32[2,3]{1,0} parameter(0)
  ROOT fusion = f16[2,3]{1,0} fusion(param_0), kind=kLoop, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(
      GpuCompiler::FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
}

TEST_F(FusionCanShareBufferHintTest, BufferCannotBeSharedShapeBitcastConvert) {
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  param_0.1 = s32[3]{0} parameter(0)
  neg = s32[3]{0} negate(param_0.1)
  mul = s32[3]{0} multiply(param_0.1, neg)
  ROOT bitcast-convert = s16[3,2]{1,0} bitcast-convert(mul)
}

ENTRY main {
  param_0 = s32[3]{0} parameter(0)
  ROOT fusion = s16[3,2]{1,0} fusion(param_0), kind=kLoop, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(
      GpuCompiler::FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
}

TEST_F(FusionCanShareBufferHintTest, BufferCannotBeSharedDueToCopy) {
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  param_0.1 = s32[2,3]{0,1} parameter(0)
  copy = s32[2,3]{1,0} copy(param_0.1)
  ROOT neg = s32[2,3]{1,0} negate(copy)
}

ENTRY main {
  param_0 = s32[2,3]{0,1} parameter(0)
  ROOT fusion = s32[2,3]{1,0} fusion(param_0), kind=kLoop, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(
      GpuCompiler::FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
}

TEST_F(FusionCanShareBufferHintTest, BufferCannotBeSharedDueToTranspose) {
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  param_0.1 = s32[2,3]{1,0} parameter(0)
  transpose = s32[3,2]{1,0} transpose(param_0.1), dimensions={1,0}
  ROOT neg = s32[3,2]{1,0} negate(transpose)
}

ENTRY main {
  param_0 = s32[2,3]{1,0} parameter(0)
  ROOT fusion = s32[3,2]{1,0} fusion(param_0), kind=kLoop, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(
      GpuCompiler::FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
}

TEST_F(FusionCanShareBufferHintTest,
       BufferCannotBeSharedDueToReduceAndBroadcast) {
  const char* const kModuleString = R"(
HloModule fusion

add {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

fused_computation {
  param_0.1 = s32[3]{0} parameter(0)
  broadcast = s32[3,2]{1,0} broadcast(param_0.1), dimensions={0}
  zero = s32[] constant(0)
  ROOT reduce = s32[3]{0} reduce(broadcast, zero), to_apply=add, dimensions={1}
}

ENTRY main {
  param_0 = s32[3]{0} parameter(0)
  ROOT fusion = s32[3]{0} fusion(param_0), kind=kInput, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(
      GpuCompiler::FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
