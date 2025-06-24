/* Copyright 2023 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/statusor.h"

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

class GpuCopyInsertionTest : public HloHardwareIndependentTestBase {
 public:
  CopyInsertion CreateCopyInsertion() const {
    return CopyInsertion(&alias_info_,
                         /*use_region_based_live_range_analysis=*/0);
  }

 private:
  const se::DeviceDescription device_description_{
      xla::gpu::TestGpuDeviceInfo::CudaOrRocmDeviceInfo()};
  GpuAliasInfo alias_info_{&device_description_};
};

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
  add = s32[] add(param_2, constant_one), control-predecessors={fusion.549}
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

  CopyInsertion copy_insertion = CreateCopyInsertion();
  ASSERT_IS_OK(copy_insertion.Run(module.get(), {"foobar"}).status());
  VLOG(2) << module->ToString();
  // Copy insertion adds two copies inside the entry computation.
  EXPECT_EQ(CountCopies(*module->entry_computation()), 2);
  // We expect that for fusion.549, no further copy needs to be added to the
  // module.
  EXPECT_EQ(CountCopies(*module), 2);
}

class FusionCanShareBufferHintTest : public HloHardwareIndependentTestBase {
 public:
  std::optional<bool> FusionCanShareBufferHint(const HloInstruction* fusion,
                                               const HloInstruction* operand,
                                               const ShapeIndex& user_index) {
    return alias_info_.MayAlias(operand, {}, fusion, user_index);
  }

 private:
  const se::DeviceDescription device_description_{
      xla::gpu::TestGpuDeviceInfo::CudaOrRocmDeviceInfo()};
  GpuAliasInfo alias_info_{&device_description_};
};

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
  ExpectOptionalTrue(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
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
  ExpectOptionalTrue(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
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
  ExpectOptionalTrue(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
}

TEST_F(FusionCanShareBufferHintTest, BufferCanBeSharedMultiOutputFusion) {
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  param_0.1 = f32[2,3]{1,0} parameter(0)
  param_1.1 = f32[2,3]{1,0} parameter(1)
  neg = f32[2,3]{1,0} negate(param_1.1)
  mul = f32[2,3]{1,0} multiply(param_0.1, neg)
  transpose = f32[3,2]{1,0} transpose(neg), dimensions={1,0}
  ROOT tuple = (f32[2,3]{1,0}, f32[2,3]{1,0}, f32[3,2]{1,0}) tuple(mul, neg, transpose)
}

ENTRY main {
  param_0 = f32[2,3]{1,0} parameter(0)
  param_1 = f32[2,3]{1,0} parameter(1)
  ROOT fusion = (f32[2,3]{1,0}, f32[2,3]{1,0}, f32[3,2]{1,0}) fusion(param_0, param_1), kind=kLoop, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(FusionCanShareBufferHint(fusion, fusion->operand(0), {0}));
  // The second operand cannot share the buffer with the second fusion output,
  // because the 'neg' op is also used by a non-elementwise op.
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(1), {1}));
  // The first operand cannot share the buffer with the second fusion output,
  // because there is no path between them.
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(0), {1}));
}

TEST_F(FusionCanShareBufferHintTest,
       BufferCanBeSharedMultiOutputFusionTwoReachableOutputs) {
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
  ExpectOptionalTrue(FusionCanShareBufferHint(fusion, fusion->operand(0), {0}));
  // The first operand cannot share the buffer with the second fusion output,
  // because there is no path between them.
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(0), {1}));
  // The second operand can share the buffer with the second fusion output and
  // the first fusion output.
  ExpectOptionalTrue(FusionCanShareBufferHint(fusion, fusion->operand(1), {0}));
  ExpectOptionalTrue(FusionCanShareBufferHint(fusion, fusion->operand(1), {1}));
}

TEST_F(FusionCanShareBufferHintTest, BufferCanBeSharedReductionEmitter) {
  constexpr char kModuleString[] = R"(
HloModule TestModule

%maximum {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %res = f32[] maximum(%lhs, %rhs)
}

%fused_computation {
  %lhs = f32[3,40] parameter(0)
  %rhs = f32[3,40] parameter(1)
  %add = f32[3,40] add(%lhs, %rhs)
  %bc = f32[120] bitcast(%add)
  %init = f32[] constant(-inf)
  %max = f32[] reduce(%bc, %init), dimensions={0}, to_apply=%maximum
  ROOT %result = (f32[], f32[3,40]) tuple(%max, %add)
}

ENTRY %main {
  %lhs = f32[3,40] parameter(0)
  %rhs = f32[3,40] parameter(1)
  ROOT %fusion = (f32[], f32[3,40]) fusion(%lhs, %rhs),
      kind=kLoop, calls=%fused_computation
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(FusionCanShareBufferHint(fusion, fusion->operand(0), {1}));
}

TEST_F(FusionCanShareBufferHintTest,
       BufferCannotBeSharedScatterMultiOutputFusion) {
  // This is a fusion that we would normally not create because it cannot be
  // emitted in-place. Still check whether buffer sharing logic would handle it
  // correctly.
  const char* const kModuleString = R"(
    HloModule fusion

    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    fused_computation {
      p0 = s32[3,3] parameter(0)
      p1 = s32[3] parameter(1)
      indices = s32[3] add(p1, p1)
      p2 = s32[3,3] parameter(2)
      updates = s32[3,3] add(p2, p2)
      add = s32[3,3] add(p0, p0)
      scatter = s32[3,3] scatter(p0, indices, updates),
          to_apply=add,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
      ROOT output = (s32[3,3], s32[3,3]) tuple(scatter, add)
    }

    ENTRY main {
      parameter0 = s32[3,3] parameter(0)
      parameter1 = s32[3] parameter(1)
      parameter2 = s32[3,3] parameter(2)
      ROOT fusion = (s32[3,3], s32[3,3]) fusion(parameter0, parameter1, parameter2), kind=kInput, calls=fused_computation
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  // We expect that no buffer can be shared, because when Scatter is involved,
  // the only buffer we can potentially share is the first operand of scatter,
  // but if that is also used for a different fusion output, it will not work
  // due to potentially different access patterns.
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(0), {0}));
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(0), {1}));
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(1), {0}));
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(1), {1}));
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(2), {0}));
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(2), {1}));
}

TEST_F(FusionCanShareBufferHintTest, BufferCanBeSharedScatterFusion) {
  const char* const kModuleString = R"(
    HloModule fusion

    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    fused_computation {
      p0 = s32[3,3] parameter(0)
      p1 = s32[3] parameter(1)
      indices = s32[3] add(p1, p1)
      p2 = s32[3,3] parameter(2)
      updates = s32[3,3] add(p2, p2)
      ROOT scatter = s32[3,3] scatter(p0, indices, updates),
          to_apply=add,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }

    ENTRY main {
      parameter0 = s32[3,3] parameter(0)
      parameter1 = s32[3] parameter(1)
      parameter2 = s32[3,3] parameter(2)
      ROOT fusion = s32[3,3] fusion(parameter0, parameter1, parameter2), kind=kInput, calls=fused_computation
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
  ExpectOptionalFalse(FusionCanShareBufferHint(fusion, fusion->operand(1), {}));
  ExpectOptionalFalse(FusionCanShareBufferHint(fusion, fusion->operand(2), {}));
}

TEST_F(FusionCanShareBufferHintTest, BufferCannotBeSharedScatterFusion) {
  // This is a fusion that we would normally not create because it cannot be
  // emitted in-place. Still check whether buffer sharing logic would handle it
  // correctly.
  const char* const kModuleString = R"(
    HloModule fusion

    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    fused_computation {
      p0 = s32[3,3] parameter(0)
      p1 = s32[3] parameter(1)
      indices = s32[3] add(p1, p1)
      updates = s32[3,3] add(p0, p0)
      ROOT scatter = s32[3,3] scatter(p0, indices, updates),
          to_apply=add,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }

    ENTRY main {
      parameter0 = s32[3,3] parameter(0)
      parameter1 = s32[3] parameter(1)
      ROOT fusion = s32[3,3] fusion(parameter0, parameter1), kind=kInput, calls=fused_computation
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
  ExpectOptionalFalse(FusionCanShareBufferHint(fusion, fusion->operand(1), {}));
}

TEST_F(FusionCanShareBufferHintTest, BufferCanBeSharedVariadicScatterFusion) {
  // We currently don't have variadic scatter fusions on GPU, but just in case
  // we verify here that buffer sharing logic could handle it.
  const char* const kModuleString = R"(
    HloModule fusion

    add_mul {
      lhs_0 = s32[] parameter(0)
      rhs_0 = s32[] parameter(2)
      add = s32[] add(lhs_0, rhs_0)
      lhs_1 = s32[] parameter(1)
      rhs_1 = s32[] parameter(3)
      mul = s32[] multiply(lhs_1, rhs_1)
      ROOT tuple = (s32[], s32[]) tuple(add, mul)
    }

    fused_computation {
      p0 = s32[3,3] parameter(0)
      p1 = s32[3,3] parameter(1)
      p2 = s32[3] parameter(2)
      p3 = s32[3,2] parameter(3)
      p4 = s32[3,2] parameter(4)
      indices = s32[3] add(p2, p2)
      ROOT scatter = (s32[3,3], s32[3,3]) scatter(p0, p1, indices, p3, p4),
          to_apply=add_mul,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }

    ENTRY main {
      parameter0 = s32[3,3] parameter(0)
      parameter1 = s32[3,3] parameter(1)
      parameter2 = s32[3] parameter(2)
      parameter3 = s32[3,2] parameter(3)
      parameter4 = s32[3,2] parameter(4)
      ROOT fusion = (s32[3,3], s32[3,3]) fusion(parameter0, parameter1, parameter2, parameter3, parameter4), kind=kInput, calls=fused_computation
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(FusionCanShareBufferHint(fusion, fusion->operand(0), {0}));
  ExpectOptionalTrue(FusionCanShareBufferHint(fusion, fusion->operand(1), {1}));
}

TEST_F(FusionCanShareBufferHintTest,
       BufferCannotBeSharedScatterFusionDuplicateScatterOperand) {
  // This is a fusion that we would normally not create because it cannot be
  // emitted in-place. Still check whether buffer sharing logic would handle it
  // correctly.
  const char* const kModuleString = R"(
    HloModule fusion

    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    fused_computation {
      p0 = s32[3,3] parameter(0)
      p1 = s32[3] parameter(1)
      indices = s32[3] add(p1, p1)
      ROOT scatter = s32[3,3] scatter(p0, indices, p0),
          to_apply=add,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }

    ENTRY main {
      parameter0 = s32[3,3] parameter(0)
      parameter1 = s32[3] parameter(1)
      ROOT fusion = s32[3,3] fusion(parameter0, parameter1), kind=kInput, calls=fused_computation
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
  ExpectOptionalFalse(FusionCanShareBufferHint(fusion, fusion->operand(1), {}));
}

TEST_F(FusionCanShareBufferHintTest,
       BufferCannotBeSharedVariadicScatterFusion) {
  // This is a fusion that we would normally not create because it cannot be
  // emitted in-place. Still check whether buffer sharing logic would handle it
  // correctly.
  const char* const kModuleString = R"(
    HloModule fusion

    add_mul {
      lhs_0 = s32[] parameter(0)
      rhs_0 = s32[] parameter(2)
      add = s32[] add(lhs_0, rhs_0)
      lhs_1 = s32[] parameter(1)
      rhs_1 = s32[] parameter(3)
      mul = s32[] multiply(lhs_1, rhs_1)
      ROOT tuple = (s32[], s32[]) tuple(add, mul)
    }

    fused_computation {
      p0 = s32[3,3] parameter(0)
      p1 = s32[3,3] parameter(1)
      p2 = s32[3] parameter(2)
      indices = s32[3] add(p2, p2)
      ROOT scatter = (s32[3,3], s32[3,3]) scatter(p0, p1, indices, p0, p1),
          to_apply=add_mul,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }

    ENTRY main {
      parameter0 = s32[3,3] parameter(0)
      parameter1 = s32[3,3] parameter(1)
      parameter2 = s32[3] parameter(2)
      ROOT fusion = (s32[3,3], s32[3,3]) fusion(parameter0, parameter1, parameter2), kind=kInput, calls=fused_computation
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(0), {0}));
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(1), {1}));
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
  ExpectOptionalFalse(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
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
  ExpectOptionalFalse(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
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
  ExpectOptionalFalse(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
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
  ExpectOptionalFalse(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
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
  ExpectOptionalFalse(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
}

TEST_F(FusionCanShareBufferHintTest,
       BufferCanBeSharedBecauseDUSAndDSAccessSameSlice) {
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  param_0.1 = s32[6]{0} parameter(0)
  bitcast = s32[2,3]{1,0} bitcast(param_0.1)
  zero = s32[] constant(0)
  param_1.1 = s32[] parameter(1)
  dynamic-slice = s32[1,2]{1,0} dynamic-slice(bitcast, param_1.1, zero), dynamic_slice_sizes={1,2}
  one = s32[] constant(1)
  broadcast = s32[1,2]{1,0} broadcast(one), dimensions={}
  add = s32[1,2] add(dynamic-slice, broadcast)
  dynamic-update-slice = s32[2,3]{1,0} dynamic-update-slice(bitcast, add, param_1.1, zero)
  ROOT bitcast.1 = s32[6]{0} bitcast(dynamic-update-slice)
}

ENTRY main {
  param_0 = s32[6]{0} parameter(0)
  param_1 = s32[] parameter(1)
  ROOT fusion = s32[6]{0} fusion(param_0, param_1), kind=kInput, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
}

TEST_F(FusionCanShareBufferHintTest,
       BufferCannotBeSharedWhenOtherUserIsTransposeUser) {
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  p0 = f32[100,110,120]{2,1,0} parameter(0)
  p1 = f32[120,110,100]{2,1,0} parameter(1)
  zero = f32[] constant(0.0)
  broadcast = f32[120,110,100]{2,1,0} broadcast(zero), dimensions={}
  maximum = f32[120,110,100]{2,1,0} maximum(broadcast, p1)
  t = f32[120,110,100]{2,1,0} transpose(p0), dimensions={2,1,0}
  add = f32[120,110,100]{2,1,0} add(t, maximum)
  ROOT res = (f32[120,110,100]{2,1,0}, f32[120,110,100]{2,1,0}) tuple(add, maximum)
}

ENTRY main {
  param_0 = f32[100,110,120]{2,1,0} parameter(0)
  param_1 = f32[120,110,100]{2,1,0} parameter(1)
  ROOT fusion = (f32[120,110,100]{2,1,0}, f32[120,110,100]{2,1,0}) fusion(param_0, param_1), kind=kInput, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(0), {0}));
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(0), {1}));
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(1), {0}));
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(1), {1}));
}

TEST_F(FusionCanShareBufferHintTest,
       BufferCannotBeSharedDynamicUpdateSliceAndOtherUser) {
  // This is a fusion that we would normally not create because it cannot be
  // emitted in-place. Still check whether buffer sharing logic would handle it
  // correctly.
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  param_0.1 = s32[6]{0} parameter(0)
  bitcast = s32[2,3]{1,0} bitcast(param_0.1)
  zero = s32[] constant(0)
  param_1.1 = s32[] parameter(1)
  dynamic-slice = s32[1,2]{1,0} dynamic-slice(bitcast, param_1.1, zero), dynamic_slice_sizes={1,2}
  one = s32[] constant(1)
  broadcast = s32[1,2]{1,0} broadcast(one), dimensions={}
  add = s32[1,2] add(dynamic-slice, broadcast)
  dynamic-update-slice = s32[2,3]{1,0} dynamic-update-slice(bitcast, add, param_1.1, zero)
  bitcast.1 = s32[6]{0} bitcast(dynamic-update-slice)
  neg = s32[2,3]{1,0} negate(bitcast)
  ROOT output = (s32[6]{0}, s32[2,3]{1,0}) tuple(bitcast.1, neg)
}

ENTRY main {
  param_0 = s32[6]{0} parameter(0)
  param_1 = s32[] parameter(1)
  ROOT fusion = (s32[6]{0},s32[2,3]{1,0}) fusion(param_0, param_1), kind=kInput, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(0), {0}));
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(0), {1}));
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(1), {0}));
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(1), {1}));
}

TEST_F(FusionCanShareBufferHintTest,
       BufferCannotBeSharedBecauseDUSAndDSAccessDifferentSliceSizes) {
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  param_0.1 = s32[6]{0} parameter(0)
  bitcast = s32[2,3]{1,0} bitcast(param_0.1)
  zero = s32[] constant(0)
  param_1.1 = s32[] parameter(1)
  dynamic-slice = s32[1,2]{1,0} dynamic-slice(bitcast, param_1.1, zero), dynamic_slice_sizes={1,2}
  param_2.1 = s32[1,1]{1,0} parameter(2)
  dynamic-update-slice = s32[2,3]{1,0} dynamic-update-slice(bitcast, param_2.1, param_1.1, zero)
  param_3.1 = s32[2,3]{1,0} parameter(3)
  dynamic-update-slice.1 = s32[2,3]{1,0} dynamic-update-slice(param_3.1, dynamic-slice, param_1.1, zero)
  ROOT tuple = (s32[2,3]{1,0}, s32[2,3]{1,0}) tuple(dynamic-update-slice, dynamic-update-slice.1)
}

ENTRY main {
  param_0 = s32[6]{0} parameter(0)
  param_1 = s32[] parameter(1)
  param_2 = s32[1,1]{1,0} parameter(2)
  param_3 = s32[2,3]{1,0} parameter(3)
  ROOT fusion = (s32[2,3]{1,0}, s32[2,3]{1,0}) fusion(param_0, param_1, param_2, param_3), kind=kInput, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(
      FusionCanShareBufferHint(fusion, fusion->operand(0), {0}));
}

TEST_F(FusionCanShareBufferHintTest,
       BufferCanBeSharedBecauseDUSAndDSAccessSlicesOfSizeOne) {
  const char* const kModuleString = R"(
HloModule fusion

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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
}

TEST_F(FusionCanShareBufferHintTest,
       BufferCannotBeSharedBecauseDUSAndDSAccessDifferentOperands) {
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  param_0.1 = s32[6]{0} parameter(0)
  bitcast = s32[2,3]{1,0} bitcast(param_0.1)
  zero = s32[] constant(0)
  param_1.1 = s32[] parameter(1)
  dynamic-slice = s32[1]{0} dynamic-slice(param_0.1, param_1.1), dynamic_slice_sizes={1}
  one = s32[1]{0} constant({1})
  add = s32[1] add(dynamic-slice, one)
  bitcasted_add = s32[1,1]{1,0} bitcast(add)
  dynamic-update-slice = s32[2,3]{1,0} dynamic-update-slice(bitcast, bitcasted_add, param_1.1, zero)
  ROOT bitcast.1 = s32[6]{0} bitcast(dynamic-update-slice)
}

ENTRY main {
  param_0 = s32[6]{0} parameter(0)
  param_1 = s32[] parameter(1)
  ROOT fusion = s32[6]{0} fusion(param_0, param_1), kind=kInput, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
}

TEST_F(FusionCanShareBufferHintTest,
       BufferCannotBeSharedBecauseDUSAndDSAccessDifferentOverlappingOffsets) {
  const char* const kModuleString = R"(
HloModule fusion

fused_computation {
  param_0.1 = s32[6]{0} parameter(0)
  bitcast = s32[2,3]{1,0} bitcast(param_0.1)
  zero = s32[] constant(0)
  param_1.1 = s32[] parameter(1)
  dynamic-slice = s32[1,2]{1,0} dynamic-slice(bitcast, param_1.1, zero), dynamic_slice_sizes={1,2}
  one = s32[] constant(1)
  broadcast = s32[1,2]{1,0} broadcast(one), dimensions={}
  add = s32[1,2] add(dynamic-slice, broadcast)
  dynamic-update-slice = s32[2,3]{1,0} dynamic-update-slice(bitcast, add, param_1.1, one)
  ROOT bitcast.1 = s32[6]{0} bitcast(dynamic-update-slice)
}

ENTRY main {
  param_0 = s32[6]{0} parameter(0)
  param_1 = s32[] parameter(1)
  ROOT fusion = s32[6]{0} fusion(param_0, param_1), kind=kInput, calls=fused_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(FusionCanShareBufferHint(fusion, fusion->operand(0), {}));
}

// For loops unrolled with double buffering,
// copyInsertion should not insert any copy.
TEST_F(GpuCopyInsertionTest, UnrolledLoopShouldNotHaveCopy) {
  const char* const kModuleString = R"(
HloModule all_gather_overlapping, entry_computation_layout={(f32[1,128]{1,0}, f32[2,128]{1,0})->(f32[1,128]{1,0}, f32[1,128]{1,0}, f32[2,128]{1,0}, s32[])}

body {
  input_tuple_while = (f32[1,128]{1,0}, f32[1,128]{1,0}, f32[2,128]{1,0}, s32[]) parameter(0)
  param_1 = f32[2,128]{1,0} get-tuple-element(input_tuple_while), index=2
  c1_s32 = s32[] constant(1)
  c0_s32 = s32[] constant(0)
  dynamic-slice = f32[1,128]{1,0} dynamic-slice(param_1, c1_s32, c0_s32), dynamic_slice_sizes={1,128}
  param_0 = f32[1,128]{1,0} get-tuple-element(input_tuple_while), index=0
  cond.1 = s32[] get-tuple-element(input_tuple_while), index=3
  cond_plus_1 = s32[] add(cond.1, c1_s32)
  c0 = f32[] constant(0)
  splat_c0 = f32[1,128]{1,0} broadcast(c0), dimensions={}
  add = f32[1,128]{1,0} add(splat_c0, param_0)
  all-gather-start = (f32[1,128]{1,0}, f32[2,128]{1,0}) all-gather-start(add), channel_id=1337, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true
  all-gather-done = f32[2,128]{1,0} all-gather-done(all-gather-start)
  dynamic-slice.double_buffer_clone = f32[1,128]{1,0} dynamic-slice(all-gather-done, c1_s32, c0_s32), dynamic_slice_sizes={1,128}
  splat_c0_unrolled = f32[1,128]{1,0} broadcast(c0), dimensions={}
  add.double_buffer_clone = f32[1,128]{1,0} add(splat_c0_unrolled, param_0)
  all-gather-start-unrolled = (f32[1,128]{1,0}, f32[2,128]{1,0}) all-gather-start(add.double_buffer_clone), channel_id=1339, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true
  all-gather-done-unrolled = f32[2,128]{1,0} all-gather-done(all-gather-start-unrolled)
  one.2 = s32[] constant(1)
  cond_plus_1.double_buffer_clone = s32[] add(cond_plus_1, one.2)
  ROOT output_tuple = (f32[1,128]{1,0}, f32[1,128]{1,0}, f32[2,128]{1,0}, s32[]) tuple(param_0, dynamic-slice.double_buffer_clone, all-gather-done-unrolled, cond_plus_1.double_buffer_clone)
}

condition {
  input_tuple = (f32[1,128]{1,0}, f32[1,128]{1,0}, f32[2,128]{1,0}, s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=3
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

ENTRY main {
  input_param_0 = f32[1,128]{1,0} parameter(0)
  input_param_1 = f32[2,128]{1,0} parameter(1)
  constant_1 = s32[] constant(1)
  constant_0 = s32[] constant(0)
  dynamic-slice-main = f32[1,128]{1,0} dynamic-slice(input_param_1, constant_1, constant_0), dynamic_slice_sizes={1,128}
  float0 = f32[] constant(0)
  splat_float0 = f32[1,128]{1,0} broadcast(float0), dimensions={}
  add.peeled_double_buffer = f32[1,128]{1,0} add(splat_float0, input_param_0)
  all-gather-start-main = (f32[1,128]{1,0}, f32[2,128]{1,0}) all-gather-start(add.peeled_double_buffer), channel_id=1338, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true
  all-gather-done-main = f32[2,128]{1,0} all-gather-done(all-gather-start-main)
  param_2 = s32[] constant(0)
  cond_plus_1.peeled_double_buffer = s32[] add(param_2, constant_1)
  tuple = (f32[1,128]{1,0}, f32[1,128]{1,0}, f32[2,128]{1,0}, s32[]) tuple(input_param_0, dynamic-slice-main, all-gather-done-main, cond_plus_1.peeled_double_buffer)
  ROOT while = (f32[1,128]{1,0}, f32[1,128]{1,0}, f32[2,128]{1,0}, s32[]) while(tuple), condition=condition, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  CopyInsertion copy_insertion = CreateCopyInsertion();
  ASSERT_IS_OK(copy_insertion.Run(module.get(), {"foobar"}).status());
  VLOG(2) << module->ToString();
  EXPECT_EQ(CountCopies(*module), 0);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
