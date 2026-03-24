/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/alias_info.h"

#include <memory>
#include <optional>

#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/service/copy_insertion.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

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

class AliasInfoTest : public HloHardwareIndependentTestBase {
 public:
  std::optional<bool> MayAlias(const HloInstruction* fusion,
                               const HloInstruction* operand,
                               const ShapeIndex& user_index) {
    return alias_info_.MayAlias(operand, {}, fusion, user_index);
  }

 private:
  const se::DeviceDescription device_description_{
      xla::gpu::TestGpuDeviceInfo::CudaOrRocmDeviceInfo()};
  GpuAliasInfo alias_info_{device_description_};
};

TEST_F(AliasInfoTest, BufferCanBeSharedSameShape) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(MayAlias(fusion, fusion->operand(0), {}));
}

TEST_F(AliasInfoTest, BufferCanBeSharedBitcastedShape) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(MayAlias(fusion, fusion->operand(0), {}));
}

TEST_F(AliasInfoTest, BufferCanBeSharedConvertedShapeSameByteWidth) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(MayAlias(fusion, fusion->operand(0), {}));
}

TEST_F(AliasInfoTest, BufferCanBeSharedMultiOutputFusion) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(MayAlias(fusion, fusion->operand(0), {0}));
  // The second operand cannot share the buffer with the second fusion output,
  // because the 'neg' op is also used by a non-elementwise op.
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(1), {1}));
  // The first operand cannot share the buffer with the second fusion output,
  // because there is no path between them.
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {1}));
}

TEST_F(AliasInfoTest, BufferCanBeSharedMultiOutputFusionTwoReachableOutputs) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(MayAlias(fusion, fusion->operand(0), {0}));
  // The first operand cannot share the buffer with the second fusion output,
  // because there is no path between them.
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {1}));
  // The second operand can share the buffer with the second fusion output and
  // the first fusion output.
  ExpectOptionalTrue(MayAlias(fusion, fusion->operand(1), {0}));
  ExpectOptionalTrue(MayAlias(fusion, fusion->operand(1), {1}));
}

TEST_F(AliasInfoTest, BufferCanBeSharedReductionEmitter) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(MayAlias(fusion, fusion->operand(0), {1}));
}

TEST_F(AliasInfoTest, BufferCannotBeSharedScatterMultiOutputFusion) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  // We expect that no buffer can be shared, because when Scatter is involved,
  // the only buffer we can potentially share is the first operand of scatter,
  // but if that is also used for a different fusion output, it will not work
  // due to potentially different access patterns.
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {0}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {1}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(1), {0}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(1), {1}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(2), {0}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(2), {1}));
}

TEST_F(AliasInfoTest, BufferCanBeSharedScatterFusion) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(MayAlias(fusion, fusion->operand(0), {}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(1), {}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(2), {}));
}

TEST_F(AliasInfoTest, BufferCannotBeSharedScatterFusion) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(1), {}));
}

TEST_F(AliasInfoTest, BufferCanBeSharedVariadicScatterFusion) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(MayAlias(fusion, fusion->operand(0), {0}));
  ExpectOptionalTrue(MayAlias(fusion, fusion->operand(1), {1}));
}

TEST_F(AliasInfoTest,
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(1), {}));
}

TEST_F(AliasInfoTest, BufferCannotBeSharedVariadicScatterFusion) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {0}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(1), {1}));
}

TEST_F(AliasInfoTest, BufferCanBeSharedSortFusion) {
  const char* const kModuleString = R"(
    HloModule module

    sorting_computation {
      %lhs_key = s32[] parameter(0)
      %rhs_key = s32[] parameter(1)
      %lhs_update_0 = s32[] parameter(2)
      %rhs_update_0 = s32[] parameter(3)
      %lhs_permutation = s32[] parameter(4)
      %rhs_permutation = s32[] parameter(5)
      ROOT %compare = pred[] compare(%lhs_key, %rhs_key), direction=LT
    }

    sort_fusion {
      p0 = s32[16384]{0} parameter(0)
      iota = s32[16384]{0} iota(), iota_dimension=0
      ROOT sort = (s32[16384]{0}, s32[16384]{0}, s32[16384]{0}) sort(p0, iota, iota), dimensions={0}, is_stable=true, to_apply=sorting_computation
    }

    ENTRY main {
      p = s32[16384]{0} parameter(0)
      ROOT fusion = (s32[16384]{0}, s32[16384]{0}, s32[16384]{0}) fusion(p), kind=kInput, calls=sort_fusion
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(MayAlias(fusion, fusion->operand(0), {0}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {1}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {2}));
}

TEST_F(AliasInfoTest, BufferCannotBeSharedSortFusionDuplicateSortOperand) {
  const char* const kModuleString = R"(
    HloModule module

    sorting_computation {
      %lhs_key = s32[] parameter(0)
      %rhs_key = s32[] parameter(1)
      %lhs_update_0 = s32[] parameter(2)
      %rhs_update_0 = s32[] parameter(3)
      %lhs_permutation = s32[] parameter(4)
      %rhs_permutation = s32[] parameter(5)
      ROOT %compare = pred[] compare(%lhs_key, %rhs_key), direction=LT
    }

    sort_fusion {
      p0 = s32[16384]{0} parameter(0)
      iota = s32[16384]{0} iota(), iota_dimension=0
      ROOT sort = (s32[16384]{0}, s32[16384]{0}, s32[16384]{0}) sort(p0, iota, p0), dimensions={0}, is_stable=true, to_apply=sorting_computation
    }

    ENTRY main {
      p = s32[16384]{0} parameter(0)
      ROOT fusion = (s32[16384]{0}, s32[16384]{0}, s32[16384]{0}) fusion(p), kind=kInput, calls=sort_fusion
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {0}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {1}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {2}));
}

TEST_F(AliasInfoTest, BufferCannotBeSharedConvertedShapeDifferentByteWidth) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {}));
}

TEST_F(AliasInfoTest, BufferCannotBeSharedShapeBitcastConvert) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {}));
}

TEST_F(AliasInfoTest, BufferCannotBeSharedDueToCopy) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {}));
}

TEST_F(AliasInfoTest, BufferCannotBeSharedDueToTranspose) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {}));
}

TEST_F(AliasInfoTest, BufferCannotBeSharedDueToReduceAndBroadcast) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {}));
}

TEST_F(AliasInfoTest, BufferCanBeSharedBecauseDUSAndDSAccessSameSlice) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(MayAlias(fusion, fusion->operand(0), {}));
}

TEST_F(AliasInfoTest, BufferCannotBeSharedWhenOtherUserIsTransposeUser) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {0}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {1}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(1), {0}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(1), {1}));
}

TEST_F(AliasInfoTest, BufferCannotBeSharedDynamicUpdateSliceAndOtherUser) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {0}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {1}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(1), {0}));
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(1), {1}));
}

TEST_F(AliasInfoTest,
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {0}));
}

TEST_F(AliasInfoTest, BufferCanBeSharedBecauseDUSAndDSAccessSlicesOfSizeOne) {
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(MayAlias(fusion, fusion->operand(0), {}));
}

TEST_F(AliasInfoTest,
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {}));
}

TEST_F(AliasInfoTest,
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(0), {}));
}

TEST_F(AliasInfoTest, BufferCannotBeSharedForFusionWithUnknownCustomCall) {
  const char* const kModuleString = R"(
HloModule dynamic_slice_fusion

%dynamic-slice-fusion {
  %p0 = f32[8,8]{1,0} parameter(0)
  %slice.0 = f32[1,8]{1,0} slice(%p0), slice={[0:1], [0:8]}
  %p1 = f32[128]{0} parameter(1)
  %p2 = f32[256]{0} parameter(2)
  %tuple.0 = (f32[128]{0}, f32[256]{0}) tuple(%p1, %p2)
  %custom-call.0 = (f32[8]{0}, (f32[128]{0}, f32[256]{0})) custom-call(%slice.0, %tuple.0), custom_call_target="__xla_test$$subbuffers", api_version=API_VERSION_TYPED_FFI
  %get-tuple-element = f32[8]{0} get-tuple-element(%custom-call.0), index=0
  %get-tuple-element.1 = (f32[128]{0}, f32[256]{0}) get-tuple-element(%custom-call.0), index=1
  %get-tuple-element.2 = f32[128]{0} get-tuple-element(%get-tuple-element.1), index=0
  %get-tuple-element.3 = f32[256]{0} get-tuple-element(%get-tuple-element.1), index=1
  %tuple.1 = (f32[128]{0}, f32[256]{0}) tuple(%get-tuple-element.2, %get-tuple-element.3)
  ROOT %tuple.3 = (f32[8]{0}, (f32[128]{0}, f32[256]{0})) tuple(%get-tuple-element, %tuple.1)
}

ENTRY main {
  p0 = f32[8,8]{1,0} parameter(0)
  p1 = f32[128]{0} parameter(1)
  p2 = f32[256]{0} parameter(2)
  ROOT %address_computation = (f32[8]{0}, (f32[128]{0}, f32[256]{0})) fusion(p0, p1, p2), kind=kCustom, calls=%dynamic-slice-fusion, backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"address_computation","kernel_index":0}},"force_earliest_schedule":false,"reification_cost":[]}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(fusion, fusion->operand(1), {1, 0}));
}

}  // namespace
}  // namespace xla::gpu
