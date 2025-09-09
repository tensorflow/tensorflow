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

#include "xla/codegen/ir_emission_utils.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace {

class IrEmissionUtilsTest : public HloHardwareIndependentTestBase {};

TEST_F(IrEmissionUtilsTest,
       CanEmitFusedDynamicUpdateSliceInPlace_HandlesBitcasts) {
  const char* hlo = R"(
HloModule fusion, is_scheduled=true

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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  auto fusion = module->entry_computation()->root_instruction();
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0, 10);
  auto adaptor = HloFusionAdaptor::ForInstruction(fusion);
  EXPECT_THAT(CanEmitFusedDynamicUpdateSliceInPlace(
                  *adaptor,
                  [&slice0](const HloInstruction*, const ShapeIndex&) {
                    return slice0;
                  },
                  fusion),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(IrEmissionUtilsTest,
       CanEmitFusedDynamicUpdateSliceInPlace_ElementwiseOnPathToParameter) {
  const char* hlo = R"(
HloModule fusion, is_scheduled=true

fused_computation {
  param_0.1 = s32[2,3]{1,0} parameter(0)
  bitcast = s32[2,3]{1,0} negate(param_0.1)
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
  param_0 = s32[2,3]{1,0} parameter(0)
  param_1 = s32[] parameter(1)
  ROOT fusion = s32[6]{0} fusion(param_0, param_1), kind=kInput, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  auto fusion = module->entry_computation()->root_instruction();
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0, 10);
  auto adaptor = HloFusionAdaptor::ForInstruction(fusion);
  EXPECT_THAT(CanEmitFusedDynamicUpdateSliceInPlace(
                  *adaptor,
                  [&slice0](const HloInstruction*, const ShapeIndex&) {
                    return slice0;
                  },
                  fusion),
              absl_testing::IsOkAndHolds(false));
}

// Same test as above, but different allocation slices for parameter and output.
TEST_F(IrEmissionUtilsTest,
       CanEmitFusedDynamicUpdateSliceInPlace_SlicesDifferent) {
  const char* hlo = R"(
HloModule fusion, is_scheduled=true

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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  auto fusion = module->entry_computation()->root_instruction();
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0, 10);
  BufferAllocation::Slice slice1(&alloc, 10, 20);
  auto adaptor = HloFusionAdaptor::ForInstruction(fusion);
  EXPECT_THAT(CanEmitFusedDynamicUpdateSliceInPlace(
                  *adaptor,
                  [fusion, &slice0, &slice1](const HloInstruction* instr,
                                             const ShapeIndex&) {
                    if (instr == fusion) {
                      return slice0;
                    }
                    return slice1;
                  },
                  fusion),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(
    IrEmissionUtilsTest,
    CanEmitFusedDynamicUpdateSliceInPlace_DynamicUpdateSliceWithDifferentDynamicSliceAccess) {  // NOLINT
  const char* hlo = R"(
HloModule fusion, input_output_alias={ {}: (0, {}) }

fused_computation {
  param_0.1 = s32[6]{0} parameter(0)
  bitcast = s32[2,3]{1,0} bitcast(param_0.1)
  zero = s32[] constant(0)
  one = s32[] constant(1)
  param_1.1 = s32[] parameter(1)
  dynamic-slice = s32[2,2]{1,0} dynamic-slice(bitcast, param_1.1, one), dynamic_slice_sizes={2,2}
  broadcasted_one = s32[2,2]{1,0} broadcast(one), dimensions={}
  add = s32[2,2] add(dynamic-slice, broadcasted_one)
  dynamic-update-slice = s32[2,3]{1,0} dynamic-update-slice(bitcast, add, param_1.1, zero)
  ROOT bitcast.1 = s32[6]{0} bitcast(dynamic-update-slice)
}

ENTRY main {
  param_0 = s32[6]{0} parameter(0)
  param_1 = s32[] parameter(1)
  ROOT fusion = s32[6]{0} fusion(param_0, param_1), kind=kInput, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  auto fusion = module->entry_computation()->root_instruction();
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0, 10);
  auto adaptor = HloFusionAdaptor::ForInstruction(fusion);
  EXPECT_THAT(CanEmitFusedDynamicUpdateSliceInPlace(
                  *adaptor,
                  [&slice0](const HloInstruction*, const ShapeIndex&) {
                    return slice0;
                  },
                  fusion),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(IrEmissionUtilsTest,
       CanEmitFusedDynamicUpdateSliceInPlace_HandlesMultiOutputFusion) {
  const char* hlo = R"(
HloModule MultipleInplaceDus, is_scheduled=true, input_output_alias={ {0}: (0, {}), {1}: (2, {}) }

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
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  auto fusion = module->entry_computation()->root_instruction();
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0, 10);
  auto adaptor = HloFusionAdaptor::ForInstruction(fusion);
  EXPECT_THAT(CanEmitFusedDynamicUpdateSliceInPlace(
                  *adaptor,
                  [&slice0](const HloInstruction*, const ShapeIndex&) {
                    return slice0;
                  },
                  fusion),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(
    IrEmissionUtilsTest,
    CanEmitFusedDynamicUpdateSliceInPlace_HandlesMultiOutputFusionSharedParameter) {  // NOLINT
  const char* hlo = R"(
HloModule MultipleInplaceDus, is_scheduled=true, input_output_alias={ {0}: (0, {}), {1}: (2, {}) }

fused_computation {
  p0 = bf16[10,11,12] parameter(0)
  p1 = bf16[1,11,12] parameter(1)
  p2 = bf16[1,11,12] parameter(2)
  p3 = s32[] parameter(3)
  c0 = s32[] constant(0)
  cmp = pred[] compare(p3, c0), direction=EQ
  broadcast = pred[1,11,12] broadcast(cmp), dimensions={}
  select = bf16[1,11,12] select(broadcast, p1, p2)
  dus0 = bf16[10,11,12] dynamic-update-slice(p0, select, c0, c0, c0)
  dus1 = bf16[10,11,12] dynamic-update-slice(p0, select, c0, c0, c0)
  ROOT tuple = (bf16[10,11,12], bf16[10,11,12]) tuple(dus0, dus1)
}

ENTRY main {
  p0 = bf16[10,11,12] parameter(0)
  p1 = bf16[1,11,12] parameter(1)
  p2 = bf16[1,11,12] parameter(2)
  p3 = s32[] parameter(3)
  ROOT fusion_root_multiple = (bf16[10,11,12], bf16[10,11,12]) fusion(p0, p1, p2, p3), kind=kLoop, calls=fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  auto fusion = module->entry_computation()->root_instruction();
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0, 10);
  auto adaptor = HloFusionAdaptor::ForInstruction(fusion);
  EXPECT_THAT(CanEmitFusedDynamicUpdateSliceInPlace(
                  *adaptor,
                  [&slice0](const HloInstruction*, const ShapeIndex&) {
                    return slice0;
                  },
                  fusion),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(
    IrEmissionUtilsTest,
    CanEmitFusedDynamicUpdateSliceInPlace_HandlesMultiOutputFusionWithTransposeBitcasts) {  // NOLINT
  const char* hlo = R"(
HloModule MultipleInplaceDusWithTransposeBitcastToTheRoot, is_scheduled=true, input_output_alias={ {0}: (0, {}), {1}: (2, {}) }

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
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  auto fusion = module->entry_computation()->root_instruction();
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0, 10);
  auto adaptor = HloFusionAdaptor::ForInstruction(fusion);
  EXPECT_THAT(CanEmitFusedDynamicUpdateSliceInPlace(
                  *adaptor,
                  [&slice0](const HloInstruction*, const ShapeIndex&) {
                    return slice0;
                  },
                  fusion),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(
    IrEmissionUtilsTest,
    CanEmitFusedDynamicUpdateSliceInPlace_HandlesTransposeBitcastToTheRoot) {  // NOLINT
  const char* hlo = R"(
HloModule SingleInplaceDusWithTransposeBitcastToTheRoot, is_scheduled=true, input_output_alias={ {}: (0, {}) }

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
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  auto fusion = module->entry_computation()->root_instruction();
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0, 10);
  auto adaptor = HloFusionAdaptor::ForInstruction(fusion);
  EXPECT_THAT(CanEmitFusedDynamicUpdateSliceInPlace(
                  *adaptor,
                  [&slice0](const HloInstruction*, const ShapeIndex&) {
                    return slice0;
                  },
                  fusion),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(
    IrEmissionUtilsTest,
    CanEmitFusedDynamicUpdateSliceInPlace_HandlesReshapeBitcastToTheRoot) {  // NOLINT
  const char* hlo = R"(
HloModule SingleInplaceDusWithReshapeBitcastToTheRoot, is_scheduled=true, input_output_alias={ {}: (0, {}) }

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
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  auto fusion = module->entry_computation()->root_instruction();
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0, 10);
  auto adaptor = HloFusionAdaptor::ForInstruction(fusion);
  EXPECT_THAT(CanEmitFusedDynamicUpdateSliceInPlace(
                  *adaptor,
                  [&slice0](const HloInstruction*, const ShapeIndex&) {
                    return slice0;
                  },
                  fusion),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(
    IrEmissionUtilsTest,
    CanEmitFusedDynamicUpdateSliceInPlace_HandlesBitcastToTheRootAndFromParameter) {  // NOLINT
  const char* hlo = R"(
HloModule SingleInplaceDusWithBitcastToTheRootAndFromTheParameter, is_scheduled=true, input_output_alias={ {}: (0, {}) }

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
  ROOT fusion_root_bitcast_both_ways = bf16[10,11,6,2] fusion(p0, p1, p2, p3),
    kind=kLoop, calls=single_inplace_dus_with_bitcast_to_the_root_and_from_the_parameter
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  auto fusion = module->entry_computation()->root_instruction();
  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice0(&alloc, 0, 10);
  auto adaptor = HloFusionAdaptor::ForInstruction(fusion);
  EXPECT_THAT(CanEmitFusedDynamicUpdateSliceInPlace(
                  *adaptor,
                  [&slice0](const HloInstruction*, const ShapeIndex&) {
                    return slice0;
                  },
                  fusion),
              absl_testing::IsOkAndHolds(true));
}

}  // namespace
}  // namespace xla
