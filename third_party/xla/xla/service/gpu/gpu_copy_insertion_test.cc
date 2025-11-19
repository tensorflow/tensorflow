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

class GpuCopyInsertionTest : public HloHardwareIndependentTestBase {
 public:
  CopyInsertion CreateCopyInsertion() const {
    return CopyInsertion(&alias_info_,
                         /*use_region_based_live_range_analysis=*/0);
  }

 private:
  const se::DeviceDescription device_description_{
      xla::gpu::TestGpuDeviceInfo::CudaOrRocmDeviceInfo()};
  GpuAliasInfo alias_info_{device_description_};
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
