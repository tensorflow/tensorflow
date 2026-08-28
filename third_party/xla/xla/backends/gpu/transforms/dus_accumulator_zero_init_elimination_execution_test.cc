/* Copyright 2026 The OpenXLA Authors.

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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

class DusAccumulatorZeroInitEliminationExecutionTest : public HloTestBase {
 protected:
  HloModuleConfig ConfigWithFlag(bool enable) {
    HloModuleConfig config = GetModuleConfigForTest();
    DebugOptions debug_options = config.debug_options();
    debug_options.set_xla_gpu_enable_dus_accumulator_zero_init_elimination(
        enable);
    config.set_debug_options(debug_options);
    return config;
  }
};

// Single-slot raw-DUS scan: trip count == accumulator slot count, so every
// slot gets written by the body. Eliminating the broadcast(0) init must not
// change the observable output.
TEST_F(DusAccumulatorZeroInitEliminationExecutionTest, RawDusScan) {
  constexpr absl::string_view kHlo = R"(
HloModule scan_dus

body {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  acc = bf16[4,8] get-tuple-element(p), index=1
  it_bf16 = bf16[] convert(it)
  update = bf16[1,8] broadcast(it_bf16), dimensions={}
  z = s32[] constant(0)
  dus = bf16[4,8] dynamic-update-slice(acc, update, it, z)
  one = s32[] constant(1)
  next = s32[] add(it, one)
  ROOT out = (s32[], bf16[4,8]) tuple(next, dus)
}

cond {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(4)
  ROOT cmp = pred[] compare(it, limit), direction=LT
}

ENTRY main {
  init_it = s32[] constant(0)
  c0 = bf16[] constant(0)
  zero_init = bf16[4,8] broadcast(c0), dimensions={}
  init_tuple = (s32[], bf16[4,8]) tuple(init_it, zero_init)
  w = (s32[], bf16[4,8]) while(init_tuple), condition=cond, body=body
  ROOT result = bf16[4,8] get-tuple-element(w), index=1
}
)";
  EXPECT_TRUE(RunAndCompareTwoModules(
      kHlo, kHlo, ConfigWithFlag(/*enable=*/false),
      ConfigWithFlag(/*enable=*/true), ErrorSpec(/*aabs=*/0.0, /*arel=*/0.0)));
}

// Multi-slot scan: two DUS accumulators in the same loop. Both must produce
// identical results with the pass on vs off.
TEST_F(DusAccumulatorZeroInitEliminationExecutionTest, MultiSlotScan) {
  constexpr absl::string_view kHlo = R"(
HloModule multi_slot_scan

body {
  p = (s32[], bf16[4,8], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  acc_a = bf16[4,8] get-tuple-element(p), index=1
  acc_b = bf16[4,8] get-tuple-element(p), index=2
  it_bf16 = bf16[] convert(it)
  update_a = bf16[1,8] broadcast(it_bf16), dimensions={}
  two = s32[] constant(2)
  it2 = s32[] add(it, it)
  it2_bf16 = bf16[] convert(it2)
  update_b = bf16[1,8] broadcast(it2_bf16), dimensions={}
  z = s32[] constant(0)
  dus_a = bf16[4,8] dynamic-update-slice(acc_a, update_a, it, z)
  dus_b = bf16[4,8] dynamic-update-slice(acc_b, update_b, it, z)
  one = s32[] constant(1)
  next = s32[] add(it, one)
  ROOT out = (s32[], bf16[4,8], bf16[4,8]) tuple(next, dus_a, dus_b)
}

cond {
  p = (s32[], bf16[4,8], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(4)
  ROOT cmp = pred[] compare(it, limit), direction=LT
}

ENTRY main {
  init_it = s32[] constant(0)
  c0 = bf16[] constant(0)
  zi_a = bf16[4,8] broadcast(c0), dimensions={}
  zi_b = bf16[4,8] broadcast(c0), dimensions={}
  init_tuple = (s32[], bf16[4,8], bf16[4,8]) tuple(init_it, zi_a, zi_b)
  w = (s32[], bf16[4,8], bf16[4,8]) while(init_tuple),
      condition=cond, body=body
  out_a = bf16[4,8] get-tuple-element(w), index=1
  out_b = bf16[4,8] get-tuple-element(w), index=2
  ROOT result = (bf16[4,8], bf16[4,8]) tuple(out_a, out_b)
}
)";
  EXPECT_TRUE(RunAndCompareTwoModules(
      kHlo, kHlo, ConfigWithFlag(/*enable=*/false),
      ConfigWithFlag(/*enable=*/true), ErrorSpec(/*aabs=*/0.0, /*arel=*/0.0)));
}

}  // namespace
}  // namespace xla::gpu
