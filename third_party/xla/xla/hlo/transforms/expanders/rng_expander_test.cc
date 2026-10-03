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

#include "xla/hlo/transforms/expanders/rng_expander.h"

#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using RngExpanderTest = HloHardwareIndependentTestBase;

TEST_F(RngExpanderTest, ReplacesRngWithGetRngSeedCustomCall) {
  const char* const hlo_string = R"(
    HloModule m
    ENTRY entry {
      min = f32[] constant(0)
      max = f32[] constant(1)
      ROOT result = f32[2,4,8] rng(min, max), distribution=rng_uniform
    })";

  const char* const expected = R"(
    // CHECK: %[[rng_comp:[^ ]+]] ({{.*}}) -> f32[2,4,8] {
    // CHECK-DAG:   %[[op_seed:[^ ]+]] = u64[] parameter(0)
    // CHECK-DAG:   %[[state:[^ ]+]] = u64[2]{0} parameter(1)
    // CHECK-DAG:   %[[seed:[^ ]+]] = u64[] custom-call(), custom_call_target="GetRngSeed", frontend_attributes={_xla_cse_safe_zero_operand="true"}
    // CHECK:   %[[xor:[^ ]+]] = u64[] xor(%[[seed]], %[[op_seed]])
    // CHECK:   ROOT %[[result:[^ ]+]] = f32[2,4,8]{2,1,0}
    // CHECK: }

    // CHECK-LABEL: ENTRY %entry
    // CHECK-DAG:   %[[min:[^ ]+]] = f32[] constant(0)
    // CHECK-DAG:   %[[max:[^ ]+]] = f32[] constant(1)
    // CHECK-DAG:   %[[op_seed_const:[^ ]+]] = u64[] constant({{[0-9]+}})
    // CHECK-DAG:   %[[state_inst:[^ ]+]] = u64[2]{0} rng-get-and-update-state(){{.*}}
    // CHECK:   ROOT %[[call:[^ ]+]] = f32[2,4,8]{2,1,0} call(%[[op_seed_const]], %[[state_inst]], %[[min]], %[[max]]), to_apply=%[[rng_comp]]
  )";

  RunAndFilecheckHloRewrite(hlo_string, RngExpander{}, expected);
}

}  // namespace
}  // namespace xla
