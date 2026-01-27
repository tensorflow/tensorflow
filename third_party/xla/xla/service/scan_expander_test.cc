/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/scan_expander.h"

#include <memory>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/hlo_module_config.h"

namespace xla {
namespace {

class ScanExpanderTest : public HloHardwareIndependentTestBase {};

TEST_F(ScanExpanderTest, ExpandsScan) {
  const char* kModuleStr = R"(
    HloModule scan_module

    add {
      input = f32[] parameter(0)
      acc = f32[] parameter(1)
      add = f32[] add(acc, input)
      ROOT t = (f32[], f32[]) tuple(add, add)
    }

    ENTRY Scan {
      input = f32[4]{0} parameter(0)
      init = f32[] constant(0)
      ROOT scan = (f32[4]{0}, f32[]) scan(input, init), dimensions={0}, is_reverse=false, to_apply=add
    }
  )";

  RunAndFilecheckHloRewrite(kModuleStr, ScanExpander(), R"(
    // CHECK-NOT: scan(
    // CHECK: while(
  )");
}

TEST_F(ScanExpanderTest, ExpandsScanComplex) {
  const char* kModuleStr = R"(
    HloModule complex_scan

    body {
      p0 = s8[] parameter(0)
      p1 = s16[2] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s64[3,4] parameter(3)
      carry0 = u16[] parameter(4)
      carry1 = u32[5] parameter(5)
      carry2 = u64[] parameter(6)
      out0 = f32[6] constant(1.5)
      out1 = f64[] constant(2.5)
      ROOT t = (f32[6], f64[], u16[], u32[5], u64[])
               tuple(out0, out1, carry0, carry1, carry2)
    }

    ENTRY entry {
      in0 = s8[8] parameter(0)
      in1 = s16[8,2] parameter(1)
      in2 = s32[8] parameter(2)
      in3 = s64[8,3,4] parameter(3)
      init0 = u16[] constant(0)
      init1 = u32[5] constant(0)
      init2 = u64[] constant(0)
      ROOT scan = (f32[8,6], f64[8], u16[], u32[5], u64[])
                  scan(in0, in1, in2, in3, init0, init1, init2),
                  dimensions={0}, to_apply=body
    }
  )";

  HloModuleConfig config;
  auto module = ParseAndReturnUnverifiedModule(kModuleStr, config).value();

  ScanExpander expander;
  ASSERT_TRUE(expander.Run(module.get()).value());

  auto hlo_string = module->ToString();
  RunAndFilecheckHloRewrite(kModuleStr, ScanExpander(), R"(
    // CHECK-LABEL: %scan_body
    // CHECK: [[ITER:%.*]] = s64[] get-tuple-element({{.*}}), index=0
    // CHECK: [[INCR:%.*]] = s64[] add([[ITER]], {{.*}})
    // CHECK: dynamic-update-slice
    // CHECK: dynamic-update-slice
    // CHECK: ROOT {{.*}} tuple([[INCR]],

    // CHECK-LABEL: %scan_condition
    // CHECK: compare({{.*}}), direction=LT

    // CHECK-LABEL: ENTRY
    // CHECK-NOT: scan(
    // CHECK: while({{.*}}), condition=%scan_condition, body=%scan_body
  )");
}

}  // namespace
}  // namespace xla
