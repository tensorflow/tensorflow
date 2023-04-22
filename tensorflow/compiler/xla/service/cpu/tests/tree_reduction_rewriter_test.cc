/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>

#include "tensorflow/compiler/xla/service/cpu/tests/cpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/llvm_irgen_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace cpu {

namespace {

class TreeReductionRewriterTest : public CpuCodegenTest {};

TEST_F(TreeReductionRewriterTest, SimpleRewrite) {
  const char* hlo_text = R"(
HloModule SimpleReduction

add {
  acc = f32[] parameter(1)
  op = f32[] parameter(0)
  ROOT out = f32[] add(acc, op)
}

ENTRY main {
  input = f32[1000] parameter(0)
  zero = f32[] constant(0)
  ROOT out = f32[] reduce(input, zero), dimensions={0}, to_apply=add
}
  )";

  MatchOptimizedHlo(hlo_text,
                    R"(
; CHECK-LABEL: ENTRY %main (input: f32[1000]) -> f32[] {
; CHECK-NEXT:    %input = f32[1000]{0} parameter(0)
; CHECK-NEXT:    %zero = f32[] constant(0)
; CHECK-NEXT:    %reduce-window = f32[32]{0} reduce-window(%input, %zero)
; CHECK-NEXT:    %reduce-window.1 = f32[1]{0} reduce-window(%reduce-window, %zero), window={size=32 stride=32}, to_apply=%add
; CHECK-NEXT:    ROOT %bitcast = f32[] bitcast(%reduce-window.1)
      )");
}

}  // namespace
}  // namespace cpu
}  // namespace xla
