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

#include "xla/service/gpu/rename_fusions.h"

#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {

class RenameFusionsTest : public HloTestBase {
 protected:
  RenameFusions rename_fusions_;
};

TEST_F(RenameFusionsTest, FusionInstructionNames) {
  absl::string_view kHlo = R"(
      HloModule test_module

      square {
        p = f32[16384] parameter(0)
        ROOT m = f32[16384] multiply(p, p)
      }

      exp {
        p = f32[16384] parameter(0)
        ROOT e = f32[16384] exponential(p)
      }

      log {
        p = f32[16384] parameter(0)
        ROOT l = f32[16384] log(p)
      }

      add {
        p0 = f32[] parameter(0)
        p1 = f32[] parameter(1)
        ROOT add = f32[] add(p0, p1)
      }

      ENTRY main {
        p0 = bf16[1024,8192] parameter(0)
        p1 = f32[8192] parameter(1)
        p2 = f32[16384] parameter(2)
        convert = f32[1024,8192] convert(p0)
        broadcast = f32[1024,8192] broadcast(p1), dimensions={1}
        c0 = f32[] constant(0)
        multiply = f32[1024,8192] multiply(broadcast, convert)
        reduce = f32[1024] reduce(multiply, c0), dimensions={1}, to_apply=add
        convert.1 = bf16[1024] convert(reduce)
        s = f32[16384] fusion(p2), kind=kLoop, calls=square
        e = f32[16384] fusion(s), kind=kLoop, calls=exp
        l = f32[16384] fusion(s), kind=kInput, calls=log
        ROOT result = (bf16[1024]{0}, f32[16384]{0}, f32[16384]{0}) tuple(convert.1, l, e)
      })";

  RunAndFilecheckHloRewrite(kHlo, std::move(rename_fusions_), R"(
CHECK: ENTRY %main
CHECK: %loop_multiply_fusion{{.*}} calls=%fused_multiply
CHECK: %input_log_fusion{{.*}} calls=%fused_log
CHECK: %loop_exponential_fusion{{.*}} calls=%fused_exponential
CHECK: ROOT %result
  )");
}

}  // namespace gpu
}  // namespace xla
