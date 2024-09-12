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

#include "xla/tools/hlo_decomposer.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class HloDecomposerTest : public HloTestBase {
 protected:
  std::unique_ptr<HloModule> GetModule() {
    absl::string_view kHlo = R"(
HloModule test_module, entry_computation_layout={(bf16[1024,8192]{1,0}, f32[8192]{0}, f32[16384]{0})->(bf16[1024]{0}, bf16[1024]{0}, f32[16384]{0}, f32[16384]{0})}

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add.1 = f32[] add(p0, p1)
}

fused_computation.1 {
  param_1.3 = f32[8192]{0} parameter(1)
  broadcast.2 = f32[1024,8192]{1,0} broadcast(param_1.3), dimensions={1}
  param_0.3 = bf16[1024,8192]{1,0} parameter(0)
  convert.5 = f32[1024,8192]{1,0} convert(param_0.3)
  multiply.2 = f32[1024,8192]{1,0} multiply(broadcast.2, convert.5)
  c0_1 = f32[] constant(0)
  reduce.1 = f32[1024]{0} reduce(multiply.2, c0_1), dimensions={1}, to_apply=add
  ROOT convert.4 = bf16[1024]{0} convert(reduce.1)
}

fused_computation.2 {
  p0.0 = bf16[1024,8192]{1,0} parameter(0)
  c.0 = f32[1024,8192]{1,0} convert(p0.0)
  co0_1.1 = f32[] constant(0)
  p.0 = f32[8192]{0} parameter(1)
  b.0 = f32[1024,8192]{1,0} broadcast(p.0), dimensions={1}
  m.0 = f32[1024,8192]{1,0} multiply(b.0, c.0)
  r.0 = f32[1024]{0} reduce(m.0, co0_1.1), dimensions={1}, to_apply=add
  ROOT c.1 = bf16[1024]{0} convert(r.0)
}

exp {
  param_0.5 = f32[16384]{0} parameter(0)
  m.4 = f32[16384]{0} multiply(param_0.5, param_0.5)
  e = f32[16384]{0} exponential(m.4)
  l.clone.1 = f32[16384]{0} log(m.4)
  ROOT tuple = (f32[16384]{0}, f32[16384]{0}) tuple(e, l.clone.1)
}

ENTRY main {
  p0.1 = bf16[1024,8192]{1,0} parameter(0)
  p1.1 = f32[8192]{0} parameter(1)
  fusion.1 = bf16[1024]{0} fusion(p0.1, p1.1), kind=kInput, calls=fused_computation.1
  fusion.2 = bf16[1024]{0} fusion(p0.1, p1.1), kind=kInput, calls=fused_computation.2
  p2 = f32[16384]{0} parameter(2)
  e.1 = (f32[16384]{0}, f32[16384]{0}) fusion(p2), kind=kInput, calls=exp
  get-tuple-element.1 = f32[16384]{0} get-tuple-element(e.1), index=1
  get-tuple-element = f32[16384]{0} get-tuple-element(e.1), index=0
  ROOT result = (bf16[1024]{0}, bf16[1024]{0}, f32[16384]{0}, f32[16384]{0}) tuple(fusion.1, fusion.2, get-tuple-element.1, get-tuple-element)
})";
    return ParseAndReturnVerifiedModule(kHlo).value();
  }

  void FindAndCompare(const std::vector<std::unique_ptr<HloModule>>& modules,
                      absl::string_view module_name,
                      absl::string_view pattern) {
    auto iter =
        absl::c_find_if(modules, [&](const std::unique_ptr<HloModule>& module) {
          return module->name() == module_name;
        });
    EXPECT_NE(iter, modules.end()) << "No module named " << module_name;
    if (iter == modules.end()) {
      return;
    }
    EXPECT_TRUE(*RunFileCheck((*iter)->ToString(), pattern));
  }
};

TEST_F(HloDecomposerTest, DecomposeNoDedup) {
  auto module = GetModule();
  TF_ASSERT_OK_AND_ASSIGN(
      auto decomposed,
      DecomposeHloModule(*module, /*deduplicate_modules=*/false));
  EXPECT_EQ(decomposed.size(), 3);

  FindAndCompare(decomposed, "fusion.1", R"(
CHECK: %add{{.*}} {
CHECK: %fused_computation.1
CHECK: ENTRY
CHECK-THEN: %parameter.0 = bf16[1024,8192]{1,0} parameter(0)
CHECK-THEN: %parameter.1 = f32[8192]{0} parameter(1)
CHECK-THEN: ROOT %fusion.1
)");

  FindAndCompare(decomposed, "fusion.2", R"(
CHECK: %add{{.*}} {
CHECK: %fused_computation.2
CHECK: ENTRY
CHECK-THEN: %parameter.0 = bf16[1024,8192]{1,0} parameter(0)
CHECK-THEN: %parameter.1 = f32[8192]{0} parameter(1)
CHECK-THEN: ROOT %fusion.2
)");

  FindAndCompare(decomposed, "e.1", R"(
CHECK: %exp{{.*}} {
CHECK: ENTRY
CHECK-THEN: %parameter.0 = f32[16384]{0} parameter(0)
CHECK-THEN: ROOT %e.1
)");
}

TEST_F(HloDecomposerTest, DecomposeDedup) {
  auto module = GetModule();
  TF_ASSERT_OK_AND_ASSIGN(
      auto decomposed,
      DecomposeHloModule(*module, /*deduplicate_modules=*/true));
  EXPECT_EQ(decomposed.size(), 2);

  FindAndCompare(decomposed, "fusion.1", R"(
CHECK: %add{{.*}} {
CHECK: %fused_computation.1
CHECK: ENTRY
CHECK-THEN: %parameter.0 = bf16[1024,8192]{1,0} parameter(0)
CHECK-THEN: %parameter.1 = f32[8192]{0} parameter(1)
CHECK-THEN: ROOT %fusion.1
)");

  FindAndCompare(decomposed, "e.1", R"(
CHECK: %exp{{.*}} {
CHECK: ENTRY
CHECK-THEN: %parameter.0 = f32[16384]{0} parameter(0)
CHECK-THEN: ROOT %e.1
)");
}

}  // namespace
}  // namespace xla
