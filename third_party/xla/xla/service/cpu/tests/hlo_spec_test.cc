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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm-c/Target.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/cpu/tests/cpu_pjrt_codegen_test.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

struct HloTestSpec {
  absl::string_view name;
  absl::string_view hlo_text;
  bool match_optimized_ir;
  absl::string_view triple;
  absl::string_view features;
  absl::string_view check_lines;
};

class HloSpecTest : public CpuPjRtCodegenTest,
                    public ::testing::WithParamInterface<HloTestSpec> {
 public:
  static std::string Name(const ::testing::TestParamInfo<HloTestSpec>& info) {
    auto spec = info.param;

    std::string triple{spec.triple.data(), spec.triple.size()};
    std::replace(triple.begin(), triple.end(), '-', '_');

    std::string features{spec.features.data(), spec.features.size()};
    if (!features.empty()) {
      std::replace_if(
          features.begin(), features.end(),
          [](char c) { return c != '_' && !absl::ascii_isalnum(c); }, '_');
    } else {
      features = "Default";
    }

    return absl::StrCat(spec.name, "_On_", triple, "_With_", features);
  }
};

TEST_P(HloSpecTest, DoIt) {
  HloTestSpec spec = GetParam();
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64TargetMC();

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(spec.hlo_text));

  std::string triple{spec.triple.data(), spec.triple.size()};
  std::string features{spec.features.data(), spec.features.size()};

  CpuAotCompilationOptions options{
      /*triple=*/triple, /*cpu_name=*/"", /*features=*/features,
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  std::string check_lines{spec.check_lines.data(), spec.check_lines.size()};

  CompileAheadOfTimeAndVerifyIr(std::move(hlo_module), options, check_lines,
                                spec.match_optimized_ir);
}

const char* const kSoftmaxFusionHlo = R"(
HloModule FusedComputationModule

 ENTRY fused_computation.133 (param_0.199: f32[128,1,8,32], param_1.179: bf16[128,32,256]) -> f32[128,1,8,32,32] {
   %param_1.179 = bf16[128,32,256]{2,1,0} parameter(1)
   %convert.1735 = f32[128,32,256]{2,1,0} convert(%param_1.179)
   %bitcast.263 = f32[128,1,32,32,8]{4,3,2,0,1} reshape(%convert.1735)
   %copy.215 = f32[128,1,32,32,8]{4,3,2,1,0} copy(%bitcast.263)
   %transpose.383 = f32[128,1,8,32,32]{2,3,4,1,0} transpose(%copy.215), dimensions={0,1,4,3,2}
   %copy.214 = f32[128,1,8,32,32]{4,3,2,1,0} copy(%transpose.383)
   %param_0.199 = f32[128,1,8,32]{3,2,1,0} parameter(0)
   %bitcast.262 = f32[128,8,32]{2,1,0} reshape(%param_0.199)
   %sub.49 = f32[128,1,8,32,32]{4,3,2,1,0} broadcast(%bitcast.262), dimensions={0,2,3}
   %sub.48 = f32[128,1,8,32,32]{4,3,2,1,0} subtract(%copy.214, %sub.49)
   ROOT %exp.0 = f32[128,1,8,32,32]{4,3,2,1,0} exponential(%sub.48)
 }
)";

const char* const kBroadcastSubtractExpHlo = R"(
HloModule BroadcastSubtractExp

ENTRY broadcast_subtract_exp {
  param.0 = f32[2,32,32] parameter(0)
  param.1 = f32[2,32] parameter(1)
  broadcast.0 = f32[2,32,32] broadcast(param.1), dimensions={0,1}
  sub.0 = f32[2,32,32] subtract(param.0, broadcast.0)
  ROOT exp.0 = f32[2,32,32] exponential(sub.0)
}
)";

HloTestSpec HloTestCases[] = {
    // TODO(b/495870398): The compiler should be generating <4 x float> (or <8 x
    // float>) here instead of <1 x float>
    HloTestSpec{
        /*name=*/"BroadcastSubtractExp",
        /*hlo_text=*/kBroadcastSubtractExpHlo,
        /*match_optimized_ir=*/true,
        /*triple=*/"aarch64-unknown-linux-gnu",
        /*features=*/"+neon,+sve2",
        // Check if it still vectorizes
        /*check_lines=*/R"(
CHECK: fmul <1 x float>
)",
    },
    HloTestSpec{
        /*name=*/"BroadcastSubtractExp_x86",
        /*hlo_text=*/kBroadcastSubtractExpHlo,
        /*match_optimized_ir=*/true,
        /*triple=*/"x86_64-unknown-linux-gnu",
        /*features=*/"+avx2",
        // Check if it still vectorizes
        /*check_lines=*/R"(
CHECK: fmul <8 x float>
)",
    },
    // TODO(b/495870398): The compiler should be generating <4 x float> (or <8 x
    // float>) here instead of <1 x float>
    HloTestSpec{
        /*name=*/"SoftmaxFusion",
        /*hlo_text=*/kSoftmaxFusionHlo,
        /*match_optimized_ir=*/true,
        /*triple=*/"aarch64-unknown-linux-gnu",
        /*features=*/"+neon,+sve2",
        /*check_lines=*/R"(
CHECK: fmul <1 x float>
)",
    },
};

INSTANTIATE_TEST_SUITE_P(HloSpecTestInstantiation, HloSpecTest,
                         ::testing::ValuesIn(HloTestCases), HloSpecTest::Name);

}  // namespace
}  // namespace cpu
}  // namespace xla
