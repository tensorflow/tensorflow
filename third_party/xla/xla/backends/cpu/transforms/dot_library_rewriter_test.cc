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

#include "xla/backends/cpu/transforms/dot_library_rewriter.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/codegen/target_machine_test_base.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

struct XnnDotRewriteTestSpec {
  std::string in_dtype;
  std::string out_dtype;
  std::string cpu_name;
  std::string features;
  bool changed;
};

class CpuDotLibraryTest
    : public TargetMachineTestBase,
      public ::testing::WithParamInterface<XnnDotRewriteTestSpec> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<XnnDotRewriteTestSpec>& info) {
    return absl::StrCat(info.param.in_dtype, "_", info.param.out_dtype, "_",
                        info.param.cpu_name);
  }

 protected:
  void RunTest(absl::string_view hlo_template) {
    // Create TargetMachineFeatures.
    XnnDotRewriteTestSpec spec = GetParam();
    std::unique_ptr<TargetMachineFeatures> features =
        CreateTargetMachineFeatures(
            /*triple_string=*/"x86_64-unknown-linux-gnu", spec.cpu_name,
            spec.features);

    // Create an HLO module with the specified input and output data types.
    std::string hlo_text = absl::StrReplaceAll(
        hlo_template,
        {{"$in_dtype", spec.in_dtype}, {"$out_dtype", spec.out_dtype}});
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text));

    // Run the pass.
    DotLibraryRewriterOptions options = {/*use_onednn=*/false,
                                         /*use_xnnpack=*/true};
    DotLibraryRewriter rewriter(features.get(), options);
    EXPECT_EQ(spec.changed, rewriter.Run(module.get()).value());
    if (!spec.changed) {
      return;  // No further checks if the module was not changed.
    }
    VLOG(3) << module->ToString();

    // Verify that there is a custom fusion.
    HloInstruction* result = FindInstruction(module.get(), HloOpcode::kFusion);
    EXPECT_NE(result, nullptr);
    HloFusionInstruction* fusion = Cast<HloFusionInstruction>(result);
    EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kCustom);

    // The fusion root must be a dot or a convert because we don't fuse other
    // elementwise ops yet.
    HloInstruction* fusion_root = fusion->fused_expression_root();
    EXPECT_EQ(fusion_root->opcode(),
              spec.out_dtype == "bf16" ? HloOpcode::kConvert : HloOpcode::kDot);
  }
};

TEST_P(CpuDotLibraryTest, MatMul) {
  const absl::string_view hlo_template = R"(
    HloModule matmul

    ENTRY %main {
      %input = $in_dtype[64,64]{1,0} parameter(0)
      %weight = $in_dtype[64,262144]{1,0} parameter(1)
      ROOT %dot = $out_dtype[64,262144]{1,0} dot(%input, %weight),
                  lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  RunTest(hlo_template);
}

TEST_P(CpuDotLibraryTest, MatMulAndAdd) {
  const absl::string_view hlo_template = R"(
    HloModule matmul

    ENTRY %main {
      %input = $in_dtype[64,64]{1,0} parameter(0)
      %weight = $in_dtype[64,262144]{1,0} parameter(1)
      %addend = $out_dtype[64,262144]{1,0} parameter(2)
      %dot = $out_dtype[64,262144]{1,0} dot(%input, %weight),
                  lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT %add = $out_dtype[64,262144]{1,0} add(%dot, %addend)
    })";

  RunTest(hlo_template);
}

std::vector<XnnDotRewriteTestSpec> GetXnnDotRewriteTestSpecs() {
  const std::string kZen3Features = "+avx,+avx2";
  const std::string kSapphireRapidsFeatures =
      "+avx512vnni,+avx512bf16,+amx-bf16,+amx-int8,+amx-tile,+amx-transpose";
  return std::vector<XnnDotRewriteTestSpec>{
      XnnDotRewriteTestSpec{"f32", "f32", "znver3", kZen3Features, true},
      XnnDotRewriteTestSpec{"bf16", "f32", "znver3", kZen3Features, false},
      XnnDotRewriteTestSpec{"f32", "f32", "sapphirerapids",
                            kSapphireRapidsFeatures, true},
      XnnDotRewriteTestSpec{"bf16", "f32", "sapphirerapids",
                            kSapphireRapidsFeatures, true},
      XnnDotRewriteTestSpec{"bf16", "bf16", "sapphirerapids",
                            kSapphireRapidsFeatures, true},
  };
}

INSTANTIATE_TEST_SUITE_P(CpuDotLibraryTestSuite, CpuDotLibraryTest,
                         ::testing::ValuesIn(GetXnnDotRewriteTestSpecs()),
                         CpuDotLibraryTest::Name);

}  // namespace
}  // namespace xla::cpu
