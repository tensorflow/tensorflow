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
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
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
  std::string lib;
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
    return absl::StrCat(info.param.lib, "_", info.param.in_dtype, "_",
                        info.param.out_dtype, "_", info.param.cpu_name);
  }

 protected:
  struct FusionProperties {
    HloOpcode fusion_root;
    int num_fusion_params;
    int num_instructions_in_fused_computation;
  };

  void RunTest(absl::string_view hlo_template, FusionProperties expected) {
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
    DotLibraryRewriterOptions options = {/*use_onednn=*/spec.lib == "onednn",
                                         /*use_xnnpack=*/spec.lib == "xnn"};
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

    // A fusion that ends with dot may have a convert as root if the dot output
    // must be converted.
    HloInstruction* fusion_root = fusion->fused_expression_root();
    if (spec.out_dtype == "bf16") {
      if (expected.fusion_root == HloOpcode::kDot) {
        expected.fusion_root = HloOpcode::kConvert;
      }
      expected.num_instructions_in_fused_computation++;
    }
    EXPECT_EQ(fusion_root->opcode(), expected.fusion_root);
    EXPECT_EQ(fusion->operand_count(), expected.num_fusion_params);
    EXPECT_EQ(fusion->fused_instructions_computation()->instruction_count(),
              expected.num_instructions_in_fused_computation);
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

  RunTest(hlo_template, {HloOpcode::kDot, 2, 3});
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

  RunTest(hlo_template, {HloOpcode::kAdd, 3, 5});
}

TEST_P(CpuDotLibraryTest, MatMulAddSubMulSameInputs) {
  const absl::string_view hlo_template = R"(
    HloModule matmul

    ENTRY %main {
      %input = $in_dtype[64,64]{1,0} parameter(0)
      %weight = $in_dtype[64,262144]{1,0} parameter(1)
      %addend = $out_dtype[64,262144]{1,0} parameter(2)
      %dot = $out_dtype[64,262144]{1,0} dot(%input, %weight),
             lhs_contracting_dims={1}, rhs_contracting_dims={0}
      %add = $out_dtype[64,262144]{1,0} add(%dot, %addend)
      %sub = $out_dtype[64,262144]{1,0} subtract(%add, %addend)
      ROOT %mul = $out_dtype[64,262144]{1,0} multiply(%sub, %addend)
    })";

  RunTest(hlo_template, GetParam().lib == "xnn"
                            ? FusionProperties{HloOpcode::kMultiply, 3, 7}
                            : FusionProperties{HloOpcode::kAdd, 3, 5});
}

TEST_P(CpuDotLibraryTest, MatMulAddSubMulDifferentInputs) {
  const absl::string_view hlo_template = R"(
    HloModule matmul

    ENTRY %main {
      %input = $in_dtype[64,64]{1,0} parameter(0)
      %weight = $in_dtype[64,262144]{1,0} parameter(1)
      %addend = $out_dtype[64,262144]{1,0} parameter(2)
      %subtractor = $out_dtype[64,262144]{1,0} parameter(3)
      %multiplier = $out_dtype[64,262144]{1,0} parameter(4)
      %dot = $out_dtype[64,262144]{1,0} dot(%input, %weight),
             lhs_contracting_dims={1}, rhs_contracting_dims={0}
      %add = $out_dtype[64,262144]{1,0} add(%dot, %addend)
      %sub = $out_dtype[64,262144]{1,0} subtract(%add, %subtractor)
      ROOT %mul = $out_dtype[64,262144]{1,0} multiply(%sub, %multiplier)
    })";

  RunTest(hlo_template, GetParam().lib == "xnn"
                            ? FusionProperties{HloOpcode::kMultiply, 5, 9}
                            : FusionProperties{HloOpcode::kAdd, 3, 5});
}

TEST_P(CpuDotLibraryTest, MatMulAddMinExpSort) {
  const absl::string_view hlo_template = R"(
    HloModule matmul

    compare {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT result = pred[] compare(lhs, rhs), direction=LT
    }

    ENTRY %main {
      %input = $in_dtype[64,64]{1,0} parameter(0)
      %weight = $in_dtype[64,262144]{1,0} parameter(1)
      %addend = $out_dtype[64,262144]{1,0} parameter(2)
      %threshold = $out_dtype[64,262144]{1,0} parameter(3)
      %dot = $out_dtype[64,262144]{1,0} dot(%input, %weight),
             lhs_contracting_dims={1}, rhs_contracting_dims={0}
      %add = $out_dtype[64,262144]{1,0} add(%dot, %addend)
      %min = $out_dtype[64,262144]{1,0} minimum(%add, %threshold)
      %exp = $out_dtype[64,262144]{1,0} exponential(%min)
      ROOT %sorted = $out_dtype[64,262144] sort(%exp),
                     dimensions={0}, to_apply=compare
    })";

  // Sort is not supported by xnn_emitter and should not be in the fusion.
  RunTest(hlo_template, GetParam().lib == "xnn"
                            ? FusionProperties{HloOpcode::kExp, 4, 8}
                            : FusionProperties{HloOpcode::kAdd, 3, 5});
}

TEST_P(CpuDotLibraryTest, DoNotFuseMultiOutputs) {
  const absl::string_view hlo_template = R"(
    HloModule matmul

    ENTRY %main {
      %input = $in_dtype[64,64]{1,0} parameter(0)
      %weight = $in_dtype[64,262144]{1,0} parameter(1)
      %addend = $out_dtype[64,262144]{1,0} parameter(2)
      %val1 = $out_dtype[64,262144]{1,0} parameter(3)
      %val2 = $out_dtype[64,262144]{1,0} parameter(4)
      %dot = $out_dtype[64,262144]{1,0} dot(%input, %weight),
             lhs_contracting_dims={1}, rhs_contracting_dims={0}
      %add = $out_dtype[64,262144]{1,0} add(%dot, %addend)
      %sub1 = $out_dtype[64,262144]{1,0} subtract(%add, %val1)
      %sub2 = $out_dtype[64,262144]{1,0} subtract(%add, %val2)
      ROOT %mul = $out_dtype[64,262144]{1,0} multiply(%sub1, %sub2)
    })";

  // `dot + add` fusion has 2 users, so we cannot fuse further.
  RunTest(hlo_template, {HloOpcode::kAdd, 3, 5});
}

std::vector<XnnDotRewriteTestSpec> GetXnnDotRewriteTestSpecs() {
  // CPUs to test with.
  absl::flat_hash_map<std::string, std::string> cpu_to_features = {
      {"znver3", "+avx,+avx2"},
      {"sapphirerapids",
       "+avx512vnni,+avx512bf16,+amx-bf16,+amx-int8,+amx-tile,+amx-transpose"},
  };

  // Input and output data types to test per each library + CPU combination.
  using StrPair = std::pair<std::string, std::string>;
  absl::flat_hash_map<StrPair, std::vector<StrPair>> dtype_map = {
      {{"xnn", "znver3"}, {{"f32", "f32"}, {"bf16", "f32"}}},
      {{"xnn", "sapphirerapids"},
       {{"f32", "f32"}, {"bf16", "f32"}, {"bf16", "bf16"}}},
      {{"onednn", "sapphirerapids"}, {{"f32", "f32"}}},
  };

  std::vector<XnnDotRewriteTestSpec> specs;
  for (auto& [lib_cpu, dtype_pairs] : dtype_map) {
    auto& [lib, cpu] = lib_cpu;
    for (auto& [in_dtype, out_dtype] : dtype_pairs) {
      std::string& features = cpu_to_features.at(cpu);
      bool changed =
          in_dtype != "bf16" || absl::StrContains(features, "+avx512bf16");
      specs.push_back(XnnDotRewriteTestSpec{lib, in_dtype, out_dtype, cpu,
                                            features, changed});
    }
  }
  return specs;
}

INSTANTIATE_TEST_SUITE_P(CpuDotLibraryTestSuite, CpuDotLibraryTest,
                         ::testing::ValuesIn(GetXnnDotRewriteTestSpecs()),
                         CpuDotLibraryTest::Name);

}  // namespace
}  // namespace xla::cpu
