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
#include "xla/backends/cpu/xnn_gemm_config.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

struct DotRewriteTestSpec {
  std::string lib;
  std::string in_dtype;
  std::string out_dtype;
  std::string cpu_name;
  std::string features;
  std::string fusion_mode;
};

class CpuLibraryTest : public TargetMachineTestBase {
 protected:
  struct FusionProperties {
    HloOpcode fusion_root;
    int num_fusion_params;
    int num_instructions_in_fused_computation;
    bool changed;
  };

  virtual void RunTest(absl::string_view hlo_template,
                       FusionProperties expected) {}

  void RunTestInternal(DotRewriteTestSpec spec, absl::string_view hlo_template,
                       FusionProperties expected) {
    // Create TargetMachineFeatures.
    std::unique_ptr<TargetMachineFeatures> features =
        CreateTargetMachineFeatures(
            /*triple_string=*/"x86_64-unknown-linux-gnu", spec.cpu_name,
            spec.features);

    // Override XnnGemmConfig.
    GetXnnGemmConfig().SetTestFilter([](const XnnGemm&) { return true; });

    // Create an HLO module with the specified input and output data types.
    std::string hlo_text = absl::StrReplaceAll(
        hlo_template,
        {{"$in_dtype", spec.in_dtype}, {"$out_dtype", spec.out_dtype}});
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text));

    // Run the pass.
    tsl::protobuf::RepeatedField<int> fusion_types;
    fusion_types.Add(DebugOptions::LIBRARY_FUSION_TYPE_DOT);
    if (spec.fusion_mode == "greedy") {
      fusion_types.Add(DebugOptions::LIBRARY_FUSION_TYPE_ELTWISE);
    }
    tsl::protobuf::RepeatedField<int> empty_fusion_types;
    bool use_onednn = spec.lib == "onednn";
    bool use_xnnpack = spec.lib == "xnn";
    DotLibraryRewriterOptions options = {
        use_onednn, use_xnnpack,
        /*onednn_fusion_types=*/
        use_onednn ? &fusion_types : &empty_fusion_types,
        /*xnn_fusion_types=*/use_xnnpack ? &fusion_types : &empty_fusion_types};
    DotLibraryRewriter rewriter(features.get(), options);
    EXPECT_EQ(expected.changed, rewriter.Run(module.get()).value());
    if (!expected.changed) {
      return;  // No further checks if the module was not changed.
    }
    VLOG(3) << module->ToString();

    // Verify that there is a custom fusion.
    HloInstruction* result = FindInstruction(module.get(), HloOpcode::kFusion);
    EXPECT_NE(result, nullptr);
    HloFusionInstruction* fusion = Cast<HloFusionInstruction>(result);
    EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kCustom);

    // Adjust the expected values if a convert is auto-inserted.
    if (spec.out_dtype == "bf16" &&
        hlo_query::FindInstruction(fusion->fused_instructions_computation(),
                                   HloOpcode::kDot)) {
      ++expected.num_instructions_in_fused_computation;
      if (expected.fusion_root == HloOpcode::kDot) {
        expected.fusion_root = HloOpcode::kConvert;
      }
    }
    HloInstruction* fusion_root = fusion->fused_expression_root();
    EXPECT_EQ(fusion_root->opcode(), expected.fusion_root);
    EXPECT_EQ(fusion->operand_count(), expected.num_fusion_params);
    EXPECT_EQ(fusion->fused_instructions_computation()->instruction_count(),
              expected.num_instructions_in_fused_computation);
  }
};

class CpuLibraryFullParamTest
    : public CpuLibraryTest,
      public ::testing::WithParamInterface<DotRewriteTestSpec> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<DotRewriteTestSpec>& info) {
    return absl::StrCat(info.param.lib, "_", info.param.fusion_mode, "_",
                        info.param.in_dtype, "_", info.param.out_dtype, "_",
                        info.param.cpu_name);
  }

 protected:
  void RunTest(absl::string_view hlo_template,
               FusionProperties expected) override {
    RunTestInternal(GetParam(), hlo_template, expected);
  }

  bool IsDotEnabledOnCPU() {
    DotRewriteTestSpec spec = GetParam();
    bool bf16_dot_supported = absl::StrContains(spec.features, "+avx512bf16");
    return spec.in_dtype != "bf16" || bf16_dot_supported;
  }
};

TEST_P(CpuLibraryFullParamTest, AddMatMul) {
  const absl::string_view hlo_template = R"(
    HloModule matmul

    ENTRY %main {
      %a = $in_dtype[64,64] parameter(0)
      %b = $in_dtype[64,64] parameter(1)
      %c = $in_dtype[64,64] parameter(2)
      %x = $in_dtype[64,64] add(%a, %b)
      %y = $in_dtype[64,64] add(%a, %c)
      ROOT %dot = $out_dtype[64,64]{1,0} dot(%x, %y),
                  lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  DotRewriteTestSpec spec = GetParam();
  FusionProperties expected = {HloOpcode::kDot, 0, 0, false};
  if (IsDotEnabledOnCPU()) {
    expected = FusionProperties{HloOpcode::kDot, 3, 6, true};
  } else if (spec.fusion_mode == "greedy") {
    expected = FusionProperties{HloOpcode::kAdd, 2, 3, true};
  }
  RunTest(hlo_template, expected);
}

TEST_P(CpuLibraryFullParamTest, MatMul) {
  const absl::string_view hlo_template = R"(
    HloModule matmul

    ENTRY %main {
      %input = $in_dtype[64,64]{1,0} parameter(0)
      %weight = $in_dtype[64,262144]{1,0} parameter(1)
      ROOT %dot = $out_dtype[64,262144]{1,0} dot(%input, %weight),
                  lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  RunTest(hlo_template, {HloOpcode::kDot, 2, 3, IsDotEnabledOnCPU()});
}

TEST_P(CpuLibraryFullParamTest, MatMulAndAdd) {
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

  FusionProperties expected = {HloOpcode::kAdd, 0, 0, false};
  if (IsDotEnabledOnCPU()) {
    // Dot and Add in the fusion.
    expected = {HloOpcode::kAdd, 3, 5, true};
  } else if (GetParam().fusion_mode == "greedy") {
    // Only Add in the fusion.
    expected = {HloOpcode::kAdd, 2, 3, true};
  }
  RunTest(hlo_template, expected);
}

TEST_P(CpuLibraryFullParamTest, MatMulAddSubMulSameInputs) {
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

  DotRewriteTestSpec spec = GetParam();
  FusionProperties expected = {HloOpcode::kMultiply, 0, 0, false};
  if (IsDotEnabledOnCPU()) {
    // {Dot, Add, Sub, Mul} for XNN, {Dot, Add} for oneDNN.
    expected = spec.lib == "xnn"
                   ? FusionProperties{HloOpcode::kMultiply, 3, 7, true}
                   : FusionProperties{HloOpcode::kAdd, 3, 5, true};
  } else if (spec.fusion_mode == "greedy") {
    // Only Add, Sub, and Mul in the fusion.
    expected = {HloOpcode::kMultiply, 2, 5, true};
  }
  RunTest(hlo_template, expected);
}

TEST_P(CpuLibraryFullParamTest, MatMulAddSubMulDifferentInputs) {
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

  DotRewriteTestSpec spec = GetParam();
  FusionProperties expected = {HloOpcode::kMultiply, 0, 0, false};
  if (IsDotEnabledOnCPU()) {
    // {Dot, Add, Sub, Mul} for XNN, {Dot, Add} for oneDNN.
    expected = spec.lib == "xnn"
                   ? FusionProperties{HloOpcode::kMultiply, 5, 9, true}
                   : FusionProperties{HloOpcode::kAdd, 3, 5, true};
  } else if (spec.fusion_mode == "greedy") {
    // Only Add, Sub, and Mul in the fusion.
    expected = {HloOpcode::kMultiply, 4, 7, true};
  }
  RunTest(hlo_template, expected);
}

TEST_P(CpuLibraryFullParamTest, MatMulAddMinExpSort) {
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
  DotRewriteTestSpec spec = GetParam();
  FusionProperties expected = {HloOpcode::kExp, 0, 0, false};
  if (IsDotEnabledOnCPU()) {
    // {Dot, Add, Min, Exp} for XNN, {Dot, Add} for oneDNN.
    expected = spec.lib == "xnn"
                   ? FusionProperties{HloOpcode::kExp, 4, 8, true}
                   : FusionProperties{HloOpcode::kAdd, 3, 5, true};
  } else if (spec.fusion_mode == "greedy") {
    // Only {Add, Min, Exp} in the fusion.
    expected = {HloOpcode::kExp, 3, 6, true};
  }
  RunTest(hlo_template, expected);
}

TEST_P(CpuLibraryFullParamTest, DoNotFuseMultiOutputs) {
  //   weight   addend   val2
  //         \        \     \
  // input -- dot --- add -- sub2 -- tuple
  //                      \         /
  //                val1 -- sub1 ---
  const absl::string_view hlo_template = R"(
    HloModule matmul

    ENTRY %main {
      %input = $in_dtype[64,64] parameter(0)
      %weight = $in_dtype[64,262144] parameter(1)
      %addend = $out_dtype[64,262144] parameter(2)
      %val1 = $out_dtype[64,262144] parameter(3)
      %val2 = $out_dtype[64,262144] parameter(4)
      %dot = $out_dtype[64,262144] dot(%input, %weight),
             lhs_contracting_dims={1}, rhs_contracting_dims={0}
      %add = $out_dtype[64,262144] add(%dot, %addend)
      %sub1 = $out_dtype[64,262144] subtract(%val1, %add)
      %sub2 = $out_dtype[64,262144] subtract(%add, %val2)
      ROOT %tuple = ($out_dtype[64,262144], $out_dtype[64,262144])
                    tuple(%sub1, %sub2)
    })";

  // Many cases will have multiple fusions. We only check the first one.
  DotRewriteTestSpec spec = GetParam();
  FusionProperties expected = {HloOpcode::kAdd, 0, 0, false};
  if (IsDotEnabledOnCPU()) {
    // {Dot, Add} fusion has 2 users, so we cannot fuse further.
    expected = {HloOpcode::kAdd, 3, 5, true};
  } else if (spec.fusion_mode == "greedy") {
    // Only {Add} in the fusion.
    expected = {HloOpcode::kSubtract, 2, 3, true};
  }
  RunTest(hlo_template, expected);
}

std::vector<DotRewriteTestSpec> GetDotRewriteTestSpecs() {
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

  // Fusion modes to test for each library.
  // We temporarily use XNN_GRAPH_FUSION_MODE_DISABLED to denote the dot fusion
  // mode (starting fusion nodes with dots).
  absl::flat_hash_map<std::string, std::vector<std::string>> fusion_modes = {
      {"xnn", {"dot", "greedy"}},
      {"onednn", {"dot"}},
  };

  std::vector<DotRewriteTestSpec> specs;
  for (auto& [lib_cpu, dtype_pairs] : dtype_map) {
    auto& [lib, cpu] = lib_cpu;
    for (auto& [in_dtype, out_dtype] : dtype_pairs) {
      if (out_dtype == "bf16" && cpu == "znver3") {
        continue;
      }
      std::string& features = cpu_to_features.at(cpu);
      for (auto& fusion_mode : fusion_modes.at(lib)) {
        specs.push_back(DotRewriteTestSpec{lib, in_dtype, out_dtype, cpu,
                                           features, fusion_mode});
      }
    }
  }
  return specs;
}

INSTANTIATE_TEST_SUITE_P(CpuLibraryFullParamTestSuite, CpuLibraryFullParamTest,
                         ::testing::ValuesIn(GetDotRewriteTestSpecs()),
                         CpuLibraryFullParamTest::Name);

class CpuLibraryFusionTypeTest
    : public CpuLibraryTest,
      public ::testing::WithParamInterface<std::string> {
 public:
  static std::string Name(const ::testing::TestParamInfo<std::string>& info) {
    return absl::StrCat(kLib, "_", info.param, "_", kDType, "_", kCpuName);
  }

 protected:
  void RunTest(absl::string_view hlo_template,
               FusionProperties expected) override {
    DotRewriteTestSpec spec = {std::string(kLib),      std::string(kDType),
                               std::string(kDType),    std::string(kCpuName),
                               std::string(kFeatures), GetParam()};
    RunTestInternal(spec, hlo_template, expected);
  }
  static constexpr absl::string_view kLib = "xnn";
  static constexpr absl::string_view kDType = "f32";
  static constexpr absl::string_view kCpuName = "znver3";
  static constexpr absl::string_view kFeatures = "+avx,+avx2";
};

TEST_P(CpuLibraryFusionTypeTest, AllEltwiseFusion) {
  //   b    c -- exp
  //    \           \
  // a -- mul ------ add
  const absl::string_view hlo_template = R"(
    HloModule matmul

    ENTRY %main {
      %a = $in_dtype[64,64] parameter(0)
      %b = $in_dtype[64,64] parameter(1)
      %c = $in_dtype[64,64] parameter(2)
      %mul = $in_dtype[64,64] multiply(%a, %b)
      %exp = $in_dtype[64,64] exponential(%c)
      %add = $in_dtype[64,64] add(%mul, %exp)
    })";

  RunTest(hlo_template, GetParam() == "greedy"
                            ? FusionProperties{HloOpcode::kAdd, 3, 6, true}
                            : FusionProperties{HloOpcode::kAdd, 0, 0, false});
}

TEST_P(CpuLibraryFusionTypeTest, ForkJoin) {
  //   b      c
  //    \      \
  // a -- mul -- add1 -- exp -- add2
  //          \________________/
  const absl::string_view hlo_template = R"(
    HloModule matmul

    ENTRY %main {
      %a = $in_dtype[64,64] parameter(0)
      %b = $in_dtype[64,64] parameter(1)
      %c = $in_dtype[64,64] parameter(2)
      %mul = $in_dtype[64,64] multiply(%a, %b)
      %add1 = $in_dtype[64,64] add(%mul, %c)
      %exp = $in_dtype[64,64] exponential(%add1)
      ROOT %add2 = $in_dtype[64,64] add(%mul, %exp)
    })";

  RunTest(hlo_template, GetParam() == "greedy"
                            ? FusionProperties{HloOpcode::kAdd, 3, 7, true}
                            : FusionProperties{HloOpcode::kAdd, 0, 0, false});
}

TEST_P(CpuLibraryFusionTypeTest, NoCycle) {
  //   b      c
  //    \      \
  // a -- mul -- add1 -- chol -- add2
  //          \________________/
  //
  // `chol` is not supported by libraries. We check that we don't fuse all
  // {mul, add1, add2} together, which would create a cycle.
  const absl::string_view hlo_template = R"(
    HloModule matmul

    ENTRY %main {
      %a = $in_dtype[64,64] parameter(0)
      %b = $in_dtype[64,64] parameter(1)
      %c = $in_dtype[64,64] parameter(2)
      %mul = $in_dtype[64,64] multiply(%a, %b)
      %add1 = $in_dtype[64,64] add(%mul, %c)
      %chol = $in_dtype[64,64] cholesky(%add1)
      ROOT %add2 = $in_dtype[64,64] add(%mul, %chol)
    })";

  RunTest(hlo_template, GetParam() == "greedy"
                            ? FusionProperties{HloOpcode::kAdd, 2, 3, true}
                            : FusionProperties{HloOpcode::kAdd, 0, 0, false});
}

TEST_P(CpuLibraryFusionTypeTest, JoiningFusions) {
  //   b        c ---- add2 ----
  //    \        \   /          \
  // a -- add1 -- dot -- exp -- mul
  //
  // In "dot" mode, the fusion grown from `dot` will only have {add1, dot} since
  // it needs multi-output support to include `add2` or `exp`. But in "greedy"
  // mode, the fusion grown upwards from `mul` will be able to merge with the
  // dot fusion into a single fusion.
  const absl::string_view hlo_template = R"(
    HloModule matmul

    ENTRY %main {
      %a = $in_dtype[64,64] parameter(0)
      %b = $in_dtype[64,64] parameter(1)
      %c = $in_dtype[64,64] parameter(2)
      %add1 = $in_dtype[64,64] add(%a, %b)
      %dot = $in_dtype[64,64] dot(%add1, %c), lhs_contracting_dims={1},
                                              rhs_contracting_dims={0}
      %exp = $in_dtype[64,64] exponential(%dot)
      %add2 = $in_dtype[64,64] add(%dot, %c)
      ROOT %mul = $in_dtype[64,64] multiply(%exp, %add2)
    })";

  RunTest(hlo_template, GetParam() == "greedy"
                            ? FusionProperties{HloOpcode::kMultiply, 3, 8, true}
                            : FusionProperties{HloOpcode::kDot, 3, 5, true});
}

INSTANTIATE_TEST_SUITE_P(CpuLibraryFusionTypeTestSuite,
                         CpuLibraryFusionTypeTest,
                         ::testing::ValuesIn({std::string("dot"),
                                              std::string("greedy")}),
                         CpuLibraryFusionTypeTest::Name);

}  // namespace
}  // namespace xla::cpu
