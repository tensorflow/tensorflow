/* Copyright 2019 The OpenXLA Authors.

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

#include <cctype>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "llvm-c/Target.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/platform.h"

namespace xla {
namespace cpu {
namespace {

const char* const kTriple_x86_64 = "x86_64-pc-linux";
const char* const kTriple_android_arm = "armv7-none-android";

struct VectorizationTestSpec {
  HloOpcode opcode;
  std::string triple;
  std::string features;
  std::string check_lines;
};

// Tests that the vectorizer does what we want.
class CpuVectorizationTest
    : public CpuCodegenTest,
      public ::testing::WithParamInterface<VectorizationTestSpec> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<VectorizationTestSpec>& info) {
    auto spec = info.param;

    std::string opcode(HloOpcodeString(spec.opcode));
    opcode[0] = toupper(opcode[0]);

    std::string triple{spec.triple.data(), spec.triple.size()};
    if (triple == kTriple_x86_64) {
      triple = "x86_64";
    } else if (triple == kTriple_android_arm) {
      triple = "android_arm";
    } else {
      triple = "Unknown";
    }

    std::string features = spec.features;
    if (!features.empty()) {
      absl::c_replace_if(
          features, [](char c) { return c != '_' && !absl::ascii_isalnum(c); },
          '_');
    }

    return absl::StrCat(opcode, "_on_", triple,
                        (features.empty() ? "" : "_With"), features);
  }

 private:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    HloTestBase::SetAotFastMathDebugOptions(&debug_options);
    return debug_options;
  }
};

TEST_P(CpuVectorizationTest, DoIt) {
  HloComputation::Builder builder(TestName());
  VectorizationTestSpec spec = GetParam();

  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTargetMC();

  auto param_shape = ShapeUtil::MakeShape(F32, {1024});
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "input1"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(param_shape, spec.opcode, param0, param1));
  std::unique_ptr<HloComputation> computation = builder.Build();

  CpuAotCompilationOptions options{
      /*triple=*/spec.triple, /*cpu_name=*/"", /*features=*/spec.features,
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  std::string check_lines{spec.check_lines.data(), spec.check_lines.size()};

  hlo_module->mutable_config()
      .mutable_debug_options()
      .set_xla_cpu_use_thunk_runtime(false);

  CompileAheadOfTimeAndVerifyIr(std::move(hlo_module), options, check_lines,
                                /*match_optimized_ir=*/true);
}

VectorizationTestSpec CpuVectorizationTestCases[] = {
    VectorizationTestSpec{HloOpcode::kMultiply, kTriple_x86_64, "",
                          R"(CHECK: fmul fast <4 x float>)"},

    VectorizationTestSpec{HloOpcode::kMultiply, kTriple_x86_64, "+avx",
                          R"(CHECK: fmul fast <8 x float>)"},

    VectorizationTestSpec{HloOpcode::kMultiply, kTriple_android_arm,
                          "-vfp,-neon", R"(CHECK: fmul fast float)"},

    // Neon is not IEEE754-compliant (no denormals). We want vectorized code
    // anyways.
    VectorizationTestSpec{HloOpcode::kMultiply, kTriple_android_arm,
                          "+neon,-vfp", R"(CHECK: fmul fast <4 x float>)"}};

INSTANTIATE_TEST_SUITE_P(CpuVectorizationTestInstantiation,
                         CpuVectorizationTest,
                         ::testing::ValuesIn(CpuVectorizationTestCases),
                         CpuVectorizationTest::Name);

struct MaxIsaTestSpec {
  std::string max_isa;
  std::string feature;
  bool should_enable;
};

class MaxIsaTest : public CpuCodegenTest,
                   public ::testing::WithParamInterface<MaxIsaTestSpec> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<MaxIsaTestSpec>& info) {
    // Test names cannot contain '-'. Replace it with '_'.
    std::string feature = info.param.feature;
    absl::c_replace_if(
        feature, [](char c) { return c != '_' && !absl::ascii_isalnum(c); },
        '_');
    return absl::StrCat(info.param.max_isa, "_feature_", feature);
  }
};

class X86MaxIsaTest : public MaxIsaTest {};

TEST_P(X86MaxIsaTest, ShouldEnableFeature) {
  HloComputation::Builder builder(TestName());
  MaxIsaTestSpec spec = GetParam();
  if (!tsl::port::IsX86CPU()) {
    GTEST_SKIP() << "This test is for x86 CPUs.";
  }

  auto max_feature = CpuFeatureFromString(spec.max_isa);
  bool should_enable = ShouldEnableCpuFeature(spec.feature, *max_feature);
  EXPECT_EQ(should_enable, spec.should_enable);
}

std::vector<MaxIsaTestSpec> GetX86MaxIsaTestCases() {
  return std::vector<MaxIsaTestSpec>({
      MaxIsaTestSpec{"AVX2", "avx", true},
      MaxIsaTestSpec{"AVX2", "avx2", true},
      MaxIsaTestSpec{"AVX2", "avx512f", false},
      MaxIsaTestSpec{"AVX2", "avx512vnni", false},
      MaxIsaTestSpec{"AVX2", "evex512", false},
      MaxIsaTestSpec{"AVX512", "avx512f", true},
      MaxIsaTestSpec{"AVX512", "avx512vnni", false},
      MaxIsaTestSpec{"AVX512", "amx-bf16", false},
  });
}

INSTANTIATE_TEST_SUITE_P(X86MaxIsaTestInstantiation, X86MaxIsaTest,
                         ::testing::ValuesIn(GetX86MaxIsaTestCases()),
                         X86MaxIsaTest::Name);

class AArch64MaxIsaTest : public MaxIsaTest {};

TEST_P(AArch64MaxIsaTest, ShouldEnableFeature) {
  HloComputation::Builder builder(TestName());
  MaxIsaTestSpec spec = GetParam();
  if (!tsl::port::IsAarch64CPU()) {
    GTEST_SKIP() << "This test is for AArch64 CPUs.";
  }

  auto max_feature = CpuFeatureFromString(spec.max_isa);
  bool should_enable = ShouldEnableCpuFeature(spec.feature, *max_feature);
  EXPECT_EQ(should_enable, spec.should_enable);
}

std::vector<MaxIsaTestSpec> GetAArch64MaxIsaTestCases() {
  return std::vector<MaxIsaTestSpec>({
      MaxIsaTestSpec{"NEON", "neon", true},
      MaxIsaTestSpec{"NEON", "sve", false},
      MaxIsaTestSpec{"NEON", "sve2", false},
      MaxIsaTestSpec{"SVE", "neon", true},
      MaxIsaTestSpec{"SVE", "sve", true},
      MaxIsaTestSpec{"SVE", "sve2", false},
      MaxIsaTestSpec{"SVE2", "neon", true},
      MaxIsaTestSpec{"SVE2", "sve", true},
      MaxIsaTestSpec{"SVE2", "sve2", true},
  });
}

INSTANTIATE_TEST_SUITE_P(AArch64MaxIsaTestInstantiation, AArch64MaxIsaTest,
                         ::testing::ValuesIn(GetAArch64MaxIsaTestCases()),
                         AArch64MaxIsaTest::Name);

class DefaultMaxIsaTest : public CpuCodegenTest {};

TEST_F(DefaultMaxIsaTest, NeonForOssAArch64) {
  if (!tsl::port::IsAarch64CPU()) {
    GTEST_SKIP() << "This test is for AArch64 CPUs.";
  }
  DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
  EXPECT_EQ(debug_options.xla_cpu_max_isa(), "NEON");
}

struct JitVectorizationTestSpec {
  HloOpcode opcode;
  std::string max_isa;
  std::string check_template;
  int num_vector_elements;
};

class JitVectorizationTest
    : public CpuCodegenTest,
      public ::testing::WithParamInterface<JitVectorizationTestSpec> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<JitVectorizationTestSpec>& info) {
    std::string op_name(HloOpcodeString(info.param.opcode));
    op_name[0] = toupper(op_name[0]);
    return absl::StrCat(op_name, "_max_", info.param.max_isa);
  }

 private:
  DebugOptions GetDebugOptionsForTest() const override {
    JitVectorizationTestSpec spec = GetParam();
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_cpu_max_isa(spec.max_isa);
    // For AVX512, we have to override the default `prefer_vector_width=256`
    // setting. Otherwise, LLVM won't generate AVX512.
    // TODO(penporn): Change the setting for actual AVX512 codegen too.
    if (spec.max_isa == "AVX512") {
      debug_options.set_xla_cpu_prefer_vector_width(512);
    }
    return debug_options;
  }
};

// Most Aarch64 CPUs are still using 128-bit registers so we don't have this
// test for Aarch64.
TEST_P(JitVectorizationTest, JitX86UpToIsa) {
  if (!tsl::port::IsX86CPU()) {
    GTEST_SKIP() << "This feature only works for x86 CPUs.";
  }
  HloComputation::Builder builder(TestName());
  JitVectorizationTestSpec spec = GetParam();

  // If the CPU doesn't have the `max_isa` feature, e.g., `max_isa=AVX512` but
  // we are running on an AVX2 machine, update the `check_lines` accordingly.
  using tsl::port::CPUFeature;
  auto feature = CpuFeatureFromString(spec.max_isa);
  if (!tsl::port::TestCPUFeature(*feature)) {
    if (tsl::port::TestCPUFeature(CPUFeature::AVX)) {
      spec.num_vector_elements = 8;
    } else {
      spec.num_vector_elements = 4;
    }
  }
  std::string check_lines = absl::StrReplaceAll(
      spec.check_template, {{"%d", absl::StrCat(spec.num_vector_elements)}});

  // Build HLO module.
  auto shape = ShapeUtil::MakeShape(F32, {1024});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(shape, spec.opcode, a, b));
  std::unique_ptr<HloComputation> computation = builder.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  CompileAndVerifyIr(std::move(hlo_module), check_lines,
                     /*match_optimized_ir=*/true);
}

std::vector<JitVectorizationTestSpec> GetJitVectorizationTestCases() {
  return std::vector<JitVectorizationTestSpec>({
      JitVectorizationTestSpec{HloOpcode::kMultiply, "SSE4_2",
                               R"(CHECK: fmul <%d x float>)", 4},
      JitVectorizationTestSpec{HloOpcode::kMultiply, "AVX2",
                               R"(CHECK: fmul <%d x float>)", 8},
      JitVectorizationTestSpec{HloOpcode::kMultiply, "AVX512",
                               R"(CHECK: fmul <%d x float>)", 16},
  });
}

INSTANTIATE_TEST_SUITE_P(JitVectorizationTestInstantiation,
                         JitVectorizationTest,
                         ::testing::ValuesIn(GetJitVectorizationTestCases()),
                         JitVectorizationTest::Name);

}  // namespace
}  // namespace cpu
}  // namespace xla
