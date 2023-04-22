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

#include <algorithm>
#include <string>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "llvm-c/Target.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/cpu/tests/cpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/core/platform/test.h"

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

    std::string opcode = HloOpcodeString(spec.opcode);
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

    return absl::StrCat(opcode, "_On_", triple,
                        (features.empty() ? "" : "_With"), features);
  }

 private:
  DebugOptions GetDebugOptionsForTest() override {
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

  string check_lines{spec.check_lines.data(), spec.check_lines.size()};

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

}  // namespace
}  // namespace cpu
}  // namespace xla
