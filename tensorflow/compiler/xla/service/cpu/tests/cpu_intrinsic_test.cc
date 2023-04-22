/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

struct IntrinsicTestSpec {
  HloOpcode opcode;
  absl::string_view triple;
  absl::string_view features;
  absl::string_view check_lines;
};

// Tests that unary functions get lowered using intrinsic calls.
class CpuUnaryIntrinsicTest
    : public CpuCodegenTest,
      public ::testing::WithParamInterface<IntrinsicTestSpec> {
 public:
  static string Name(const ::testing::TestParamInfo<IntrinsicTestSpec>& info) {
    auto spec = info.param;

    string opcode = HloOpcodeString(spec.opcode);
    opcode[0] = toupper(opcode[0]);

    string triple{spec.triple.data(), spec.triple.size()};
    if (triple == kTriple_x86_64) {
      triple = "x86_64";
    } else if (triple == kTriple_android_arm) {
      triple = "android_arm";
    } else {
      triple = "Unknown";
    }

    string features{spec.features.data(), spec.features.size()};
    if (!features.empty()) {
      std::replace_if(
          features.begin(), features.end(),
          [](char c) { return c != '_' && !absl::ascii_isalnum(c); }, '_');
    } else {
      features = "";
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

// Creates a module with a call to the unary op, and tests if the
// compiler replaced it with a call to the intrinsic.
TEST_P(CpuUnaryIntrinsicTest, DoIt) {
  HloComputation::Builder builder(TestName());
  IntrinsicTestSpec spec = GetParam();

  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTargetMC();

  auto param_shape = ShapeUtil::MakeShape(F32, {1024});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input"));
  builder.AddInstruction(
      HloInstruction::CreateUnary(param_shape, spec.opcode, param));
  std::unique_ptr<HloComputation> computation = builder.Build();

  string triple{spec.triple.data(), spec.triple.size()};
  string features{spec.features.data(), spec.features.size()};

  CpuAotCompilationOptions options{
      /*triple=*/triple, /*cpu_name=*/"", /*features=*/features,
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  string check_lines{spec.check_lines.data(), spec.check_lines.size()};

  CompileAheadOfTimeAndVerifyIr(std::move(hlo_module), options, check_lines,
                                /*match_optimized_ir=*/true);
}

IntrinsicTestSpec CpuUnaryIntrinsicTestCases[] = {
    // The intrinsics are always inlined, so we match a line from it instead of
    // a function call.

    IntrinsicTestSpec{
        HloOpcode::kExp, kTriple_x86_64, "",
        R"(CHECK: fmul fast <4 x float> <float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000>)"},

    IntrinsicTestSpec{
        HloOpcode::kExp, kTriple_x86_64, "+avx",
        R"(CHECK: fmul fast <8 x float> <float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000>)"},

    IntrinsicTestSpec{
        HloOpcode::kExp, kTriple_android_arm, "+neon",
        R"(CHECK: fmul fast <4 x float> <float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000>)"},

    IntrinsicTestSpec{
        HloOpcode::kTanh, kTriple_x86_64, "",
        R"(CHECK: fcmp fast uge <4 x float> %wide.load, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>)"},

    IntrinsicTestSpec{
        HloOpcode::kTanh, kTriple_x86_64, "+avx",
        R"(CHECK: fcmp fast uge <8 x float> %wide.load, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>)"},

    IntrinsicTestSpec{
        HloOpcode::kTanh, kTriple_android_arm, "",
        R"(CHECK: fcmp fast uge <4 x float> %wide.load, <float -9.000000e+00, float -9.000000e+00, float -9.000000e+00, float -9.000000e+00>)"},

    IntrinsicTestSpec{
        HloOpcode::kLog, kTriple_x86_64, "",
        R"(CHECK: fadd fast <4 x float> <float 0x3FBDE4A340000000, float 0x3FBDE4A340000000, float 0x3FBDE4A340000000, float 0x3FBDE4A340000000>)"},

    IntrinsicTestSpec{
        HloOpcode::kLog, kTriple_x86_64, "+avx",
        R"(CHECK: fadd fast <8 x float> <float 0x3FBDE4A340000000, float 0x3FBDE4A340000000, float 0x3FBDE4A340000000, float 0x3FBDE4A340000000, float 0x3FBDE4A340000000, float 0x3FBDE4A340000000, float 0x3FBDE4A340000000, float 0x3FBDE4A340000000>)"},

    IntrinsicTestSpec{
        HloOpcode::kLog, kTriple_android_arm, "",
        R"(CHECK: fadd fast <4 x float> <float 0x3FBDE4A340000000, float 0x3FBDE4A340000000, float 0x3FBDE4A340000000, float 0x3FBDE4A340000000>)"}};

INSTANTIATE_TEST_SUITE_P(CpuUnaryIntrinsicTestInstantiation,
                         CpuUnaryIntrinsicTest,
                         ::testing::ValuesIn(CpuUnaryIntrinsicTestCases),
                         CpuUnaryIntrinsicTest::Name);

}  // namespace
}  // namespace cpu
}  // namespace xla
