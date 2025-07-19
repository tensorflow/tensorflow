/* Copyright 2018 The OpenXLA Authors.

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
#include <cctype>
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
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {
namespace {

const char* const kTriple_x86_64 = "x86_64-pc-linux";
const char* const kTriple_android_arm = "armv7-none-android";

struct IntrinsicTestSpec {
  HloOpcode opcode;
  PrimitiveType type;
  bool match_optimized_ir;
  absl::string_view triple;
  absl::string_view features;
  absl::string_view check_lines;
};

// Tests that unary functions get lowered using intrinsic calls.
class CpuUnaryIntrinsicTest
    : public CpuCodegenTest,
      public ::testing::WithParamInterface<IntrinsicTestSpec> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<IntrinsicTestSpec>& info) {
    auto spec = info.param;

    std::string opcode(HloOpcodeString(spec.opcode));
    opcode[0] = toupper(opcode[0]);

    std::string type(PrimitiveType_Name(spec.type));
    type[0] = toupper(type[0]);

    std::string triple{spec.triple.data(), spec.triple.size()};
    if (triple == kTriple_x86_64) {
      triple = "x86_64";
    } else if (triple == kTriple_android_arm) {
      triple = "android_arm";
    } else {
      triple = "Unknown";
    }

    std::string features{spec.features.data(), spec.features.size()};
    if (!features.empty()) {
      std::replace_if(
          features.begin(), features.end(),
          [](char c) { return c != '_' && !absl::ascii_isalnum(c); }, '_');
    } else {
      features = "";
    }

    std::string opt = spec.match_optimized_ir ? "" : "_PreOpt_";

    return absl::StrCat(opcode, "_", type, opt, "_On_", triple,
                        (features.empty() ? "" : "_With"), features);
  }

 private:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloHardwareIndependentTestBase::GetDebugOptionsForTest();
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

  auto param_shape = ShapeUtil::MakeShape(spec.type, {1024});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input"));
  builder.AddInstruction(
      HloInstruction::CreateUnary(param_shape, spec.opcode, param));
  std::unique_ptr<HloComputation> computation = builder.Build();

  std::string triple{spec.triple.data(), spec.triple.size()};
  std::string features{spec.features.data(), spec.features.size()};

  CpuAotCompilationOptions options{
      /*triple=*/triple, /*cpu_name=*/"", /*features=*/features,
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  std::string check_lines{spec.check_lines.data(), spec.check_lines.size()};

  hlo_module->mutable_config()
      .mutable_debug_options()
      .set_xla_cpu_use_thunk_runtime(false);

  CompileAheadOfTimeAndVerifyIr(std::move(hlo_module), options, check_lines,
                                spec.match_optimized_ir);
}

IntrinsicTestSpec CpuUnaryIntrinsicTestCases[] = {
    // The intrinsics are always inlined, so we match a line from it instead of
    // a function call.

    IntrinsicTestSpec{
        HloOpcode::kExp, F32, true, kTriple_x86_64, "",
        R"(CHECK: fmul fast <4 x float> splat (float 0xBF2BD01060000000)"},

    // Check that we see inlined vectorized exp.f64 code
    IntrinsicTestSpec{HloOpcode::kExp, F64, true, kTriple_x86_64, "",
                      R"(
                      CHECK-NOT: define {{[a-z]* ?}}<4 x double> @local_xla.exp.v4f32
                      CHECK-NOT: define {{[a-z]* ?}}<4 x double> @local_xla.exp.v4f64
                      CHECK: fmul <2 x double> {{.*}}splat (double 0x3FF71547652B82FE)
                      CHECK-NOT: define {{[a-z]* ?}}<2 x double> @local_xla.exp.v2f32
                      CHECK-NOT: define {{[a-z]* ?}}<4 x double> @local_xla.exp.v4f64
    )"},

    IntrinsicTestSpec{HloOpcode::kExp, F64, true, kTriple_x86_64, "+avx",
                      R"(
                      CHECK-NOT: define {{[a-z]* ?}}<2 x double> @local_xla.exp.v2f64
                      CHECK-NOT: define {{[a-z]* ?}}<4 x float> @local_xla.exp.v4f32
                      CHECK: fmul <4 x double> {{.*}}splat (double 0x3FF71547652B82FE)
                      CHECK-NOT: define {{[a-z]* ?}}<4 x float> @local_xla.exp.v4f32
                      CHECK-NOT: define {{[a-z]* ?}}<2 x double> @local_xla.exp.v2f64
    )"},

    IntrinsicTestSpec{HloOpcode::kExp, F64, false, kTriple_x86_64, "",
                      R"(CHECK: call fast double @local_xla.exp.f64(double %4)"},

    IntrinsicTestSpec{
        HloOpcode::kExp, F32, true, kTriple_x86_64, "+avx",
        R"(CHECK: fmul fast <8 x float> splat (float 0xBF2BD01060000000)"},

    IntrinsicTestSpec{
        HloOpcode::kExp, F32, true, kTriple_android_arm, "+neon",
        R"(CHECK: fmul fast <4 x float> splat (float 0xBF2BD01060000000)"},

    IntrinsicTestSpec{
        HloOpcode::kTanh, F32, true, kTriple_x86_64, "",
        R"(CHECK: fcmp fast uge <4 x float> %wide.load, splat (float
        0xC01FFEC880000000)"},

    IntrinsicTestSpec{
        HloOpcode::kTanh, F32, true, kTriple_x86_64, "+avx",
        R"(CHECK: fcmp fast uge <8 x float> %wide.load, splat (float
        0xC01FFEC880000000)"},

    IntrinsicTestSpec{
        HloOpcode::kTanh, F32, true, kTriple_android_arm, "",
        R"(CHECK: fcmp fast uge <4 x float> %wide.load, splat (float
        0xC01FFEC880000000)"},

    IntrinsicTestSpec{
        HloOpcode::kLog, F32, true, kTriple_x86_64, "",
        R"(CHECK: fadd fast <4 x float> splat (float 0x3FBDE4A340000000)"},

    IntrinsicTestSpec{
        HloOpcode::kLog, F32, true, kTriple_x86_64, "+avx",
        R"(CHECK: fadd fast <8 x float> splat (float 0x3FBDE4A340000000)"},

    IntrinsicTestSpec{
        HloOpcode::kLog, F32, true, kTriple_android_arm, "",
        R"(CHECK: fadd fast <4 x float> splat (float 0x3FBDE4A340000000)"}};

INSTANTIATE_TEST_SUITE_P(CpuUnaryIntrinsicTestInstantiation,
                         CpuUnaryIntrinsicTest,
                         ::testing::ValuesIn(CpuUnaryIntrinsicTestCases),
                         CpuUnaryIntrinsicTest::Name);

}  // namespace
}  // namespace cpu
}  // namespace xla
