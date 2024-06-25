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

// Tests that we call into Eigen for dot operations as needed.

#include <algorithm>
#include <string>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/test_target_triple_helper.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/shape_util.h"
#include "xla/tests/test_utils.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

struct DotTestSpec {
  PrimitiveType primitive_type;
  std::string filecheck_lines;
};

std::string DotTestSpecToString(
    const ::testing::TestParamInfo<DotTestSpec>& info) {
  return PrimitiveType_Name(info.param.primitive_type);
}

class CpuEigenDotOperationTest
    : public CpuCodegenTest,
      public ::testing::WithParamInterface<DotTestSpec> {
 protected:
  void CompileAndCheck(std::unique_ptr<HloComputation> entry_computation,
                       const std::string& filecheck_lines) {
    CpuAotCompilationOptions options{
        /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
        /*features=*/"",
        /*entry_point_name=*/"entry",
        /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

    auto hlo_module = CreateNewVerifiedModule();
    hlo_module->AddEntryComputation(std::move(entry_computation));

    CompileAheadOfTimeAndVerifyIr(std::move(hlo_module), options,
                                  filecheck_lines,
                                  /*match_optimized_ir=*/true);
  }
};

TEST_P(CpuEigenDotOperationTest, SimpleDotOp) {
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)
  GTEST_SKIP() << "OneDNN rewrites dot instruction to custom-call.";
#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
  HloComputation::Builder builder(TestName());
  DotTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {128, 128});

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "input"));

  builder.AddInstruction(CreateCanonicalDot(param_shape, lhs, rhs));
  CompileAndCheck(builder.Build(), spec.filecheck_lines);
}

TEST_P(CpuEigenDotOperationTest, DotTransposeOp) {
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)
  GTEST_SKIP() << "OneDNN rewrites dot instruction to custom-call.";
#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
  HloComputation::Builder builder(TestName());
  DotTestSpec spec = GetParam();

  auto param_shape = ShapeUtil::MakeShape(spec.primitive_type, {128, 128});

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "input"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "input"));
  HloInstruction* lhs_transposed = builder.AddInstruction(
      HloInstruction::CreateTranspose(param_shape, lhs, {1, 0}));

  builder.AddInstruction(CreateCanonicalDot(param_shape, lhs_transposed, rhs));
  CompileAndCheck(builder.Build(), spec.filecheck_lines);
}

std::vector<DotTestSpec> GetDotTestCases() {
  std::vector<DotTestSpec> result;
  // The fp16 test runs a 32-bit matmul because we promote fp16 gemms to fp32
  // (they run much faster).
  result.push_back(
      {F16, R"(CHECK: call void @__xla_cpu_runtime_EigenMatMulF32)"});
  result.push_back(
      {F32, R"(CHECK: call void @__xla_cpu_runtime_EigenMatMulF32)"});
  result.push_back(
      {F64, R"(CHECK: call void @__xla_cpu_runtime_EigenMatMulF64)"});
  return result;
}

INSTANTIATE_TEST_SUITE_P(CpuEigenDotOperationTestInstantiation,
                         CpuEigenDotOperationTest,
                         ::testing::ValuesIn(GetDotTestCases()),
                         DotTestSpecToString);

}  // namespace
}  // namespace cpu
}  // namespace xla
