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

#include <cstdint>
#include <memory>
#include <utility>

#include "xla/array2d.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

class CpuExternalConstantsTest : public CpuCodegenTest {
 public:
  void TestWithArray(int64_t rows, int64_t cols,
                     const char* filecheck_pattern) {
    HloComputation::Builder builder(TestName());

    Array2D<float> backing_array(rows, cols);
    backing_array.FillUnique();

    auto shape = ShapeUtil::MakeValidatedShape(F32, {rows, cols}).value();

    HloInstruction* constant =
        builder.AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR2FromArray2D(backing_array)));
    HloInstruction* param =
        builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param, constant));

    std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
    module->AddEntryComputation(builder.Build());

    CompileAndVerifyIr(std::move(module), filecheck_pattern,
                       /*match_optimized_ir=*/false);
  }
};

TEST_F(CpuExternalConstantsTest, DoNotExternalizeConstants) {
  TestWithArray(/*rows=*/4, /*cols=*/4, R"(
CHECK-NOT: external unnamed_addr constant [16 x float]
CHECK: @[[CST:.+]] = private unnamed_addr constant [64 x i8] {{.*}}, align {{[0-9]+}}
)");
}

}  // namespace
}  // namespace xla::cpu
