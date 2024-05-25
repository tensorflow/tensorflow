/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/cpu/ir_emitter2.h"

#include <memory>

#include "absl/status/statusor.h"
#include "llvm/IR/LLVMContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

using IrEmitter2Test = HloTestBase;

TEST_F(IrEmitter2Test, EmitElementalKernel) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  const char* hlo_text = R"(
    HloModule m
    ENTRY main {
      p0 = f32[2,2] parameter(0)
      ROOT convert = s32[2,2] convert(p0)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));
  HloInstruction* convert = FindInstruction(hlo.get(), "convert");
  ASSERT_NE(convert, nullptr);

  IrEmitter2 ir_emitter(module.get());
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2::HostKernelSym sym,
                          ir_emitter.EmitElementalHostKernel(convert));

  ASSERT_TRUE(*RunFileCheck(llvm_ir::DumpToString(module.get()), R"(
    CHECK: define ptr @convert(ptr %0) {
    CHECK:   fptosi float {{.*}} to i32
    CHECK: }
  )"));
}

}  // namespace
}  // namespace xla::cpu
