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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/test_target_triple_helper.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {
class CodegenReduceOnArchWithNoVectorRegisters
    : public HloHardwareIndependentTestBase {};

absl::StatusOr<unsigned int> GetTargetVectorRegisterByteSize(
    std::string triple) {
  // Unfortunately we need a lot of boilerplate to get to an
  // llvm::TargetMachine.

  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (target == nullptr) {
    return Internal("TargetRegistry::lookupTarget failed: %s", error);
  }

  llvm::LLVMContext context;
  llvm::Module module("test", context);
  llvm::Function* function = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(context), {}),
      llvm::GlobalValue::ExternalLinkage, "test", &module);

  std::unique_ptr<llvm::TargetMachine> target_machine =
      absl::WrapUnique(target->createTargetMachine(
          /*TT=*/triple, /*CPU=*/"", /*Features=*/"", llvm::TargetOptions{},
          /*RM=*/std::nullopt));
  cpu::TargetMachineFeatures target_machine_features(target_machine.get());
  return target_machine_features.vector_register_byte_size(*function);
}

TEST_F(CodegenReduceOnArchWithNoVectorRegisters, Test) {
  absl::string_view text = R"(
HloModule Reduce

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY main {
  input = f32[1000,1000] parameter(0)
  constant = f32[] constant(0)
  ROOT reduce = f32[1000] reduce(input, constant), dimensions={0}, to_apply=add
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(text));
  cpu::CpuCompiler cpu_compiler;
  auto module_group = std::make_unique<HloModuleGroup>("group");
  module_group->push_back(std::move(hlo_module));

  // Check that the GetTargetVectorRegisterByteSize is itself working.
  TF_ASSERT_OK_AND_ASSIGN(
      unsigned vector_register_byte_size_for_x86_64,
      GetTargetVectorRegisterByteSize(kTargetTripleForHost));
  ASSERT_EQ(vector_register_byte_size_for_x86_64, 16);

  std::string triple = "i686-none-android";

  TF_ASSERT_OK_AND_ASSIGN(unsigned vector_register_byte_size,
                          GetTargetVectorRegisterByteSize(triple));

  // This test is supposed to check whether the XLA CPU vectorized reduction
  // codegen works correctly for architectures that do not have vector
  // registers.  So first ASSERT that `triple` is actually a target with no
  // vector registers, as otherwise the test isn't actually testing anything
  // interesting.

  ASSERT_EQ(vector_register_byte_size, 0);

  cpu::CpuAotCompilationOptions aot_compilation_options(
      /*triple=*/triple, /*cpu_name=*/"", /*features=*/"",
      /*entry_point_name=*/"main",
      cpu::CpuAotCompilationOptions::RelocationModel::BigPic);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_compilation_result,
      cpu_compiler.CompileAheadOfTime(std::move(module_group),
                                      aot_compilation_options));
  EXPECT_EQ(aot_compilation_result.size(), 1);
}
}  // namespace
}  // namespace xla
