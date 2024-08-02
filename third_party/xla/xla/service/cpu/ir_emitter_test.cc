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

#include "xla/service/cpu/ir_emitter.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Casting.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/ir_function.h"
#include "xla/service/cpu/target_machine_features_fake.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_ordering.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/logical_buffer.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

using IrEmitterTest = HloTestBase;

static std::pair<llvm::Function*, llvm::BasicBlock*> CreateFunction(
    llvm::LLVMContext& context, llvm::Module* module, llvm::IRBuilder<>* b) {
  llvm::PointerType* ptrtype = llvm::PointerType::getUnqual(context);
  llvm::FunctionType* ftype = llvm::FunctionType::get(ptrtype, ptrtype, false);

  llvm::Function* function = llvm::dyn_cast<llvm::Function>(
      module->getOrInsertFunction("func2", ftype).getCallee());

  llvm::BasicBlock* return_block =
      llvm::BasicBlock::Create(context, "", function);
  b->SetInsertPoint(return_block);
  [[maybe_unused]] llvm::ReturnInst* ret = b->CreateRet(
      llvm::ConstantPointerNull::get(llvm::PointerType::getUnqual(context)));

  return std::make_pair(function, return_block);
}

TEST_F(IrEmitterTest, ComputeFuncStack) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  const char* hlo_text = R"(
    HloModule m
    ENTRY main {
      ROOT %zero = f32[] constant(0)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));
  const HloInstruction* zero = FindInstruction(hlo.get(), "zero");
  ASSERT_NE(zero, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssigner::Run(
          hlo.get(), std::make_unique<DependencyHloOrdering>(hlo.get()),
          backend().compiler()->BufferSizeBytesFunction(),
          [](LogicalBuffer::Color) { return /*alignment=*/1; }));

  TargetMachineFeaturesWithFakeAlignmentLogic target_machine(
      [](int64_t size) { return 1; });

  IrEmitter ir_emitter(nullptr, *hlo, *buffer_assignment, module.get(), {}, {},
                       {}, &target_machine, false);

  llvm::IRBuilder<>* b = ir_emitter.b();
  ASSERT_NE(b, nullptr);

  const std::pair<llvm::Function*, llvm::BasicBlock*> fb =
      CreateFunction(context, module.get(), b);

  llvm::Function* function = fb.first;
  llvm::BasicBlock* return_block = fb.second;

  ASSERT_NE(function, nullptr);
  ASSERT_NE(return_block, nullptr);

  const auto funcname = "func1";
  const auto linkagetype = llvm::GlobalValue::LinkageTypes::ExternalLinkage;
  const HloModuleConfig module_config;
  ir_emitter.PushComputeFunction(funcname, linkagetype, module_config,
                                 module.get(), 0);
  ASSERT_EQ(ir_emitter.compute_function()->function()->getName().str(),
            funcname);

  ir_emitter.PushComputeFunction(b, module.get(), 0, function, nullptr,
                                 return_block);
  ASSERT_EQ(ir_emitter.compute_function()->function(), function);

  ir_emitter.PopComputeFunction();
  ASSERT_EQ(ir_emitter.compute_function()->function()->getName().str(),
            funcname);

  ir_emitter.PopComputeFunction();
}

}  // namespace
}  // namespace xla::cpu
