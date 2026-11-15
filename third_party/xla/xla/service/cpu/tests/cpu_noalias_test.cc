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

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/fusion_emitter.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

std::unique_ptr<Compiler> GetCpuCompiler() {
  absl::StatusOr<std::string> name = PlatformUtil::CanonicalPlatformName("cpu");
  CHECK_OK(name.status());
  absl::StatusOr<stream_executor::Platform::Id> platform_id =
      PlatformUtil::GetPlatformIdFromCanonicalName(*name);
  CHECK_OK(platform_id.status());
  absl::StatusOr<std::unique_ptr<Compiler>> compiler =
      Compiler::GetForPlatform(*platform_id);
  CHECK_OK(compiler.status());
  return std::move(*compiler);
}

class CpuNoAliasTest : public HloHardwareIndependentTestBase {};

TEST_F(CpuNoAliasTest, InlinedKernelsDoNotFoldLoadsAcrossWrites) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule m, input_output_alias={ {0}: (0, {}, must-alias) }

    body_fusion {
      state = s32[] parameter(0)
      ten = s32[] constant(10)
      ROOT next = s32[] add(state, ten)
    }

    cond_fusion {
      state = s32[] parameter(0)
      limit = s32[] constant(10)
      lt = pred[] compare(state, limit), direction=LT
      ROOT lt_i32 = s32[] convert(lt)
    }

    ENTRY main {
      init = s32[] parameter(0)
      next = s32[] fusion(init), kind=kLoop, calls=body_fusion
      cond = s32[] fusion(next), kind=kLoop, calls=cond_fusion
      ROOT out = (s32[], s32[]) tuple(next, cond)
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(kHlo));
  AliasInfo alias_info;
  std::unique_ptr<Compiler> compiler = GetCpuCompiler();
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BufferAssignment> buffer_assn,
      BufferAssigner::Run(
          hlo_module.get(),
          std::make_unique<DependencyHloOrdering>(hlo_module.get()),
          compiler->BufferSizeBytesFunction(), &alias_info,
          [](LogicalBuffer::Color) { return /*alignment=*/MinAlign(); },
          BufferAssigner::Options{}));

  const HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  const auto* body_instr = Cast<HloFusionInstruction>(root->operand(0));
  const auto* cond_instr = Cast<HloFusionInstruction>(root->operand(1));

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> cond_def,
      EmitFusionKernel(*mlir_context, *cond_instr, buffer_assn.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));
  ASSERT_OK_AND_ASSIGN(
      KernelDefinition<MlirKernelSource> body_def,
      EmitFusionKernel(*mlir_context, *body_instr, buffer_assn.get(),
                       /*use_unique_c_name=*/false,
                       /*enable_tiled_emitter=*/false));

  llvm::LLVMContext context;
  FusionCompiler fusion_compiler(
      mlir_context.get(),
      FusionCompiler::Options{/*vector_width=*/256, /*verification_level=*/1,
                              /*fast_min_max=*/false});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<llvm::Module> combined,
      fusion_compiler.Compile(context, cond_def.source().module()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<llvm::Module> body_module,
      fusion_compiler.Compile(context, body_def.source().module()));
  ASSERT_FALSE(llvm::Linker::linkModules(*combined, std::move(body_module)));

  for (llvm::Function& fn : *combined) {
    if (!fn.isDeclaration()) {
      fn.setLinkage(llvm::GlobalValue::InternalLinkage);
      fn.removeFnAttr(llvm::Attribute::NoInline);
      fn.addFnAttr(llvm::Attribute::AlwaysInline);
    }
  }

  llvm::Function* cond_fn = combined->getFunction(cond_def.spec().name());
  llvm::Function* body_fn = combined->getFunction(body_def.spec().name());
  ASSERT_NE(cond_fn, nullptr);
  ASSERT_NE(body_fn, nullptr);

  llvm::PointerType* ptr_ty = llvm::PointerType::getUnqual(context);
  llvm::IntegerType* i32_ty = llvm::Type::getInt32Ty(context);
  llvm::IntegerType* i64_ty = llvm::Type::getInt64Ty(context);
  llvm::StructType* dim3_ty =
      llvm::StructType::get(context, {i64_ty, i64_ty, i64_ty});
  llvm::StructType* arg_ty = llvm::StructType::get(context, {ptr_ty, i64_ty});
  llvm::ArrayType* args_ty = llvm::ArrayType::get(arg_ty, 2);
  llvm::StructType* frame_ty =
      llvm::StructType::get(context, {ptr_ty, ptr_ty, i64_ty, ptr_ty});

  auto init_frame = [&](llvm::IRBuilder<>& b) {
    llvm::AllocaInst* num_wgs = b.CreateAlloca(dim3_ty);
    b.CreateStore(llvm::ConstantStruct::get(
                      dim3_ty, {b.getInt64(1), b.getInt64(1), b.getInt64(1)}),
                  num_wgs);
    llvm::AllocaInst* wg_id = b.CreateAlloca(dim3_ty);
    b.CreateStore(llvm::ConstantAggregateZero::get(dim3_ty), wg_id);
    llvm::AllocaInst* args = b.CreateAlloca(args_ty);
    llvm::AllocaInst* frame = b.CreateAlloca(frame_ty);
    b.CreateStore(num_wgs, b.CreateInBoundsGEP(frame_ty, frame,
                                               {b.getInt32(0), b.getInt32(0)}));
    b.CreateStore(wg_id, b.CreateInBoundsGEP(frame_ty, frame,
                                             {b.getInt32(0), b.getInt32(1)}));
    b.CreateStore(
        b.getInt64(2),
        b.CreateInBoundsGEP(frame_ty, frame, {b.getInt32(0), b.getInt32(2)}));
    b.CreateStore(args, b.CreateInBoundsGEP(frame_ty, frame,
                                            {b.getInt32(0), b.getInt32(3)}));
    return std::make_pair(frame, args);
  };

  auto emit_call = [&](llvm::IRBuilder<>& b, llvm::Value* frame,
                       llvm::Value* args, llvm::Function* callee,
                       llvm::Value* arg0, llvm::Value* arg1) {
    b.CreateStore(arg0, b.CreateInBoundsGEP(
                            args_ty, args,
                            {b.getInt32(0), b.getInt32(0), b.getInt32(0)}));
    b.CreateStore(
        b.getInt64(4),
        b.CreateInBoundsGEP(args_ty, args,
                            {b.getInt32(0), b.getInt32(0), b.getInt32(1)}));
    b.CreateStore(arg1, b.CreateInBoundsGEP(
                            args_ty, args,
                            {b.getInt32(0), b.getInt32(1), b.getInt32(0)}));
    b.CreateStore(
        b.getInt64(4),
        b.CreateInBoundsGEP(args_ty, args,
                            {b.getInt32(0), b.getInt32(1), b.getInt32(1)}));
    b.CreateCall(callee, {frame});
  };

  // Call cond(state=5) -> 1, body(state) -> 15, cond(state=15) -> 0, and
  // cond(other=3) -> 1 while reusing one call frame. Folding either the buffer
  // load or the call frame pointer load across writes changes the result.
  {
    llvm::Function* eval_fn = llvm::Function::Create(
        llvm::FunctionType::get(i32_ty, /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, "eval_shared_buffer_and_frame",
        combined.get());
    llvm::IRBuilder<> b(llvm::BasicBlock::Create(context, "entry", eval_fn));
    auto [frame, args] = init_frame(b);
    llvm::AllocaInst* state = b.CreateAlloca(i32_ty);
    state->setAlignment(llvm::Align(MinAlign()));
    llvm::AllocaInst* other = b.CreateAlloca(i32_ty);
    other->setAlignment(llvm::Align(MinAlign()));
    llvm::AllocaInst* cond_out = b.CreateAlloca(i32_ty);
    cond_out->setAlignment(llvm::Align(MinAlign()));
    b.CreateStore(b.getInt32(5), state);
    b.CreateStore(b.getInt32(3), other);

    emit_call(b, frame, args, cond_fn, state, cond_out);
    llvm::Value* c0 = b.CreateLoad(i32_ty, cond_out);
    emit_call(b, frame, args, body_fn, state, state);
    emit_call(b, frame, args, cond_fn, state, cond_out);
    llvm::Value* c1 = b.CreateLoad(i32_ty, cond_out);
    emit_call(b, frame, args, cond_fn, other, cond_out);
    llvm::Value* c2 = b.CreateLoad(i32_ty, cond_out);

    llvm::Value* result =
        b.CreateAdd(c0, b.CreateAdd(b.CreateMul(c1, b.getInt32(2)),
                                    b.CreateMul(c2, b.getInt32(4))));
    b.CreateRet(result);
  }

  // Inline cond into a loop whose body modifies the shared buffer through a
  // call, verifying that the condition load is neither folded nor hoisted.
  {
    llvm::Function* modify_fn = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), {ptr_ty},
                                /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, "modify", combined.get());
    llvm::Function* loop_fn = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), {ptr_ty},
                                /*isVarArg=*/false),
        llvm::GlobalValue::ExternalLinkage, "loop_with_modifying_call",
        combined.get());
    loop_fn->addParamAttr(
        0, llvm::Attribute::getWithAlignment(context, llvm::Align(MinAlign())));
    loop_fn->addDereferenceableParamAttr(0, 4);
    llvm::Value* state_arg = loop_fn->getArg(0);
    state_arg->setName("state");

    llvm::BasicBlock* entry_bb =
        llvm::BasicBlock::Create(context, "entry", loop_fn);
    llvm::BasicBlock* loop_bb =
        llvm::BasicBlock::Create(context, "loop", loop_fn);
    llvm::BasicBlock* exit_bb =
        llvm::BasicBlock::Create(context, "exit", loop_fn);

    llvm::IRBuilder<> b(entry_bb);
    auto [frame, args] = init_frame(b);
    llvm::AllocaInst* cond_out = b.CreateAlloca(i32_ty);
    cond_out->setAlignment(llvm::Align(MinAlign()));
    emit_call(b, frame, args, cond_fn, state_arg, cond_out);
    llvm::Value* c0 = b.CreateLoad(i32_ty, cond_out);
    b.CreateCondBr(b.CreateICmpNE(c0, b.getInt32(0)), loop_bb, exit_bb);

    b.SetInsertPoint(loop_bb);
    b.CreateCall(modify_fn, {state_arg});
    emit_call(b, frame, args, cond_fn, state_arg, cond_out);
    llvm::Value* c1 = b.CreateLoad(i32_ty, cond_out);
    b.CreateCondBr(b.CreateICmpNE(c1, b.getInt32(0)), loop_bb, exit_bb);

    b.SetInsertPoint(exit_bb);
    b.CreateRetVoid();
  }

  ASSERT_FALSE(llvm::verifyModule(*combined, &llvm::errs()));

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;
  llvm::PassBuilder pb(context);
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);
  llvm::ModulePassManager mpm =
      pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
  mpm.run(*combined, mam);

  ASSERT_FALSE(llvm::verifyModule(*combined, &llvm::errs()));

  std::string optimized_ir = llvm_ir::DumpToString(combined.get());
  EXPECT_EQ(optimized_ir.find("!invariant.load"), std::string::npos);
  ASSERT_OK_AND_ASSIGN(bool filecheck_match, RunFileCheck(optimized_ir, R"(
    CHECK-LABEL: define {{.*}}i32 @eval_shared_buffer_and_frame(
    CHECK:         ret i32 5

    CHECK-LABEL: define void @loop_with_modifying_call(
    CHECK-SAME:    ptr {{.*}}[[STATE:%[a-zA-Z0-9_.]+]])
    CHECK:         load i32, ptr [[STATE]]
    CHECK:         call void @modify(ptr {{.*}}[[STATE]])
    CHECK:         load i32, ptr [[STATE]]
  )"));
  EXPECT_TRUE(filecheck_match) << optimized_ir;
}

}  // namespace
}  // namespace cpu
}  // namespace xla
