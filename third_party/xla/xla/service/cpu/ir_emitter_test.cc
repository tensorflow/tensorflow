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
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/ir_function.h"
#include "xla/service/cpu/target_machine_features_fake.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_ordering.h"
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

TEST_F(IrEmitterTest, CheckNativeConvertSupportOnTargetCPU) {
  std::string spr_feature_string =
      "+prfchw,+cldemote,+avx,+aes,+sahf,+pclmul,-xop,+crc32,+xsaves,+"
      "avx512fp16,-usermsr,-sm4,-egpr,+sse4.1,+avx512ifma,+xsave,+sse4.2,+"
      "tsxldtrk,-sm3,+ptwrite,-widekl,+invpcid,+64bit,+xsavec,-avx10.1-512,+"
      "avx512vpopcntdq,+cmov,-avx512vp2intersect,+avx512cd,+movbe,-avxvnniint8,"
      "-ccmp,+amx-int8,-kl,-avx10.1-256,+evex512,+avxvnni,-rtm,+adx,+avx2,-"
      "hreset,+movdiri,+serialize,-sha512,+vpclmulqdq,+avx512vl,+uintr,-cf,+"
      "clflushopt,-raoint,-cmpccxadd,+bmi,+amx-tile,+sse,-avx10.2-256,+gfni,-"
      "avxvnniint16,-amx-fp16,-zu,-ndd,+xsaveopt,+rdrnd,+avx512f,+amx-bf16,+"
      "avx512bf16,+avx512vnni,-push2pop2,+cx8,+avx512bw,+sse3,+pku,-nf,+"
      "fsgsbase,-clzero,-mwaitx,-lwp,+lzcnt,+sha,+movdir64b,-ppx,+wbnoinvd,+"
      "enqcmd,-avx10.2-512,-avxneconvert,-tbm,-pconfig,-amx-complex,+ssse3,+"
      "cx16,+bmi2,+fma,+popcnt,-avxifma,+f16c,+avx512bitalg,-rdpru,+clwb,+mmx,+"
      "sse2,+rdseed,+avx512vbmi2,-prefetchi,+rdpid,-fma4,+avx512vbmi,+shstk,+"
      "vaes,+waitpkg,-sgx,+fxsr,+avx512dq,-sse4a";

  std::string skx_feature_string =
      "+prfchw,-cldemote,+avx,+aes,+sahf,+pclmul,-xop,+crc32,+xsaves,-"
      "avx512fp16,-usermsr,-sm4,-egpr,+sse4.1,-avx512ifma,+xsave,+sse4.2,-"
      "tsxldtrk,-sm3,-ptwrite,-widekl,+invpcid,+64bit,+xsavec,-avx10.1-512,-"
      "avx512vpopcntdq,+cmov,-avx512vp2intersect,+avx512cd,+movbe,-avxvnniint8,"
      "-ccmp,-amx-int8,-kl,-avx10.1-256,+evex512,-avxvnni,+rtm,+adx,+avx2,-"
      "hreset,-movdiri,-serialize,-sha512,-vpclmulqdq,+avx512vl,-uintr,-cf,+"
      "clflushopt,-raoint,-cmpccxadd,+bmi,-amx-tile,+sse,-avx10.2-256,-gfni,-"
      "avxvnniint16,-amx-fp16,-zu,-ndd,+xsaveopt,+rdrnd,+avx512f,-amx-bf16,-"
      "avx512bf16,-avx512vnni,-push2pop2,+cx8,+avx512bw,+sse3,+pku,-nf,+"
      "fsgsbase,-clzero,-mwaitx,-lwp,+lzcnt,-sha,-movdir64b,-ppx,-wbnoinvd,-"
      "enqcmd,-avx10.2-512,-avxneconvert,-tbm,-pconfig,-amx-complex,+ssse3,+"
      "cx16,+bmi2,+fma,+popcnt,-avxifma,+f16c,-avx512bitalg,-rdpru,+clwb,+mmx,+"
      "sse2,+rdseed,-avx512vbmi2,-prefetchi,-rdpid,-fma4,-avx512vbmi,-shstk,-"
      "vaes,-waitpkg,-sgx,+fxsr,+avx512dq,-sse4a";

  std::string srf_feature_string =
      "+prfchw,+cldemote,+avx,+aes,+sahf,+pclmul,-xop,+crc32,+xsaves,-"
      "avx512fp16,-usermsr,-sm4,-egpr,+sse4.1,-avx512ifma,+xsave,+sse4.2,-"
      "tsxldtrk,-sm3,+ptwrite,-widekl,+invpcid,+64bit,+xsavec,-avx10.1-512,-"
      "avx512vpopcntdq,+cmov,-avx512vp2intersect,-avx512cd,+movbe,+avxvnniint8,"
      "-ccmp,-amx-int8,-kl,-avx10.1-256,-sha512,+avxvnni,-rtm,+adx,+avx2,-"
      "hreset,+movdiri,+serialize,+vpclmulqdq,-avx512vl,+uintr,-cf,+clflushopt,"
      "-raoint,+cmpccxadd,+bmi,-amx-tile,+sse,-avx10.2-256,+gfni,-avxvnniint16,"
      "-amx-fp16,-zu,-ndd,+xsaveopt,+rdrnd,-avx512f,-amx-bf16,-avx512bf16,-"
      "avx512vnni,-push2pop2,+cx8,-avx512bw,+sse3,+pku,-nf,+fsgsbase,-clzero,-"
      "mwaitx,-lwp,+lzcnt,+sha,+movdir64b,-ppx,+wbnoinvd,+enqcmd,-avx10.2-512,+"
      "avxneconvert,-tbm,+pconfig,-amx-complex,+ssse3,+cx16,+bmi2,+fma,+popcnt,"
      "+avxifma,+f16c,-avx512bitalg,-rdpru,+clwb,+mmx,+sse2,+rdseed,-"
      "avx512vbmi2,-prefetchi,+rdpid,-fma4,-avx512vbmi,+shstk,+vaes,+waitpkg,+"
      "sgx,+fxsr,-avx512dq,-sse4a";

  // Testing sapphire-rapids target
  ASSERT_TRUE(IsNativeConvertSupportedOnTargetCPU(spr_feature_string));

  // Testing skylake target
  ASSERT_FALSE(IsNativeConvertSupportedOnTargetCPU(skx_feature_string));

  // Testing sierra-forest target
  ASSERT_TRUE(IsNativeConvertSupportedOnTargetCPU(srf_feature_string));
}

}  // namespace
}  // namespace xla::cpu
