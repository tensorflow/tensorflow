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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/cpu_options.h"
#include "xla/service/cpu/ir_function.h"
#include "xla/service/cpu/runtime_symbol_generator.h"
#include "xla/service/cpu/target_machine_features_stub.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla::cpu {
namespace {

using IrEmitterTest = HloTestBase;

static std::pair<llvm::Function*, llvm::BasicBlock*> CreateFunction(
    llvm::LLVMContext& context, llvm::Module* module, llvm::IRBuilderBase* b) {
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

  TargetMachineFeaturesStub target_machine([](int64_t size) { return 1; });

  IrEmitter ir_emitter(nullptr, *hlo, *buffer_assignment, module.get(), {}, {},
                       {}, &target_machine, false);

  llvm::IRBuilderBase* b = ir_emitter.b();
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

// Used to keep all dependencies of IrEmitter alive.
struct IrEmitterWrapper {
  std::unique_ptr<IrEmitter> ir_emitter;
  std::unique_ptr<BufferAssignment> buffer_assignment;
  std::unique_ptr<TargetMachineFeatures> target_machine_features;
  std::unique_ptr<mlir::MLIRContext> mlir_context;
};

static absl::StatusOr<std::unique_ptr<IrEmitterWrapper>>
CreateIrEmitterForConstantEmissionTests(HloModule& module,
                                        llvm::Module& llvm_module) {
  const DebugOptions& debug_options = module.config().debug_options();

  const HloModuleConfig& config = module.config();

  // Options for compiling LLVM IR to machine code.
  IrCompiler::Options ir_compiler_options{
      /*optimization_level=*/llvm::CodeGenOptLevel::Default,
      /*optimize_for_size=*/options::OptimizeForSizeRequested(config),
      /*fast_math_flags=*/llvm_ir::GetCpuFastMathFlags(config),
      /*disable_expensive_passes=*/
      debug_options.xla_llvm_disable_expensive_passes(),
      /*slp_vectorizer_disabled=*/options::SlpVectorizerDisabled(config),
  };

  // Definition generator to link with XLA:CPU host runtime symbols.
  JitCompiler::DefinitionGenerator definition_generator =
      [](llvm::TargetMachine* target_machine) {
        return std::make_unique<RuntimeSymbolGenerator>(
            target_machine->createDataLayout());
      };

  // Options for orchestrating the JIT compilation process.
  JitCompiler::Options jit_compiler_options{
      std::move(ir_compiler_options),
      {},
      /*num_dylibs=*/1,
      /*definition_generator=*/std::move(definition_generator),
      /*max_cpu_isa=*/CpuFeatureFromString(debug_options.xla_cpu_max_isa()),
  };

  llvm::TargetOptions target_options;
  target_options.AllowFPOpFusion = llvm::FPOpFusion::Fast;

  // Returns a global (per-process) thread pool for XLA CPU compilation tasks.
  auto compilation_task_runner = [](cpu::JitCompiler::Task task) {
    static auto* thread_pool =
        new tsl::thread::ThreadPool(tsl::Env::Default(), "ir-emitter-test", 1);

    thread_pool->Schedule(std::move(task));
  };

  TF_ASSIGN_OR_RETURN(
      JitCompiler jit_compiler,
      JitCompiler::Create(target_options, std::move(jit_compiler_options),
                          compilation_task_runner));

  auto scheduler =
      debug_options.xla_cpu_enable_concurrency_optimized_scheduler()
          ? BFSMemoryScheduler
          : DFSMemoryScheduler;

  auto buffer_size_bytes_function = [](const BufferValue& buffer) {
    return CpuExecutable::ShapeSizeBytes(buffer.shape());
  };
  TF_ASSIGN_OR_RETURN(
      HloSchedule schedule,
      ScheduleModule(&module, buffer_size_bytes_function,
                     ComputationSchedulerToModuleScheduler(scheduler)));
  TF_RETURN_IF_ERROR(module.set_schedule(schedule));

  auto memory_alignment = [](LogicalBuffer::Color) {
    return cpu_function_runtime::MinAlign();
  };
  // Run buffer allocation on the HLO graph.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(&module,
                          std::make_unique<SequentialHloOrdering>(schedule),
                          buffer_size_bytes_function, memory_alignment,
                          /*allocate_buffers_for_constants=*/true));

  auto target_machine_features =
      std::make_unique<TargetMachineFeatures>(jit_compiler.target_machine());

  std::unique_ptr<mlir::MLIRContext> mlir_context;
  auto ir_emitter = std::make_unique<IrEmitter>(
      mlir_context.get(), module, *assignment, &llvm_module,
      absl::flat_hash_map<const HloInstruction*, int64_t>{},
      absl::flat_hash_map<const HloComputation*, int64_t>{},
      absl::flat_hash_map<const HloComputation*, bool>{},
      target_machine_features.get(),
      /*emit_code_for_msan=*/false);

  return std::make_unique<IrEmitterWrapper>(IrEmitterWrapper{
      std::move(ir_emitter), std::move(assignment),
      std::move(target_machine_features), std::move(mlir_context)});
}

TEST_F(IrEmitterTest, SmallConstantsAreEmittedAsGlobalsLargeAreNot) {
  constexpr size_t kNumberOfSmallConstants = 1;
  absl::string_view module_string = R"(
HloModule module

ENTRY main {
  a = f32[1000,1000]{1,0} parameter(0)
  b = f32[1000,1000]{1,0} constant({...})
  a_plus_b = f32[1000,1000]{1,0} add(a, b)
  c = f32[1,1]{1,0} constant({...})
  broadcast = f32[1000,1000]{1,0} broadcast(c), dimensions={}
  ROOT result = f32[1000,1000]{1,0} add(a_plus_b, broadcast)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(module_string));

  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  auto llvm_module = std::make_unique<llvm::Module>("test", *llvm_context);

  TF_ASSERT_OK_AND_ASSIGN(
      auto wrapped_ir_emitter,
      CreateIrEmitterForConstantEmissionTests(*module, *llvm_module));

  TF_ASSERT_OK(wrapped_ir_emitter->ir_emitter->EmitSmallConstantGlobals());

  EXPECT_EQ(
      std::distance(llvm_module->global_begin(), llvm_module->global_end()),
      kNumberOfSmallConstants);
}

}  // namespace
}  // namespace xla::cpu
