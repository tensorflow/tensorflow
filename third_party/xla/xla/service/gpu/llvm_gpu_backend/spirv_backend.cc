/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/llvm_gpu_backend/spirv_backend.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/base/no_destructor.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/lib/Target/SPIRV/MCTargetDesc/SPIRVBaseInfo.h"
#include "llvm/lib/Target/SPIRV/SPIRVAPI.h"
#include "llvm/lib/Target/SPIRV/SPIRVSubtarget.h"
#include "llvm/lib/Target/SPIRV/SPIRVTargetMachine.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "tsl/platform/errors.h"

namespace xla::gpu::spirv {

namespace {

// Default inline threshold value to use in llvm.
const int kDefaultInlineThreshold = 1100;

void SPIRVBackendInit() {
  LLVMInitializeSPIRVTargetInfo();
  LLVMInitializeSPIRVTarget();
  LLVMInitializeSPIRVTargetMC();
  LLVMInitializeSPIRVAsmPrinter();

  // Initialize the LLVM optimization passes.
  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  InitializePasses(registry);
}

absl::Status SPIRVTargetModuleLinker(
    llvm::Module* module, stream_executor::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options, const std::string& device_bitcode_path) {
  return absl::OkStatus();
}

std::string EmitModuleToSPIRV(llvm::Module* module,
                              llvm::TargetMachine* target_machine) {
  std::string spirv_binary;
  llvm::raw_string_ostream string_stream(spirv_binary);
  llvm::buffer_ostream buffered_stream(string_stream);
  llvm::legacy::PassManager pm;
  // Unlike other GPU backends like NVTPTX and AMDGPU, SPIRV does not have
  // address inference pass in the TargetPassConfig. So we do it here
  // explicitly.
  pm.add(llvm::createInferAddressSpacesPass(0));
  pm.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(module->getTargetTriple())));
  std::unique_ptr<llvm::MachineModuleInfoWrapperPass> mmiwp(
      new llvm::MachineModuleInfoWrapperPass(target_machine));
  target_machine->getObjFileLowering()->Initialize(mmiwp->getMMI().getContext(),
                                                   *target_machine);
  target_machine->addPassesToEmitFile(pm, buffered_stream, nullptr,
                                      llvm::CodeGenFileType::ObjectFile);

  pm.run(*module);
  return spirv_binary;
}

}  // namespace

std::vector<std::string> SPIRVExtensionsEnumToString(
    const llvm::ExtensionSet& enum_extensions) {
  std::vector<std::string> str_extensions;
  for (auto& ext : enum_extensions) {
    str_extensions.push_back(llvm::getSymbolicOperandMnemonic(
        llvm::SPIRV::OperandCategory::ExtensionOperand, ext));
  }
  return str_extensions;
}

std::vector<std::string> GetSPIRVBackendOptions(
    const DebugOptions& debug_options) {
  // Feed all customized flags here, so we can override them with llvm_cl_opts
  // without redeploy the compiler for development purpose.
  std::vector<std::string> backend_llvm_opts;

  auto backend_extra_llvm_opts = llvm_ir::ExtractXlaBackendExtraOptions(
      debug_options.xla_backend_extra_options());
  backend_llvm_opts.insert(backend_llvm_opts.end(),
                           backend_extra_llvm_opts.cbegin(),
                           backend_extra_llvm_opts.cend());

  return backend_llvm_opts;
}

absl::StatusOr<std::string> CompileToSPIRV(
    llvm::Module* module, stream_executor::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options) {
  static absl::once_flag backend_init_flag;
  static absl::NoDestructor<std::vector<std::string>> spirv_extensions;
  absl::call_once(backend_init_flag, SPIRVBackendInit);
  auto llvm_opts = GetSPIRVBackendOptions(debug_options);
  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_opts);

  // SPIRV Kernel functions expect their arguments' address spaces to be global,
  // i.e., addrspace(1). Here we only change kernel argument's address space if
  // it is addrspace(0). Then an address space cast to original is applied so
  // that users still have old address space.
  llvm::LLVMContext& context = module->getContext();
  llvm::SmallVector<llvm::Function*> kernel_funcs;
  for (auto& func : *module) {
    if (func.getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
      kernel_funcs.push_back(&func);
    }
  }
  for (auto old_func : kernel_funcs) {
    if (old_func->getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
      std::vector<llvm::Type*> new_arg_types;
      for (auto& old_arg : old_func->args()) {
        llvm::Type* old_arg_type = old_arg.getType();
        auto ptr_type = llvm::dyn_cast<llvm::PointerType>(old_arg_type);
        if (ptr_type->getAddressSpace() == 0) {
          auto new_arg_type = llvm::PointerType::get(context, 1);
          new_arg_types.push_back(new_arg_type);
        } else {
          new_arg_types.push_back(old_arg_type);
        }
      }
      auto new_func_type = llvm::FunctionType::get(
          old_func->getReturnType(), new_arg_types, old_func->isVarArg());

      auto new_func = llvm::Function::Create(
          new_func_type, old_func->getLinkage(), old_func->getName(), module);

      // We do not want to modify type of arguments of old func at the uses,
      // hence using identity map.
      llvm::ValueToValueMapTy identity_map;
      for (auto& old_arg : old_func->args()) {
        identity_map[&old_arg] = &old_arg;
      }
      llvm::SmallVector<llvm::ReturnInst*> returns;
      llvm::CloneFunctionInto(new_func, old_func, identity_map,
                              llvm::CloneFunctionChangeType::LocalChangesOnly,
                              returns);

      llvm::IRBuilder<> builder(&(*new_func->begin()->begin()));
      auto new_arg_it = new_func->arg_begin();
      auto old_arg_it = old_func->arg_begin();
      for (; old_arg_it != old_func->arg_end(); ++old_arg_it, ++new_arg_it) {
        if (auto old_ptr_type =
                llvm::dyn_cast<llvm::PointerType>(old_arg_it->getType())) {
          auto cast = builder.CreateAddrSpaceCast(new_arg_it, old_ptr_type);
          old_arg_it->replaceAllUsesWith(cast);
        }
      }
      // TODO: Update kernel function's uses. Currently, we are assuming that
      // kernel function is not called by any other functions in the current
      // LLVM module.
      new_func->takeName(old_func);
      old_func->eraseFromParent();
    }
  }

  llvm::Triple default_target_triple("spirv64-unknown-unknown");
  std::unique_ptr<llvm::TargetMachine> target_machine =
      GetTargetMachine(default_target_triple, "", debug_options, "");
  // Set datalayout and spirv extenstions.
  module->setDataLayout(target_machine->createDataLayout());
  llvm::SPIRVTargetMachine* sub_target =
      static_cast<llvm::SPIRVTargetMachine*>(target_machine.get());
  const_cast<llvm::SPIRVSubtarget*>(sub_target->getSubtargetImpl())
      ->initAvailableExtensions(common_spirv_extensions);

  TF_RETURN_IF_ERROR(LinkAndOptimizeModule(
      module, gpu_version, debug_options, "", SPIRVTargetModuleLinker,
      default_target_triple, target_machine.get(), kDefaultInlineThreshold));

  // The LLVM SPIR-V backend removes unused globals during its passes for
  // translation to SPIR-V. To prevent this, we create a fake use of those
  // globals with a minimal fake use function. We first create a global pointer
  // list with appending linkage containing pointers to all globals in the
  // module. And then the fake use function uses getelementptr and load
  // instruction to load the first element of the global pointer list.
  llvm::Type* ptr_type = llvm::PointerType::getUnqual(context);
  llvm::SmallVector<llvm::Constant*> global_ptrs;
  for (llvm::GlobalVariable& global_var : module->globals()) {
    global_ptrs.push_back(
        llvm::ConstantExpr::getPointerCast(&global_var, ptr_type));
  }
  if (!global_ptrs.empty()) {
    auto* arr_type = llvm::ArrayType::get(ptr_type, global_ptrs.size());
    auto* arr_init = llvm::ConstantArray::get(arr_type, global_ptrs);
    auto* global_ptr_arr = new llvm::GlobalVariable(
        arr_type, /*isConstant=*/false, llvm::GlobalValue::AppendingLinkage,
        /*Initializer=*/arr_init, "global_ptr_list",
        /*ThreadLocalMode=*/llvm::GlobalValue::NotThreadLocal,
        /*AddressSpace=*/1, /*isExternallyInitialized=*/false);
    global_ptr_arr->setAlignment(llvm::Align(kConstantBufferAlignBytes));
    module->insertGlobalVariable(global_ptr_arr);

    // Create a fake use function that loads the first element of
    // global_ptr_list to prevent it from being optimized away.
    auto* fake_use_func_type =
        llvm::FunctionType::get(ptr_type, /*isVarArg=*/false);
    auto* fake_use_func = llvm::Function::Create(
        fake_use_func_type, llvm::GlobalValue::ExternalLinkage,
        "fake_use_globals", module);
    fake_use_func->setCallingConv(llvm::CallingConv::SPIR_FUNC);
    auto* bb = llvm::BasicBlock::Create(context, "entry", fake_use_func);
    llvm::IRBuilder<> ir_builder(bb);
    auto* gep = ir_builder.CreateConstGEP2_64(arr_type, global_ptr_arr,
                                              /*Idx0=*/0, /*Idx1=*/0);
    auto* load = ir_builder.CreateLoad(ptr_type, gep);
    ir_builder.CreateRet(load);
  }

  return EmitModuleToSPIRV(module, target_machine.get());
}

}  // namespace xla::gpu::spirv
