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
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/lib/Target/SPIRV/MCTargetDesc/SPIRVBaseInfo.h"
#include "llvm/lib/Target/SPIRV/SPIRVAPI.h"
#include "llvm/lib/Target/SPIRV/SPIRVCommandLine.h"
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

}  // namespace

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

std::vector<std::string> RemoveUnsupportedExtensionsFromAll(
    llvm::Triple triple,
    const std::vector<std::string> unsupported_extensions) {
  std::set<std::string> extensions;
  auto valid_extensions =
      llvm::SPIRVExtensionsParser::getValidExtensions(triple);

  for (auto& ext : valid_extensions) {
    extensions.insert(llvm::getSymbolicOperandMnemonic(
        llvm::SPIRV::OperandCategory::ExtensionOperand, ext));
  }

  for (auto& ext : unsupported_extensions) {
    extensions.erase(ext);
  }
  return std::vector<std::string>(extensions.begin(), extensions.end());
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
  std::vector<llvm::Type*> new_arg_types;
  llvm::LLVMContext& context = module->getContext();
  llvm::SmallVector<llvm::Function*> kernel_funcs;
  for (auto& func : *module) {
    if (func.getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
      kernel_funcs.push_back(&func);
    }
  }
  for (auto old_func : kernel_funcs) {
    if (old_func->getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
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
  // List of extensions to block during module translation.
  // Block SPV_KHR_float_controls2 extension because the current L0 drivers lack
  // the needed translation support.
  const std::vector<std::string> unsupported_extensions = {
      "SPV_KHR_float_controls2"};
  if (spirv_extensions->empty()) {
    *spirv_extensions = RemoveUnsupportedExtensionsFromAll(
        default_target_triple, unsupported_extensions);
  }
  std::unique_ptr<llvm::TargetMachine> target_machine =
      GetTargetMachine(default_target_triple, "", debug_options, "");
  TF_RETURN_IF_ERROR(LinkAndOptimizeModule(
      module, gpu_version, debug_options, "", SPIRVTargetModuleLinker,
      default_target_triple, target_machine.get(), kDefaultInlineThreshold));

  // Unlike other GPU backends like NVTPTX and AMDGPU, SPIRV does not have
  // address inference pass in the TargetPassConfig. So we do it here
  // explicitly.
  llvm::legacy::PassManager pm;
  pm.add(llvm::createInferAddressSpacesPass(0));
  pm.run(*module);

  std::string spirv_str;
  std::string spirv_err_msg;
  std::vector<std::string> spirv_options{default_target_triple.str()};
  bool spirv_success = llvm::SPIRVTranslateModule(
      module, spirv_str, spirv_err_msg, *spirv_extensions, spirv_options);
  if (!spirv_success) {
    return absl::AbortedError("Failed to translate LLVM module to SPIRV.");
  }
  return spirv_str;
}

}  // namespace xla::gpu::spirv
