/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/llvm_ir/kernel_support_library.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/llvm_loop.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
absl::Status KernelSupportLibrary::ForWithStatus(
    absl::string_view name, llvm::Value* start, llvm::Value* end,
    llvm::Value* step,
    const std::function<absl::Status(llvm::Value*, bool)>& for_body_generator) {
  return IfWithStatus(b_->CreateICmpSLT(start, end), [&]() -> absl::Status {
    TF_RETURN_IF_ERROR(for_body_generator(start, /*is_first_iteration=*/true));
    return ForWithStatus(
        name, b_->CreateAdd(start, step), end, step,
        [&](llvm::Value* iv) { return for_body_generator(iv, false); });
  });
}

absl::Status KernelSupportLibrary::ForWithStatus(
    absl::string_view name, llvm::Value* start, llvm::Value* end,
    llvm::Value* step,
    const std::function<absl::Status(llvm::Value*)>& for_body_generator) {
  std::unique_ptr<llvm_ir::ForLoop> loop = llvm_ir::ForLoop::EmitForLoop(
      name, start, end, step, b_,
      /*unroll_mode=*/unroll_mode_,
      /*prevent_vectorization=*/prevent_vectorization_);
  b_->SetInsertPoint(&loop->GetBodyBasicBlock()->back());
  TF_RETURN_IF_ERROR(for_body_generator(loop->GetIndVarValue()));
  llvm_ir::SetToLastInsertPoint(loop->GetExitBasicBlock(), b_);
  return absl::OkStatus();
}

absl::Status KernelSupportLibrary::IfWithStatus(
    absl::string_view name, llvm::Value* condition,
    const std::function<absl::Status()>& true_block_generator,
    const std::function<absl::Status()>& false_block_generator) {
  llvm_ir::LlvmIfData if_data =
      llvm_ir::EmitIfThenElse(condition, name, b_,
                              /*emit_else=*/false_block_generator != nullptr);
  b_->SetInsertPoint(&if_data.true_block->back());
  TF_RETURN_IF_ERROR(true_block_generator());
  if (false_block_generator != nullptr) {
    b_->SetInsertPoint(&if_data.false_block->back());
    TF_RETURN_IF_ERROR(false_block_generator());
  }
  llvm_ir::SetToLastInsertPoint(if_data.after_block, b_);
  return absl::OkStatus();
}

void KernelSupportLibrary::EmitAndCallOutlinedKernel(
    const HloModuleConfig& module_config, llvm::IRBuilderBase* b,
    absl::string_view kernel_name,
    KernelSupportLibrary::ArgumentVector arguments,
    const std::function<void(KernelSupportLibrary::ArgumentVector)>&
        kernel_body_generator) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  llvm::Function* function =
      module->getFunction(llvm_ir::AsStringRef(kernel_name));

  int64_t null_arg_idx = -1;
  std::vector<llvm::Value*> sanitized_args;
  sanitized_args.reserve(arguments.size());
  for (int64_t i = 0, e = arguments.size(); i < e; i++) {
    if (arguments[i]) {
      sanitized_args.push_back(arguments[i]);
    } else {
      CHECK_EQ(null_arg_idx, -1);
      null_arg_idx = i;
    }
  }

  if (!function) {
    VLOG(2) << "Generating kernel for " << kernel_name;
    std::vector<llvm::Type*> arg_types;
    std::transform(sanitized_args.begin(), sanitized_args.end(),
                   std::back_inserter(arg_types),
                   [](llvm::Value* arg) { return arg->getType(); });

    auto* function_type =
        llvm::FunctionType::get(b->getVoidTy(), arg_types, /*isVarArg=*/false);

    function = llvm_ir::CreateCpuFunction(function_type,
                                          llvm::GlobalValue::InternalLinkage,
                                          module_config, kernel_name, module);

    llvm::IRBuilderBase::InsertPointGuard guard(*b);

    auto* entry_bb =
        llvm::BasicBlock::Create(b->getContext(), "entry", function);
    auto* return_inst = llvm::ReturnInst::Create(b->getContext(),
                                                 /*retVal=*/nullptr, entry_bb);
    // Set the insert point to before return_inst.
    b->SetInsertPoint(return_inst);

    std::vector<llvm::Value*> arg_values;
    /*
     * clang on OSX doesn't like std::transform or range for loop here.
     * See https://github.com/tensorflow/tensorflow/issues/15196
     */
    for (llvm::Function::arg_iterator arg = function->arg_begin(),
                                      arg_e = function->arg_end();
         arg != arg_e; ++arg) {
      arg_values.push_back(arg);
    }
    if (null_arg_idx != -1) {
      arg_values.insert(arg_values.begin() + null_arg_idx, nullptr);
    }
    kernel_body_generator(arg_values);
  } else {
    VLOG(3) << "Re-using kernel for " << kernel_name;
  }

  b->CreateCall(function, llvm_ir::AsArrayRef(sanitized_args));
}

}  // namespace xla
