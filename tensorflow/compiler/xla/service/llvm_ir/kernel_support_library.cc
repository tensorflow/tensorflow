/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h"

#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
Status KernelSupportLibrary::For(
    tensorflow::StringPiece name, llvm::Value* start, llvm::Value* end,
    llvm::Value* step,
    const std::function<Status(llvm::Value*, bool)>& for_body_generator) {
  return If(ir_builder_->CreateICmpSLT(start, end), [&]() -> Status {
    TF_RETURN_IF_ERROR(for_body_generator(start, /*is_first_iteration=*/true));
    return For(name, ir_builder_->CreateAdd(start, step), end, step,
               [&](llvm::Value* iv) { return for_body_generator(iv, false); });
  });
}

Status KernelSupportLibrary::For(
    tensorflow::StringPiece name, llvm::Value* start, llvm::Value* end,
    llvm::Value* step, bool peel_first_iteration,
    const std::function<Status(llvm::Value*, llvm::Value*)>&
        for_body_generator) {
  if (peel_first_iteration) {
    return For(name, start, end, step, true,
               [&](llvm::Value* indvar, bool is_first_iteration) -> Status {
                 return for_body_generator(
                     indvar, ir_builder_->getInt1(is_first_iteration));
               });
  } else {
    std::unique_ptr<llvm_ir::ForLoop> loop = llvm_ir::ForLoop::EmitForLoop(
        name, start, end, step, ir_builder_,
        /*unroll_mode=*/unroll_mode_,
        /*prevent_vectorization=*/prevent_vectorization_);
    ir_builder_->SetInsertPoint(&loop->GetBodyBasicBlock()->back());
    TF_RETURN_IF_ERROR(
        for_body_generator(loop->GetIndVarValue(),
                           /*is_first_iteration=*/ir_builder_->CreateICmpEQ(
                               loop->GetIndVarValue(), start)));
    llvm_ir::SetToLastInsertPoint(loop->GetExitBasicBlock(), ir_builder_);
    return Status::OK();
  }
}

Status KernelSupportLibrary::If(
    tensorflow::StringPiece name, llvm::Value* condition,
    const std::function<Status()>& true_block_generator,
    const std::function<Status()>& false_block_generator) {
  llvm_ir::LlvmIfData if_data =
      llvm_ir::EmitIfThenElse(condition, name, ir_builder_);
  ir_builder_->SetInsertPoint(&if_data.true_block->back());
  TF_RETURN_IF_ERROR(true_block_generator());
  ir_builder_->SetInsertPoint(&if_data.false_block->back());
  TF_RETURN_IF_ERROR(false_block_generator());
  llvm_ir::SetToLastInsertPoint(if_data.after_block, ir_builder_);
  return Status::OK();
}

void KernelSupportLibrary::EmitAndCallOutlinedKernel(
    bool enable_fast_math, bool optimize_for_size,
    llvm::IRBuilder<>* ir_builder, tensorflow::StringPiece kernel_name,
    KernelSupportLibrary::ArgumentVector arguments,
    const std::function<void(KernelSupportLibrary::ArgumentVector)>&
        kernel_body_generator) {
  llvm::Module* module = ir_builder->GetInsertBlock()->getModule();
  llvm::Function* function =
      module->getFunction(llvm_ir::AsStringRef(kernel_name));

  int64 null_arg_idx = -1;
  std::vector<llvm::Value*> sanitized_args;
  sanitized_args.reserve(arguments.size());
  for (int64 i = 0, e = arguments.size(); i < e; i++) {
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

    auto* function_type = llvm::FunctionType::get(
        ir_builder->getVoidTy(), arg_types, /*isVarArg=*/false);

    function = llvm_ir::CreateFunction(
        function_type, llvm::GlobalValue::InternalLinkage,
        /*enable_fast_math=*/enable_fast_math,
        /*optimize_for_size=*/optimize_for_size, kernel_name, module);

    llvm::IRBuilder<>::InsertPointGuard guard(*ir_builder);

    auto* entry_bb =
        llvm::BasicBlock::Create(ir_builder->getContext(), "entry", function);
    auto* return_inst = llvm::ReturnInst::Create(ir_builder->getContext(),
                                                 /*retVal=*/nullptr, entry_bb);
    // Set the insert point to before return_inst.
    ir_builder->SetInsertPoint(return_inst);

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

  ir_builder->CreateCall(function, llvm_ir::AsArrayRef(sanitized_args));
}

}  // namespace xla
