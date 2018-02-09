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

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/ir_emitter_nested.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace gpu {

IrEmitterNested::IrEmitterNested(const HloModuleConfig& hlo_module_config,
                                 const HloComputation& nested_computation,
                                 IrEmitterContext* ir_emitter_context)
    : IrEmitter(hlo_module_config, ir_emitter_context, /*is_nested=*/true) {
  std::vector<const HloInstruction*> io_hlos;
  emitted_function_ =
      EmitBasePointersForNestedComputation(nested_computation, &io_hlos);
}

llvm::Function* IrEmitterNested::EmitBasePointersForNestedComputation(
    const HloComputation& nested_computation,
    std::vector<const HloInstruction*>* io_hlos) {
  std::vector<llvm::Type*> argument_types;
  std::vector<int64> argument_dereferenceable_bytes;
  for (const HloInstruction* param :
       nested_computation.parameter_instructions()) {
    io_hlos->push_back(param);
    const Shape& param_shape = param->shape();
    argument_types.push_back(
        llvm_ir::ShapeToIrType(param_shape, module_)->getPointerTo());
    int64 param_size =
        llvm_ir::ByteSizeOf(param_shape, module_->getDataLayout());
    argument_dereferenceable_bytes.push_back(param_size);
  }
  {
    const HloInstruction* root = nested_computation.root_instruction();
    io_hlos->push_back(root);
    const Shape& root_shape = root->shape();
    argument_types.push_back(
        llvm_ir::ShapeToIrType(root_shape, module_)->getPointerTo());
    int64 root_size = llvm_ir::ByteSizeOf(
        root_shape, ir_emitter_context_->llvm_module()->getDataLayout());
    argument_dereferenceable_bytes.push_back(root_size);
  }
  // The base pointer of the memory block for all pre-allocated temp buffers.
  argument_types.push_back(ir_builder_.getInt8PtrTy());

  llvm::FunctionType* function_type =
      llvm::FunctionType::get(ir_builder_.getVoidTy(), argument_types, false);
  llvm::Function* function = llvm::Function::Create(
      function_type,                       // The function type.
      llvm::GlobalValue::InternalLinkage,  // The linkage type.
      llvm_ir::AsStringRef(ir_emitter_context_->name_uniquer()->GetUniqueName(
          llvm_ir::SanitizeFunctionName(
              nested_computation.name()))),  // The name of the function.
      ir_emitter_context_->llvm_module());   // The parent LLVM module.
  for (size_t arg_no = 0; arg_no < argument_dereferenceable_bytes.size();
       ++arg_no) {
    int64 arg_size = argument_dereferenceable_bytes[arg_no];
    if (arg_size > 0) {
      function->addDereferenceableAttr(arg_no + 1, arg_size);
    }
  }

  // TODO(b/65380986): Investigate if adding fast math flags for generated
  // kernels makes sense.

  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(function->getContext(), "entry", function);
  // Emit a "return void" at entry_bb's end, and sets the insert point before
  // that return instruction.
  ir_builder_.SetInsertPoint(
      llvm::ReturnInst::Create(function->getContext(), entry_bb));

  std::vector<const HloInstruction*> non_io_hlos;
  for (const auto* hlo : nested_computation.instructions()) {
    if (hlo->opcode() != HloOpcode::kParameter &&
        hlo != nested_computation.root_instruction()) {
      non_io_hlos.push_back(hlo);
    }
  }
  bindings_.EmitBasePointersForHlos(*io_hlos, non_io_hlos);
  return function;
}

Status IrEmitterNested::HandleParameter(HloInstruction* parameter) {
  return Status::OK();
}

Status IrEmitterNested::EmitTargetElementLoop(
    const HloInstruction& hlo,
    const llvm_ir::ElementGenerator& element_generator) {
  return llvm_ir::LoopEmitter(element_generator, GetIrArray(hlo, hlo),
                              &ir_builder_)
      .EmitLoop();
}

}  // namespace gpu
}  // namespace xla
