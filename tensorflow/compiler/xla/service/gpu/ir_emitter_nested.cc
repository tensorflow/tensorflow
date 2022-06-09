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

#include "tensorflow/compiler/xla/service/gpu/ir_emitter_nested.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace gpu {

IrEmitterNested::IrEmitterNested(const HloModuleConfig& hlo_module_config,
                                 const HloComputation& nested_computation,
                                 IrEmitterContext* ir_emitter_context)
    : IrEmitter(hlo_module_config, ir_emitter_context, /*is_nested=*/true),
      nested_computation_(nested_computation) {}

StatusOr<std::unique_ptr<IrEmitterNested>> IrEmitterNested::Create(
    const HloModuleConfig& hlo_module_config,
    const HloComputation& nested_computation,
    IrEmitterContext* ir_emitter_context) {
  std::unique_ptr<IrEmitterNested> emitter(new IrEmitterNested(
      hlo_module_config, nested_computation, ir_emitter_context));
  TF_RETURN_IF_ERROR(emitter->EmitConstants(nested_computation));
  return emitter;
}

// Nested function serves the same purpose on GPU as a thread-local function on
// a CPU.
Status IrEmitterNested::CodegenNestedComputation() {
  std::vector<const HloInstruction*> io_hlos;
  std::vector<llvm::Type*> argument_types;
  std::vector<int64_t> argument_dereferenceable_bytes;
  const auto& params = nested_computation_.parameter_instructions();
  const auto n = params.size() + 1;
  io_hlos.reserve(n - 1);
  argument_types.reserve(n);
  argument_dereferenceable_bytes.reserve(n);
  for (const HloInstruction* param : params) {
    io_hlos.push_back(param);
    const Shape& param_shape = param->shape();
    argument_types.push_back(
        llvm_ir::ShapeToIrType(param_shape, module_)->getPointerTo());
    int64_t param_size =
        llvm_ir::ByteSizeOf(param_shape, module_->getDataLayout());
    argument_dereferenceable_bytes.push_back(param_size);
  }

  const HloInstruction* root = nested_computation_.root_instruction();
  {
    const Shape& root_shape = root->shape();
    argument_types.push_back(
        llvm_ir::ShapeToIrType(root_shape, module_)->getPointerTo());
    int64_t root_size = llvm_ir::ByteSizeOf(
        root_shape, ir_emitter_context_->llvm_module()->getDataLayout());
    argument_dereferenceable_bytes.push_back(root_size);
  }

  llvm::FunctionType* function_type =
      llvm::FunctionType::get(b_.getVoidTy(), argument_types, false);
  llvm::Function* function = llvm::Function::Create(
      function_type,                       // The function type.
      llvm::GlobalValue::InternalLinkage,  // The linkage type.
      ir_emitter_context_->name_uniquer()->GetUniqueName(
          llvm_ir::SanitizeFunctionName(
              nested_computation_.name())),  // The name of the function.
      ir_emitter_context_->llvm_module());   // The parent LLVM module.
  for (size_t arg_no = 0; arg_no < argument_dereferenceable_bytes.size();
       ++arg_no) {
    int64_t arg_size = argument_dereferenceable_bytes[arg_no];
    if (arg_size > 0) {
      function->addDereferenceableParamAttr(arg_no, arg_size);
    }
  }

  // TODO(b/65380986): Investigate if adding fast math flags for generated
  // kernels makes sense.

  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(function->getContext(), "entry", function);
  // Emit a "return void" at entry_bb's end, and sets the insert point before
  // that return instruction.
  llvm::ReturnInst* ret_instr =
      llvm::ReturnInst::Create(function->getContext(), entry_bb);
  b_.SetInsertPoint(ret_instr);

  std::vector<const HloInstruction*> non_io_hlos;
  non_io_hlos.push_back(root);
  for (const auto* hlo : nested_computation_.instructions()) {
    if (hlo->opcode() != HloOpcode::kParameter &&
        hlo != nested_computation_.root_instruction()) {
      non_io_hlos.push_back(hlo);
    }
  }
  bindings_.EmitBasePointersForHlos(io_hlos, non_io_hlos);

  TF_RETURN_IF_ERROR(nested_computation_.root_instruction()->Accept(this));
  b_.SetInsertPoint(ret_instr);

  // Function epilogue: copy the output value back.
  {
    // TODO(cheshire) Duplication vs. EmitThreadLocalFunctionEpilogue
    const HloInstruction* root_instruction =
        nested_computation_.root_instruction();
    llvm::Value* root_value = bindings_.GetBasePointer(*root_instruction);
    const Shape& return_shape = root_instruction->shape();

    // Last argument is the out parameter.
    llvm::Argument* out_parameter = std::prev(function->arg_end(), 1);

    if (ShapeUtil::IsScalar(return_shape)) {
      llvm::Value* ret_value =
          Load(llvm_ir::ShapeToIrType(return_shape, module_), root_value,
               "load_ret_value");
      Store(ret_value,
            BitCast(out_parameter, root_value->getType(), "bitcast_ret_value"));
    } else {
      CHECK(return_shape.IsTuple());
      llvm::Type* tuple_type = llvm_ir::ShapeToIrType(return_shape, module_);
      llvm::Type* tuple_type_ptr = tuple_type->getPointerTo();
      llvm::Value* tuple_ptr = BitCast(out_parameter, tuple_type_ptr);

      for (int i = 0; i < return_shape.tuple_shapes_size(); i++) {
        const Shape& element_shape = return_shape.tuple_shapes(i);
        llvm::Value* destination = llvm_ir::EmitGetTupleElement(
            element_shape,
            /*index=*/i,
            /*alignment=*/1, tuple_ptr, tuple_type, &b_);
        llvm::Value* source = llvm_ir::EmitGetTupleElement(
            element_shape,
            /*index=*/i,
            /*alignment=*/1, root_value,
            llvm_ir::ShapeToIrType(root_instruction->shape(), module_), &b_);
        Store(Load(llvm_ir::ShapeToIrType(element_shape, module_), source),
              destination);
      }
    }
  }
  b_.SetInsertPoint(ret_instr);
  emitted_function_ = function;
  return OkStatus();
}

Status IrEmitterNested::HandleParameter(HloInstruction* parameter) {
  return OkStatus();
}

Status IrEmitterNested::EmitTargetElementLoop(
    const HloInstruction& hlo,
    const llvm_ir::ElementGenerator& element_generator) {
  // For MOF we give the loop emitter an array for every output it should
  // generate.
  if (hlo.shape().IsTuple()) {
    std::vector<llvm_ir::IrArray> target_arrays =
        ConstructIrArrayForOutputs(hlo);
    TF_RETURN_IF_ERROR(
        llvm_ir::LoopEmitter(element_generator, target_arrays, &b_).EmitLoop());
    llvm_ir::EmitTuple(GetIrArray(hlo, hlo), target_arrays, &b_);
    return OkStatus();
  }
  return llvm_ir::LoopEmitter(element_generator, GetIrArray(hlo, hlo), &b_)
      .EmitLoop();
}

Status IrEmitterNested::EmitConstants(const HloComputation& computation) {
  for (HloInstruction* instr : computation.instructions()) {
    if (instr->opcode() != HloOpcode::kConstant) {
      continue;
    }
    Literal& literal = *Cast<HloConstantInstruction>(instr)->mutable_literal();
    const bool should_emit_initializer = ShouldEmitLiteralInLlvmIr(literal);
    llvm::ArrayType* global_type =
        llvm::ArrayType::get(b_.getInt8Ty(), literal.size_bytes());
    llvm::Constant* initializer =
        should_emit_initializer
            ? llvm_ir::ConvertLiteralToIrConstant(literal, module_)
            : llvm::ConstantAggregateZero::get(global_type);
    if (should_emit_initializer) {
      VLOG(3) << "Emitted initializer for constant with shape "
              << ShapeUtil::HumanString(literal.shape());
    }

    // These globals will be looked up by name by GpuExecutable so we need to
    // give them an external linkage.  Not all of their uses are visible in
    // the LLVM IR (e.g. TupleThunk) so we can't give then a linkage that
    // merely preserves their names (like available_externally), we also need
    // to ensure that they stick around even if they're "unused".
    //
    // We may have to be more clever here in the future if we notice that we're
    // keeping around too many globals because of their linkage.
    std::string global_name = llvm_ir::ConstantHloToGlobalName(*instr);

    llvm::GlobalVariable* global_for_const = new llvm::GlobalVariable(
        global_type, /*isConstant=*/should_emit_initializer,
        llvm::GlobalValue::ExternalLinkage,
        /*Initializer=*/initializer, global_name,
        /*TLMode=*/llvm::GlobalValue::NotThreadLocal,
        /*AddressSpace=*/0,
        /*isExternallyInitialized=*/false);
    global_for_const->setAlignment(llvm::Align(kConstantBufferAlignBytes));
    ir_emitter_context_->llvm_module()->getGlobalList().push_back(
        global_for_const);

    GpuExecutable::ConstantInfo info;
    info.symbol_name = global_name;

    if (!should_emit_initializer) {
      auto base = static_cast<const uint8_t*>(literal.untyped_data());
      info.content.assign(base, base + literal.size_bytes());
    }
    ir_emitter_context_->constants().push_back(std::move(info));
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
