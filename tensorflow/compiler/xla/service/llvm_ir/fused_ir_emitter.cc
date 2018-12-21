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

#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"

#include <functional>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using llvm_ir::IrArray;

Status FusedIrEmitter::DefaultAction(HloInstruction* hlo) {
  indexed_generators_[hlo] =
      [=](const IrArray::Index& index) -> StatusOr<llvm::Value*> {
    if (generated_value_cache_[hlo].count(index.multidim()) > 0) {
      llvm::Value* generated_value =
          generated_value_cache_[hlo][index.multidim()];
      llvm::BasicBlock* generated_value_bb = nullptr;
      if (auto* generated_instruction =
              llvm::dyn_cast<llvm::Instruction>(generated_value)) {
        generated_value_bb = generated_instruction->getParent();
      }
      // Ideally, we should be able to reuse the cached generated value if it
      // dominates the current insertion block. However, the check for dominance
      // can be expensive and unreliable when the function is being constructed.
      //
      // It's also worth experimenting what if we don't do caching at all.
      // LLVM's CSE or GVN should be able to easily merge common subexpressions
      // that would be regenerated without caching. But this might increase the
      // JIT compilation time.
      if (generated_value_bb == nullptr ||
          generated_value_bb == b_->GetInsertBlock()) {
        VLOG(3) << "The cached generated value is reused.";
        return generated_value;
      }
      VLOG(3) << "The cached generated value can't be reused, because it is in "
                 "a different BB ("
              << llvm_ir::AsString(generated_value_bb->getName())
              << ") from the current insertion block ("
              << llvm_ir::AsString(b_->GetInsertBlock()->getName()) << ").";
    }

    TF_ASSIGN_OR_RETURN(generated_value_cache_[hlo][index.multidim()],
                        elemental_emitter_->MakeElementGenerator(
                            hlo, indexed_generators_)(index));
    return generated_value_cache_[hlo][index.multidim()];
  };
  return Status::OK();
}

Status FusedIrEmitter::HandleConstant(HloInstruction* constant) {
  indexed_generators_[constant] = [=](const IrArray::Index& index) {
    const Literal& literal = constant->literal();
    llvm::Constant* initializer =
        llvm_ir::ConvertLiteralToIrConstant(literal, module_);
    llvm::GlobalVariable* global = new llvm::GlobalVariable(
        *b_->GetInsertBlock()->getModule(), initializer->getType(),
        /*isConstant=*/true, llvm::GlobalValue::ExternalLinkage, initializer,
        /*Name=*/"");
    llvm::Constant* shape_constant = llvm::ConstantExpr::getBitCast(
        global,
        llvm_ir::ShapeToIrType(literal.shape(), module_)->getPointerTo());
    return IrArray(shape_constant, constant->shape())
        .EmitReadArrayElement(index, b_);
  };

  return Status::OK();
}

Status FusedIrEmitter::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  auto emit_tuple_element_ptr = [=]() -> StatusOr<llvm::Value*> {
    const HloInstruction* tuple_operand = get_tuple_element->operand(0);
    llvm::Value* tuple_ptr;
    if (tuple_operand->opcode() == HloOpcode::kGetTupleElement) {
      TF_ASSIGN_OR_RETURN(tuple_ptr, non_indexed_generators_[tuple_operand]());
    } else {
      if (tuple_operand->opcode() != HloOpcode::kParameter) {
        return Unimplemented(
            "GetTupleElement fusion currently only supports parameter or "
            "nested"
            "GetTupleElement as tuple operand, found an exception: %s",
            tuple_operand->name());
      }
      tuple_ptr =
          GetBasePointerForFusedParameter(tuple_operand->parameter_number());
    }

    // Lookup tuple element pointer.
    return llvm_ir::EmitGetTupleElement(
        get_tuple_element->shape(), get_tuple_element->tuple_index(),
        /*alignment=*/1, tuple_ptr, b_, module_);
  };

  if (!get_tuple_element->shape().IsTuple()) {
    indexed_generators_[get_tuple_element] =
        [=](const IrArray::Index& index) -> StatusOr<llvm::Value*> {
      // TODO(b/34080002) Add aliasing information to tuple element IrArray.
      TF_ASSIGN_OR_RETURN(llvm::Value * tuple_element_ptr,
                          emit_tuple_element_ptr());
      return IrArray(tuple_element_ptr, get_tuple_element->shape())
          .EmitReadArrayElement(index, b_);
    };
  } else {
    non_indexed_generators_[get_tuple_element] = emit_tuple_element_ptr;
  }
  return Status::OK();
}

Status FusedIrEmitter::HandleParameter(HloInstruction* parameter) {
  indexed_generators_[parameter] =
      [=](const IrArray::Index& index) -> llvm::Value* {
    if (tiled_parameter_info_) {
      if (llvm::Value* param_tile_buffer =
              tiled_parameter_info_->GetBufferForParameter(
                  parameter->parameter_number())) {
        // TODO(jlebar): Add AA metadata to this load.  Tile buffers are global
        // variables, so LLVM's points-to analysis doesn't help us much.  And we
        // want the AA info to be present before address spaces are inferred
        // (which is pretty late in the pipeline), so even if we had
        // address-space-based AA in LLVM, it wouldn't help us much here.
        return b_->CreateLoad(
            b_->CreateGEP(param_tile_buffer, {index.GetConstantWithIndexType(0),
                                              tiled_parameter_info_->x(),
                                              tiled_parameter_info_->y()}),
            "tiled_buffer");
      }
    }
    return GetIrArrayForFusedParameter(parameter->parameter_number())
        .EmitReadArrayElement(index, b_);
  };
  return Status::OK();
}

Status FusedIrEmitter::HandleTuple(HloInstruction* tuple) {
  absl::Span<HloInstruction* const> operands(tuple->operands());
  std::vector<llvm::Type*> operand_elemental_ir_types;
  for (HloInstruction* operand : operands) {
    operand_elemental_ir_types.push_back(llvm_ir::PrimitiveTypeToIrType(
        operand->shape().element_type(), module_));
  }
  indexed_generators_[tuple] =
      [=](const IrArray::Index& index) -> StatusOr<llvm::Value*> {
    llvm::Value* ret = llvm::UndefValue::get(
        llvm::StructType::get(b_->getContext(), operand_elemental_ir_types));
    for (size_t i = 0; i < ShapeUtil::TupleElementCount(tuple->shape()); ++i) {
      TF_ASSIGN_OR_RETURN(llvm::Value * val_i,
                          indexed_generators_[operands[i]](index));
      ret = b_->CreateInsertValue(ret, val_i, i);
    }
    return ret;
  };
  return Status::OK();
}

Status FusedIrEmitter::FinishVisit(HloInstruction* root) {
  fused_root_ = root;
  return Status::OK();
}

FusedIrEmitter::IndexedGenerator FusedIrEmitter::GetRootGenerator() const {
  CHECK_NE(nullptr, fused_root_)
      << "GetRootGenerator should be called after Accept.";
  return indexed_generators_.at(fused_root_);
}

FusedIrEmitter::IndexedGenerator FusedIrEmitter::GetGenerator(
    const HloInstruction* instruction) const {
  return indexed_generators_.at(instruction);
}

}  // namespace xla
