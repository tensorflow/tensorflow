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

#include <algorithm>
#include <functional>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using llvm_ir::IrArray;

Status FusedIrEmitter::DefaultAction(const HloInstruction* hlo) {
  indexed_generators_[hlo] =
      [=](const IrArray::Index& index) -> StatusOr<llvm::Value*> {
    if (llvm::Value* generated_value = FindOrDefault(
            generated_value_cache_[hlo], index.multidim(), nullptr)) {
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
              << generated_value_bb->getName().str()
              << ") from the current insertion block ("
              << b_->GetInsertBlock()->getName().str() << ").";
    }

    TF_ASSIGN_OR_RETURN(llvm::Value* const generated_value,
                        elemental_emitter_->MakeElementGenerator(
                            hlo, indexed_generators_)(index));
    generated_value_cache_[hlo][index.multidim()] = generated_value;
    return generated_value;
  };
  return Status::OK();
}

Status FusedIrEmitter::HandleConstant(const HloInstruction* constant) {
  indexed_generators_[constant] = [=](const IrArray::Index& index) {
    const Literal& literal = constant->literal();
    llvm::Constant* initializer =
        llvm_ir::ConvertLiteralToIrConstant(literal, module_);
    llvm::GlobalVariable* global = new llvm::GlobalVariable(
        *b_->GetInsertBlock()->getModule(), initializer->getType(),
        /*isConstant=*/true,
        /*Linkage=*/llvm::GlobalValue::PrivateLinkage,
        /*Initializer=*/initializer,
        /*Name=*/"", /*InsertBefore=*/nullptr,
        /*TLMode=*/llvm::GlobalValue::NotThreadLocal,
        /*AddressSpace=*/0,
        /*isExternallyInitialized=*/false);

    global->setUnnamedAddr(llvm::GlobalVariable::UnnamedAddr::Global);
    llvm::Constant* shape_constant =
        llvm::ConstantExpr::getPointerBitCastOrAddrSpaceCast(
            global,
            llvm_ir::ShapeToIrType(literal.shape(), module_)->getPointerTo());
    return IrArray(shape_constant, constant->shape())
        .EmitReadArrayElement(index, b_, constant->name());
  };

  return Status::OK();
}

Status FusedIrEmitter::HandleGetTupleElement(
    const HloInstruction* get_tuple_element) {
  return InternalError("Tuple parameters are not supported for fusion");
}

Status FusedIrEmitter::HandleParameter(const HloInstruction* parameter) {
  if (indexed_generators_.find(parameter) == indexed_generators_.end()) {
    return InvalidArgument("Unbound parameter: %s", parameter->ToString());
  }
  return Status::OK();
}

Status FusedIrEmitter::HandleTuple(const HloInstruction* tuple) {
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

bool FusedIrEmitter::IsFusedIrEmitterInefficient(
    const HloInstruction* consumer, const HloInstruction* producer) {
  if (consumer->opcode() != HloOpcode::kFusion) {
    return false;
  }
  FusionNodeIndexingEvaluation eval_consumer(consumer);
  if (producer->opcode() != HloOpcode::kFusion) {
    return eval_consumer.CodeDuplicationTooHigh(producer);
  }
  // If 'producer' is a fusion node as well, also evaluate it. Pass the
  // evaluated duplication of the fusion node if it is merged into consumer.
  FusionNodeIndexingEvaluation eval_producer(
      producer, eval_consumer.EvaluateEmittedInstructions(producer));
  return eval_producer.MaxCodeDuplicationTooHigh();
}

StatusOr<FusedIrEmitter::IndexedGenerator> FusedIrEmitter::GetGenerator(
    const HloInstruction* instruction) {
  std::vector<const HloInstruction*> stack;
  stack.push_back(instruction);
  while (!stack.empty()) {
    const HloInstruction* instr = stack.back();
    stack.pop_back();
    if (indexed_generators_.count(instr)) {
      continue;
    }
    for (const HloInstruction* operand : instr->operands()) {
      stack.push_back(operand);
    }
    switch (instr->opcode()) {
      case HloOpcode::kConstant:
        TF_RETURN_IF_ERROR(HandleConstant(instr));
        break;
      case HloOpcode::kGetTupleElement:
        TF_RETURN_IF_ERROR(HandleGetTupleElement(instr));
        break;
      case HloOpcode::kParameter:
        TF_RETURN_IF_ERROR(HandleParameter(instr));
        break;
      case HloOpcode::kTuple:
        TF_RETURN_IF_ERROR(HandleTuple(instr));
        break;
      default:
        TF_RETURN_IF_ERROR(DefaultAction(instr));
        break;
    }
    CHECK(indexed_generators_.count(instr));
  }
  return indexed_generators_.at(instruction);
}

}  // namespace xla
