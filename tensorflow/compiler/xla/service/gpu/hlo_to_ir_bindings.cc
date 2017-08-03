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

#include "tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ops.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

void HloToIrBindings::EmitBasePointersForHlos(
    tensorflow::gtl::ArraySlice<const HloInstruction*> io_hlos,
    tensorflow::gtl::ArraySlice<const HloInstruction*> non_io_hlos) {
  // I/O HLOs are bound to the arguments of the current IR function. I.e.,
  //
  // void IrFunction(io_0, io_1, ..., io_{m-1}, temp_buffer_base) {
  llvm::Function* function = ir_builder_->GetInsertBlock()->getParent();
  CHECK_EQ(io_hlos.size() + 1, function->arg_size());

  // An HLO can have duplicated operands. This data structure remembers which
  // operand HLOs are already bound to avoid rebinding the same HLO.
  std::set<const HloInstruction*> already_bound_for_this_function;
  auto arg_iter = function->arg_begin();
  for (const HloInstruction* io_hlo : io_hlos) {
    if (!already_bound_for_this_function.count(io_hlo)) {
      if (!is_nested_ && io_hlo->opcode() == HloOpcode::kGetTupleElement) {
        BindHloToIrValue(*io_hlo, EmitGetTupleElement(io_hlo, &*arg_iter));
      } else {
        BindHloToIrValue(*io_hlo, &*arg_iter);
      }
      already_bound_for_this_function.insert(io_hlo);
    }
    ++arg_iter;
  }

  temp_buffer_base_ = &*arg_iter;
  temp_buffer_base_->setName("temp_buffer");

  for (const HloInstruction* non_io_hlo : non_io_hlos) {
    if (already_bound_for_this_function.count(non_io_hlo)) {
      continue;
    }
    already_bound_for_this_function.insert(non_io_hlo);

    if (non_io_hlo->opcode() == HloOpcode::kGetTupleElement) {
      if (!is_nested_) {
        // Lookup allocation GetTupleElement operand.
        const BufferAllocation::Slice slice =
            buffer_assignment_
                ->GetUniqueTopLevelSlice(LatestNonGteAncestor(non_io_hlo))
                .ConsumeValueOrDie();
        // We are not in a nested context, so check non-thread-local allocation.
        CHECK(!slice.allocation()->is_thread_local());
        const int64 offset = slice.offset();
        CHECK_NE(nullptr, temp_buffer_base_);
        // Emit IR for GetTupleElement instruction and bind to emitted value.
        llvm::Value* base_ptr = ir_builder_->CreateInBoundsGEP(
            temp_buffer_base_, ir_builder_->getInt64(offset));
        BindHloToIrValue(*non_io_hlo,
                         EmitGetTupleElement(non_io_hlo, base_ptr));
      }
      continue;
    }

    if (!buffer_assignment_->HasTopLevelAllocation(non_io_hlo)) {
      continue;
    }

    ShapeUtil::ForEachSubshape(
        non_io_hlo->shape(),
        [&](const Shape& /*subshape*/, const ShapeIndex& index) {
          // A non-IO HLO with a buffer is bound to
          // (1) an alloca if it is thread-local, or
          // (2) an internal pointer in temp_buffer_base according to its
          // offset.
          auto slice_result =
              buffer_assignment_->GetUniqueSlice(non_io_hlo, index);
          if (!slice_result.ok()) {
            return;
          }
          const BufferAllocation::Slice slice =
              slice_result.ConsumeValueOrDie();
          if (slice.allocation()->is_thread_local()) {
            llvm::Type* pointee_type =
                llvm_ir::ShapeToIrType(non_io_hlo->shape(), ir_builder_);
            BindHloToIrValue(*non_io_hlo,
                             ir_builder_->CreateAlloca(pointee_type), index);
          } else {
            const int64 offset = slice.offset();
            CHECK_NE(nullptr, temp_buffer_base_);
            BindHloToIrValue(
                *non_io_hlo,
                ir_builder_->CreateInBoundsGEP(temp_buffer_base_,
                                               ir_builder_->getInt64(offset)),
                index);
          }
        });
  }
}

llvm::Value* HloToIrBindings::EmitGetTupleElement(const HloInstruction* gte,
                                                  llvm::Value* base_ptr) {
  // TODO(b/26344050): tighten the alignment based on the real element type.
  if (gte->operand(0)->opcode() != HloOpcode::kGetTupleElement) {
    return llvm_ir::EmitGetTupleElement(
        gte->shape(), gte->tuple_index(), /*alignment=*/1,
        GetTypedIrValue(*gte->operand(0), {}, base_ptr), ir_builder_);
  }
  return llvm_ir::EmitGetTupleElement(
      gte->shape(), gte->tuple_index(), /*alignment=*/1,
      EmitGetTupleElement(gte->operand(0), base_ptr), ir_builder_);
}

llvm::Value* HloToIrBindings::GetTypedIrValue(const HloInstruction& hlo,
                                              const ShapeIndex& shape_index,
                                              llvm::Value* ir_value) {
  llvm::Type* pointee_type = llvm_ir::ShapeToIrType(
      ShapeUtil::GetSubshape(hlo.shape(), shape_index), ir_builder_);
  llvm::Type* dest_type = pointee_type->getPointerTo();

  llvm::Value* typed_ir_value;
  if (llvm::isa<llvm::GlobalVariable>(ir_value)) {
    typed_ir_value = llvm::ConstantExpr::getBitCast(
        llvm::cast<llvm::GlobalVariable>(ir_value), dest_type);
  } else {
    typed_ir_value =
        ir_builder_->CreateBitCast(ir_value, pointee_type->getPointerTo());
  }
  string ir_value_name = llvm_ir::SanitizeIrName(hlo.name());
  ir_value->setName(llvm_ir::AsStringRef(ir_value_name + ".raw"));
  typed_ir_value->setName(llvm_ir::AsStringRef(ir_value_name + ".typed"));
  return typed_ir_value;
}

void HloToIrBindings::BindHloToIrValue(const HloInstruction& hlo,
                                       llvm::Value* ir_value,
                                       const ShapeIndex& shape_index) {
  VLOG(2) << "Binding " << hlo.ToString();

  const Shape& hlo_shape = hlo.shape();
  llvm::Value* typed_ir_value = GetTypedIrValue(hlo, shape_index, ir_value);

  if (!BoundToIrValue(hlo)) {
    // Set the root of ShapeTree first before assigning the element ir value.
    InsertOrDie(&base_ptrs_, &hlo, ShapeTree<llvm::Value*>(hlo_shape, nullptr));
  }
  *(base_ptrs_[&hlo].mutable_element(shape_index)) = typed_ir_value;
}

llvm_ir::IrArray HloToIrBindings::GetIrArray(const HloInstruction& hlo,
                                             const ShapeIndex& shape_index) {
  llvm_ir::IrArray ir_array(GetBasePointer(hlo, shape_index),
                            ShapeUtil::GetSubshape(hlo.shape(), shape_index));
  alias_analysis_.AddAliasingInformationToIrArray(hlo, &ir_array);
  return ir_array;
}

void HloToIrBindings::UnbindAllLocalIrValues() {
  std::vector<const HloInstruction*> hlos_to_unbind;
  for (auto& key_value : base_ptrs_) {
    if (!llvm::isa<llvm::GlobalVariable>(
            (key_value.second.element({}))->stripPointerCasts())) {
      hlos_to_unbind.push_back(key_value.first);
    }
  }
  for (const HloInstruction* hlo_to_unbind : hlos_to_unbind) {
    VLOG(2) << "Unbinding " << hlo_to_unbind->ToString();
    base_ptrs_.erase(hlo_to_unbind);
  }
}

}  // namespace gpu
}  // namespace xla
