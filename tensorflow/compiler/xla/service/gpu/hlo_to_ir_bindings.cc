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

#include "absl/strings/str_cat.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

using absl::StrAppend;
using absl::StrCat;

void HloToIrBindings::EmitBasePointersForHlos(
    absl::Span<const HloInstruction* const> io_hlos,
    absl::Span<const HloInstruction* const> non_io_hlos) {
  // I/O HLOs are bound to the arguments of the current IR function. I.e.,
  //
  // void IrFunction(io_0, io_1, ..., io_{m-1}, temp_buffer_base) {
  llvm::Function* function = b_->GetInsertBlock()->getParent();
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
                ->GetUniqueTopLevelSlice(non_io_hlo->LatestNonGteAncestor())
                .ConsumeValueOrDie();
        // We are not in a nested context, so check non-thread-local allocation.
        CHECK(!slice.allocation()->is_thread_local());
        const int64 offset = slice.offset();
        CHECK_NE(nullptr, temp_buffer_base_);
        // Emit IR for GetTupleElement instruction and bind to emitted value.
        llvm::Value* base_ptr =
            b_->CreateInBoundsGEP(temp_buffer_base_, b_->getInt64(offset));
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
                llvm_ir::ShapeToIrType(non_io_hlo->shape(), module_);
            BindHloToIrValue(*non_io_hlo, b_->CreateAlloca(pointee_type),
                             index);
          } else if (slice.allocation()->is_constant()) {
            llvm::Value* global_for_constant =
                module_->getGlobalVariable(llvm_ir::AsStringRef(
                    llvm_ir::ConstantBufferAllocationToGlobalName(
                        *slice.allocation())));
            BindHloToIrValue(*non_io_hlo, global_for_constant);
          } else {
            const int64 offset = slice.offset();
            CHECK_NE(nullptr, temp_buffer_base_);
            BindHloToIrValue(
                *non_io_hlo,
                b_->CreateInBoundsGEP(temp_buffer_base_, b_->getInt64(offset)),
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
        GetTypedIrValue(*gte->operand(0), {}, base_ptr), b_, module_);
  }
  return llvm_ir::EmitGetTupleElement(
      gte->shape(), gte->tuple_index(), /*alignment=*/1,
      EmitGetTupleElement(gte->operand(0), base_ptr), b_, module_);
}

// Returns true if `value` has a name that should not be changed.
static bool HasMeaningfulName(llvm::Value* value) {
  if (auto* global = llvm::dyn_cast<llvm::GlobalValue>(value)) {
    return global->getLinkage() != llvm::GlobalValue::PrivateLinkage;
  }
  return false;
}

llvm::Value* HloToIrBindings::GetTypedIrValue(const HloInstruction& hlo,
                                              ShapeIndexView shape_index,
                                              llvm::Value* ir_value) {
  llvm::Type* pointee_type = llvm_ir::ShapeToIrType(
      ShapeUtil::GetSubshape(hlo.shape(), shape_index), module_);
  llvm::Type* dest_type = pointee_type->getPointerTo();

  llvm::Value* typed_ir_value;
  if (llvm::isa<llvm::GlobalVariable>(ir_value)) {
    typed_ir_value = llvm::ConstantExpr::getPointerBitCastOrAddrSpaceCast(
        llvm::cast<llvm::GlobalVariable>(ir_value), dest_type);
  } else {
    typed_ir_value = b_->CreatePointerBitCastOrAddrSpaceCast(
        ir_value, pointee_type->getPointerTo());
  }
  if (!HasMeaningfulName(ir_value)) {
    ir_value->setName(llvm_ir::AsStringRef(llvm_ir::IrName(&hlo, "raw")));
  }
  if (!HasMeaningfulName(typed_ir_value)) {
    typed_ir_value->setName(
        llvm_ir::AsStringRef(llvm_ir::IrName(&hlo, "typed")));
  }
  return typed_ir_value;
}

void HloToIrBindings::BindHloToIrValue(const HloInstruction& hlo,
                                       llvm::Value* ir_value,
                                       ShapeIndexView shape_index) {
  VLOG(2) << "Binding " << hlo.ToString();

  const Shape& hlo_shape = hlo.shape();
  llvm::Value* typed_ir_value = GetTypedIrValue(hlo, shape_index, ir_value);

  if (!BoundToIrValue(hlo)) {
    // Set the root of ShapeTree first before assigning the element ir value.
    InsertOrDie(&base_ptrs_, &hlo, ShapeTree<llvm::Value*>(hlo_shape, nullptr));
  }
  *(base_ptrs_[&hlo].mutable_element(shape_index)) = typed_ir_value;
}

// Determines whether hlo's buffers are never modified within the execution of
// consumer.
static bool BuffersInvariantWithinConsumer(
    const HloInstruction& hlo, const HloInstruction& consumer,
    const BufferAssignment* buffer_assignment) {
  // Check if consumer is inside a fusion node -- if so, "dereference" it until
  // we get to a non-fusion node.
  const HloInstruction* c = &consumer;
  while (c->IsFused()) {
    c = c->parent()->FusionInstruction();
  }

  // If, after dereferencing c, we end up with a node that's not inside our
  // module's top-level computation (say our node is inside a while loop), we
  // give up on marking array as invariant, because this HLO may be run multiple
  // times (e.g. multiple while loop iterations, or multiple invocations of a
  // reducer's computation).  TODO(jlebar): We could relax this constraint if we
  // emitted an llvm.invariant.group.barrier at the end of the computation.
  return c->parent() == c->GetModule()->entry_computation() &&
         buffer_assignment->HaveDisjointSlices(&hlo, &consumer);
}

llvm_ir::IrArray HloToIrBindings::GetIrArray(const HloInstruction& hlo,
                                             const HloInstruction& consumer,
                                             const ShapeIndex& shape_index) {
  llvm::Value* base_ptr = GetBasePointer(hlo, shape_index);
  CHECK_NE(base_ptr, nullptr)
      << "Buffer not assigned for shape_index " << shape_index.ToString()
      << " of " << hlo.ToString();
  llvm_ir::IrArray ir_array(base_ptr,
                            ShapeUtil::GetSubshape(hlo.shape(), shape_index));
  alias_analysis_.AddAliasingInformationToIrArray(hlo, &ir_array, shape_index);

  // The GPU backend emits one kernel per top-level HLO, and LLVM views
  // execution of one kernel as the "whole program" executed on the GPU.
  // Therefore if hlo's output buffer is not modified within consumer, and if
  // consumer runs hlo only once (so that it doesn't create two different
  // outputs), then we can mark ir_array as invariant over the whole program.
  if (BuffersInvariantWithinConsumer(hlo, consumer, buffer_assignment_)) {
    VLOG(2) << "Marking " << hlo.name() << " as invariant within "
            << consumer.name();
    ir_array.MarkInvariantOverWholeProgram(&module_->getContext());
  }

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

string HloToIrBindings::ToString() const {
  string s = StrCat("** HloToIrBindings **\n");
  StrAppend(&s, "  is_nested_=", is_nested_, "\n");
  StrAppend(&s,
            "  temp_buffer_base_=", llvm_ir::DumpToString(*temp_buffer_base_),
            "\n");

  if (base_ptrs_.empty()) {
    return s;
  }

  // Iterate over all computations in the module in topological order, and print
  // out the base pointers we have in each computation in topological order.
  for (const HloComputation* computation :
       base_ptrs_.begin()->first->GetModule()->MakeComputationPostOrder()) {
    bool is_first = true;
    for (const HloInstruction* instr :
         computation->MakeInstructionPostOrder()) {
      auto it = base_ptrs_.find(instr);
      if (it == base_ptrs_.end()) {
        continue;
      }
      if (is_first) {
        StrAppend(&s, "  Base pointers for computation ", computation->name(),
                  ":\n");
        is_first = false;
      }
      StrAppend(&s, "    ", instr->ToString());

      const ShapeTree<llvm::Value*>& shape_tree = it->second;
      if (!ShapeUtil::IsTuple(instr->shape())) {
        const llvm::Value* val = shape_tree.begin()->second;
        StrAppend(&s, " -> ", llvm_ir::DumpToString(*val), "\n");
        continue;
      }

      StrAppend(&s, "\n");
      for (auto shape_it = shape_tree.begin(); shape_it != shape_tree.end();
           ++shape_it) {
        llvm::Value* val = shape_it->second;
        StrAppend(&s, "      ", shape_it->first.ToString(), " -> ",
                  (val != nullptr ? llvm_ir::DumpToString(*val) : "null"),
                  "\n");
      }
    }
  }
  return s;
}

}  // namespace gpu
}  // namespace xla
