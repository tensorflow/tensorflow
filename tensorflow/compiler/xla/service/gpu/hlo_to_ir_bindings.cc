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

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {
namespace gpu {

using absl::StrAppend;
using absl::StrCat;

void HloToIrBindings::EmitBasePointersForHlos(
    absl::Span<const HloInstruction* const> io_hlos,
    absl::Span<const HloInstruction* const> non_io_hlos) {
  CHECK(is_nested_);

  // I/O HLOs are bound to the arguments of the current IR function,
  // *excluding* the output argument, which is added to non-I/O HLOs.
  // I.e.,
  //
  // void IrFunction(io_0, io_1, ..., io_{m-1}, output_arg);
  llvm::Function* function = b_->GetInsertBlock()->getParent();
  CHECK_EQ(io_hlos.size() + 1, function->arg_size());

  // An HLO can have duplicated operands. This data structure remembers which
  // operand HLOs are already bound to avoid rebinding the same HLO.
  absl::flat_hash_set<const HloInstruction*> already_bound_for_this_function;
  auto arg_iter = function->arg_begin();
  for (const HloInstruction* io_hlo : io_hlos) {
    CHECK(io_hlo == io_hlo->parent()->root_instruction() ||
          !absl::c_count(non_io_hlos, io_hlo))
        << "IO HLOs and non-IO HLOs should be disjoint";
    if (!already_bound_for_this_function.contains(io_hlo)) {
      BindHloToIrValue(*io_hlo, &*arg_iter);
      already_bound_for_this_function.insert(io_hlo);
    }
    ++arg_iter;
  }

  // Name and skip the output parameter.
  arg_iter->setName("output_arg");
  ++arg_iter;

  for (const HloInstruction* non_io_hlo : non_io_hlos) {
    if (already_bound_for_this_function.contains(non_io_hlo)) {
      continue;
    }
    already_bound_for_this_function.insert(non_io_hlo);

    if (non_io_hlo->opcode() == HloOpcode::kGetTupleElement) {
      continue;
    }

    ShapeUtil::ForEachSubshape(
        non_io_hlo->shape(),
        [&](const Shape& /*subshape*/, const ShapeIndex& index) {
          if (non_io_hlo->opcode() == HloOpcode::kConstant) {
            llvm::Value* global_for_constant = module_->getGlobalVariable(
                llvm_ir::ConstantHloToGlobalName(*non_io_hlo));
            CHECK(global_for_constant)
                << llvm_ir::ConstantHloToGlobalName(*non_io_hlo);
            BindHloToIrValue(*non_io_hlo, global_for_constant);
          } else {
            llvm::Type* pointee_type =
                llvm_ir::ShapeToIrType(non_io_hlo->shape(), module_);
            BindHloToIrValue(*non_io_hlo,
                             llvm_ir::EmitAllocaAtFunctionEntry(
                                 pointee_type, /*name=*/"", b_),
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
        GetTypedIrValue(*gte->operand(0), {}, base_ptr),
        llvm_ir::ShapeToIrType(gte->operand(0)->shape(), module_), b_);
  }
  return llvm_ir::EmitGetTupleElement(
      gte->shape(), gte->tuple_index(), /*alignment=*/1,
      EmitGetTupleElement(gte->operand(0), base_ptr),
      llvm_ir::ShapeToIrType(gte->operand(0)->shape(), module_), b_);
}

// Returns true if `value` has a name that should not be changed.
static bool HasMeaningfulName(llvm::Value* value) {
  if (auto* global = llvm::dyn_cast<llvm::GlobalValue>(value)) {
    return global->getLinkage() != llvm::GlobalValue::PrivateLinkage;
  }
  return false;
}

llvm::Value* CastToTypedValue(const Shape& shape, llvm::Value* ir_value,
                              llvm::IRBuilder<>* b) {
  llvm::Type* pointee_type =
      llvm_ir::ShapeToIrType(shape, b->GetInsertBlock()->getModule());

  llvm::Type* dest_type = pointee_type->getPointerTo();

  llvm::Value* typed_ir_value;
  if (llvm::isa<llvm::GlobalVariable>(ir_value)) {
    typed_ir_value = llvm::ConstantExpr::getPointerBitCastOrAddrSpaceCast(
        llvm::cast<llvm::GlobalVariable>(ir_value), dest_type);
  } else {
    typed_ir_value = b->CreatePointerBitCastOrAddrSpaceCast(
        ir_value, pointee_type->getPointerTo());
  }
  return typed_ir_value;
}

llvm::Value* HloToIrBindings::GetTypedIrValue(const HloInstruction& hlo,
                                              ShapeIndexView shape_index,
                                              llvm::Value* ir_value) {
  auto typed_ir_value = CastToTypedValue(
      ShapeUtil::GetSubshape(hlo.shape(), shape_index), ir_value, b_);
  if (!HasMeaningfulName(ir_value)) {
    ir_value->setName(llvm_ir::IrName(&hlo, "raw"));
  }
  if (!HasMeaningfulName(typed_ir_value)) {
    typed_ir_value->setName(llvm_ir::IrName(&hlo, "typed"));
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

llvm_ir::IrArray HloToIrBindings::GetIrArray(const HloInstruction& hlo,
                                             const HloInstruction& consumer,
                                             const ShapeIndex& shape_index) {
  CHECK(is_nested_)
      << "IrEmitterUnnested should instead use LMHLO to get the IrArray";

  llvm::Value* base_ptr = GetBasePointer(hlo, shape_index);
  Shape new_shape = ShapeUtil::GetSubshape(hlo.shape(), shape_index);
  llvm::Type* pointee_type = llvm_ir::ShapeToIrType(new_shape, module_);
  CHECK_NE(base_ptr, nullptr)
      << "Buffer not assigned for shape_index " << shape_index.ToString()
      << " of " << hlo.ToString();
  llvm_ir::IrArray ir_array(base_ptr, pointee_type, new_shape);

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

std::string HloToIrBindings::ToString() const {
  std::string s = StrCat("** HloToIrBindings **\n");
  StrAppend(&s, "  is_nested_=", is_nested_, "\n");
  StrAppend(&s,
            "  temp_buffer_base_=", llvm_ir::DumpToString(temp_buffer_base_),
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
      if (!instr->shape().IsTuple()) {
        const llvm::Value* val = shape_tree.begin()->second;
        StrAppend(&s, " -> ", llvm_ir::DumpToString(val), "\n");
        continue;
      }

      StrAppend(&s, "\n");
      for (auto shape_it = shape_tree.begin(); shape_it != shape_tree.end();
           ++shape_it) {
        llvm::Value* val = shape_it->second;
        StrAppend(&s, "      ", shape_it->first.ToString(), " -> ",
                  (val != nullptr ? llvm_ir::DumpToString(val) : "null"), "\n");
      }
    }
  }
  return s;
}

}  // namespace gpu
}  // namespace xla
