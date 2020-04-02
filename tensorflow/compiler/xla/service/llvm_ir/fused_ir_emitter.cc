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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
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
    if (generated_value_cache_[hlo].contains(index.multidim())) {
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
              << generated_value_bb->getName().str()
              << ") from the current insertion block ("
              << b_->GetInsertBlock()->getName().str() << ").";
    }

    TF_ASSIGN_OR_RETURN(generated_value_cache_[hlo][index.multidim()],
                        elemental_emitter_->MakeElementGenerator(
                            hlo, indexed_generators_)(index));
    return generated_value_cache_[hlo][index.multidim()];
  };
  return Status::OK();
}

Status FusedIrEmitter::HandleConstant(const HloInstruction* constant) {
  unsigned global_address_space =
      llvm_ir::GetGlobalMemoryAddressSpace(*module_);
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
        /*AddressSpace=*/global_address_space,
        /*isExternallyInitialized=*/false);

    global->setUnnamedAddr(llvm::GlobalVariable::UnnamedAddr::Global);
    llvm::Constant* shape_constant =
        llvm::ConstantExpr::getPointerBitCastOrAddrSpaceCast(
            global,
            llvm_ir::ShapeToIrType(literal.shape(), module_)->getPointerTo());
    return IrArray(shape_constant, constant->shape())
        .EmitReadArrayElement(index, b_);
  };

  return Status::OK();
}

Status FusedIrEmitter::HandleGetTupleElement(
    const HloInstruction* get_tuple_element) {
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
    return llvm_ir::EmitGetTupleElement(get_tuple_element->shape(),
                                        get_tuple_element->tuple_index(),
                                        /*alignment=*/1, tuple_ptr, b_);
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

Status FusedIrEmitter::HandleParameter(const HloInstruction* parameter) {
  indexed_generators_[parameter] =
      [=](const IrArray::Index& index) -> llvm::Value* {
    int64 param_num = parameter->parameter_number();
    if (param_shmem_buffers_.size() > param_num) {
      if (llvm::Value* param_tile_buffer = param_shmem_buffers_[param_num]) {
        // TODO(jlebar): Add AA metadata to this load.  Tile buffers are global
        // variables, so LLVM's points-to analysis doesn't help us much.  And we
        // want the AA info to be present before address spaces are inferred
        // (which is pretty late in the pipeline), so even if we had
        // address-space-based AA in LLVM, it wouldn't help us much here.
        return b_->CreateLoad(
            b_->CreateGEP(param_tile_buffer, {index.GetConstantWithIndexType(0),
                                              thread_id_x_, thread_id_y_}),
            "tiled_buffer");
      }
    }
    return GetIrArrayForFusedParameter(param_num).EmitReadArrayElement(index,
                                                                       b_);
  };
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

Status FusedIrEmitter::FinishVisit(const HloInstruction* root) {
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

bool FusedIrEmitter::IsFusedIrEmitterInefficient(
    const HloInstruction* consumer, const HloInstruction* producer) {
  if (consumer->opcode() != HloOpcode::kFusion) {
    return false;
  }
  // Collects for each instruction in the fusion node from which (indirect)
  // users newly created index values are passed. Roughly speaking, we reuse
  // index values if the shapes are equal when ignoring the element type (we may
  // reuse also if the shape change is a bitcast, but we don't consider that
  // here). By ignoring potential reuses our estimate whether the fusion emitter
  // is inefficient is a bit more conservative than necessary.
  absl::flat_hash_map<const HloInstruction*,
                      absl::flat_hash_set<const HloInstruction*>>
      indexing_users;
  // Stores the number of different index accesses for each instruction in the
  // fusion node. The fusion emitter caches access with the same index, so this
  // value indicates how many times a specific instruction will be emitted.
  absl::flat_hash_map<const HloInstruction*, int64> index_usage_count;
  index_usage_count[consumer] = 1;

  auto evaluate_fusion_computation = [&indexing_users, &index_usage_count](
                                         const HloInstruction* fusion) {
    auto postorder =
        fusion->fused_instructions_computation()->MakeInstructionPostOrder();
    std::reverse(postorder.begin(), postorder.end());
    for (const auto* instruction : postorder) {
      if (instruction->opcode() == HloOpcode::kParameter) {
        continue;
      }
      int64& total = index_usage_count[instruction];
      if (indexing_users[instruction].empty()) {
        total = index_usage_count[fusion];
      } else {
        total = 0;
        for (const auto* user : indexing_users[instruction]) {
          int64 weight = 1;
          // Concatenate is special: the index differs for each operand, so
          // in the worst case we have to deal with as many index values as
          // the number of operands of Concatenate. By considering the worst
          // case, we are more conservative than necessary regarding
          // refusing to fuse.
          if (user->opcode() == HloOpcode::kConcatenate) {
            weight = user->operand_count();
          }
          total += index_usage_count[user] * weight;
        }
      }
      for (const auto* operand : instruction->operands()) {
        // For simplicity we assume that all shape and layout changing
        // operations invalidate index reuse.
        if (Shape::Equal().IgnoreElementType()(operand->shape(),
                                               instruction->shape())) {
          // If the index is reused, it means the operand gets index values
          // from the same set of (indirect) users as 'instruction' itself.
          indexing_users[operand].insert(indexing_users[instruction].begin(),
                                         indexing_users[instruction].end());
        } else {
          // If the index is not reused, it means 'instruction' computes a
          // new index derived from the index it gets.
          indexing_users[operand].insert(instruction);
        }
      }
    }
  };
  evaluate_fusion_computation(consumer);

  // Also account for the 'producer' if it would be fused. Find the operand it
  // corresponds to.
  for (int64 operand_num = 0; operand_num < consumer->operand_count();
       ++operand_num) {
    if (consumer->operand(operand_num) == producer) {
      auto instruction = consumer->fused_parameter(operand_num);
      int64& total = index_usage_count[producer];
      total = 0;
      for (const auto* user : indexing_users[instruction]) {
        total += index_usage_count[user];
      }
      break;
    }
  }

  // If 'producer' is a fusion node as well, also evaluate it.
  if (producer->opcode() == HloOpcode::kFusion) {
    evaluate_fusion_computation(producer);
  }

  // Sum up the total number of emitted ops.
  int64 total = 0;
  for (const auto& entry : index_usage_count) {
    total += entry.second;
  }

  // Check that the code duplication has at most a factor of 15 (where 15 is an
  // arbitrary constant that seems to work).
  return total > 15 * index_usage_count.size();
}

}  // namespace xla
