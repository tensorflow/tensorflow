/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/evaluator/hlo_evaluator_interpreter.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/evaluator/hlo_evaluator_interpreter_deferred_ops.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
/*
 * LinearizedInterpreter Overview
 * ==============================
 *
 * The LinearizedInterpreter executes a flattened sequence of operations on
 * elements in batches in a unified scratchpad memory. Higher-level reduction
 * loops (like Reduce and ReduceWindow) are handled by dedicated runner
 * classes (e.g., ReduceRunner, ReduceWindowRunner) that use this interpreter
 * to evaluate the core computation.
 *
 * The interpreter distinguishes between:
 * - **Deferred Ops** (e.g., Slice, Broadcast): These operations do not compute
 *   values directly. Instead, they map requested output indices back to input
 *   indices in reverse flow.
 * - **Data Ops** (e.g., Add, Maximum) and terminal nodes of deferred chains
 *   (e.g., Iota, Lookup): These operations compute or fetch actual values.
 *
 * Sequence of Events:
 * -------------------
 * 1. Build():
 *    - Processes deferred operations first by tracing chains from the root down
 *      to terminal nodes.
 *    - Allocates **Index Buffers** for deferred ops to map indices in reverse
 *      flow (from requested result indices back to operand indices).
 *    - Allocates **Value Buffers** for terminal nodes of deferred chains to
 *      store evaluated values.
 *    - Flattens the main computation in post-order.
 *    - Allocates **Value Buffers** for all other instructions (Parameters,
 *      Constants, and Intermediate results of data ops).
 *    - Generates a sequence of executable 'Steps'.
 *
 * 2. CreateScratchpad():
 *    - Allocates the physical memory for the scratchpad based on calculated
 *      size.
 *    - Executes constant steps once to populate constant values in the
 *      scratchpad.
 *
 * 3. Execution (Driven by a runner class, e.g., ReduceRunner):
 *    - Setup:
 *      - Populates parameters in the scratchpad with input values (e.g.,
 *        operands, initial values, or current accumulator values).
 *    - Execution:
 *      - Calls ExecuteSteps() to run all non-constant steps on the
 *        scratchpad. This includes steps that map indices for deferred ops
 *        and steps that compute values for data ops.
 *    - Readback:
 *      - Reads results from the scratchpad's result slots to update
 *        accumulators or produce final outputs.
 *
 * Scratchpad Structure (Memory Layout):
 * -------------------------------------
 * The scratchpad is a single contiguous block of memory. Elements are allocated
 * in the following order during Build():
 *
 * +---------------------------------------------------------------------------+
 * | 1. Deferred Operation Buffers (Allocated first in ProcessDeferredOpChain) |
 * |    - Contains buffers for deferred operation chains.                      |
 * |    - Interleaved per chain: [Chain 0 Indices] [Chain 0 Val] ...          |
 * |                                                                           |
 * |  +---------------------------------------------------------------------+  |
 * |  | Index Buffers (Used by deferred ops like Slice, Broadcast)          |  |
 * |  | - Maps output indices back to input indices.                        |  |
 * |  | - Occupies: batch_size * rank * sizeof(int64_t) bytes               |  |
 * |  +---------------------------------------------------------------------+  |
 * |  | Value Buffers (Used by terminal nodes, e.g., Iota, Lookup)          |  |
 * |  | - Stores actual values computed or looked up.                       |  |
 * |  | - Occupies: batch_size * 1 * sizeof(type) bytes                     |  |
 * |  +---------------------------------------------------------------------+  |
 * +---------------------------------------------------------------------------+
 * | 2. Linearized Computation Buffers (Allocated in LinearizeComputation)     |
 * |    - Contains Value Buffers for all instructions in post-order.           |
 * |    - Interleaved: [Param 0] [Constant 0] [Step 0 Res] ...                 |
 * |                                                                           |
 * |  +---------------------------------------------------------------------+  |
 * |  | Value Buffers (Used by "data ops" like Add, Maximum, etc.)          |  |
 * |  | - Stores actual values for parameters, constants, and intermediates. | |
 * |  | - Occupies: batch_size * element_count * sizeof(promoted_type) bytes | |
 * |  +---------------------------------------------------------------------+  |
 * +---------------------------------------------------------------------------+
 */

namespace xla {

namespace {

size_t AlignmentOfPrimitiveType(PrimitiveType type) {
  size_t alignment = 1;
  primitive_util::PrimitiveTypeSwitch<void>(
      [&](auto type_constant) {
        constexpr PrimitiveType kType = decltype(type_constant)::value;
        if constexpr (primitive_util::IsArrayType(kType)) {
          using T = primitive_util::NativeTypeOf<kType>;
          alignment = alignof(T);
        }
      },
      type);
  return alignment;
}

}  // namespace

void LinearizedInterpreter::RecordInstructionOffset(const HloInstruction* instr,
                                                    size_t offset) {
  ShapeTree<size_t> offsets(instr->shape());
  *offsets.mutable_element({}) = offset;
  instruction_offsets_.emplace(instr, std::move(offsets));
}

size_t LinearizedInterpreter::AllocateValueBuffer(PrimitiveType type,
                                                  size_t element_count,
                                                  int batch_size,
                                                  size_t& current_offset) {
  current_offset =
      xla::RoundUpTo(current_offset, AlignmentOfPrimitiveType(type));
  size_t value_size =
      primitive_util::ByteWidth(type) * element_count * batch_size;
  size_t offset = current_offset;
  current_offset += value_size;
  return offset;
}

LinearizedInterpreter::Scratchpad LinearizedInterpreter::CreateScratchpad()
    const {
  Scratchpad s(scratchpad_size_);
  char* base = static_cast<char*>(s.data());
  for (const auto& step : steps_) {
    if (step.opcode == HloOpcode::kConstant) {
      step.execute_fn(&step, base);
    }
  }
  return s;
}

void LinearizedInterpreter::AddStep(std::optional<HloOpcode> opcode,
                                    size_t result_offset,
                                    absl::Span<const size_t> operand_offsets,
                                    StepOpMetadata op_metadata,
                                    Step::ExecuteFn execute_fn,
                                    PrimitiveType type) {
  Step step;
  step.opcode = opcode;
  step.result_offset = result_offset;
  step.operand_offsets.assign(operand_offsets.begin(), operand_offsets.end());
  step.op_metadata = std::move(op_metadata);
  step.execute_fn = execute_fn;
  step.type = type;
  step.batch_size = batch_size_;
  steps_.push_back(std::move(step));
}

void LinearizedInterpreter::ExecuteSteps(Scratchpad& scratchpad) const {
  char* base = static_cast<char*>(scratchpad.data());
  for (const Step& step : steps_) {
    if (step.opcode != HloOpcode::kConstant) {
      step.execute_fn(&step, base);
    }
  }
}

void LinearizedInterpreter::OpRegistry::Register(HloOpcode opcode,
                                                 PopulatorFn populator) {
  registry_[opcode] = std::move(populator);
}

absl::Status LinearizedInterpreter::OpRegistry::Populate(
    Step& step, const HloInstruction* instr,
    PrimitiveType promoted_type) const {
  HloOpcode opcode = step.opcode.value();
  auto it = registry_.find(opcode);
  if (it == registry_.end()) {
    return absl::UnimplementedError(absl::StrCat(
        "Unsupported op in interpreter: ", HloOpcodeString(opcode)));
  }
  return it->second(step, instr, promoted_type);
}

absl::Status LinearizedInterpreter::PopulateStepExecuteFn(
    Step& step, const HloInstruction* instr,
    PrimitiveType promoted_type) const {
  if (!step.opcode) {
    return absl::InvalidArgumentError("Step opcode must be set");
  }
  return op_registry_.Populate(step, instr, promoted_type);
}

absl::Status LinearizedInterpreter::LinearizeComputation(
    const HloComputation* computation, int batch_size,
    const absl::flat_hash_map<int, size_t>& param_to_offset,
    PromotionPolicy promotion_policy, size_t& current_offset) {
  computation_ = computation;
  param_is_double_.assign(computation->num_parameters(), false);

  param_slots_.assign(computation->num_parameters(), std::nullopt);

  std::vector<HloInstruction*> post_order =
      computation->MakeInstructionPostOrder();

  absl::flat_hash_map<const HloInstruction*, PrimitiveType> promoted_types;

  // Process instructions in post-order to ensure operands are processed before
  // users.
  for (const HloInstruction* instr : post_order) {
    // Tuples are handled by creating a ShapeTree of offsets.
    if (instr->opcode() == HloOpcode::kTuple) {
      ShapeTree<size_t> tuple_offsets(instr->shape());
      for (int i = 0; i < instr->operand_count(); ++i) {
        tuple_offsets.CopySubtreeFrom(
            instruction_offsets_.at(instr->operand(i)),
            /*src_index=*/{},
            /*dst_index=*/{i});
      }
      instruction_offsets_.emplace(instr, std::move(tuple_offsets));
      continue;
    }

    // GetTupleElement extracts the offset for the specific tuple element.
    if (instr->opcode() == HloOpcode::kGetTupleElement) {
      const ShapeTree<size_t>& operand_offsets =
          instruction_offsets_.at(instr->operand(0));
      ShapeTree<size_t> gte_offsets(instr->shape());
      gte_offsets.CopySubtreeFrom(operand_offsets,
                                  /*src_index=*/{instr->tuple_index()},
                                  /*dst_index=*/{});
      instruction_offsets_.emplace(instr, std::move(gte_offsets));
      continue;
    }

    PrimitiveType promoted_type = promotion_policy(instr, promoted_types);
    if (promoted_type == F64 && instr->opcode() == HloOpcode::kParameter) {
      param_is_double_[instr->parameter_number()] = true;
    }
    promoted_types[instr] = promoted_type;

    // Allocate space for normal array instructions.
    size_t size_bytes = primitive_util::ByteWidth(promoted_type) *
                        ShapeUtil::ElementsIn(instr->shape()) * batch_size;
    size_t value_offset = AllocateValueBuffer(
        promoted_type, ShapeUtil::ElementsIn(instr->shape()), batch_size,
        current_offset);

    if (instr->shape().IsTuple()) {
      VLOG(1) << "Cannot linearize: Tuples are only supported for kTuple and "
                 "kGetTupleElement. Found: "
              << instr->ToString();
      return absl::UnimplementedError(
          "Tuples are only supported for kTuple and kGetTupleElement");
    }

    RecordInstructionOffset(instr, value_offset);

    // Handle parameters. If they correspond to deferred operands, use their
    // already allocated offsets.
    if (instr->opcode() == HloOpcode::kParameter) {
      int param_no = instr->parameter_number();
      auto it = param_to_offset.find(param_no);
      if (it != param_to_offset.end()) {
        size_t val_offset = it->second;

        param_slots_[param_no] = ParamSlot{
            val_offset, size_bytes,
            static_cast<size_t>(primitive_util::ByteWidth(promoted_type))};

        *instruction_offsets_.at(instr).mutable_element({}) = val_offset;

        continue;
      }

      param_slots_[param_no] = ParamSlot{
          value_offset, size_bytes,
          static_cast<size_t>(primitive_util::ByteWidth(promoted_type))};
      continue;
    }

    // Create a new step for this instruction.
    Step step;
    step.opcode = instr->opcode();
    step.type = promoted_type;
    step.result_offset = value_offset;
    step.element_count = ShapeUtil::ElementsIn(instr->shape()) * batch_size;
    step.batch_size = batch_size;

    for (const HloInstruction* operand : instr->operands()) {
      step.operand_offsets.push_back(
          instruction_offsets_.at(operand).element({}));
      step.operand_types.push_back(promoted_types[operand]);
    }

    // Find and populate the execution function for this step.
    TF_RETURN_IF_ERROR(PopulateStepExecuteFn(step, instr, promoted_type));

    if (step.execute_fn) {
      steps_.push_back(step);
    } else {
      VLOG(1) << "Cannot linearize: Unsupported op in interpreter: "
              << HloOpcodeString(instr->opcode()) << " for type "
              << primitive_util::LowercasePrimitiveTypeName(step.type);
      return absl::UnimplementedError(absl::StrCat(
          "Unsupported op in interpreter: ", HloOpcodeString(instr->opcode())));
    }
  }

  for (int64_t i = 0; i < computation->num_parameters(); ++i) {
    if (!param_slots_[i].has_value()) {
      const HloInstruction* param = computation->parameter_instruction(i);
      size_t size_bytes = ShapeUtil::ByteSizeOf(param->shape()) * batch_size;
      current_offset = xla::RoundUpTo(
          current_offset,
          AlignmentOfPrimitiveType(param->shape().element_type()));
      param_slots_[i] =
          ParamSlot{current_offset, size_bytes,
                    static_cast<size_t>(ShapeUtil::ByteSizeOf(param->shape()))};
      current_offset += size_bytes;
    }
  }

  const HloInstruction* root = computation->root_instruction();
  const ShapeTree<size_t>& root_offsets = instruction_offsets_.at(root);
  root_offsets.ForEachElement([&](const ShapeIndex& index, size_t offset) {
    if (ShapeUtil::IsLeafIndex(root->shape(), index)) {
      const Shape& sub_shape = ShapeUtil::GetSubshape(root->shape(), index);
      size_t size_bytes = ShapeUtil::ByteSizeOf(sub_shape);
      result_slots_.push_back({offset, size_bytes});
    }
  });

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<LinearizedInterpreter>>
LinearizedInterpreter::Build(
    const HloComputation* computation,
    absl::Span<const HloInstruction* const> deferred_instructions,
    LeafLiteralResolver resolver, const OpRegistry& op_registry,
    const absl::flat_hash_map<int, const HloInstruction*>& param_to_operand,
    PromotionPolicy promotion_policy, int batch_size) {
  if (batch_size <= 0) {
    return absl::InvalidArgumentError("batch_size must be > 0");
  }
  if (batch_size > kMaxBatchSize) {
    return absl::InvalidArgumentError(
        absl::StrCat("batch_size must be <= ", kMaxBatchSize));
  }
  auto interpreter = absl::WrapUnique(new LinearizedInterpreter());
  interpreter->batch_size_ = batch_size;
  interpreter->op_registry_ = op_registry;
  interpreter->computation_ = computation;

  size_t current_offset = 0;
  absl::flat_hash_map<const HloInstruction*, size_t> operand_value_offsets;

  // Process deferred ops first.
  for (const HloInstruction* instr : deferred_instructions) {
    auto [it, inserted] = operand_value_offsets.try_emplace(instr);
    if (inserted) {
      TF_ASSIGN_OR_RETURN(size_t val_offset,
                          ProcessDeferredOpChain(interpreter.get(), instr,
                                                 resolver, current_offset));
      it->second = val_offset;
    }
  }

  absl::flat_hash_map<int, size_t> param_to_offset;
  for (const auto& [param_idx, instr] : param_to_operand) {
    auto it = operand_value_offsets.find(instr);
    if (it != operand_value_offsets.end()) {
      param_to_offset[param_idx] = it->second;
    }
  }

  TF_RETURN_IF_ERROR(interpreter->LinearizeComputation(
      computation, batch_size, param_to_offset, promotion_policy,
      current_offset));

  interpreter->scratchpad_size_ =
      xla::RoundUpTo(current_offset, sizeof(double));
  return std::move(interpreter);
}

}  // namespace xla
