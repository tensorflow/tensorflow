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
#ifndef XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_H_
#define XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_H_

#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

inline DimensionVector MakeDimMultipliers(const Shape& shape) {
  absl::Span<const int64_t> minor_to_major = LayoutUtil::MinorToMajor(
      shape.has_layout() ? shape : LayoutUtil::GetWithDefaultLayout(shape));
  DimensionVector v(shape.dimensions().size());
  int64_t scale = 1;
  for (auto dim : minor_to_major) {
    v[dim] = scale;
    scale *= shape.dimensions(dim);
  }
  return v;
}

// LinearizedInterpreter evaluates a subset of HLO instructions by linearizing
// the computation graph into a sequence of flat steps. This approach avoids the
// overhead of recursive evaluation and repeated index delinearization
// (converting linear indices to multi-dimensional indices and back) by
// processing elements in a flat buffer. It serves as a pure interpreter for
// flattened computations, executing scheduled steps on a scratchpad.
// Higher-level logic, such as reduction loops, is handled by dedicated runners
// (like ReduceRunner) that use this interpreter to process elements in batches.
class LinearizedInterpreter {
 public:
  using LeafLiteralResolver =
      absl::FunctionRef<const Literal&(const HloInstruction*)>;

  using PromotionPolicy = absl::FunctionRef<PrimitiveType(
      const HloInstruction*,
      const absl::flat_hash_map<const HloInstruction*, PrimitiveType>&)>;

  using StepOpMetadata = std::any;

  struct Step {
    using ExecuteFn = void (*)(const Step*, void* /*scratchpad_base*/);
    ExecuteFn execute_fn = nullptr;
    std::optional<HloOpcode> opcode;
    PrimitiveType type;
    size_t result_offset;
    std::vector<size_t> operand_offsets;
    std::vector<PrimitiveType> operand_types;
    size_t element_count = 0;
    const void* aux_data = nullptr;
    int batch_size = 1;
    StepOpMetadata op_metadata;
  };

  class OpRegistry {
   public:
    using PopulatorFn = std::function<absl::Status(Step&, const HloInstruction*,
                                                   PrimitiveType)>;
    void Register(HloOpcode opcode, PopulatorFn populator);
    absl::Status Populate(Step& step, const HloInstruction* instr,
                          PrimitiveType promoted_type) const;

   private:
    absl::flat_hash_map<HloOpcode, PopulatorFn> registry_;
  };

  static const OpRegistry& GetDefaultOpRegistry();

  // Scratchpad is the unified working memory for the interpreter. It holds
  // operand pointers, result pointers, saved accumulators for reduction edge
  // cases, and temporary indices. By consolidating all intermediate buffers
  // into a single allocation (or a few vectors), it minimizes heap allocations
  // during interpretation.
  struct ResultSlot {
    size_t offset;
    size_t size;
  };

  class Scratchpad {
   public:
    // size is in bytes.
    explicit Scratchpad(size_t size_bytes) : buffer_(size_bytes) {}
    void* data() { return buffer_.data(); }

    // Helper function to get a pointer to a type T at the given offset in the
    // scratchpad.
    template <typename T>
    static T* GetPointerFromBase(void* base, size_t offset) {
      return reinterpret_cast<T*>(static_cast<char*>(base) + offset);  // NOLINT
    }

    // Helper function to get a pointer to an array of int64_t indices in the
    // scratchpad.
    int64_t* GetIndicesPointer(size_t offset, int batch_idx, int rank) {
      return GetPointerFromBase<int64_t>(
          data(), offset + batch_idx * rank * sizeof(int64_t));
    }

   private:
    friend class LinearizedInterpreter;

    std::vector<char> buffer_;
  };

  static constexpr int kMaxBatchSize = 64;

  struct ParamSlot {
    size_t offset;
    size_t size;
    size_t elem_size;
  };

  // Builds a LinearizedInterpreter for the given computation.
  // resolver is used to fetch literals for operands that are not deferred.
  // batch_size determines how many elements are processed together (must be <=
  // kMaxBatchSize).
  static absl::StatusOr<std::unique_ptr<LinearizedInterpreter>> Build(
      const HloComputation* computation,
      absl::Span<const HloInstruction* const> deferred_instructions,
      LeafLiteralResolver resolver, const OpRegistry& op_registry,
      const absl::flat_hash_map<int, const HloInstruction*>& param_to_operand,
      PromotionPolicy promotion_policy =
          [](const HloInstruction* instr,
             const absl::flat_hash_map<const HloInstruction*, PrimitiveType>&) {
            return instr->shape().element_type();
          },
      int batch_size = 1);

  Scratchpad CreateScratchpad() const;

  // Executes the scheduled steps on the scratchpad.
  void ExecuteSteps(Scratchpad& scratchpad) const;

  // Accessors
  size_t scratchpad_size() const { return scratchpad_size_; }
  int batch_size() const { return batch_size_; }
  const HloComputation* computation() const { return computation_; }
  size_t GetInstructionOffset(const HloInstruction* instr) const {
    return instruction_offsets_.at(instr).element({});
  }
  const std::vector<std::optional<ParamSlot>>& param_slots() const {
    return param_slots_;
  }
  const std::vector<ResultSlot>& result_slots() const { return result_slots_; }
  const std::vector<bool>& param_is_double() const { return param_is_double_; }

  // ProcessDeferredOpChain is a friend free function defined in
  // hlo_evaluator_interpreter_deferred_ops.h
  friend absl::StatusOr<size_t> ProcessDeferredOpChain(
      LinearizedInterpreter* interpreter, const HloInstruction* instr,
      LeafLiteralResolver resolver, size_t& current_offset);

  // Adds a step to the execution plan.
  void AddStep(std::optional<HloOpcode> opcode, size_t result_offset,
               absl::Span<const size_t> operand_offsets,
               StepOpMetadata op_metadata, Step::ExecuteFn execute_fn,
               PrimitiveType type = PRIMITIVE_TYPE_INVALID);

  // Patches the result offset of a previously added step.

  // Allocates a buffer in the scratchpad for values.
  static size_t AllocateValueBuffer(PrimitiveType type, size_t element_count,
                                    int batch_size, size_t& current_offset);

  class Ops;

 private:
  LinearizedInterpreter() = default;

  // Populates the execution function for a step based on the HLO instruction.
  absl::Status PopulateStepExecuteFn(Step& step, const HloInstruction* instr,
                                     PrimitiveType promoted_type) const;

  // Copies computed results from the flat batch buffer back to the final
  // output literals, handling potential layout differences.
  absl::Status CopyBackResults(
      absl::Span<Literal> results, const Scratchpad& scratchpad,
      const std::vector<std::vector<char>>& temp_results,
      const std::vector<int64_t>& output_indices, int64_t start_elem,
      int actual_batch_size) const;

  // Records the scratchpad offset for the result of an instruction.
  void RecordInstructionOffset(const HloInstruction* instr, size_t offset);

  // Populates a step for a simple binary operation.
  template <typename Op>
  static absl::Status PopulateSimpleBinary(Step& step, const char* op_name);

  // Populates a step for a logical binary operation.
  template <typename Op>
  static absl::Status PopulateLogicalBinary(Step& step, const char* op_name);

  // Flattens the computation into a sequence of steps.
  absl::Status LinearizeComputation(
      const HloComputation* computation, int batch_size,
      const absl::flat_hash_map<int, size_t>& param_to_offset,
      PromotionPolicy promotion_policy, size_t& current_offset);

  std::vector<Step> steps_;
  size_t scratchpad_size_ = 0;
  OpRegistry op_registry_;

  std::vector<bool> param_is_double_;

  // Mapping from instruction to its allocated offsets in the scratchpad.
  absl::flat_hash_map<const HloInstruction*, ShapeTree<size_t>>
      instruction_offsets_;

  std::vector<std::optional<ParamSlot>> param_slots_;

  // Result slots, one per leaf index of the result shape tree.
  std::vector<ResultSlot> result_slots_;

  const HloComputation* computation_ = nullptr;
  int batch_size_ = 1;
  int output_rank_ = 0;
};

}  // namespace xla

#endif  // XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_H_
