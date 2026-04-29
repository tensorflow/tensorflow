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

// Deferred operations allow the LinearizedInterpreter to efficiently evaluate
// certain HLO operations by avoiding materialization of intermediate tensors.
// Instead of reading the Literal values of the operands of an operation, the
// HloEvaluator defers their evaluation, allowing the interpreter to process
// the operations as part of the main interpreter loop.

#ifndef XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_DEFERRED_OPS_H_
#define XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_DEFERRED_OPS_H_

#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/hlo/evaluator/hlo_evaluator_interpreter.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Result of tracing a deferred operation.
struct DeferredOpResult {
  // The operand to continue tracing with. Null if this is a terminal operation
  // (e.g., Iota or parameter lookup).
  const HloInstruction* next_operand = nullptr;
  // The step index created in the interpreter for this operation. Used for
  // patching the result offset in the next iteration of the trace loop.
  size_t step_idx = 0;
  // The offset in the value buffer if this was a terminal operation that
  // produced a final value.
  std::optional<size_t> val_offset;
};

// Function type for processing a specific HLO instruction as a deferred
// operation. It returns a DeferredOpResult to guide the iterative loop in
// ProcessDeferredOpChain.
using DeferredOpHandlerFn = std::function<absl::StatusOr<DeferredOpResult>(
    LinearizedInterpreter* interpreter, const HloInstruction* instr,
    size_t input_offset, std::optional<size_t> result_offset,
    LinearizedInterpreter::LeafLiteralResolver resolver,
    size_t& current_offset)>;

// Registry mapping HloOpcode to DeferredOpHandlerFn to allow dynamic lookup
// of handlers for deferred operations.
class DeferredOpRegistry {
 public:
  void Register(HloOpcode opcode, DeferredOpHandlerFn handler);
  absl::StatusOr<DeferredOpResult> Process(
      LinearizedInterpreter* interpreter, const HloInstruction* instr,
      size_t input_offset, std::optional<size_t> result_offset,
      LinearizedInterpreter::LeafLiteralResolver resolver,
      size_t& current_offset) const;

 private:
  absl::flat_hash_map<HloOpcode, DeferredOpHandlerFn> registry_;
};

const DeferredOpRegistry& GetDefaultDeferredOpRegistry();

absl::StatusOr<size_t> ProcessDeferredOpChain(
    LinearizedInterpreter* interpreter, const HloInstruction* instr,
    LinearizedInterpreter::LeafLiteralResolver resolver,
    size_t& current_offset);

// Metadata structs moved from hlo_evaluator_interpreter.h
// Metadata for Iota operation, extracting dimension and rank needed to
// compute iota values during execution.
struct IotaMetadata {
  explicit IotaMetadata(const HloInstruction* instr) {
    const auto* iota = Cast<HloIotaInstruction>(instr);
    dimension = iota->iota_dimension();
    rank = instr->shape().dimensions().size();
  }
  int64_t dimension;
  int rank;
};

// Metadata for Broadcast operation, storing mapping dimensions and ranks
// to map output indices back to input indices.
struct BroadcastMetadata {
  explicit BroadcastMetadata(const HloInstruction* instr) {
    const auto* broadcast = Cast<HloBroadcastInstruction>(instr);
    dimensions.assign(broadcast->dimensions().begin(),
                      broadcast->dimensions().end());
    result_rank = instr->shape().dimensions().size();
    operand_rank = broadcast->operand(0)->shape().dimensions().size();
  }
  DimensionVector dimensions;
  int result_rank;
  int operand_rank;
};

// Metadata for Slice operation, storing starts and strides to map output
// indices back to input indices.
struct SliceMetadata {
  explicit SliceMetadata(const HloInstruction* instr) {
    const auto* slice = Cast<HloSliceInstruction>(instr);
    starts.assign(slice->slice_starts().begin(), slice->slice_starts().end());
    strides.assign(slice->slice_strides().begin(),
                   slice->slice_strides().end());
    rank = instr->shape().dimensions().size();
  }
  DimensionVector starts;
  DimensionVector strides;
  int rank;
};

// Metadata for Leaf Literal Lookup, storing literal reference and multipliers
// to compute linear index from multidimensional indices.
struct LookupMetadata {
  LookupMetadata(const HloInstruction* instr,
                 LinearizedInterpreter::LeafLiteralResolver resolver) {
    literal = &resolver(instr);
    rank = instr->shape().dimensions().size();
    raw_data = literal->untyped_data();
    dim_multipliers = MakeDimMultipliers(literal->shape());
  }
  const Literal* literal;
  int rank;
  const void* raw_data = nullptr;
  DimensionVector dim_multipliers;
};

// Execution functions for deferred ops
void ExecuteSlice(const LinearizedInterpreter::Step* step,
                  void* scratchpad_base);
void ExecuteBroadcast(const LinearizedInterpreter::Step* step,
                      void* scratchpad_base);

// Executes Iota operation. It takes requested output indices and computes
// the corresponding iota value (which is the index along the iota dimension).
template <typename T>
void ExecuteIota(const LinearizedInterpreter::Step* step,
                 void* scratchpad_base) {
  const auto& data = std::any_cast<const IotaMetadata&>(step->op_metadata);
  const int64_t* indices =
      LinearizedInterpreter::Scratchpad::GetPointerFromBase<int64_t>(
          scratchpad_base, step->operand_offsets[0]);
  T* result = LinearizedInterpreter::Scratchpad::GetPointerFromBase<T>(
      scratchpad_base, step->result_offset);

  int rank = data.rank;
  for (int b = 0; b < step->batch_size; ++b) {
    int64_t idx = indices[b * rank + data.dimension];
    result[b] = static_cast<T>(idx);
  }
}

// Executes Lookup operation. It takes requested output indices, computes the
// linear index, and looks up the value in the source literal.
template <typename T>
void ExecuteLookup(const LinearizedInterpreter::Step* step,
                   void* scratchpad_base) {
  const auto& data = std::any_cast<const LookupMetadata&>(step->op_metadata);
  const int64_t* indices =
      LinearizedInterpreter::Scratchpad::GetPointerFromBase<int64_t>(
          scratchpad_base, step->operand_offsets[0]);
  T* result = LinearizedInterpreter::Scratchpad::GetPointerFromBase<T>(
      scratchpad_base, step->result_offset);

  int rank = data.rank;
  const T* raw_data = static_cast<const T*>(data.raw_data);

  int b = 0;
  for (; b <= step->batch_size - 4; b += 4) {
    int64_t linear_index0 = 0;
    int64_t linear_index1 = 0;
    int64_t linear_index2 = 0;
    int64_t linear_index3 = 0;
    for (int d = 0; d < rank; ++d) {
      int64_t m = data.dim_multipliers[d];
      linear_index0 += indices[b * rank + d] * m;
      linear_index1 += indices[(b + 1) * rank + d] * m;
      linear_index2 += indices[(b + 2) * rank + d] * m;
      linear_index3 += indices[(b + 3) * rank + d] * m;
    }
    result[b] = raw_data[linear_index0];
    result[b + 1] = raw_data[linear_index1];
    result[b + 2] = raw_data[linear_index2];
    result[b + 3] = raw_data[linear_index3];
  }

  for (; b < step->batch_size; ++b) {
    int64_t linear_index = 0;
    for (int d = 0; d < rank; ++d) {
      int64_t m = data.dim_multipliers[d];
      linear_index += indices[b * rank + d] * m;
    }
    result[b] = raw_data[linear_index];
  }
}

}  // namespace xla

#endif  // XLA_HLO_EVALUATOR_HLO_EVALUATOR_INTERPRETER_DEFERRED_OPS_H_
