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

#include "xla/hlo/evaluator/hlo_evaluator_interpreter_deferred_ops.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/evaluator/hlo_evaluator_interpreter.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"

namespace xla {

using ::xla::LinearizedInterpreter;

namespace {
size_t AllocateIndexBuffer(const Shape& shape, int batch_size,
                           size_t& current_offset) {
  current_offset = xla::RoundUpTo(current_offset, alignof(int64_t));
  size_t index_size = shape.dimensions().size() * batch_size * sizeof(int64_t);
  size_t offset = current_offset;
  current_offset += index_size;
  return offset;
}
}  // namespace

void DeferredOpRegistry::Register(HloOpcode opcode,
                                  DeferredOpHandlerFn handler) {
  registry_[opcode] = std::move(handler);
}

absl::StatusOr<DeferredOpResult> DeferredOpRegistry::Process(
    LinearizedInterpreter* interpreter, const HloInstruction* instr,
    size_t input_offset, std::optional<size_t> result_offset,
    LinearizedInterpreter::LeafLiteralResolver resolver,
    size_t& current_offset) const {
  auto it = registry_.find(instr->opcode());
  if (it != registry_.end()) {
    return it->second(interpreter, instr, input_offset, result_offset, resolver,
                      current_offset);
  }
  return absl::NotFoundError(absl::StrCat("No handler registered for opcode: ",
                                          HloOpcodeString(instr->opcode())));
}

// Executes Slice operation. It takes requested output indices and computes
// the corresponding input indices in the operand, which is the reverse of
// standard HLO execution.
void ExecuteSlice(const LinearizedInterpreter::Step* step,
                  void* scratchpad_base) {
  const auto& data = std::any_cast<const SliceMetadata&>(step->op_metadata);
  // We map from the output shape (result) back to the operand shape (input)
  // due to reverse execution flow.
  const int64_t* result_indices =
      LinearizedInterpreter::Scratchpad::GetPointerFromBase<int64_t>(
          scratchpad_base, step->operand_offsets[0]);
  int64_t* operand_indices =
      LinearizedInterpreter::Scratchpad::GetPointerFromBase<int64_t>(
          scratchpad_base, step->result_offset);

  int rank = data.rank;
  for (int b = 0; b < step->batch_size; ++b) {
    for (int d = 0; d < rank; ++d) {
      operand_indices[b * rank + d] =
          data.starts[d] + result_indices[b * rank + d] * data.strides[d];
    }
  }
}

// Executes Broadcast operation. It takes requested output indices and computes
// the corresponding input indices in the operand, which is the reverse of
// standard HLO execution.
void ExecuteBroadcast(const LinearizedInterpreter::Step* step,
                      void* scratchpad_base) {
  const auto& data = std::any_cast<const BroadcastMetadata&>(step->op_metadata);
  // We map from the output shape (result) back to the operand shape (input)
  // due to reverse execution flow.
  const int64_t* result_indices =
      LinearizedInterpreter::Scratchpad::GetPointerFromBase<int64_t>(
          scratchpad_base, step->operand_offsets[0]);
  int64_t* operand_indices =
      LinearizedInterpreter::Scratchpad::GetPointerFromBase<int64_t>(
          scratchpad_base, step->result_offset);

  int operand_rank = data.operand_rank;
  int result_rank = data.result_rank;
  for (int b = 0; b < step->batch_size; ++b) {
    for (int d = 0; d < operand_rank; ++d) {
      int mapped_dim = data.dimensions[d];
      operand_indices[b * operand_rank + d] =
          result_indices[b * result_rank + mapped_dim];
    }
  }
}

// Processes Slice operation. Avoids recursion by not calling Process
// themselves. Instead, adds a step to the interpreter and returns the next
// operand to be processed by the loop in ProcessDeferredOpChain.
absl::StatusOr<DeferredOpResult> DeferredOpSlice(
    LinearizedInterpreter* interpreter, const HloInstruction* instr,
    size_t input_offset, std::optional<size_t> result_offset,
    LinearizedInterpreter::LeafLiteralResolver resolver,
    size_t& current_offset) {
  const HloInstruction* operand = instr->operand(0);

  interpreter->AddStep(HloOpcode::kSlice, *result_offset, {input_offset},
                       SliceMetadata(instr), &ExecuteSlice);

  return DeferredOpResult{operand, 0, std::nullopt};
}

// Processes Broadcast operation. Avoids recursion by not calling Process
// themselves. Instead, adds a step to the interpreter and returns the next
// operand to be processed by the loop in ProcessDeferredOpChain.
absl::StatusOr<DeferredOpResult> DeferredOpBroadcast(
    LinearizedInterpreter* interpreter, const HloInstruction* instr,
    size_t input_offset, std::optional<size_t> result_offset,
    LinearizedInterpreter::LeafLiteralResolver resolver,
    size_t& current_offset) {
  const HloInstruction* operand = instr->operand(0);

  interpreter->AddStep(HloOpcode::kBroadcast, *result_offset, {input_offset},
                       BroadcastMetadata(instr), &ExecuteBroadcast);

  return DeferredOpResult{operand, 0, std::nullopt};
}

// Processes Iota operation. Avoids recursion by not calling Process themselves.
// Instead, adds a step to the interpreter and returns the next operand to be
// processed by the loop in ProcessDeferredOpChain.
absl::StatusOr<DeferredOpResult> DeferredOpIota(
    LinearizedInterpreter* interpreter, const HloInstruction* instr,
    size_t input_offset, std::optional<size_t> result_offset,
    LinearizedInterpreter::LeafLiteralResolver resolver,
    size_t& current_offset) {
  size_t value_offset = LinearizedInterpreter::AllocateValueBuffer(
      instr->shape().element_type(), 1, interpreter->batch_size(),
      current_offset);

  IotaMetadata iota_data(instr);

  PrimitiveType type = instr->shape().element_type();
  LinearizedInterpreter::Step::ExecuteFn execute_fn = nullptr;
  TF_RETURN_IF_ERROR(primitive_util::PrimitiveTypeSwitch<absl::Status>(
      [&](auto type_constant) -> absl::Status {
        constexpr PrimitiveType kType = decltype(type_constant)::value;
        if constexpr (kType == S32 || kType == F32 || kType == S64 ||
                      kType == BF16 || kType == F64 || kType == PRED) {
          using T = primitive_util::NativeTypeOf<kType>;
          execute_fn = &ExecuteIota<T>;
          return absl::OkStatus();
        }
        return absl::UnimplementedError("Unsupported Iota type");
      },
      type));

  interpreter->AddStep(HloOpcode::kIota, value_offset, {input_offset},
                       std::move(iota_data), execute_fn, type);

  return DeferredOpResult{nullptr, 0, value_offset};
}

// Processes Lookup operation. Avoids recursion by not calling Process
// themselves. Instead, adds a step to the interpreter and returns the next
// operand to be processed by the loop in ProcessDeferredOpChain.
absl::StatusOr<DeferredOpResult> DeferredOpLookup(
    LinearizedInterpreter* interpreter, const HloInstruction* instr,
    size_t input_offset, std::optional<size_t> result_offset,
    LinearizedInterpreter::LeafLiteralResolver resolver,
    size_t& current_offset) {
  size_t value_offset = LinearizedInterpreter::AllocateValueBuffer(
      instr->shape().element_type(), 1, interpreter->batch_size(),
      current_offset);

  LookupMetadata lookup_data(instr, resolver);

  PrimitiveType type = instr->shape().element_type();
  LinearizedInterpreter::Step::ExecuteFn execute_fn = nullptr;
  TF_RETURN_IF_ERROR(primitive_util::PrimitiveTypeSwitch<absl::Status>(
      [&](auto type_constant) -> absl::Status {
        constexpr PrimitiveType kType = decltype(type_constant)::value;
        if constexpr (kType == S32 || kType == F32 || kType == S64 ||
                      kType == BF16 || kType == F64 || kType == PRED) {
          using T = primitive_util::NativeTypeOf<kType>;
          execute_fn = &ExecuteLookup<T>;
          return absl::OkStatus();
        }
        return absl::UnimplementedError("Unsupported Lookup type");
      },
      type));

  interpreter->AddStep(std::nullopt, value_offset, {input_offset},
                       std::move(lookup_data), execute_fn, type);

  return DeferredOpResult{nullptr, 0, value_offset};
}

absl::StatusOr<size_t> ProcessDeferredOpChain(
    LinearizedInterpreter* interpreter, const HloInstruction* instr,
    LinearizedInterpreter::LeafLiteralResolver resolver,
    size_t& current_offset) {
  const DeferredOpRegistry& registry = GetDefaultDeferredOpRegistry();

  const HloInstruction* current_instr = instr;

  size_t current_input_offset = AllocateIndexBuffer(
      instr->shape(), interpreter->batch_size(), current_offset);
  interpreter->RecordInstructionOffset(instr, current_input_offset);

  while (current_instr != nullptr) {
    if (current_instr->opcode() == HloOpcode::kSlice ||
        current_instr->opcode() == HloOpcode::kBroadcast) {
      const HloInstruction* operand = current_instr->operand(0);
      size_t next_input_offset = AllocateIndexBuffer(
          operand->shape(), interpreter->batch_size(), current_offset);
      interpreter->RecordInstructionOffset(operand, next_input_offset);

      auto status_or_result =
          registry.Process(interpreter, current_instr, current_input_offset,
                           next_input_offset, resolver, current_offset);

      if (!status_or_result.ok()) {
        return status_or_result.status();
      }

      current_input_offset = next_input_offset;
      current_instr = operand;
    } else {
      auto status_or_result =
          registry.Process(interpreter, current_instr, current_input_offset,
                           std::nullopt, resolver, current_offset);

      if (!status_or_result.ok()) {
        if (absl::IsNotFound(status_or_result.status())) {
          // Fallback to Materialized lookup
          size_t value_offset = interpreter->AllocateValueBuffer(
              current_instr->shape().element_type(), 1,
              interpreter->batch_size(), current_offset);

          LookupMetadata lookup_data(current_instr, resolver);

          PrimitiveType type = current_instr->shape().element_type();
          LinearizedInterpreter::Step::ExecuteFn execute_fn = nullptr;
          TF_RETURN_IF_ERROR(primitive_util::PrimitiveTypeSwitch<absl::Status>(
              [&](auto type_constant) -> absl::Status {
                constexpr PrimitiveType kType = decltype(type_constant)::value;
                if constexpr (kType == S32 || kType == F32 || kType == S64 ||
                              kType == BF16 || kType == F64 || kType == PRED) {
                  using T = primitive_util::NativeTypeOf<kType>;
                  execute_fn = &ExecuteLookup<T>;
                  return absl::OkStatus();
                }
                return absl::UnimplementedError("Unsupported Lookup type");
              },
              type));

          interpreter->AddStep(std::nullopt, value_offset,
                               {current_input_offset}, std::move(lookup_data),
                               execute_fn, type);

          return value_offset;
        }
        return status_or_result.status();
      }

      const DeferredOpResult& result = status_or_result.value();
      return *result.val_offset;
    }
  }

  return absl::InternalError(
      "ProcessDeferredOpChain ended without terminal node");
}

const DeferredOpRegistry& GetDefaultDeferredOpRegistry() {
  static absl::NoDestructor<DeferredOpRegistry> registry([] {
    DeferredOpRegistry r;
    r.Register(HloOpcode::kSlice, DeferredOpSlice);
    r.Register(HloOpcode::kBroadcast, DeferredOpBroadcast);
    r.Register(HloOpcode::kIota, DeferredOpIota);
    r.Register(HloOpcode::kParameter, DeferredOpLookup);
    return r;
  }());
  return *registry;
}

}  // namespace xla
