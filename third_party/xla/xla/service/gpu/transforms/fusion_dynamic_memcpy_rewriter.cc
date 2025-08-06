/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/fusion_dynamic_memcpy_rewriter.h"

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/copy.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class DynamicOffsetEvaluator {
 public:
  // Evaluates the clamped array index for the given offset.
  absl::StatusOr<int64_t> EvaluateArrayIndexForOffset(
      const DynamicMemcpyThunk::MemcpyDescriptor::DynamicOffset& offset,
      const WhileLoopBackendConfig& loop_config, int64_t iteration) {
    TF_ASSIGN_OR_RETURN(auto call_stack, ComputeCallStack(offset));

    // Walk up the call stack and compute the required parameter's values at
    // each step, using them as the substitutions for the next call. By
    // definition, the first call can only depend on the induction variable.
    TF_ASSIGN_OR_RETURN(
        auto substitutions,
        GetInductionVariableSubstitutions(offset, loop_config, iteration));
    HloEvaluator evaluator(/*max_loop_iterations=*/0);
    for (auto it = call_stack.rbegin(), e = call_stack.rend(); it != e; ++it) {
      const HloInstruction* caller = *it;
      VLOG(3) << "Evaluating required operands of caller " << caller->name()
              << ".";
      if (VLOG_IS_ON(4)) {
        VLOG(4) << "Current substitutions:";
        for (auto [instr, value] : substitutions) {
          VLOG(4) << "  " << instr->name() << " -> " << value->ToString();
        }
      }
      absl::flat_hash_map<const HloInstruction*, const LiteralBase*>
          next_substitutions;
      for (auto [parameter, operand] :
           GetRequiredParametersAndOperands(offset, caller)) {
        // Only compute the value if we didn't already need it for a different
        // offset.
        if (!known_values_.contains(operand)) {
          TF_ASSIGN_OR_RETURN(
              known_values_[operand],
              evaluator.Evaluate(operand, {}, true, substitutions));
        }
        next_substitutions[parameter] = &known_values_[operand];
      }

      std::swap(substitutions, next_substitutions);
    }

    // We now have the parameter values for the innermost call, so we can
    // compute the offset.
    TF_ASSIGN_OR_RETURN(
        auto array_index_literal,
        evaluator.Evaluate(offset.offset, {}, true, substitutions));

    std::optional<int64_t> array_index =
        LiteralUtil::LiteralAsScalarInt64(array_index_literal);
    if (!array_index) {
      return absl::InternalError("Failed to evaluate offset");
    }

    int64_t clamped_index =
        std::max<int64_t>(0, std::min(*array_index, offset.dimension_size - 1));
    VLOG(3) << "Computed dynamic array index " << clamped_index << ".";

    return clamped_index;
  }

 private:
  // Computes the call stack between `offset`'s while loop and the derived
  // value. Typically, there will be up to three items in the stack: 1) a
  // fusion, 2) optionally an async-start, 3) optionally a command buffer. The
  // while loop instruction is not included.
  static absl::StatusOr<absl::InlinedVector<HloInstruction*, 4>>
  ComputeCallStack(
      const DynamicMemcpyThunk::MemcpyDescriptor::DynamicOffset& offset) {
    VLOG(3) << "Computing call stack for " << offset.offset->name() << ".";
    const HloComputation* current_computation = offset.offset->parent();
    const HloComputation* while_body = offset.induction_variable->parent();

    absl::InlinedVector<HloInstruction*, 4> call_stack;
    while (current_computation && current_computation != while_body) {
      VLOG(3) << "Current computation: " << current_computation->name() << ".";
      auto callers = current_computation->caller_instructions();

      // If there isn't a single caller, the thunk was not constructed
      // correctly.
      TF_RET_CHECK(callers.size() == 1);

      call_stack.push_back(callers.front());
      current_computation = callers.front()->parent();
    }

    // If we didn't arrive at the while body, the thunk was not constructed
    // correctly.
    TF_RET_CHECK(current_computation == while_body);
    return call_stack;
  }

  // Returns the pairs of {computation parameter, computation caller operand}
  // that are required in the given computation to compute the given offset.
  static absl::InlinedVector<
      std::pair<const HloInstruction*, const HloInstruction*>, 1>
  GetRequiredParametersAndOperands(
      const DynamicMemcpyThunk::MemcpyDescriptor::DynamicOffset& offset,
      const HloInstruction* caller) {
    absl::InlinedVector<std::pair<const HloInstruction*, const HloInstruction*>,
                        1>
        result;
    const HloComputation* callee = caller->called_computations().front();
    if (auto maybe_required = offset.required_parameters.find(callee);
        maybe_required != offset.required_parameters.end()) {
      const auto& required_parameters = maybe_required->second;
      for (int i = 0; i < required_parameters.size(); ++i) {
        if (required_parameters[i]) {
          result.push_back(
              {callee->parameter_instruction(i), caller->operand(i)});
        }
      }
    }
    return result;
  }

  absl::StatusOr<absl::flat_hash_map<const HloInstruction*, const LiteralBase*>>
  GetInductionVariableSubstitutions(
      const DynamicMemcpyThunk::MemcpyDescriptor::DynamicOffset& offset,
      const WhileLoopBackendConfig& loop_config, int64_t iteration) {
    // Set the value of the induction variable, if it's not known yet.
    if (!known_values_.contains(offset.induction_variable)) {
      int64_t induction_variable =
          loop_config.known_init_step().init() +
          iteration * loop_config.known_init_step().step();

      Literal induction_variable_literal(offset.induction_variable->shape());
      TF_RETURN_IF_ERROR(
          induction_variable_literal.SetIntegralAsS64({}, induction_variable));
      known_values_[offset.induction_variable] =
          std::move(induction_variable_literal);
    }

    return {{{offset.induction_variable,
              &known_values_.at(offset.induction_variable)}}};
  }

  absl::node_hash_map<const HloInstruction*, Literal> known_values_;
};

absl::StatusOr<int64_t> EvaluateDynamicOffsets(
    absl::Span<const DynamicMemcpyThunk::MemcpyDescriptor::DynamicOffset>
        offsets,
    const WhileLoopBackendConfig& loop_config, int64_t iteration) {
  int64_t offset_sum = 0;
  DynamicOffsetEvaluator evaluator;
  for (const auto& offset : offsets) {
    TF_ASSIGN_OR_RETURN(
        int64_t clamped_index,
        evaluator.EvaluateArrayIndexForOffset(offset, loop_config, iteration));
    offset_sum += clamped_index * offset.byte_stride;
  }
  return offset_sum;
}

absl::StatusOr<WhileLoopBackendConfig> GetLoopConfig(
    const DynamicMemcpyThunk::MemcpyDescriptor& descriptor) {
  const HloInstruction* loop = nullptr;
  for (const auto& offsets :
       {descriptor.src_dynamic_offsets, descriptor.dst_dynamic_offsets}) {
    for (const auto& offset : offsets) {
      if (loop == nullptr) {
        loop = offset.while_loop;
      } else if (loop != offset.while_loop) {
        // The while loop must be the same in all dynamic offsets. This should
        // always be the case.
        return absl::InternalError(
            "Loops in dynamic memcpy descriptor are not consistent.");
      }
    }
  }

  if (!loop) {
    return absl::InternalError("Did not find a loop.");
  }

  TF_ASSIGN_OR_RETURN(auto config,
                      loop->backend_config<WhileLoopBackendConfig>());
  if (!config.has_known_init_step() || !config.has_known_trip_count()) {
    return absl::InternalError(
        "Loop is not a for loop with a static trip count.");
  }
  return config;
}

absl::Status ComputeOffsetsForLoop(
    absl::Span<const DynamicMemcpyThunk::MemcpyDescriptor::DynamicOffset>
        dynamic_offsets,
    int64_t static_offset, const WhileLoopBackendConfig& loop_config,
    tsl::protobuf::RepeatedField<int64_t>* output_offsets) {
  for (int64_t i = 0; i < loop_config.known_trip_count().n(); ++i) {
    TF_ASSIGN_OR_RETURN(
        auto dynamic_offset,
        EvaluateDynamicOffsets(dynamic_offsets, loop_config, i));
    output_offsets->Add(static_offset + dynamic_offset);
  }
  return absl::OkStatus();
}

absl::Status SetLoopMemcpyConfig(
    const DynamicMemcpyThunk::MemcpyDescriptor& descriptor,
    DynamicMemcpyConfig* config) {
  TF_ASSIGN_OR_RETURN(auto loop_config, GetLoopConfig(descriptor));

  config->set_depends_on_loop(true);
  TF_RETURN_IF_ERROR(ComputeOffsetsForLoop(
      descriptor.src_dynamic_offsets, descriptor.src_byte_static_offset,
      loop_config, config->mutable_src_offset_bytes()));
  TF_RETURN_IF_ERROR(ComputeOffsetsForLoop(
      descriptor.dst_dynamic_offsets, descriptor.dst_byte_static_offset,
      loop_config, config->mutable_dst_offset_bytes()));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> FusionDynamicMemcpyRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool has_changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    if (!computation->IsFusionComputation()) {
      continue;
    }

    HloFusionInstruction* fusion =
        ::xla::Cast<HloFusionInstruction>(computation->FusionInstruction());
    auto descriptor =
        DynamicMemcpyFusion::GetMemcpyDescriptorForFusion(*fusion);
    if (!descriptor) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(auto backend_config,
                        fusion->backend_config<GpuBackendConfig>());
    auto* fusion_config = backend_config.mutable_fusion_backend_config();
    fusion_config->set_kind(std::string(kDynamicMemcpyFusionKind));
    auto* memcpy_config = fusion_config->mutable_dynamic_memcpy_config();

    if (descriptor->src_dynamic_offsets.size() +
            descriptor->dst_dynamic_offsets.size() ==
        0) {
      memcpy_config->add_src_offset_bytes(descriptor->src_byte_static_offset);
      memcpy_config->add_dst_offset_bytes(descriptor->dst_byte_static_offset);
    } else {
      auto status = SetLoopMemcpyConfig(*descriptor, memcpy_config);
      if (!status.ok()) {
        LOG(INFO) << "Failed to produce memcpy configuration: " << status;
        continue;
      }
    }

    TF_RETURN_IF_ERROR(fusion->set_backend_config(backend_config));
    has_changed = true;
  }

  return has_changed;
}

}  // namespace gpu
}  // namespace xla
