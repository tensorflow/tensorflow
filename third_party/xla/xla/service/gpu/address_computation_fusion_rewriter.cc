/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/service/gpu/address_computation_fusion_rewriter.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

bool IsNoOp(const HloInstruction* hlo) {
  return HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kTuple,
                          HloOpcode::kGetTupleElement>(hlo);
}

bool IsCustomCall(const HloInstruction* hlo, absl::string_view platform_name) {
  auto* custom_call = DynCast<HloCustomCallInstruction>(hlo);
  if (custom_call == nullptr) return false;

  // TODO(vuson): properly handle token by following
  // `LhloDialectEmitter::EmitCustomCallOp`'s `CreateOperands` logic for
  // `LhloDialectEmitter::EmitFusionOp`'s `RewriteFusionOperand`
  if (custom_call->shape().IsTuple() &&
      absl::c_any_of(
          custom_call->shape().tuple_shapes(),
          [&](const Shape& sub_shape) { return sub_shape.IsToken(); }))
    return false;

  const std::string call_target_name = custom_call->custom_call_target();

  bool is_ffi_custom_call =
      custom_call->api_version() == CustomCallApiVersion::API_VERSION_TYPED_FFI;

  void* call_target = CustomCallTargetRegistry::Global()->Lookup(
      call_target_name, std::string(platform_name));

  absl::StatusOr<ffi::HandlerRegistration> handler_registration =
      ffi::FindHandler(call_target_name, platform_name);

  // At least one implementation should be available at run time.
  bool found_custom_call = !is_ffi_custom_call && call_target != nullptr;
  bool found_ffi_handler = is_ffi_custom_call && handler_registration.ok();

  return found_custom_call || found_ffi_handler;
}

// Returns true if the slice is 128-byte-aligned. The slice starting
// address is determined by the product of all non-sliced dimensions and an
// offset defined by `slice_starts` of the slice op.
//
// For dynamic cases, we don't have info about the start indices, so we have to
// be conservative by only accepting sliced shapes that have the product of all
// non-sliced dimensions being a multiple of `kXlaAllocatedBufferAlignBytes`.
bool IsAlignedSlice(const Shape& src_shape, const Shape& dst_shape,
                    const HloSliceInstruction* slice) {
  if (!IsContiguousSlice(src_shape, dst_shape)) return false;

  auto strides = ShapeUtil::ByteStrides(dst_shape);
  if (!strides.has_value()) return false;

  for (auto dim : dst_shape.layout().minor_to_major()) {
    if ((strides.value()[dim] % kXlaAllocatedBufferAlignBytes) == 0)
      return true;
    if (dst_shape.dimensions(dim) < src_shape.dimensions(dim)) {
      return (slice != nullptr &&
              ((strides.value()[dim] * slice->slice_starts(dim)) %
                   kXlaAllocatedBufferAlignBytes ==
               0));
    }
  }
  return true;
}

absl::InlinedVector<HloInstruction*, 8> GetSlicedChains(
    const HloInstruction* instr, bool dynamic,
    absl::flat_hash_map<const HloInstruction*, const HloInstruction*>&
        replacement_map) {
  replacement_map[instr] = instr;
  absl::InlinedVector<HloInstruction*, 8> dyn_slice_chains = {
      const_cast<HloInstruction*>(instr)};
  absl::InlinedVector<HloInstruction*, 8> dus_chain;
  auto fusion = HloFusionAdaptor::ForComputation(instr->parent());
  // This set is used to avoid duplicates in the matched results. It contains
  // the matched instructions that we have seen so far.
  absl::flat_hash_set<HloInstruction*> processed_sliced_chain_set;

  const auto& aliasing_pairs =
      Cast<HloCustomCallInstruction>(instr)->output_to_operand_aliasing();
  absl::flat_hash_set<int64_t> aliased_operands;
  for (const auto& pair : aliasing_pairs) {
    aliased_operands.insert(pair.second.first);
  }

  for (auto* operand : instr->operands()) {
    // output_to_operand_aliasing means the operand is to be materialized, which
    // is against the whole idea of address computation fusion. Skip this
    // operand.
    if (aliased_operands.contains(instr->operand_index(operand))) continue;
    absl::InlinedVector<HloInstruction*, 4> maybe_sliced_operand_chain;
    bool slice_found = false;
    auto maybe_slice_adaptor =
        HloFindIf({HloInstructionAdaptor(*operand)}, *fusion, [&](auto node) {
          const HloInstruction* cur = &node.instruction();
          // If the node is a match that has been processed, stop the traversal.
          if (processed_sliced_chain_set.contains(cur)) return true;
          maybe_sliced_operand_chain.push_back(
              const_cast<HloInstruction*>(cur));
          if (dynamic) {
            if (const auto slice_instr =
                    DynCast<HloDynamicSliceInstruction>(cur)) {
              if (IsAlignedSlice(slice_instr->operand(0)->shape(),
                                 slice_instr->shape(), nullptr)) {
                slice_found = true;
                return slice_found;
              }
            }
          } else {
            if (const auto slice_instr = DynCast<HloSliceInstruction>(cur)) {
              if (IsAlignedSlice(slice_instr->operand(0)->shape(),
                                 slice_instr->shape(), slice_instr)) {
                slice_found = true;
                return slice_found;
              }
            }
          }
          // TODO(vuson): lift the first restriction by considering fusing other
          // uses of the operand to reuse the address computation. Only worth it
          // if other uses are also custom calls though.
          return cur->user_count() > 1 || !IsNoOp(cur);
        });
    if (maybe_slice_adaptor == std::nullopt) continue;
    const auto& maybe_slice_instr = maybe_slice_adaptor->instruction();
    if (slice_found ||
        processed_sliced_chain_set.contains(&maybe_slice_instr)) {
      // Even in the case of stopping at a match that has been processed, we
      // still need to add instructions encountered in the sliced operand chain
      // during the latest traversal.
      dyn_slice_chains.insert(dyn_slice_chains.end(),
                              maybe_sliced_operand_chain.begin(),
                              maybe_sliced_operand_chain.end());
      processed_sliced_chain_set.insert(maybe_sliced_operand_chain.begin(),
                                        maybe_sliced_operand_chain.end());
    }
  }

  if (dynamic) {
    for (auto* user : instr->users()) {
      absl::InlinedVector<HloInstruction*, 4> maybe_sliced_user_chain;
      bool dus_found = false;
      auto maybe_dus_adaptor = HloFindIf(
          {HloInstructionAdaptor(*user)}, *fusion,
          [&](auto node) {
            const HloInstruction* cur = &node.instruction();
            // If the node is a match that has been processed, stop the
            // traversal.
            if (processed_sliced_chain_set.contains(cur)) return true;
            maybe_sliced_user_chain.push_back(const_cast<HloInstruction*>(cur));
            if (const auto slice_instr =
                    DynCast<HloDynamicUpdateSliceInstruction>(cur)) {
              if (IsAlignedSlice(slice_instr->shape(),
                                 slice_instr->update()->shape(), nullptr)) {
                dus_found = true;
                replacement_map[instr] = cur;
                return dus_found;
              }
            }
            // TODO(vuson): lift the first restriction by considering fusing
            // other uses of the user to reuse the address computation. Only
            // worth it if other uses are also custom calls though.
            return cur->user_count() > 1 || !IsNoOp(cur);
          },
          /*visit_operands=*/false);
      if (maybe_dus_adaptor == std::nullopt) continue;
      const auto& maybe_dus_instr = maybe_dus_adaptor->instruction();
      if (dus_found || processed_sliced_chain_set.contains(&maybe_dus_instr)) {
        // Even in the case of stopping at a match that has been processed, we
        // still need to add instructions encountered in the sliced user chain
        // during the latest traversal.
        dus_chain.insert(dus_chain.end(), maybe_sliced_user_chain.rbegin(),
                         maybe_sliced_user_chain.rend());
        processed_sliced_chain_set.insert(maybe_sliced_user_chain.begin(),
                                          maybe_sliced_user_chain.end());
      }
    }
  }

  dus_chain.insert(dus_chain.end(), dyn_slice_chains.begin(),
                   dyn_slice_chains.end());
  return dus_chain;
}

absl::InlinedVector<HloInstruction*, 4> GetPatternCaptures(
    absl::Span<HloInstruction* const> matched) {
  absl::InlinedVector<HloInstruction*, 4> captures;

  absl::flat_hash_set<HloInstruction*> instructions_set(matched.begin(),
                                                        matched.end());

  for (HloInstruction* instr : matched) {
    for (HloInstruction* operand : instr->operands()) {
      if (!instructions_set.contains(operand) &&
          absl::c_find(captures, operand) == captures.end()) {
        captures.emplace_back(operand);
      }
    }
  }

  return captures;
}

absl::InlinedVector<HloInstruction*, 8> GetSortedMatched(
    absl::Span<HloInstruction* const> matched) {
  absl::InlinedVector<HloInstruction*, 8> sorted_matched;
  absl::flat_hash_set<HloInstruction*> instructions_set(matched.begin(),
                                                        matched.end());
  absl::flat_hash_set<HloInstruction*> processed_set;
  // Topologically sort `matched`
  for (auto it = matched.rbegin(); it != matched.rend(); ++it) {
    if (processed_set.contains(*it)) continue;
    for (auto* operand : (*it)->operands()) {
      if (!instructions_set.contains(operand)) {
        continue;
      }
      if (!processed_set.contains(operand)) {
        sorted_matched.emplace_back(operand);
        processed_set.insert(operand);
      }
    }
    sorted_matched.emplace_back(*it);
    processed_set.insert(*it);
  }

  return sorted_matched;
}

void CreateRootTuple(HloInstruction* root, HloComputation::Builder& builder) {
  std::vector<HloInstruction*> elements;
  elements.reserve(root->shape().tuple_shapes_size());
  for (size_t i = 0; i < root->shape().tuple_shapes_size(); ++i) {
    if (root->shape().tuple_shapes(i).IsTuple()) {
      HloInstruction* gte = builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(root, i));
      CreateRootTuple(gte, builder);
      elements.push_back(builder.last_added_instruction());
    } else {
      elements.push_back(builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(root, i)));
    }
  }
  builder.AddInstruction(HloInstruction::CreateTuple(elements));
}

absl::StatusOr<HloComputation*> CreateFusionBody(
    HloModule* module, absl::Span<HloInstruction* const> matched,
    absl::Span<HloInstruction* const> captures) {
  HloComputation::Builder builder("address-computation");

  // A mapping from original instructions to instructions in the fusion body.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> instr_mapping;

  auto mapped_operands = [&](HloInstruction* instr) {
    absl::InlinedVector<HloInstruction*, 4> operands;
    for (HloInstruction* operand : instr->operands()) {
      operands.push_back(instr_mapping.at(operand));
    }
    return operands;
  };

  // For every captured value create a parameter instruction in the computation
  // body and set up instruction mapping.
  for (const HloInstruction* capture : captures) {
    int64_t index = instr_mapping.size();
    instr_mapping[capture] =
        builder.AddInstruction(HloInstruction::CreateParameter(
            index, capture->shape(), absl::StrCat("p", index)));
  }

  // Instructions in the pattern are already topologically sorted, as we visited
  // them following use-def chain, then reverse the list.
  for (HloInstruction* instr : matched) {
    instr_mapping[instr] = builder.AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), mapped_operands(instr)));
  }

  HloInstruction* root = builder.last_added_instruction();
  // Create a root tuple if the root is a tuple to make sure there's a buffer
  // assigned for each of the elements. Make sure the tuple is not nil first.
  if (root->shape().IsTuple() && root->shape().tuple_shapes_size() > 0) {
    CreateRootTuple(root, builder);
  }

  return module->AddComputationAndUnifyNamesAndIds(builder.Build(), false);
}

absl::StatusOr<HloInstruction*> CreateFusionInstruction(
    HloModule* module, HloInstruction* orig,
    absl::Span<HloInstruction* const> captures, HloComputation* body,
    bool dynamic) {
  HloComputation* parent = orig->parent();

  // Add a fusion operation calling outlined fusion computation.
  HloInstruction* fusion = parent->AddInstruction(HloInstruction::CreateFusion(
      body->root_instruction()->shape(), HloInstruction::FusionKind::kCustom,
      captures, body));
  module->SetAndUniquifyInstrName(fusion, "address_computation");

  // We don't need to set/update output_to_operand_aliasing for the new fusion
  // instruction because all buffers are already assigned at this point.

  // Set backends config to a matched custom fusion config.
  GpuBackendConfig gpu_config;
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind("__custom_fusion");
  CustomFusionConfig config;
  config.set_name(dynamic ? "dynamic_address_computation"
                          : "address_computation");
  *backend_config.mutable_custom_fusion_config() = config;
  TF_RETURN_IF_ERROR(fusion->set_backend_config(std::move(gpu_config)));

  return fusion;
}

}  // namespace

absl::StatusOr<bool> AddressComputationFusionRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->has_schedule()) return Internal("module is not scheduled");

  auto process_slices = [&](bool dynamic) -> absl::StatusOr<bool> {
    absl::flat_hash_map<HloInstruction*,
                        absl::InlinedVector<HloInstruction*, 8>>
        matches;
    absl::flat_hash_map<const HloInstruction*, const HloInstruction*>
        replacement_map;

    // Collect all potential custom call matches in the non-fusion computations.
    for (HloComputation* computation : module->computations()) {
      if (computation->IsFusionComputation()) continue;
      for (HloInstruction* instr : computation->instructions()) {
        if (IsLegacyCublasMatmul(*instr) ||
            (!dynamic && IsCustomCall(instr, platform_name_))) {
          auto sliced_operand_chains =
              GetSlicedChains(instr, dynamic, replacement_map);
          if (!(sliced_operand_chains.size() == 1 &&
                sliced_operand_chains.front() == instr)) {
            matches[instr] = std::move(sliced_operand_chains);
          }
        }
      }
    }

    if (matches.empty()) return false;

    HloSchedule& schedule = module->schedule();
    for (auto& kv : matches) {
      auto captures = GetPatternCaptures(kv.second);
      auto sorted = GetSortedMatched(kv.second);

      TF_ASSIGN_OR_RETURN(HloComputation * fusion_body,
                          CreateFusionBody(module, sorted, captures));
      TF_ASSIGN_OR_RETURN(HloInstruction * fusion,
                          CreateFusionInstruction(module, kv.first, captures,
                                                  fusion_body, dynamic));

      // As we are running after scheduling we have to keep it valid.
      HloComputation* parent = kv.first->parent();

      // Update schedule to replace the custom call instruction with the fusion
      // instruction.
      // Removal of the rest of the instructions in the sequence is handled by
      // schedule update below.
      HloInstructionSequence& sequence = schedule.GetOrCreateSequence(parent);
      sequence.replace_instruction(kv.first, fusion);

      // TODO(vuson): handle control dependencies
      TF_RETURN_IF_ERROR(parent->ReplaceInstruction(
          const_cast<HloInstruction*>(replacement_map[kv.first]), fusion));
    }

    TF_RETURN_IF_ERROR(module->schedule().Update());

    return true;
  };

  // TODO(vuson): unify dynamic_address_computation and address_computation
  TF_ASSIGN_OR_RETURN(bool processed_pattern_with_static_slices,
                      process_slices(false));
  TF_ASSIGN_OR_RETURN(bool processed_pattern_with_dynamic_slices,
                      process_slices(true));
  return processed_pattern_with_static_slices ||
         processed_pattern_with_dynamic_slices;
}

}  // namespace gpu
}  // namespace xla
