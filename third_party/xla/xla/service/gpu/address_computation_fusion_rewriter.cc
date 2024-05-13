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
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
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
#include "xla/status.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

// A dataflow path flowing from a definition to a user.
using DefUseDataflowPath = absl::InlinedVector<HloInstruction*, 2>;

// All dataflow paths flowing from a definition to all users. Each user will
// have a separate entry in the vector.
using DefUseDataflowPaths = absl::InlinedVector<DefUseDataflowPath, 4>;

// A dataflow path flowing from a user to a definition.
using UseDefDataflowPath = absl::InlinedVector<HloInstruction*, 4>;

// All dataflow paths flowing from a user to all definitions of its operands.
using UseDefDataflowPaths = absl::InlinedVector<HloInstruction*, 8>;

using DataflowPathView = absl::Span<HloInstruction* const>;
using DataflowPathsView = absl::Span<DataflowPathView>;

using InstructionSet = absl::flat_hash_set<HloInstruction*>;

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

UseDefDataflowPaths GetSlicedOperandPaths(const HloInstruction* instr) {
  UseDefDataflowPaths sliced_operand_paths;

  // This set is used to avoid duplicates in the matched results. It contains
  // the matched instructions that we have seen so far.
  InstructionSet processed_instrs;

  const auto& aliasing_pairs =
      Cast<HloCustomCallInstruction>(instr)->output_to_operand_aliasing();
  absl::flat_hash_set<int64_t> aliased_operands;
  for (const auto& pair : aliasing_pairs) {
    aliased_operands.insert(pair.second.first);
  }

  for (const auto* operand : instr->operands()) {
    // output_to_operand_aliasing means the operand is to be materialized, which
    // is against the whole idea of address computation fusion. Skip this
    // operand.
    if (aliased_operands.contains(instr->operand_index(operand))) continue;
    UseDefDataflowPath maybe_sliced_operand_path;
    bool slice_found = false;
    // TODO: currently HloFindIf exits upon encountering the first node that
    // matches. This works well if each operand only has 1 data flow (i.e. only
    // flows through unary op). We might want to keep finding until the queue is
    // empty: if the operand is a tuple, it might have different data flows
    // (i.e. 1 for each element).
    auto maybe_slice_instr =
        HloFindIf({operand}, [&](const HloInstruction* cur) {
          // If the node is a match that has been processed, stop the traversal.
          if (processed_instrs.contains(cur)) return true;

          maybe_sliced_operand_path.push_back(const_cast<HloInstruction*>(cur));

          if (IsOpcodeAnyOf<HloOpcode::kDynamicSlice, HloOpcode::kSlice>(cur)) {
            if (IsAlignedSlice(cur->operand(0)->shape(), cur->shape(),
                               DynCast<HloSliceInstruction>(cur))) {
              slice_found = true;
              return slice_found;
            }
          }

          return !IsNoOp(cur);
        });

    if (maybe_slice_instr == std::nullopt) continue;

    if (slice_found || processed_instrs.contains(maybe_slice_instr.value())) {
      // Even in the case of stopping at a match that has been processed, we
      // still need to add instructions encountered in the sliced operand path
      // during the latest traversal.
      sliced_operand_paths.insert(sliced_operand_paths.end(),
                                  maybe_sliced_operand_path.rbegin(),
                                  maybe_sliced_operand_path.rend());
      processed_instrs.insert(maybe_sliced_operand_path.begin(),
                              maybe_sliced_operand_path.end());
    }
  }

  sliced_operand_paths.push_back(const_cast<HloInstruction*>(instr));
  return sliced_operand_paths;
}

// Each user of `instr` that goes into a DUS will have an entry in the returned
// vector.
// Each entry contains the sliced paths for that user, i.e. the sequence of ops
// following the dataflow from the user itself to the DUS (included).
DefUseDataflowPaths GetSlicedUserPaths(const HloInstruction* instr) {
  DefUseDataflowPaths sliced_user_paths;
  // This set is used to avoid duplicates in the matched results. It contains
  // the matched instructions that we have seen so far.
  InstructionSet processed_instrs;

  auto traverse_hlo_and_collect = [&](HloInstruction* start) {
    DefUseDataflowPath maybe_sliced_user_path;
    bool dus_found = false;
    auto maybe_dus_instr = HloFindIf(
        {start},
        [&](const HloInstruction* cur) {
          // If the node is a match that has been processed, stop the
          // traversal.
          if (processed_instrs.contains(cur)) return true;
          maybe_sliced_user_path.push_back(const_cast<HloInstruction*>(cur));
          if (const auto slice_instr =
                  DynCast<HloDynamicUpdateSliceInstruction>(cur)) {
            if (IsAlignedSlice(slice_instr->shape(),
                               slice_instr->update()->shape(), nullptr)) {
              dus_found = true;
              return true;
            }
          }
          return cur->user_count() > 1 || !IsNoOp(cur);
        },
        /*visit_operands=*/false);
    if (maybe_dus_instr == std::nullopt) return;
    if (dus_found || processed_instrs.contains(maybe_dus_instr.value())) {
      // Even in the case of stopping at a match that has been processed, we
      // still need to add instructions encountered in the sliced user path
      // during the latest traversal.
      processed_instrs.insert(maybe_sliced_user_path.begin(),
                              maybe_sliced_user_path.end());
      sliced_user_paths.push_back(std::move(maybe_sliced_user_path));
    }
  };

  if (instr->shape().IsTuple()) {
    for (auto* user : instr->users()) {
      if (DynCast<HloGetTupleElementInstruction>(user)) {
        traverse_hlo_and_collect(user);
      }
    }
  } else {
    if (instr->user_count() == 1) {
      traverse_hlo_and_collect(instr->users().front());
    }
  }

  return sliced_user_paths;
}

absl::InlinedVector<HloInstruction*, 4> GetPatternCaptures(
    DataflowPathView matches) {
  absl::InlinedVector<HloInstruction*, 4> captures;

  InstructionSet matched_instrs(matches.begin(), matches.end());

  for (HloInstruction* instr : matches) {
    for (HloInstruction* operand : instr->operands()) {
      if (!matched_instrs.contains(operand) &&
          absl::c_find(captures, operand) == captures.end()) {
        captures.emplace_back(operand);
      }
    }
  }

  return captures;
}

Status CreateRootTuple(HloInstruction* hero, HloComputation::Builder& builder,
                       DataflowPathsView sliced_user_paths,
                       absl::flat_hash_map<const HloInstruction*,
                                           HloInstruction*>& instr_mapping) {
  unsigned tuple_size = hero->shape().tuple_shapes_size();

  std::vector<HloInstruction*> sliced_elems(tuple_size, nullptr);
  for (auto& sliced_user_path : sliced_user_paths) {
    auto gte = Cast<HloGetTupleElementInstruction>(sliced_user_path.front());
    sliced_elems[gte->tuple_index()] = sliced_user_path.back();
  }

  std::vector<HloInstruction*> elements;
  for (size_t i = 0; i < tuple_size; ++i) {
    if (sliced_elems[i] != nullptr) {
      elements.push_back(instr_mapping[sliced_elems[i]]);
      continue;
    }
    auto* gte = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(instr_mapping[hero], i));
    if (hero->shape().tuple_shapes(i).IsTuple()) {
      instr_mapping[gte] = gte;
      TF_RETURN_IF_ERROR(CreateRootTuple(gte, builder, {}, instr_mapping));
      elements.push_back(builder.last_added_instruction());
    } else {
      elements.push_back(gte);
    }
  }
  if (elements.size() > 1)
    builder.AddInstruction(HloInstruction::CreateTuple(elements));

  return absl::OkStatus();
}

absl::StatusOr<HloComputation*> CreateFusionBody(
    HloModule* module, DataflowPathView sliced_operand_paths,
    DataflowPathsView sliced_user_paths, DataflowPathView captures) {
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
  // them following use-def path, then reverse the list.
  HloInstruction* hero;
  for (HloInstruction* instr : sliced_operand_paths) {
    instr_mapping[instr] = builder.AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), mapped_operands(instr)));
    hero = instr;
  }

  for (auto& sliced_user_path : sliced_user_paths) {
    for (HloInstruction* instr : sliced_user_path) {
      instr_mapping[instr] = builder.AddInstruction(
          instr->CloneWithNewOperands(instr->shape(), mapped_operands(instr)));
    }
  }

  // Create a tuple if the hero is a tuple to make sure there's a buffer
  // assigned for each of the elements. Make sure the tuple is not nil first.
  if (hero->shape().IsTuple() && hero->shape().tuple_shapes_size() > 0) {
    TF_RETURN_IF_ERROR(
        CreateRootTuple(hero, builder, sliced_user_paths, instr_mapping));
  }

  return module->AddComputationAndUnifyNamesAndIds(builder.Build(), false);
}

absl::StatusOr<HloInstruction*> CreateFusionInstruction(
    HloModule* module, HloInstruction* orig, DataflowPathView captures,
    HloComputation* body, bool dynamic) {
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
  absl::flat_hash_map<HloInstruction*,
                      std::pair<UseDefDataflowPaths, DefUseDataflowPaths>>
      matches;

  // Collect all potential custom call matches in the non-fusion computations.
  for (HloComputation* computation : module->computations()) {
    if (computation->IsFusionComputation()) continue;
    for (HloInstruction* instr : computation->instructions()) {
      if (IsLegacyCublasMatmul(*instr) ||
          (IsCustomCall(instr, platform_name_))) {
        UseDefDataflowPaths sliced_operand_paths = GetSlicedOperandPaths(instr);
        bool has_sliced_operand_paths = sliced_operand_paths.size() > 1;

        DefUseDataflowPaths sliced_user_paths = GetSlicedUserPaths(instr);
        bool has_sliced_user_paths = absl::c_any_of(
            sliced_user_paths,
            [&](auto& sliced_user_path) { return !sliced_user_path.empty(); });

        if (absl::c_any_of(sliced_user_paths, [&](auto& sliced_user_path) {
              return DynCast<HloDynamicUpdateSliceInstruction>(
                         sliced_user_path.back()) == nullptr;
            })) {
          return absl::InternalError(
              "Expect sliced user path to end with a DUS.");
        }

        if (has_sliced_operand_paths || has_sliced_user_paths) {
          matches[instr] = std::make_pair(std::move(sliced_operand_paths),
                                          std::move(sliced_user_paths));
        }
      }
    }
  }

  if (matches.empty()) return false;

  for (auto& [hero, paths] : matches) {
    auto& [sliced_operand_paths, sliced_user_paths] = paths;
    std::vector<HloInstruction*> matched_instrs;
    absl::c_copy(sliced_operand_paths, std::back_inserter(matched_instrs));

    std::vector<DataflowPathView> sliced_user_paths_view;
    for (auto& sliced_user_path : sliced_user_paths) {
      absl::c_copy(sliced_user_path, std::back_inserter(matched_instrs));
      DataflowPathView sliced_user_path_view{&sliced_user_path.front(),
                                             sliced_user_path.size()};
      sliced_user_paths_view.push_back(std::move(sliced_user_path_view));
    }

    auto captures = GetPatternCaptures(matched_instrs);

    TF_ASSIGN_OR_RETURN(
        HloComputation * fusion_body,
        CreateFusionBody(module, sliced_operand_paths,
                         DataflowPathsView(sliced_user_paths_view), captures));

    bool has_dynamic_slices = absl::c_any_of(matched_instrs, [&](auto* instr) {
      return DynCast<HloDynamicIndexInstruction>(instr) != nullptr;
    });
    TF_ASSIGN_OR_RETURN(
        HloInstruction * fusion,
        CreateFusionInstruction(module, hero, captures, fusion_body,
                                has_dynamic_slices));

    HloComputation* parent = hero->parent();
    if (fusion->shape().IsTuple()) {
      TF_RETURN_IF_ERROR(parent->ReplaceInstructionWithDifferentShape(
          const_cast<HloInstruction*>(hero), fusion));
      for (auto& sliced_user_path : sliced_user_paths) {
        auto old_gte =
            Cast<HloGetTupleElementInstruction>(sliced_user_path.front());
        HloInstruction* gte =
            parent->AddInstruction(HloInstruction::CreateGetTupleElement(
                fusion, old_gte->tuple_index()));
        TF_RETURN_IF_ERROR(
            parent->ReplaceInstruction(sliced_user_path.back(), gte));
      }
    } else {
      auto* instr_to_be_replaced = const_cast<HloInstruction*>(hero);
      if (sliced_user_paths.empty()) {
        // The only case where a tuple-shaped original hero op is fused into a
        // non-tuple-shaped fusion is there's only one element of the original
        // tuple being used. In that case, we need to replace that single
        // get-tuple-element (instead of the hero op) with the fusion
        // instruction.
        if (hero->shape().IsTuple()) {
          if (hero->user_count() != 1 ||
              !DynCast<HloGetTupleElementInstruction>(hero->users().front())) {
            return absl::InternalError(
                "Expect a single get-tuple-element user of the original "
                "tuple-shaped hero op when address computation fusion does "
                "not return a tuple");
          }
          instr_to_be_replaced = hero->users().front();
        }
      } else {
        instr_to_be_replaced = sliced_user_paths.front().back();
      }
      TF_RETURN_IF_ERROR(
          parent->ReplaceInstruction(instr_to_be_replaced, fusion));
    }
  }

  return true;
}

}  // namespace gpu
}  // namespace xla
