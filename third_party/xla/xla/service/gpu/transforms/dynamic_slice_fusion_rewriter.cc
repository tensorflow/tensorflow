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
#include "xla/service/gpu/transforms/dynamic_slice_fusion_rewriter.h"

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
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/while_loop_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tools/hlo_extractor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

namespace m = ::xla::match;

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

using OffsetValueMap =
    absl::flat_hash_map<HloInstruction*, std::vector<Literal>>;

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
bool IsAlignedSlice(const HloInstruction* slice) {
  DCHECK(slice->opcode() == HloOpcode::kSlice ||
         slice->opcode() == HloOpcode::kDynamicSlice ||
         slice->opcode() == HloOpcode::kDynamicUpdateSlice)
      << "Unknown slice operation: " << slice->ToString();

  if (!IsContiguousSlice(*slice)) return false;

  auto [full_shape, slice_shape] = [&] {
    if (auto* dus = DynCast<HloDynamicUpdateSliceInstruction>(slice)) {
      return std::make_pair(dus->shape(), dus->update()->shape());
    }
    return std::make_pair(slice->operand(0)->shape(), slice->shape());
  }();

  auto strides = ShapeUtil::ByteStrides(slice_shape);
  if (!strides.has_value()) return false;

  for (auto dim : slice_shape.layout().minor_to_major()) {
    if ((strides.value()[dim] % kXlaAllocatedBufferAlignBytes) == 0) {
      return true;
    }
    if (slice_shape.dimensions(dim) < full_shape.dimensions(dim)) {
      return (slice->opcode() == HloOpcode::kSlice &&
              (((*strides)[dim] * slice->slice_starts(dim)) %
                   kXlaAllocatedBufferAlignBytes ==
               0));
    }
  }
  return true;
}

// Function looks for while backend config. If this config is present, it
// returns the value of trip count, otherwise it runs the while loop analysis to
// compute trip count. `whileop` must be a while operaton. Returns
// `std::nullopt` if it cannot figure out the trip count.
std::optional<int64_t> GetWhileLoopTripCount(HloInstruction* whileop) {
  CHECK(whileop->opcode() == HloOpcode::kWhile);
  auto backend_config = whileop->backend_config<WhileLoopBackendConfig>();
  if (!backend_config.ok() || !backend_config.value().has_known_trip_count()) {
    VLOG(4) << "Backend config not ok. Computing while loop trip count for "
            << whileop->name();
    return ComputeWhileLoopTripCount(whileop);
  }
  int trip_count = backend_config.value().known_trip_count().n();
  VLOG(4) << "Found trip count in backend config for " << whileop->name()
          << ": " << trip_count;
  return trip_count;
}

// Given an HLO operation `idx`, which is wrapped by while operation, this
// function tries to find the values of the variable in all the iterations as an
// array of literals. This is done by repeatedly executing the loop update
// operation(s) and the operation(s) to calculate the value of `idx` at each
// iteration. If this is successful, then the vector of literals is returned. If
// for some reason this is not successful then `std::nullopt` is returned.
std::optional<std::vector<Literal>> GetValues(const HloInstruction* idx) {
  VLOG(3) << "Getting values for " << idx->name();
  const HloComputation* computation = idx->parent();
  if (!computation->IsWhileBodyComputation()) {
    VLOG(3) << "While calculating offset values for " << idx->name()
            << ", the parent computation(" << computation->name()
            << ") is not a while computation";
    return std::nullopt;
  }
  HloInstruction* whileop = computation->WhileCallInstruction();
  std::optional<int64_t> trip_count = GetWhileLoopTripCount(whileop);
  if (trip_count == std::nullopt) {
    VLOG(3) << "Unable to get trip count for " << whileop->name();
    return std::nullopt;
  }
  auto root_tuple = computation->root_instruction();
  if (root_tuple->opcode() != HloOpcode::kTuple) {
    VLOG(3) << "Root operation " << root_tuple->name() << " of computation "
            << computation->name()
            << " expected to be a tuple because it is a while body. Found: "
            << root_tuple->opcode();
    return std::nullopt;
  }
  std::optional<int64_t> loop_indvar_tuple_idx =
      GetLoopInductionVarTupleIdx(whileop);
  if (loop_indvar_tuple_idx == std::nullopt) {
    VLOG(3) << "Unable to find tuple index for loop induction variable";
    return std::nullopt;
  }
  auto update_operation =
      computation->root_instruction()->operand(*loop_indvar_tuple_idx);
  HloInstruction* loop_indvar = nullptr;
  for (auto instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kGetTupleElement &&
        instr->operand(0) == computation->parameter_instruction(0) &&
        instr->tuple_index() == *loop_indvar_tuple_idx) {
      loop_indvar = instr;
    }
  }
  if (loop_indvar == nullptr) {
    VLOG(3) << "Unable to find get-tuple-element("
            << computation->parameter_instruction(0)->name()
            << "), index=" << *loop_indvar_tuple_idx << " in "
            << computation->name();
    return std::nullopt;
  }

  // Extract the offset and update modules and verify that they only take the
  // loop iteration counter as parameter.
  // The operation we are extracting (update and offset) are from `computation`.
  // In the `extract_selector`, we stop at the parameter (tuple) for this
  // `computation` or at the loop induction variable and convert that to a
  // parameter. If the operation depends on the tuple parameter, then the
  // argument to the extracted module will have the shape of a tuple. So, if the
  // extracted module has only one parameter and the shape of that parameter is
  // same as the loop induction variable, then the operation only depends on the
  // loop induction variable. We also have to ensure there are no `partition-id`
  // or `replica-id` operations in the extracted module.
  auto IsValidModule =
      [loop_indvar](std::unique_ptr<HloModule>& module) -> bool {
    if (module == nullptr || module->entry_computation()->num_parameters() != 1)
      return false;
    const HloInstruction* p0 =
        module->entry_computation()->parameter_instruction(0);
    if (p0->shape() != loop_indvar->shape()) {
      VLOG(4) << "Extracted module must depend only on the loop induction "
                 "variable.";
      return false;
    };
    return llvm::all_of(module->entry_computation()->instructions(),
                        [](const HloInstruction* instr) {
                          return instr->opcode() != HloOpcode::kPartitionId &&
                                 instr->opcode() != HloOpcode::kReplicaId;
                        });
  };
  auto params = computation->parameter_instructions();
  if (params.size() != 1 || !params[0]->shape().IsTuple()) {
    VLOG(3) << "While loop parameter is expected to be a tuple.";
    return std::nullopt;
  }
  std::unique_ptr<HloModule> offset_module = ExtractModule(
      /*instruction=*/
      idx, /*height=*/-1,
      /*extract_selector=*/
      [loop_indvar, params](const HloInstruction* inst) -> bool {
        return inst != loop_indvar && llvm::find(params, inst) == params.end();
      },
      /*replace_type_selector=*/
      [](const HloInstruction* inst) -> ReplaceType {
        return ReplaceType::kReplaceParam;
      });
  std::unique_ptr<HloModule> update_module = ExtractModule(
      /*instruction=*/
      update_operation, /*height=*/-1,
      /*extract_selector=*/
      [loop_indvar, params](const HloInstruction* inst) -> bool {
        return inst != loop_indvar && llvm::find(params, inst) == params.end();
      },
      /*replace_type_selector=*/
      [](const HloInstruction* inst) -> ReplaceType {
        return ReplaceType::kReplaceParam;
      });
  if (!IsValidModule(offset_module) || !IsValidModule(update_module)) {
    return std::nullopt;
  }
  VLOG(3) << "Successfully generated offset and update modules";

  std::vector<Literal> offset_values;
  absl::Status status = [&]() -> absl::Status {
    HloEvaluator evaluator;
    const Literal& init =
        whileop->operand(0)->operand(*loop_indvar_tuple_idx)->literal();
    std::unique_ptr<Literal> updated_value = nullptr;
    for (int64_t i = 0; i < *trip_count; i++) {
      if (i == 0) {
        evaluator.ResetVisitStates();
        TF_ASSIGN_OR_RETURN(offset_values.emplace_back(),
                            evaluator.Evaluate(*offset_module, {&init}));
        CHECK(offset_values.back().shape() == idx->shape());
        evaluator.ResetVisitStates();
        TF_ASSIGN_OR_RETURN(Literal next_update_value,
                            evaluator.Evaluate(*update_module, {&init}));
        updated_value = next_update_value.CloneToUnique();
      } else {
        evaluator.ResetVisitStates();
        TF_ASSIGN_OR_RETURN(
            offset_values.emplace_back(),
            evaluator.Evaluate(*offset_module, {updated_value.get()}));
        CHECK(offset_values.back().shape() == idx->shape());
        evaluator.ResetVisitStates();
        TF_ASSIGN_OR_RETURN(
            Literal next_update_value,
            evaluator.Evaluate(*update_module, {updated_value.get()}));
        updated_value = next_update_value.CloneToUnique();
      }
    }
    VLOG(3) << "Offset values for " << idx->name() << ": "
            << absl::StrJoin(offset_values, ",",
                             [](std::string* out, const Literal& l) {
                               out->append(l.ToString());
                             });
    return absl::OkStatus();
  }();
  if (status.ok()) return offset_values;
  return std::nullopt;
}

// This function takes a while operation and adds a loop iteration counter
// variable as the last parameter in the loop. This is useful, especially
// because the loop induction variable might not be 0,1,2,3... and we need a
// variable of this form to access the array literal for offset.
absl::StatusOr<HloInstruction*> AddLoopIterationParam(HloInstruction* whileop) {
  CHECK(whileop->opcode() == HloOpcode::kWhile);
  HloComputation* while_body = whileop->while_body();
  HloComputation* while_cond = whileop->while_condition();
  const HloInstruction* while_init = whileop->operand(0);

  // First handle the initial values.
  CHECK(while_init->opcode() == HloOpcode::kTuple);
  std::vector<HloInstruction*> new_init_operands(while_init->operands().begin(),
                                                 while_init->operands().end());
  PrimitiveType indvar_type =
      whileop->while_init()
          ->operand(*GetLoopInductionVarTupleIdx(whileop))
          ->shape()
          .element_type();
  new_init_operands.push_back(whileop->parent()->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(
          whileop->while_init()
              ->operand(*GetLoopInductionVarTupleIdx(whileop))
              ->shape()
              .element_type(),
          0)),
      "zero"));
  HloInstruction* new_while_init = whileop->parent()->AddInstruction(
      HloInstruction::CreateTuple(new_init_operands));
  HloInstruction* new_whileop = whileop->parent()->AddInstruction(
      whileop->CloneWithNewOperands(new_while_init->shape(), {new_while_init}));
  if (whileop->IsRoot()) {
    absl::InlinedVector<HloInstruction*, 4> tuple_entries;
    tuple_entries.reserve(while_init->shape().tuple_shapes_size());
    for (auto i = 0; i < while_init->shape().tuple_shapes_size(); i++) {
      tuple_entries.push_back(whileop->parent()->AddInstruction(
          HloInstruction::CreateGetTupleElement(new_whileop, i)));
    }
    HloInstruction* new_whileop_result = whileop->parent()->AddInstruction(
        HloInstruction::CreateTuple(tuple_entries));
    TF_RETURN_IF_ERROR(
        whileop->parent()->ReplaceInstruction(whileop, new_whileop_result));
  } else {
    TF_RETURN_IF_ERROR(whileop->parent()->ReplaceInstructionWithDifferentShape(
        whileop, new_whileop));
  }

  // Next, lets handle the condition
  while_cond->ReplaceParameter(0, HloInstruction::CreateParameter(
                                      0, new_while_init->shape(), "new_param"));

  // Next, lets handle the body
  HloInstruction* new_body_param = while_body->ReplaceParameter(
      0,
      HloInstruction::CreateParameter(0, new_while_init->shape(), "new_param"));

  // Next, update the value of the param inside while op
  HloInstruction* gte = while_body->AddInstruction(
      HloInstruction::CreateGetTupleElement(
          new_body_param, new_while_init->shape().tuple_shapes_size() - 1),
      "loop_iteration_count");
  HloInstruction* c1 = while_body->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(indvar_type, 1)),
      "one");
  HloInstruction* add = while_body->AddInstruction(
      HloInstruction::CreateBinary(gte->shape(), HloOpcode::kAdd, gte, c1),
      "updated_loop_iteration_count");
  absl::InlinedVector<HloInstruction*, 2> old_return_tuple_operands =
      while_body->root_instruction()->operands();
  std::vector<HloInstruction*> new_return_tuple_operands(
      old_return_tuple_operands.begin(), old_return_tuple_operands.end());
  new_return_tuple_operands.push_back(add);
  HloInstruction* new_return_tuple = while_body->AddInstruction(
      HloInstruction::CreateTuple(new_return_tuple_operands));
  while_body->set_root_instruction(new_return_tuple, true);
  return gte;
}

// This function takes an array literal and gives a constant instruction with
// that literal.
std::unique_ptr<HloInstruction> GetAsConstantInstruction(
    const std::vector<Literal>& offset_values) {
  if (offset_values.empty()) return nullptr;
  std::unique_ptr<HloInstruction> value =
      primitive_util::PrimitiveTypeSwitch<std::unique_ptr<HloInstruction>>(
          [&offset_values](
              auto primitive_type_constant) -> std::unique_ptr<HloInstruction> {
            if constexpr (primitive_util::IsIntegralType(
                              primitive_type_constant)) {
              using NativeT = typename primitive_util::PrimitiveTypeToNative<
                  primitive_type_constant>::type;

              Array<NativeT> constantLiterals({(int64_t)offset_values.size()});
              std::vector<NativeT> valuesAsTy;
              valuesAsTy.reserve(offset_values.size());
              for (auto& i : offset_values) {
                valuesAsTy.push_back(
                    static_cast<NativeT>(i.data<NativeT>()[0]));
              }
              constantLiterals.SetValues(valuesAsTy);
              return HloInstruction::CreateConstant(
                  LiteralUtil::CreateFromArray(constantLiterals));
            }
            return nullptr;
          },
          offset_values[0].shape().element_type());
  return value;
}

// This function takes an operation, and a reference to a map of
// {operation: array literals containing their values}. If the operation is a
// dynamic slicing operation, we populate the value map with the values of the
// offsets. This only returns true if it can successfully find values
// corresponding to all the offsets in the `matched_instrs`. If there is a
// single offset for which we cannot find the values, then we do not add
// anything to the value map, and return false.
bool PopulateOffsetValueMap(const HloInstruction* matched_instr,
                            OffsetValueMap& value_map) {
  OffsetValueMap local_value_map;
  if (auto dyn_idx_op = DynCast<HloDynamicIndexInstruction>(matched_instr);
      dyn_idx_op) {
    for (auto indexop : dyn_idx_op->index_operands()) {
      if (indexop->IsConstant()) continue;
      if (local_value_map.contains(indexop) || value_map.contains(indexop))
        continue;
      std::optional<std::vector<Literal>> values = GetValues(indexop);
      if (values == std::nullopt) return false;
      if (values->empty() || !primitive_util::IsIntegralType(
                                 values->at(0).shape().element_type())) {
        return false;
      }
      std::transform(values->begin(), values->end(),
                     std::back_inserter(local_value_map[indexop]),
                     [](Literal& l) { return std::move(l); });
    }
  }
  for (auto& [op, values] : local_value_map) {
    std::transform(values.begin(), values.end(),
                   std::back_inserter(value_map[op]),
                   [](Literal& l) { return std::move(l); });
  }
  VLOG(2) << "Received " << local_value_map.size() << " new offsets.";
  return true;
}

// This function takes a list of fusion instructions, and a value map
// {operation: array literal containing its values across iterations}. These
// fusions take the value of offset as a input. So, the value of this offset is
// calculated outside the fusion. This function changes these fusions so that
// the fusion instead only takes the loop iteration number and the offset is
// read from a constant array. This constant array comes from the value map. On
// a high level, the transform looks like:
//
// clang-format off
//
// input-fusion(p0, p1, p2, offset, c0) {
//   ds = dynamic-slice(p0, offset, c0, c0)
//   gemm = custom-call(ds, p1)
//   ROOT dus = dynamic-update-slice(p2, gemm, offset, c0, c0)
// }
//
// changes to
//
// output-fusion(p0, p1, p2, loop_counter, c0) {
//   offset_values = constant({2,4,6,8,10})
//   offset_array = dynamic-slice(offset_values, loop_counter), slice_size={1}
//   offset = reshape(offset_array)
//   ds = dynamic-slice(p0, offset, c0, c0)
//   gemm = custom-call(ds, p1)
//   ROOT dus = dynamic-update-slice(p2, gemm, offset, c0, c0)
// }
//
// clang-format on
absl::Status ReplaceOffsetCalculationWithArrayAccess(
    PtrVec<HloInstruction*> fusions, OffsetValueMap& value_map) {
  absl::flat_hash_map<HloComputation*, HloInstruction*> loop_iteration_param;
  for (auto& [instr, _] : value_map) {
    VLOG(2) << "Handling " << instr->name();
    if (!instr->parent()->IsWhileBodyComputation()) {
      VLOG(2) << "It is not a while body computation";
      return absl::InternalError(
          absl::StrFormat("%s is expected to be a while computation.",
                          instr->parent()->name()));
    }
    if (loop_iteration_param.find(instr->parent()) !=
        loop_iteration_param.end()) {
      VLOG(2) << "This was already handled";
      continue;
    }
    VLOG(2) << "Adding loop iteration param for " << instr->parent()->name();
    TF_ASSIGN_OR_RETURN(
        loop_iteration_param[instr->parent()],
        AddLoopIterationParam(instr->parent()->WhileCallInstruction()));
  }
  for (auto fusion_instr : fusions) {
    // Check that this fusion operation has something we need to replace:
    for (auto maybe_offset : fusion_instr->operands()) {
      if (value_map.find(maybe_offset) == value_map.end()) continue;
      HloInstruction* loop_counter =
          loop_iteration_param[fusion_instr->parent()];
      HloComputation* fusion = fusion_instr->fused_instructions_computation();
      loop_iteration_param[fusion] =
          fusion_instr->AddFusionOperand(loop_counter);
      break;
    }
  }
  for (auto fusion_instr : fusions) {
    absl::flat_hash_map<HloInstruction*, HloInstruction*> param_replacement_map;
    absl::InlinedVector<HloInstruction*, 4> parameters;
    HloComputation* fusion_comp =
        fusion_instr->fused_instructions_computation();
    for (auto [idx, maybe_offset] : llvm::enumerate(fusion_instr->operands())) {
      HloInstruction* offset_param =
          fusion_instr->fused_instructions_computation()->parameter_instruction(
              idx);
      if (value_map.find(maybe_offset) == value_map.end() ||
          param_replacement_map.contains(offset_param))
        continue;
      std::vector<Literal>& values = value_map.at(maybe_offset);
      std::unique_ptr<HloInstruction> values_as_const_instruction =
          GetAsConstantInstruction(values);
      if (values_as_const_instruction == nullptr) {
        return absl::InternalError(
            "Unable to convert offsets into constant array.");
      }
      HloInstruction* array = fusion_comp->AddInstruction(
          std::move(values_as_const_instruction), "offset_values");
      HloInstruction* ds =
          fusion_comp->AddInstruction(HloInstruction::CreateDynamicSlice(
              ShapeUtil::MakeShape(offset_param->shape().element_type(), {1}),
              array, {loop_iteration_param[fusion_comp]}, {1}));
      HloInstruction* offset = fusion_comp->AddInstruction(
          HloInstruction::CreateReshape(offset_param->shape(), ds), "offset");
      param_replacement_map[offset_param] = offset;
      parameters.push_back(offset_param);
    }
    for (auto param = parameters.rbegin(); param != parameters.rend();
         param++) {
      auto offset = param_replacement_map[*param];
      TF_RETURN_IF_ERROR(fusion_comp->ReplaceInstruction(*param, offset));
    }
  }
  return absl::OkStatus();
}

UseDefDataflowPaths GetSlicedOperandPaths(const HloInstruction* instr,
                                          OffsetValueMap& value_map) {
  UseDefDataflowPaths sliced_operand_paths;

  // This set is used to avoid duplicates in the matched results. It contains
  // the matched instructions that we have seen so far.
  InstructionSet processed_instrs;

  std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
      aliasing_pairs;
  if (instr->opcode() == HloOpcode::kCustomCall) {
    aliasing_pairs =
        Cast<HloCustomCallInstruction>(instr)->output_to_operand_aliasing();
  }
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
        HloBfsFindIf({operand}, [&](const HloInstruction* cur) {
          // If the node is a match that has been processed, stop the traversal.
          if (processed_instrs.contains(cur)) return true;

          maybe_sliced_operand_path.push_back(const_cast<HloInstruction*>(cur));

          if (IsOpcodeAnyOf<HloOpcode::kDynamicSlice, HloOpcode::kSlice>(cur)) {
            if (IsAlignedSlice(cur)) {
              slice_found = true;
              return slice_found;
            }
          }

          return !IsNoOp(cur);
        });

    if (maybe_slice_instr == std::nullopt) continue;
    bool valid_slice_status =
        PopulateOffsetValueMap(*maybe_slice_instr, value_map);
    if ((valid_slice_status && slice_found) ||
        processed_instrs.contains(maybe_slice_instr.value())) {
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
DefUseDataflowPaths GetSlicedUserPaths(const HloInstruction* instr,
                                       OffsetValueMap& value_map) {
  DefUseDataflowPaths sliced_user_paths;
  // This set is used to avoid duplicates in the matched results. It contains
  // the matched instructions that we have seen so far.
  InstructionSet processed_instrs;

  auto traverse_hlo_and_collect = [&](HloInstruction* start) {
    DefUseDataflowPath maybe_sliced_user_path;
    bool dus_found = false;
    auto maybe_dus_instr = HloBfsFindIf(
        {start},
        [&](const HloInstruction* cur) {
          // If the node is a match that has been processed, stop the
          // traversal.
          if (processed_instrs.contains(cur)) return true;
          maybe_sliced_user_path.push_back(const_cast<HloInstruction*>(cur));
          if (const auto slice_instr =
                  DynCast<HloDynamicUpdateSliceInstruction>(cur)) {
            if (IsAlignedSlice(slice_instr)) {
              dus_found = true;
              return true;
            }
          }
          return cur->user_count() > 1 || !IsNoOp(cur);
        },
        /*visit_operands=*/false);
    if (maybe_dus_instr == std::nullopt) return;
    bool valid_slice_status =
        PopulateOffsetValueMap(*maybe_dus_instr, value_map);
    if ((valid_slice_status && dus_found) ||
        processed_instrs.contains(maybe_dus_instr.value())) {
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

absl::Status CreateRootTuple(
    HloInstruction* hero, HloComputation::Builder& builder,
    DataflowPathsView sliced_user_paths,
    absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
        instr_mapping) {
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
  HloComputation::Builder builder("dynamic-slice-fusion");

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

absl::StatusOr<bool> DynamicSliceFusionRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  absl::flat_hash_map<HloInstruction*,
                      std::pair<UseDefDataflowPaths, DefUseDataflowPaths>>
      matches_kv;

  std::vector<HloInstruction*> matches;
  OffsetValueMap value_map;

  // Collect all potential custom call matches in the non-fusion computations.
  for (HloComputation* computation : module->computations()) {
    if (computation->IsFusionComputation()) continue;
    for (HloInstruction* instr : computation->instructions()) {
      if ((instr->opcode() == HloOpcode::kReduceScatter &&
           instr->shape().IsArray()) ||
          IsLegacyCublasMatmul(*instr) || IsCustomCall(instr, platform_name_)) {
        UseDefDataflowPaths sliced_operand_paths =
            GetSlicedOperandPaths(instr, value_map);
        VLOG(1) << "For operation: " << instr->name() << ", operands: "
                << absl::StrJoin(
                       sliced_operand_paths, ",",
                       [](std::string* out, const HloInstruction* inst) {
                         out->append(inst->name());
                       });
        bool has_sliced_operand_paths = sliced_operand_paths.size() > 1;
        DefUseDataflowPaths sliced_user_paths =
            GetSlicedUserPaths(instr, value_map);
        VLOG(1) << "For operation: " << instr->name() << ", users: "
                << absl::StrJoin(
                       sliced_user_paths, ",",
                       [](std::string* out, const DefUseDataflowPath& path) {
                         out->append(
                             "{" +
                             absl::StrJoin(path, ",",
                                           [](std::string* out,
                                              const HloInstruction* inst) {
                                             out->append(inst->name());
                                           }) +
                             "}");
                       });
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
          matches_kv[instr] = std::make_pair(std::move(sliced_operand_paths),
                                             std::move(sliced_user_paths));
          matches.push_back(instr);
        }
      }
    }
  }

  if (matches.empty()) return false;

  PtrVec<HloInstruction*> fusions;

  for (HloInstruction* hero : matches) {
    auto& paths = matches_kv[hero];
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
    fusions.push_back(fusion);
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
      // This is required for collective operations which will not be removed.
      if (hero->parent()) {
        TF_RETURN_IF_ERROR(hero->parent()->RemoveInstruction(hero));
      }
    }
  }

  TF_RETURN_IF_ERROR(
      ReplaceOffsetCalculationWithArrayAccess(fusions, value_map));

  return true;
}

}  // namespace gpu
}  // namespace xla
