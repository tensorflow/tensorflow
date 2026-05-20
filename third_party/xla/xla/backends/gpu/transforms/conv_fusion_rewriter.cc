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

#include "xla/backends/gpu/transforms/conv_fusion_rewriter.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {
bool IsOperationSupportedByCuDNN(const HloInstruction& hlo,
                                 bool can_fuse_reduce) {
  const HloOpcode opcode = hlo.opcode();
  // Layout of all tensors in conv fusion must be the same, only allow pointwise
  // op in the fusion for now.
  switch (opcode) {
    // Pointwise
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kCeil:
    case HloOpcode::kCompare:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kDivide:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kPower:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSelect:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    // clamp = max(lower, min(value, upper));
    case HloOpcode::kClamp:
      return true;
    // hlo normalization adds bitcast to group reduction dimensions, only
    // fuse such bitcast
    case HloOpcode::kBitcast:
      return hlo.user_count() == 1 &&
             hlo.users()[0]->opcode() == HloOpcode::kReduce &&
             IsOperationSupportedByCuDNN(*hlo.users()[0], can_fuse_reduce);
    // Broadcast with scalar is allowed
    case HloOpcode::kBroadcast:
      return ShapeUtil::IsScalar(hlo.operand(0)->shape());
    // Only support scalar
    case HloOpcode::kConstant:
      return ShapeUtil::IsScalar(hlo.shape());
    case HloOpcode::kReduce:
      // one reduction to scalar in the end is allowed, useful for fp8 conv amax
      return can_fuse_reduce && ShapeUtil::IsScalar(hlo.shape());
    default:
      return false;
  }
}

HloInstruction* FuseTowardOperand(
    HloInstruction* hlo, HloComputation::Builder& builder,
    std::vector<HloInstruction*>& fusion_params,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>& fused_hlo_map) {
  if (auto it = fused_hlo_map.find(hlo); it != fused_hlo_map.end()) {
    // Check if hlo is already fused
    return it->second;
  }
  HloInstruction* fused_hlo;
  // Don't fuse reduction in the prologue
  if (IsOperationSupportedByCuDNN(*hlo, false) && hlo->user_count() == 1) {
    HloInstruction::InstructionVector new_operands;
    for (int i = 0; i < hlo->operand_count(); ++i) {
      HloInstruction* operand = hlo->mutable_operand(i);
      new_operands.push_back(
          FuseTowardOperand(operand, builder, fusion_params, fused_hlo_map));
    }
    fused_hlo = builder.AddInstruction(
        hlo->CloneWithNewOperands(hlo->shape(), new_operands));
  } else {
    // Not supported by cudnn, make it parameter
    fusion_params.push_back(hlo);
    fused_hlo = builder.AddInstruction(HloInstruction::CreateParameter(
        fusion_params.size() - 1, hlo->shape(), hlo->name()));
  }
  CHECK(fused_hlo_map.insert({hlo, fused_hlo}).second);
  return fused_hlo;
}

// FusionState manages the growth of the fusion subgraph. It relies on the
// property that HloInstructions are only ever appended to fusible_users and
// fusion_outputs, allowing RestoreSnapshot to work by simply resizing the
// vectors.
struct FusionState {
  std::vector<HloInstruction*>& fusible_users;
  std::vector<HloInstruction*>& fusion_outputs;
  bool& can_fuse_reduce;

  struct Snapshot {
    int fusible_users_size;
    int fusion_outputs_size;
    bool can_fuse_reduce;
  };

  Snapshot TakeSnapshot() const {
    return {static_cast<int>(fusible_users.size()),
            static_cast<int>(fusion_outputs.size()), can_fuse_reduce};
  }

  void RestoreSnapshot(const Snapshot& snapshot) {
    fusible_users.resize(snapshot.fusible_users_size);
    fusion_outputs.resize(snapshot.fusion_outputs_size);
    can_fuse_reduce = snapshot.can_fuse_reduce;
  }
};

bool ShouldKeepFusingUsers(HloInstruction* hlo, bool& can_fuse_reduce) {
  // Shouldn't fuse anything after reduction.
  if (hlo->user_count() == 0 || hlo->opcode() == HloOpcode::kReduce) {
    return false;
  }

  // Only keep fusing if all users are fusible.
  bool cached_can_fuse_reduce = can_fuse_reduce;
  for (HloInstruction* user : hlo->users()) {
    if (!IsOperationSupportedByCuDNN(*user, can_fuse_reduce)) {
      can_fuse_reduce = cached_can_fuse_reduce;
      return false;
    }
    can_fuse_reduce = can_fuse_reduce && user->opcode() != HloOpcode::kReduce;
  }
  return true;
}

// Grows fusion through users of fusion
void FuseTowardUsers(
    HloComputation::Builder& builder,
    std::vector<HloInstruction*>& fusion_params,
    std::vector<HloInstruction*>& fusible_users,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>& fused_hlo_map) {
  // reverse post order of all fusible users
  for (auto user : fusible_users) {
    HloInstruction::InstructionVector new_operands;
    for (int i = 0; i < user->operand_count(); ++i) {
      HloInstruction* operand = user->mutable_operand(i);
      HloInstruction* fused_operand =
          FuseTowardOperand(operand, builder, fusion_params, fused_hlo_map);
      new_operands.push_back(fused_operand);
    }

    HloInstruction* fused_user = builder.AddInstruction(
        user->CloneWithNewOperands(user->shape(), new_operands));
    CHECK(fused_hlo_map.insert({user, fused_user}).second);
  }
}

bool IsConvFusionOutputsValid(std::vector<HloInstruction*>& fusion_outputs) {
  // at most 2 outputs and at least 1 reduce if there are 2 outputs
  if (fusion_outputs.size() > 2) {
    return false;
  }
  if (fusion_outputs.size() == 2 &&
      (fusion_outputs[0]->opcode() == HloOpcode::kReduce) ==
          (fusion_outputs[1]->opcode() == HloOpcode::kReduce)) {
    return false;
  }
  return true;
}

bool DFS(HloInstruction* hlo, HloReachabilityMap* reachability,
         FusionState& state,
         absl::flat_hash_map<HloInstruction*, bool>& fusible_cache) {
  if (fusible_cache.contains(hlo)) {
    return fusible_cache[hlo];
  }

  const auto snapshot = state.TakeSnapshot();

  bool is_endpoint = !ShouldKeepFusingUsers(hlo, state.can_fuse_reduce);
  bool is_subgraph_valid = true;
  if (!is_endpoint) {
    for (HloInstruction* user : hlo->users()) {
      if (!DFS(user, reachability, state, fusible_cache)) {
        // If a consumer branch fails, we stop here and treat this node as an
        // output.
        is_subgraph_valid = false;
        is_endpoint = true;
        break;
      }
    }
  }

  if (!is_subgraph_valid) {
    state.RestoreSnapshot(snapshot);
  }

  if (is_endpoint) {
    state.fusion_outputs.push_back(hlo);
  }

  bool is_valid = IsConvFusionOutputsValid(state.fusion_outputs);
  if (is_valid && is_endpoint) {
    // New outputs must not reach instructions already inside the fusion.
    is_valid =
        std::none_of(state.fusible_users.begin(), state.fusible_users.end(),
                     [&reachability, hlo](HloInstruction* user) {
                       return reachability->IsReachable(hlo, user);
                     });
  }
  if (is_valid) {
    // Existing outputs must not reach the new internal instruction 'hlo'.
    // If 'hlo' is an endpoint, it was just added to state.fusion_outputs,
    // so we skip the last element to avoid checking hlo reachability to itself.
    is_valid = std::none_of(state.fusion_outputs.begin(),
                            state.fusion_outputs.end() - (is_endpoint ? 1 : 0),
                            [&reachability, hlo](HloInstruction* output) {
                              return reachability->IsReachable(output, hlo);
                            });
  }

  if (is_valid) {
    state.fusible_users.push_back(hlo);
  }
  return fusible_cache[hlo] = is_valid;
}

std::vector<HloInstruction*> GetAllReachableAndFusible(
    HloInstruction* convolution, std::vector<HloInstruction*>& fusion_outputs) {
  std::vector<HloInstruction*> fusible_users;
  absl::flat_hash_map<HloInstruction*, bool> fusible_cache;
  std::unique_ptr<HloReachabilityMap> reachability =
      HloReachabilityMap::Build(convolution->parent());
  bool can_fuse_reduce = true;

  FusionState state{fusible_users, fusion_outputs, can_fuse_reduce};
  DFS(convolution, reachability.get(), state, fusible_cache);

  // Remove convolution from the users.
  fusible_users.pop_back();
  std::reverse(fusible_users.begin(), fusible_users.end());
  // Make sure reduce is after the real conv output
  if (fusion_outputs.size() == 2 &&
      fusion_outputs[0]->opcode() == HloOpcode::kReduce) {
    // Swap reduce and real conv output
    std::swap(fusion_outputs[0], fusion_outputs[1]);
  }
  return fusible_users;
}

HloComputation::Builder CreateConvFusionBuilder(HloInstruction* conv) {
  HloComputation* computation = conv->parent();
  ConvolutionKind convolution_kind =
      DynCast<HloConvolutionInstruction>(conv)->convolution_kind();

  // Give the conv a user-friendly name.
  std::string name;
  if (convolution_kind == CONVOLUTION_KIND_FPROP) {
    name = "conv_fprop";
  } else if (convolution_kind == CONVOLUTION_KIND_DGRAD) {
    name = "conv_dgrad";
  } else if (convolution_kind == CONVOLUTION_KIND_WGRAD) {
    name = "conv_wgrad";
  }

  computation->parent()->SetAndUniquifyInstrName(conv, name);
  std::string fusion_name = absl::StrCat(conv->name(), "_fusion");
  std::string computation_name = absl::StrCat(fusion_name, "_comp");
  return HloComputation::Builder(computation_name);
}

// cuDNN convolutions support fp8 and int8 input types.
// Fuses the input parameters and any immediate conversions (prologue).
std::pair<HloInstruction*, HloInstruction*> TryFuseConvolutionPrologue(
    HloInstruction* convolution, HloComputation::Builder& builder,
    std::vector<HloInstruction*>& fusion_params,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>& fused_hlo_map) {
  auto is_fusable_convert = [](const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kConvert && hlo->user_count() == 1;
  };

  // Only fuse the prologue converts when both conv operands have one and
  // they produce the same source type (or neither has one). Otherwise the
  // cuDNN compiler's "consume convert before conv" path would feed the
  // conv mismatched input dtypes.
  const HloInstruction* lhs = convolution->operand(0);
  const HloInstruction* rhs = convolution->operand(1);
  const bool fuse_prologue_converts =
      is_fusable_convert(lhs) && is_fusable_convert(rhs) &&
      lhs->operand(0)->shape().element_type() ==
          rhs->operand(0)->shape().element_type();

  auto create_operand = [&](int index) {
    HloInstruction* original = convolution->mutable_operand(index);
    const bool is_convert =
        fuse_prologue_converts && is_fusable_convert(original);

    HloInstruction* source =
        is_convert ? original->mutable_operand(0) : original;

    // Create the parameter in the new computation
    HloInstruction* fused_param =
        builder.AddInstruction(HloInstruction::CreateParameter(
            fusion_params.size(), source->shape(), source->name()));
    fusion_params.push_back(source);
    fused_hlo_map[source] = fused_param;

    HloInstruction* fused = fused_param;
    // If there was a convert, clone it into the new computation
    if (is_convert) {
      fused = builder.AddInstruction(
          original->CloneWithNewOperands(original->shape(), {fused_param}));
      fused_hlo_map[original] = fused;
    }
    return fused;
  };

  return {create_operand(0), create_operand(1)};
}

HloInstruction* CreateGpuConvFusion(
    HloInstruction* convolution, std::vector<HloInstruction*>& fusion_outputs) {
  HloComputation::Builder builder = CreateConvFusionBuilder(convolution);

  // Seeding the parameters and the map for the convolution and its prologue.
  std::vector<HloInstruction*> fusion_params;
  absl::flat_hash_map<HloInstruction*, HloInstruction*> fused_hlo_map;

  // Returns fused prologue operands.
  std::pair<HloInstruction*, HloInstruction*> operands =
      TryFuseConvolutionPrologue(convolution, builder, fusion_params,
                                 fused_hlo_map);
  HloInstruction* lhs = operands.first;
  HloInstruction* rhs = operands.second;

  HloInstruction* fused_conv = builder.AddInstruction(
      convolution->CloneWithNewOperands(convolution->shape(), {lhs, rhs}));
  fused_hlo_map[convolution] = fused_conv;

  // Get all reachable and fusible instructions from a convolution in reverse
  // post order.
  std::vector<HloInstruction*> fusible_users =
      GetAllReachableAndFusible(convolution, fusion_outputs);

  FuseTowardUsers(builder, fusion_params, fusible_users, fused_hlo_map);

  HloInstruction* root = nullptr;
  Shape root_shape;
  if (fusion_outputs.size() == 1) {
    auto it = fused_hlo_map.find(fusion_outputs[0]);
    CHECK(it != fused_hlo_map.end());
    root = it->second;
    root_shape = fusion_outputs[0]->shape();
  } else {
    // Multi-output
    std::vector<HloInstruction*> roots;
    std::vector<Shape> shapes;
    for (HloInstruction* output : fusion_outputs) {
      auto it = fused_hlo_map.find(output);
      CHECK(it != fused_hlo_map.end());
      roots.push_back(it->second);
      shapes.push_back(output->shape());
    }
    root = builder.AddInstruction(HloInstruction::CreateTuple(roots));
    root_shape = ShapeUtil::MakeTupleShape(shapes);
  }

  HloComputation* new_computation =
      convolution->GetModule()->AddComputationAndUnifyNamesAndIds(
          builder.Build(root), /*is_entry=*/false);
  return convolution->parent()->AddInstruction(HloInstruction::CreateFusion(
      root_shape, HloInstruction::FusionKind::kCustom, fusion_params,
      new_computation));
}

absl::StatusOr<bool> RunOnInstruction(HloInstruction* conv) {
  CHECK_NE(DynCast<HloConvolutionInstruction>(conv)->convolution_kind(),
           CONVOLUTION_KIND_UNSET)
      << "ConvolutionKind assignment pass must run before ConvFusionRewriter "
         "pass.";

  std::vector<HloInstruction*> fusion_outputs;
  HloInstruction* conv_fusion = CreateGpuConvFusion(conv, fusion_outputs);
  if (conv_fusion == nullptr) {
    return false;
  }

  GpuBackendConfig gpu_backend_config;
  FusionBackendConfig* fusion_config =
      gpu_backend_config.mutable_fusion_backend_config();
  fusion_config->set_kind(kCuDnnFusionKind);
  TF_RETURN_IF_ERROR(conv_fusion->set_backend_config(gpu_backend_config));

  VLOG(1) << "Replacing convolution " << conv->ToString() << " with "
          << conv_fusion->ToString();
  if (fusion_outputs.size() == 1) {
    TF_RETURN_IF_ERROR(
        conv->parent()->ReplaceInstruction(fusion_outputs[0], conv_fusion));
  } else {
    for (int idx = 0; idx < fusion_outputs.size(); ++idx) {
      HloInstruction* output = fusion_outputs[idx];
      TF_RETURN_IF_ERROR(conv->parent()->ReplaceInstruction(
          output,
          conv->parent()->AddInstruction(
              HloInstruction::CreateGetTupleElement(conv_fusion, idx))));
    }
  }
  return true;
}

absl::StatusOr<bool> RunOnComputation(HloComputation* computation) {
  std::vector<HloInstruction*> convs;
  for (auto* hlo : computation->instructions()) {
    if (HloPredicateIsOp<HloOpcode::kConvolution>(hlo)) {
      convs.push_back(hlo);
    }
  }

  bool changed = false;
  for (HloInstruction* conv : convs) {
    ASSIGN_OR_RETURN(bool result, RunOnInstruction(conv));
    changed |= result;
  }
  return changed;
}
}  // namespace

absl::StatusOr<bool> ConvFusionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(2,
                 "ConvFusionRewriter::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  XLA_VLOG_LINES(2, "ConvFusionRewriter::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
