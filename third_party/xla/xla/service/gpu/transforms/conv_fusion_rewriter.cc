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

#include "xla/service/gpu/transforms/conv_fusion_rewriter.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/window_util.h"
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

bool ShouldKeepFusingUsers(HloInstruction* hlo, bool& can_fuse_reduce) {
  // shouldn't fuse anything after reduction
  if (hlo->user_count() == 0 || hlo->opcode() == HloOpcode::kReduce)
    return false;
  // only keep fusing if all users fusible
  bool can_fuse_cache = can_fuse_reduce;
  for (HloInstruction* user : hlo->users()) {
    if (!IsOperationSupportedByCuDNN(*user, can_fuse_reduce)) {
      can_fuse_reduce = can_fuse_cache;
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
  return;
}

bool IsConvFusionOutputsValid(std::vector<HloInstruction*>& fusion_outputs) {
  // at most 2 outputs and at least 1 reduce if there are 2 outputs
  if (fusion_outputs.size() > 2) return false;
  if (fusion_outputs.size() == 2 &&
      (fusion_outputs[0]->opcode() == HloOpcode::kReduce) ==
          (fusion_outputs[1]->opcode() == HloOpcode::kReduce))
    return false;
  return true;
}

bool DFSSearch(HloInstruction* hlo, HloReachabilityMap* reachability,
               std::vector<HloInstruction*>& fusible_users,
               std::vector<HloInstruction*>& fusion_outputs,
               absl::flat_hash_map<HloInstruction*, bool>& mark,
               bool& can_fuse_reduce) {
  if (auto it = mark.find(hlo); it != mark.end()) return it->second;
  int num_users = fusible_users.size();
  int num_outputs = fusion_outputs.size();
  bool is_subgraph_valid = true;
  bool is_endpoint = !ShouldKeepFusingUsers(hlo, can_fuse_reduce);
  if (!is_endpoint) {
    for (auto user : hlo->users()) {
      is_subgraph_valid &= DFSSearch(user, reachability, fusible_users,
                                     fusion_outputs, mark, can_fuse_reduce);
      if (!is_subgraph_valid) {
        is_endpoint = true;
        break;
      }
    }
  }
  if (!is_subgraph_valid) {
    // Rollback all subgraph changes
    fusible_users.resize(num_users);
    fusion_outputs.resize(num_outputs);
  }

  if (is_endpoint) {
    // Add endpoint as fusion output
    fusion_outputs.push_back(hlo);
  }

  // Check if current fusion graph is valid
  bool is_valid = IsConvFusionOutputsValid(fusion_outputs);
  // Check if new fusion output can reach fusible users, this avoids cycle
  if (is_endpoint && is_valid) {
    is_valid = std::none_of(fusible_users.begin(), fusible_users.end(),
                            [reachability, hlo](HloInstruction* user) {
                              return reachability->IsReachable(hlo, user);
                            });
  }
  // Check if new fusible can be reached from fusion outputs, this avoids
  // cycle.
  if (is_valid) {
    // Don't check endpoint, since it is the same instruction as the new fusible
    // user.
    is_valid =
        std::none_of(fusion_outputs.begin(), fusion_outputs.end() - is_endpoint,
                     [reachability, hlo](HloInstruction* output) {
                       return reachability->IsReachable(output, hlo);
                     });
  }
  // Set visited
  mark.insert({hlo, is_valid});
  if (is_valid) {
    // Everything looks good, add to fusible user sets
    fusible_users.push_back(hlo);
  }
  return is_valid;
}

void GetAllReachableAndFusible(HloInstruction* conv,
                               std::vector<HloInstruction*>& fusible_users,
                               std::vector<HloInstruction*>& fusion_outputs,
                               HloReachabilityMap* reachability) {
  absl::flat_hash_map<HloInstruction*, bool> mark;
  bool can_fuse_reduce = true;
  DFSSearch(conv, reachability, fusible_users, fusion_outputs, mark,
            can_fuse_reduce);
  // Remove conv from the users
  fusible_users.pop_back();
  std::reverse(fusible_users.begin(), fusible_users.end());
  // Make sure reduce is after the real conv output
  if (fusion_outputs.size() == 2 &&
      fusion_outputs[0]->opcode() == HloOpcode::kReduce) {
    // Swap reduce and real conv output
    std::swap(fusion_outputs[0], fusion_outputs[1]);
  }
}

using ConvKind = HloConvolutionInstruction::ConvKind;

HloInstruction* CreateGpuConvFusion(
    HloInstruction* conv, std::vector<HloInstruction*>& fusion_outputs) {
  HloComputation* computation = conv->parent();
  HloInstruction* lhs = conv->mutable_operand(0);
  HloInstruction* rhs = conv->mutable_operand(1);
  ConvKind conv_kind = DynCast<HloConvolutionInstruction>(conv)->conv_kind();

  // Give the conv a user-friendly name.
  std::string name;
  if (conv_kind == ConvKind::FPROP) {
    name = "conv_fprop";
  } else if (conv_kind == ConvKind::DGRAD) {
    name = "conv_dgrad";
  } else if (conv_kind == ConvKind::WGRAD) {
    name = "conv_wgrad";
  }

  computation->parent()->SetAndUniquifyInstrName(conv, name);
  std::string fusion_name = absl::StrCat(conv->name(), "_fusion");
  std::string computation_name = absl::StrCat(fusion_name, "_comp");

  auto builder = HloComputation::Builder(computation_name);
  // Map from hlo to its fused hlo
  absl::flat_hash_map<HloInstruction*, HloInstruction*> fused_hlo_map;
  std::vector<HloInstruction*> fusion_params;
  std::unique_ptr<HloReachabilityMap> reachability =
      HloReachabilityMap::Build(conv->parent());

  // CuDNN conv can do fp32 = conv(fp8, fp8) and int32 = conv(int8, int8)
  auto fuse_convert = [&](HloInstruction* input) {
    HloInstruction* real_input =
        input->opcode() == HloOpcode::kConvert && input->user_count() == 1
            ? input->mutable_operand(0)
            : input;
    fusion_params.push_back(real_input);
    HloInstruction* fused_real_input =
        builder.AddInstruction(HloInstruction::CreateParameter(
            fusion_params.size() - 1, real_input->shape(), real_input->name()));
    CHECK(fused_hlo_map.insert({real_input, fused_real_input}).second);
    HloInstruction* fused_input = fused_real_input;
    if (input->opcode() == HloOpcode::kConvert) {
      fused_input = builder.AddInstruction(
          input->CloneWithNewOperands(input->shape(), {fused_real_input}));
      CHECK(fused_hlo_map.insert({input, fused_input}).second);
    }
    return fused_input;
  };
  HloInstruction* fused_lhs = fuse_convert(lhs);
  HloInstruction* fused_rhs = fuse_convert(rhs);
  HloInstruction* fused_conv = builder.AddInstruction(
      conv->CloneWithNewOperands(conv->shape(), {fused_lhs, fused_rhs}));

  CHECK(fused_hlo_map.insert({conv, fused_conv}).second);
  // get all reachable and fusible instructions from conv in reverse post order
  std::vector<HloInstruction*> fusible_users;
  GetAllReachableAndFusible(conv, fusible_users, fusion_outputs,
                            reachability.get());
  FuseTowardUsers(builder, fusion_params, fusible_users, fused_hlo_map);
  Shape root_shape;
  if (fusion_outputs.size() == 1) {
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
    builder.AddInstruction(HloInstruction::CreateTuple(roots));
    root_shape = ShapeUtil::MakeTupleShape(shapes);
  }
  HloComputation* new_computation =
      conv->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                           /*is_entry=*/false);
  return computation->AddInstruction(HloInstruction::CreateFusion(
      root_shape, HloInstruction::FusionKind::kCustom, fusion_params,
      new_computation));
}

// Helper function to create a fusion instruction to replace the given
// conv instruction
static absl::StatusOr<HloInstruction*> CreateConvFusionHelper(
    HloInstruction* conv, std::vector<HloInstruction*>& fusion_outputs) {
  if (DynCast<HloConvolutionInstruction>(conv)->conv_kind() ==
      ConvKind::UNSET) {
    return nullptr;
  }
  HloInstruction* conv_fusion = CreateGpuConvFusion(conv, fusion_outputs);

  if (conv_fusion) {
    GpuBackendConfig gpu_backend_config;
    FusionBackendConfig* fusion_config =
        gpu_backend_config.mutable_fusion_backend_config();
    fusion_config->set_kind(std::string(kCuDnnFusionKind));
    TF_RETURN_IF_ERROR(conv_fusion->set_backend_config(gpu_backend_config));
  }
  return conv_fusion;
}

// Tries to rewrite convolution and fusible instructions into cudnn fusion.
absl::StatusOr<bool> RunOnInstruction(HloInstruction* conv) {
  CHECK_EQ(conv->opcode(), HloOpcode::kConvolution);
  std::vector<HloInstruction*> fusion_outputs;
  TF_ASSIGN_OR_RETURN(HloInstruction * conv_fusion,
                      CreateConvFusionHelper(conv, fusion_outputs));
  if (conv_fusion == nullptr) {
    return false;
  }

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
    TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(conv));
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
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  XLA_VLOG_LINES(2, "ConvFusionRewriter::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
