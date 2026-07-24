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
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_utils.h"
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
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {
bool IsNCHW(const HloInstruction* absl_nullable conv) {
  if (conv == nullptr || conv->opcode() != HloOpcode::kConvolution) {
    return false;
  }
  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  const Shape& shape = conv->shape();
  if (!shape.has_layout()) {
    return false;
  }
  const absl::Span<const int64_t>& minor_to_major =
      shape.layout().minor_to_major();
  if (minor_to_major.empty()) {
    return false;
  }
  return minor_to_major[0] != dnums.output_feature_dimension();
}

bool IsConvFusionOutputsValid(const std::vector<HloInstruction*>& outputs) {
  if (outputs.size() > 2) {
    return false;
  }
  if (outputs.size() == 2 && (outputs[0]->opcode() == HloOpcode::kReduce) ==
                                 (outputs[1]->opcode() == HloOpcode::kReduce)) {
    return false;
  }
  return true;
}

std::vector<HloInstruction*> GetAllReachableAndFusible(
    HloInstruction* convolution, std::vector<HloInstruction*>& fusion_outputs) {
  std::vector<HloInstruction*> fusible_users;
  // cuDNN frontend fusions do not support grouped convolutions with epilogues.
  if (convolution->feature_group_count() > 1) {
    fusion_outputs.push_back(convolution);
    return fusible_users;
  }
  absl::flat_hash_map<HloInstruction*, bool> fusible_cache;
  std::unique_ptr<HloReachabilityMap> reachability =
      HloReachabilityMap::Build(convolution->parent());
  bool can_fuse_reduce = true;

  FusionState state{fusible_users, fusion_outputs, can_fuse_reduce};
  GrowFusionDFS(convolution, reachability.get(), state, fusible_cache,
                IsNCHW(convolution), IsConvFusionOutputsValid);

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
    absl::flat_hash_map<HloInstruction*, HloInstruction*>& fused_hlo_map,
    const se::DeviceDescription& device_info) {
  const se::CudaComputeCapability* cuda_cc =
      device_info.gpu_compute_capability().cuda_compute_capability();
  auto is_fusable_convert = [cuda_cc](const HloInstruction* hlo) {
    // CuDNN only supports convert fusions starting from Ampere.
    return cuda_cc != nullptr && cuda_cc->IsAtLeastAmpere() &&
           hlo->opcode() == HloOpcode::kConvert && hlo->user_count() == 1;
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
    HloInstruction* convolution, std::vector<HloInstruction*>& fusion_outputs,
    const se::DeviceDescription& device_info) {
  HloComputation::Builder builder = CreateConvFusionBuilder(convolution);

  // Seeding the parameters and the map for the convolution and its prologue.
  std::vector<HloInstruction*> fusion_params;
  absl::flat_hash_map<HloInstruction*, HloInstruction*> fused_hlo_map;

  // Returns fused prologue operands.
  std::pair<HloInstruction*, HloInstruction*> operands =
      TryFuseConvolutionPrologue(convolution, builder, fusion_params,
                                 fused_hlo_map, device_info);
  HloInstruction* lhs = operands.first;
  HloInstruction* rhs = operands.second;

  HloInstruction* fused_conv = builder.AddInstruction(
      convolution->CloneWithNewOperands(convolution->shape(), {lhs, rhs}));
  fused_hlo_map[convolution] = fused_conv;

  // Get all reachable and fusible instructions from a convolution in reverse
  // post order.
  std::vector<HloInstruction*> fusible_users =
      GetAllReachableAndFusible(convolution, fusion_outputs);

  FuseTowardUsers(builder, fusion_params, fusible_users, fused_hlo_map,
                  IsNCHW(convolution));

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

absl::StatusOr<bool> RunOnInstruction(
    HloInstruction* conv, const se::DeviceDescription& device_info) {
  CHECK_NE(DynCast<HloConvolutionInstruction>(conv)->convolution_kind(),
           CONVOLUTION_KIND_UNSET)
      << "ConvolutionKind assignment pass must run before ConvFusionRewriter "
         "pass.";

  std::vector<HloInstruction*> fusion_outputs;
  HloInstruction* conv_fusion =
      CreateGpuConvFusion(conv, fusion_outputs, device_info);
  if (conv_fusion == nullptr) {
    return false;
  }

  GpuBackendConfig gpu_backend_config;
  FusionBackendConfig* fusion_config =
      gpu_backend_config.mutable_fusion_backend_config();
  fusion_config->set_kind(kCuDnnFusionKind);
  RETURN_IF_ERROR(conv_fusion->set_backend_config(gpu_backend_config));

  VLOG(1) << "Replacing convolution " << conv->ToString() << " with "
          << conv_fusion->ToString();
  if (fusion_outputs.size() == 1) {
    RETURN_IF_ERROR(
        conv->parent()->ReplaceInstruction(fusion_outputs[0], conv_fusion));
  } else {
    for (int idx = 0; idx < fusion_outputs.size(); ++idx) {
      HloInstruction* output = fusion_outputs[idx];
      RETURN_IF_ERROR(conv->parent()->ReplaceInstruction(
          output,
          conv->parent()->AddInstruction(
              HloInstruction::CreateGetTupleElement(conv_fusion, idx))));
    }
  }
  return true;
}

absl::StatusOr<bool> RunOnComputation(
    HloComputation* computation, const se::DeviceDescription& device_info) {
  std::vector<HloInstruction*> convs;
  for (auto* hlo : computation->instructions()) {
    if (HloPredicateIsOp<HloOpcode::kConvolution>(hlo)) {
      convs.push_back(hlo);
    }
  }

  bool changed = false;
  for (HloInstruction* conv : convs) {
    ASSIGN_OR_RETURN(bool result, RunOnInstruction(conv, device_info));
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
    ASSIGN_OR_RETURN(bool result, RunOnComputation(computation, device_info_));
    changed |= result;
  }
  XLA_VLOG_LINES(2, "ConvFusionRewriter::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
