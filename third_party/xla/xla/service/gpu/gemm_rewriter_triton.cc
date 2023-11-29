/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/gemm_rewriter_triton.h"

#include <array>
#include <cstdint>
#include <list>
#include <optional>
#include <queue>
#include <string>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_padding_requirements.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/gpu/triton_support.h"
#include "xla/service/gpu/triton_tiling_propagation.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/tensor_float_32_utils.h"

namespace xla {
namespace gpu {

namespace {

using triton_fusion::DimOrdersAndReqs;
using triton_fusion::DimOrdersAndReqsOrError;
using triton_fusion::FusionContext;
using triton_fusion::GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible;
using triton_fusion::TransformDirection;

using OldToNewHloMap =
    absl::flat_hash_map<const HloInstruction*, HloInstruction*>;

// Gets the fused HLO corresponding to `hlo` or adds a new parameter if not
// found.
HloInstruction* GetFusedHloOrAddParameter(
    HloInstruction& hlo, OldToNewHloMap& old_to_new_map,
    std::vector<HloInstruction*>& fusion_inputs,
    HloComputation::Builder& builder) {
  if (auto it = old_to_new_map.find(&hlo); it != old_to_new_map.end()) {
    return it->second;
  }
  fusion_inputs.push_back(&hlo);
  return old_to_new_map
      .insert(
          {&hlo, builder.AddInstruction(HloInstruction::CreateParameter(
                     fusion_inputs.size() - 1, hlo.shape(),
                     absl::StrCat("parameter_", fusion_inputs.size() - 1)))})
      .first->second;
}

// Clone an instruction into the fusion.
//
// For the hero dot operation in the dot fusion, please use FuseDotOnly.
void Fuse(HloInstruction& hlo, OldToNewHloMap& old_to_new_map,
          std::vector<HloInstruction*>& fusion_inputs,
          HloComputation::Builder& builder) {
  if (old_to_new_map.contains(&hlo)) {
    return;
  }
  VLOG(3) << "Fusing " << hlo.ToString();
  if (hlo.opcode() == HloOpcode::kParameter ||
      hlo.opcode() == HloOpcode::kGetTupleElement) {
    GetFusedHloOrAddParameter(hlo, old_to_new_map, fusion_inputs, builder);
  } else {
    std::vector<HloInstruction*> hlo_new_operands;
    for (HloInstruction* operand : hlo.operands()) {
      hlo_new_operands.push_back(GetFusedHloOrAddParameter(
          *operand, old_to_new_map, fusion_inputs, builder));
    }
    old_to_new_map[&hlo] = builder.AddInstruction(
        hlo.CloneWithNewOperands(hlo.shape(), hlo_new_operands));
  }
}

// Clones the hero kDot operation into the fusion.
void FuseDotOnly(HloInstruction& hlo, OldToNewHloMap& output_old_to_new_map,
                 OldToNewHloMap& lhs_old_to_new_map,
                 OldToNewHloMap& rhs_old_to_new_map,
                 std::vector<HloInstruction*>& fusion_inputs,
                 HloComputation::Builder& builder) {
  CHECK_EQ(hlo.opcode(), HloOpcode::kDot);
  CHECK_EQ(hlo.operand_count(), 2);
  VLOG(3) << "Fusing " << hlo.ToString();

  std::array<HloInstruction*, 2> hlo_new_operands = {
      GetFusedHloOrAddParameter(*hlo.mutable_operand(0), lhs_old_to_new_map,
                                fusion_inputs, builder),
      GetFusedHloOrAddParameter(*hlo.mutable_operand(1), rhs_old_to_new_map,
                                fusion_inputs, builder)};
  output_old_to_new_map[&hlo] = builder.AddInstruction(
      hlo.CloneWithNewOperands(hlo.shape(), hlo_new_operands));
}

// Tells how many new parameters does a fusion gain by fusing the operation as
// an input.
int64_t NumAddedParameters(const HloInstruction& hlo) {
  // Non-scalar constant is equivalent to a parameter: one input, one output.
  if (hlo.opcode() == HloOpcode::kConstant &&
      !ShapeUtil::IsScalar(hlo.shape())) {
    return 0;
  }
  // All other instructions add all own inputs and remove own single output.
  return hlo.operand_count() - 1;
}

// Fuse an instruction with all its fusible inputs.
// If an input is not fusible stop there and make a parameter of the new
// fusion, otherwise put it onto stack and check its own inputs first.
void TryToFuseWithInputsRecursively(HloInstruction& root,
                                    se::GpuComputeCapability gpu_version,
                                    triton_fusion::FusionContext& context,
                                    OldToNewHloMap& old_to_new_map,
                                    std::vector<HloInstruction*>& fusion_inputs,
                                    HloComputation::Builder& builder) {
  // Instructions at the fusion edge that can either get fused too or
  // become parameters of the fusion. Used to track the number of parameters.
  absl::flat_hash_set<const HloInstruction*> inputs;
  // Traverse all connected instructions that could be fused, analyze them and
  // collect ones that will be fused.
  absl::flat_hash_set<const HloInstruction*> to_fuse_set;
  std::list<HloInstruction*> to_fuse_list;
  absl::flat_hash_set<const HloInstruction*> enqueued;
  std::queue<HloInstruction*> to_visit;
  to_visit.push(&root);
  int num_requeued = 0;
  while (to_visit.size() > num_requeued) {
    HloInstruction* hlo = to_visit.front();
    to_visit.pop();
    // Watch the total number of fusion parameters.
    if (inputs.size() + NumAddedParameters(*hlo) >
        TritonFusionAnalysis::kMaxParameterPerDotScope) {
      // Re-queue: the number of parameters may go down when other instructions
      // are processed.
      to_visit.push(hlo);
      // Prevent infinite loops.
      ++num_requeued;
      continue;
    }
    num_requeued = 0;
    const DimOrdersAndReqsOrError result =
        GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
            *hlo, TransformDirection::kOutputToInput,
            /*src_operand_index=*/std::nullopt, context.dim_orders().at(hlo),
            gpu_version, context.hero_properties());
    if (!std::holds_alternative<DimOrdersAndReqs>(result) ||
        !context.CombineDimOrdersAndReqs(std::get<DimOrdersAndReqs>(result))) {
      continue;
    }
    if (hlo->opcode() != HloOpcode::kParameter) {
      inputs.erase(hlo);
    }
    inputs.insert(hlo->operands().cbegin(), hlo->operands().cend());
    to_fuse_set.insert(hlo);
    to_fuse_list.push_back(hlo);
    for (HloInstruction* operand : hlo->operands()) {
      if (enqueued.insert(operand).second) {
        VLOG(6) << "Enqueueing " << operand->ToString();
        to_visit.push(operand);
      }
    }
  }
  // Find one by one instructions that have no operands queued to be fused and
  // fuse them.
  while (!to_fuse_list.empty()) {
    for (auto it = to_fuse_list.begin(); it != to_fuse_list.end();) {
      bool ready_to_fuse = true;
      for (const HloInstruction* operand : (*it)->operands()) {
        if (to_fuse_set.contains(operand)) {
          ready_to_fuse = false;
          break;
        }
      }
      if (ready_to_fuse) {
        Fuse(**it, old_to_new_map, fusion_inputs, builder);
        to_fuse_set.erase(*it);
        it = to_fuse_list.erase(it);
      } else {
        ++it;
      }
    }
  }
}

// Fuses dot and the compatible and profitable to fuse operations around it
// into a new fusion computation constructed using the builder. fusion_inputs
// get populated with the non-fused instructions that become operands of the
// call to this fusion. fusion_output_ptr (if not nullptr) gets assigned the
// original instruction that has to be replaced by the call to the fusion.
StatusOr<FusionDecision> FuseDot(HloInstruction& dot,
                                 const se::GpuComputeCapability gpu_version,
                                 HloComputation::Builder& builder,
                                 std::vector<HloInstruction*>& fusion_inputs,
                                 HloInstruction** fusion_output_ptr) {
  VLOG(5) << dot.ToString();
  if (FusionDecision can_handle = CanTritonHandleGEMM(dot, gpu_version);
      !can_handle) {
    VLOG(3) << can_handle.Explain();
    return can_handle;
  }

  // Separate traversal from LHS and RHS inputs of the dot: they use
  // differently shaped tiles but may go through same HLO graph nodes.
  // Direct dot inputs have well defined dimension orders.

  auto fuse_inputs =
      [&](int operand_number,
          OldToNewHloMap& old_to_new_map) -> StatusOr<FusionContext> {
    const int operand_count_before = fusion_inputs.size();
    // Direct dot inputs have well defined dimension orders.
    auto context = FusionContext::FromDotOperand(dot, operand_number);
    TryToFuseWithInputsRecursively(*dot.mutable_operand(operand_number),
                                   gpu_version, context, old_to_new_map,
                                   fusion_inputs, builder);
    const int new_parameters = fusion_inputs.size() - operand_count_before;
    TF_RET_CHECK(new_parameters <=
                 TritonFusionAnalysis::kMaxParameterPerDotScope)
        << "Too many new parameters: " << new_parameters << " > "
        << TritonFusionAnalysis::kMaxParameterPerDotScope;
    return context;
  };

  // Original instruction -> fused one. Separate for each scope.
  OldToNewHloMap lhs_old_to_new_map;
  TF_ASSIGN_OR_RETURN(const FusionContext lhs_context,
                      fuse_inputs(0, lhs_old_to_new_map));

  OldToNewHloMap rhs_old_to_new_map;
  if (auto result = fuse_inputs(1, rhs_old_to_new_map); !result.ok()) {
    return result.status();
  }

  OldToNewHloMap output_old_to_new_map;
  // Fuse the dot into output_old_to_new_map and use lhs_old_to_new_map and
  // rhs_old_to_new_map to generate / determine its operands.
  FuseDotOnly(dot, output_old_to_new_map, lhs_old_to_new_map,
              rhs_old_to_new_map, fusion_inputs, builder);

  // Fusion at dot's output.

  // These describe _outputs_ of corresponding HLOs.
  auto context = FusionContext::FromDotOutput(
      dot, /*split_k=*/1, lhs_context.splittable_dimension_major_part_size());
  HloInstruction* fusion_output = &dot;
  bool output_changed = true;
  while (output_changed) {
    output_changed = false;
    if (fusion_output->user_count() != 1) {
      break;
    }
    HloInstruction* user = fusion_output->users()[0];
    if (!IsDistributiveOverAddition(*user)) {
      break;
    }
    DimOrdersAndReqsOrError result =
        GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
            *user, TransformDirection::kInputToOutput,
            user->operand_index(fusion_output),
            context.dim_orders().at(fusion_output), gpu_version,
            context.hero_properties());
    if (!std::holds_alternative<DimOrdersAndReqs>(result) ||
        !context.CombineDimOrdersAndReqs(std::get<DimOrdersAndReqs>(result))) {
      break;
    }
    for (HloInstruction* operand : user->operands()) {
      if (!output_old_to_new_map.contains(operand)) {
        TryToFuseWithInputsRecursively(*operand, gpu_version, context,
                                       output_old_to_new_map, fusion_inputs,
                                       builder);
      }
    }
    Fuse(*user, output_old_to_new_map, fusion_inputs, builder);
    fusion_output = user;
    output_changed = true;
  }
  if (fusion_output_ptr != nullptr) {
    *fusion_output_ptr = fusion_output;
  }
  if (dot.GetModule()->config().debug_options().xla_gpu_triton_gemm_any()) {
    return FusionDecision{};
  }

  for (auto* old_to_new_map : std::array<const OldToNewHloMap*, 3>{
           &lhs_old_to_new_map, &rhs_old_to_new_map, &output_old_to_new_map}) {
    for (auto [_, new_hlo] : *old_to_new_map) {
      static constexpr std::array<HloOpcode, 4> kPureOpcodes = {
          HloOpcode::kBitcast, HloOpcode::kDot, HloOpcode::kParameter,
          HloOpcode::kReshape};
      // Fuse if this is not a "pure" matmul.
      if (absl::c_find(kPureOpcodes, new_hlo->opcode()) == kPureOpcodes.end()) {
        return FusionDecision{};
      }
    }
  }
  return "No profitable operations to fuse.";
}

// Extracts into fused computations parts of HLO graph including dot()
// operations that can target the triton GEMM emitter.
class GemmRewriterTritonVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmRewriterTritonVisitor(const se::GpuComputeCapability gpu_version)
      : gpu_version_(gpu_version) {}
  // Checks that a dot() should be targeting the triton GEMM emitter;
  // if so - fuses all its compatible inputs and outputs as a new computation
  // and replaces the original dot() with a call to the computation.
  Status HandleDot(HloInstruction* dot) override {
    std::string fusion_name = absl::StrCat("triton_gemm_", dot->name());
    HloComputation::Builder builder(absl::StrCat(fusion_name, "_computation"));
    std::vector<HloInstruction*> fusion_inputs;
    HloInstruction* fusion_output = nullptr;
    TF_ASSIGN_OR_RETURN(
        const FusionDecision should_fuse,
        FuseDot(*dot, gpu_version_, builder, fusion_inputs, &fusion_output));
    if (builder.last_added_instruction() == nullptr) {
      return OkStatus();
    }
    // If a GEMM requiring padding for cuBLAS is encountered here this
    // happened because earlier ShouldTritonHandleGEMM() accepted it and padding
    // was skipped. Accept it ignoring profitability checks.
    if (!CublasRequiresPadding(
            *Cast<HloDotInstruction>(dot),
            std::get<se::CudaComputeCapability>(gpu_version_)) &&
        !should_fuse) {
      return OkStatus();
    }

    HloComputation* computation =
        dot->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                            /*is_entry=*/false);
    HloInstruction* dot_fusion =
        dot->parent()->AddInstruction(HloInstruction::CreateFusion(
            computation->root_instruction()->shape(),
            HloInstruction::FusionKind::kCustom, fusion_inputs, computation));
    dot_fusion->GetModule()->SetAndUniquifyInstrName(dot_fusion, fusion_name);

    TF_ASSIGN_OR_RETURN(auto backend_config,
                        dot_fusion->backend_config<FusionBackendConfig>());
    backend_config.set_kind(std::string(kTritonGemmFusionKind));
    TF_RETURN_IF_ERROR(dot_fusion->set_backend_config(backend_config));

    if (fusion_output->IsRoot()) {
      fusion_output->parent()->set_root_instruction(dot_fusion);
      TF_RETURN_IF_ERROR(
          fusion_output->parent()->RemoveInstructionAndUnusedOperands(
              fusion_output));
      MarkAsChanged();
    } else {
      TF_RETURN_IF_ERROR(ReplaceInstruction(fusion_output, dot_fusion));
    }
    XLA_VLOG_LINES(5, computation->ToString(HloPrintOptions::ShortParsable()));
    return OkStatus();
  }

 private:
  se::GpuComputeCapability gpu_version_;
};

StatusOr<bool> RunOnComputation(HloComputation* computation,
                                se::GpuComputeCapability gpu_version) {
  GemmRewriterTritonVisitor visitor(gpu_version);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

}  // namespace

FusionDecision CanTritonHandleGEMM(const HloInstruction& dot,
                                   const se::GpuComputeCapability gpu_version) {
  if (dot.opcode() != HloOpcode::kDot ||
      !tsl::tensor_float_32_execution_enabled() ||
      absl::c_any_of(dot.precision_config().operand_precision(),
                     [](int x) { return x != PrecisionConfig::DEFAULT; })) {
    return "Non-default precision.";
  }

  auto supported_output_type = [&](const PrimitiveType t) {
    const auto cuda_compute_capability =
        std::get<se::CudaComputeCapability>(gpu_version);
    switch (t) {
      case F16:
      case F32:
        return true;
      case BF16:
        return cuda_compute_capability.IsAtLeast(
            stream_executor::CudaComputeCapability::AMPERE);
      default:
        return false;
    }
  };

  // TODO(b/266862493): Support more output types.
  if (!supported_output_type(dot.shape().element_type())) {
    return "Unsupported output data type.";
  }

  if (!IsTritonSupportedDataType(dot.operand(0)->shape().element_type(),
                                 gpu_version) ||
      !IsTritonSupportedDataType(dot.operand(1)->shape().element_type(),
                                 gpu_version)) {
    return "Unsupported input data type.";
  }

  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  // TODO(b/269580541): support multiple batch dimensions.
  if (dim_numbers.lhs_batch_dimensions().size() > 1) {
    return "Multiple batch dimensions.";
  }

  // Cases where lhs or rhs have no non-contracting dims are not handled.
  if (dim_numbers.lhs_batch_dimensions().size() +
              dim_numbers.lhs_contracting_dimensions().size() ==
          dot.operand(0)->shape().rank() ||
      dim_numbers.rhs_batch_dimensions().size() +
              dim_numbers.rhs_contracting_dimensions().size() ==
          dot.operand(1)->shape().rank()) {
    return "No non-contracting dimensions.";
  }

  for (int operand_number = 0; operand_number <= 1; ++operand_number) {
    // This pass relies on dot decomposer which ensures that all non-contracting
    // dimensions are merged into one. Using NonContractingDimensionIndex is
    // sufficient.
    const int64_t nc_size =
        dot.operand(operand_number)
            ->shape()
            .dimensions(NonContractingDimensionIndex(dot, operand_number));
    if (nc_size <= 1) {
      return "Trivial non-contracting dimensions.";
    }
  }

  return FusionDecision{};
}

bool ShouldTritonHandleGEMM(HloInstruction& dot,
                            const se::GpuComputeCapability gpu_version) {
  std::vector<HloInstruction*> fusion_inputs;
  HloComputation::Builder builder("disposable");
  return FuseDot(dot, gpu_version, builder, fusion_inputs,
                 /*fusion_output_ptr=*/nullptr)
      ->CanFuse();
}

StatusOr<bool> GemmRewriterTriton::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result,
                        RunOnComputation(computation, gpu_version_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
