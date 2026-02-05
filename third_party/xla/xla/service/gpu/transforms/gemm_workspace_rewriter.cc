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

#include "xla/service/gpu/transforms/gemm_workspace_rewriter.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;
using se::gpu::BlasLt;

namespace {

absl::StatusOr<BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
      return BlasLt::Epilogue::kDefault;
    case GemmBackendConfig::RELU:
      return BlasLt::Epilogue::kReLU;
    case GemmBackendConfig::GELU:
      return BlasLt::Epilogue::kGELU;
    case GemmBackendConfig::GELU_AUX:
      return BlasLt::Epilogue::kGELUWithAux;
    case GemmBackendConfig::BIAS:
      return BlasLt::Epilogue::kBias;
    case GemmBackendConfig::BIAS_RELU:
      return BlasLt::Epilogue::kBiasThenReLU;
    case GemmBackendConfig::BIAS_GELU:
      return BlasLt::Epilogue::kBiasThenGELU;
    case GemmBackendConfig::BIAS_GELU_AUX:
      return BlasLt::Epilogue::kBiasThenGELUWithAux;
    default:
      return absl::InternalError("Unsupported Epilogue.");
  }
}

// Visitor that updates workspace sizes for cuBLASLt GEMM operations
// based on the selected algorithm's actual workspace requirement.
class GemmWorkspaceRewriteVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmWorkspaceRewriteVisitor(
      const se::GpuComputeCapability& gpu_version,
      se::StreamExecutor* stream_exec)
      : gpu_version_(gpu_version), stream_exec_(stream_exec) {}

  absl::Status HandleCustomCall(HloInstruction* instr) override {
    // Only handle cuBLASLt matmul calls
    if (instr->custom_call_target() != kCublasLtMatmulCallTarget &&
        instr->custom_call_target() != kCublasLtMatmulF8CallTarget) {
      return absl::OkStatus();
    }

    // Skip if stream executor is not available
    if (stream_exec_ == nullptr) {
      return absl::OkStatus();
    }

    // Get the backend config
    ASSIGN_OR_RETURN(auto gpu_config,
                     instr->backend_config<GpuBackendConfig>());
    const GemmBackendConfig& config = gpu_config.gemm_backend_config();

    // Skip if no algorithm has been selected (not autotuned yet)
    if (config.algorithm_case() != GemmBackendConfig::kSelectedAlgorithm) {
      return absl::OkStatus();
    }

    int64_t selected_algorithm = config.selected_algorithm();

    // Get the current output shape - must be a tuple with workspace as last
    // element
    if (!instr->shape().IsTuple() || instr->shape().tuple_shapes().empty()) {
      return absl::OkStatus();
    }

    // Get the current workspace size
    const Shape& current_workspace_shape = instr->shape().tuple_shapes().back();
    if (current_workspace_shape.element_type() != S8) {
      return absl::OkStatus();
    }
    int64_t current_workspace_size =
        ShapeUtil::ByteSizeOf(current_workspace_shape);

    // Create GemmConfig to get the matmul plan
    ASSIGN_OR_RETURN(GemmConfig gemm_config,
                     GemmConfig::For(instr, gpu_version_));

    // Get the epilogue
    ASSIGN_OR_RETURN(BlasLt::Epilogue epilogue,
                     AsBlasLtEpilogue(config.epilogue()));

    // Create a stream to query algorithms
    ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                     stream_exec_->CreateStream());

    // Get the matmul plan
    ASSIGN_OR_RETURN(
        std::unique_ptr<BlasLt::MatmulPlan> plan,
        se::gpu::BlasLt::GetMatmulPlan(stream.get(), gemm_config, epilogue));

    // Query algorithms with the current workspace size limit
    ASSIGN_OR_RETURN(
        std::vector<BlasLt::MatmulAlgorithm> algorithms,
        plan->GetAlgorithms(stream.get(), GemmConfig::kNumAlgorithms,
                            current_workspace_size));

    // Verify that the selected algorithm index is valid
    if (selected_algorithm < 0 ||
        selected_algorithm >= static_cast<int64_t>(algorithms.size())) {
      VLOG(3) << "Selected algorithm index " << selected_algorithm
              << " is out of range for " << instr->name()
              << ", skipping workspace update.";
      return absl::OkStatus();
    }

    // Get the actual workspace size for the selected algorithm
    int64_t actual_workspace_size =
        static_cast<int64_t>(algorithms[selected_algorithm].workspace_size);

    // If the workspace size is already optimal, nothing to do
    if (actual_workspace_size == current_workspace_size) {
      return absl::OkStatus();
    }

    // Ensure we're not increasing the workspace size
    if (actual_workspace_size > current_workspace_size) {
      VLOG(3) << "Algorithm workspace size (" << actual_workspace_size
              << ") exceeds current allocation (" << current_workspace_size
              << ") for " << instr->name() << ", skipping update.";
      return absl::OkStatus();
    }

    VLOG(2) << "Updating workspace size for " << instr->name() << " from "
            << current_workspace_size << " to " << actual_workspace_size;

    // Build the new output shape with updated workspace size
    Shape new_output_shape = instr->shape();
    *new_output_shape.mutable_tuple_shapes(
        new_output_shape.tuple_shapes().size() - 1) =
        ShapeUtil::MakeShape(S8, {actual_workspace_size});

    // Clone the instruction with the new shape
    HloInstruction* new_call = instr->AddInstruction(
        instr->CloneWithNewOperands(new_output_shape, instr->operands()));

    // Update operand aliasing if present
    auto* custom_call = Cast<HloCustomCallInstruction>(new_call);
    if (!custom_call->output_to_operand_aliasing().empty()) {
      custom_call->set_output_to_operand_aliasing(
          Cast<HloCustomCallInstruction>(instr)->output_to_operand_aliasing());
    }

    // Collect users first to avoid modifying during iteration
    std::vector<HloInstruction*> users(instr->users().begin(),
                                       instr->users().end());

    // Replace all users of the old instruction
    for (HloInstruction* user : users) {
      HloGetTupleElementInstruction* user_get_tuple =
          DynCast<HloGetTupleElementInstruction>(user);
      if (user_get_tuple == nullptr) {
        continue;
      }
      HloInstruction* get_output =
          instr->AddInstruction(HloInstruction::CreateGetTupleElement(
              new_call, user_get_tuple->tuple_index()));
      RETURN_IF_ERROR(ReplaceInstruction(user_get_tuple, get_output));
    }

    MarkAsChanged();
    return absl::OkStatus();
  }

 private:
  se::GpuComputeCapability gpu_version_;
  se::StreamExecutor* stream_exec_;
};

}  // namespace

absl::StatusOr<bool> GemmWorkspaceRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Skip if stream executor is not available
  if (stream_exec_ == nullptr) {
    VLOG(2) << "Stream executor not available, skipping workspace rewrite.";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    GemmWorkspaceRewriteVisitor visitor(gpu_version_, stream_exec_);
    RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
