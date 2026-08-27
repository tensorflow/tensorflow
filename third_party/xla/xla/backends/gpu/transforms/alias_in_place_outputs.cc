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

#include "xla/backends/gpu/transforms/alias_in_place_outputs.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

bool IsTritonFusion(const HloFusionInstruction& fusion) {
  auto config = fusion.backend_config<GpuBackendConfig>();
  if (!config.ok()) {
    return false;
  }
  absl::string_view kind = config->fusion_backend_config().kind();
  return kind == kTritonFusionKind || kind == kTritonGemmFusionKind ||
         kind == kTritonNestedGemmFusionKind;
}

bool IsSupportedCublasLtMatmul(const HloInstruction& hlo) {
  return IsCublasLtMatmul(hlo) || IsCublasLtMatmulF8(hlo) ||
         IsCublasLtMatmulMx(hlo) || IsCublasLtGroupedMatmul(hlo);
}

// Returns true if the composed output→input indexing map for the given operand
// of the fusion is an identity, i.e. output element [i,j,...] reads input
// element [i,j,...]. This covers pure elementwise chains and bitcasts that
// cancel. Operands read through multiple data-flow paths are rejected.
bool HasIdentityIndexing(const GroupedByOpIndexing& grouped,
                         const HloInstruction* operand) {
  auto it = grouped.find(operand);
  if (it == grouped.end() || it->second.size() != 1) {
    return false;
  }
  const IndexingMap& map = it->second.begin()->map();
  return map.GetSymbolicMap().IsIdentity() && map.GetRangeVars().empty() &&
         map.GetSymbolicConstraints().empty();
}

// Returns true if overwriting `operand` in place avoids rather than forces a
// copy: `operand` must be a writable intermediate (not a parameter, constant,
// or root) that `user` reads exactly once, and every other user of `operand`
// must precede `user` in the graph (e.g. a residual read before the op
// overwrites it).
bool BeneficialToAlias(const HloInstruction* operand,
                       const HloInstruction* user,
                       const HloReachabilityMap& reachability) {
  if (operand->opcode() == HloOpcode::kParameter ||
      operand->opcode() == HloOpcode::kConstant || operand->IsRoot()) {
    return false;
  }
  if (absl::c_count(user->operands(), operand) != 1) {
    // Don't alias if the same buffer is used multiple times in the user.
    return false;
  }
  return absl::c_all_of(operand->users(), [&](const HloInstruction* other) {
    return other == user || reachability.IsReachable(other, user);
  });
}

bool AliasFusion(HloFusionInstruction& fusion,
                 const HloReachabilityMap& reachability,
                 mlir::MLIRContext& mlir_ctx) {
  if (!fusion.shape().IsArray() ||
      !fusion.output_to_operand_aliasing().empty()) {
    return false;
  }
  std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
      HloFusionAdaptor::ForInstruction(&fusion);
  GroupedByOpIndexing grouped = ComputeGroupedOutputToInputIndexing(
      *fusion_adaptor, fusion_adaptor->GetRoots()[0], &mlir_ctx);
  const HloInstruction* root = fusion.fused_expression_root();
  for (int64_t i = 0; i < fusion.operand_count(); ++i) {
    const HloInstruction* operand = fusion.operand(i);
    // Aliasing requires matching element type and dimensions (layouts may
    // differ), as enforced by the HLO verifier.
    if (!ShapeUtil::Compatible(operand->shape(), root->shape())) {
      continue;
    }
    if (!HasIdentityIndexing(grouped, operand)) {
      continue;
    }
    if (!BeneficialToAlias(operand, &fusion, reachability)) {
      continue;
    }
    VLOG(2) << "Aliasing output of " << fusion.name() << " to operand " << i
            << " (" << operand->name() << ")";
    fusion.set_output_to_operand_aliasing(
        {{/*output_index=*/{}, {/*operand_number=*/i, /*operand_index=*/{}}}});
    return true;
  }
  return false;
}

bool AliasCublasLtMatmul(HloCustomCallInstruction& custom_call,
                         const HloReachabilityMap& reachability) {
  if (!custom_call.output_to_operand_aliasing().empty()) {
    return false;
  }

  const int64_t bias_idx = IsCublasLtGroupedMatmul(custom_call) ? 3 : 2;
  if (custom_call.operand_count() <= bias_idx) {
    return false;
  }

  auto config = custom_call.backend_config<GpuBackendConfig>();
  if (!config.ok() || config->gemm_backend_config().beta() == 0.0) {
    return false;
  }

  // cuBLASLt returns either the result array or a (result, scratch) tuple.
  const Shape& shape = custom_call.shape();
  ShapeIndex output_index;
  const Shape* output_shape = &shape;
  if (shape.IsTuple()) {
    if (shape.tuple_shapes().empty()) {
      return false;
    }
    output_index = ShapeIndex{0};
    output_shape = &shape.tuple_shapes(0);
  } else if (!shape.IsArray()) {
    return false;
  }

  const HloInstruction* bias = custom_call.operand(bias_idx);
  if (!ShapeUtil::Equal(bias->shape(), *output_shape)) {
    return false;
  }
  if (!BeneficialToAlias(bias, &custom_call, reachability)) {
    return false;
  }

  VLOG(2) << "Aliasing output " << output_index.ToString() << " of "
          << custom_call.name() << " to operand " << bias_idx << " ("
          << bias->name() << ")";
  custom_call.set_output_to_operand_aliasing(
      {{output_index, {/*operand_number=*/bias_idx, /*operand_index=*/{}}}});
  return true;
}

}  // namespace

absl::StatusOr<bool> AliasInPlaceOutputs::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  RegisterSymbolicExprStorage(mlir_context_);
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::unique_ptr<HloReachabilityMap> reachability;
    // Only build the reachability map if needed, reuse if already built.
    const auto reachability_of = [&]() -> const HloReachabilityMap& {
      if (reachability == nullptr) {
        reachability = HloReachabilityMap::Build(computation);
      }
      return *reachability;
    };
    for (HloInstruction* instr : computation->instructions()) {
      if (auto* fusion = DynCast<HloFusionInstruction>(instr);
          fusion != nullptr && IsTritonFusion(*fusion)) {
        changed |= AliasFusion(*fusion, reachability_of(), *mlir_context_);
      } else if (auto* call = DynCast<HloCustomCallInstruction>(instr);
                 call != nullptr && IsSupportedCublasLtMatmul(*call)) {
        changed |= AliasCublasLtMatmul(*call, reachability_of());
      }
    }
  }
  return changed;
}

}  // namespace xla::gpu
