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

#include "xla/backends/gpu/transforms/algebraic_simplifier.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace m = ::xla::match;

absl::StatusOr<bool>
GpuAlgebraicSimplifierVisitor::TryToSinkBroadcastOperandsOfChainedAdds(
    HloInstruction* add) {
  if (!options_.enable_sink_broadcast()) {
    return false;
  }

  HloInstruction *conv, *constant_0, *broadcast_0, *add_0, *constant_1,
      *broadcast_1;
  if (!Match(add, m::AddAnyOrder(
                      m::AddAnyOrder(
                          &add_0, m::Convolution(&conv, m::Op(), m::Op()),
                          m::Broadcast(&broadcast_0, m::Constant(&constant_0))),
                      m::Broadcast(&broadcast_1, m::Constant(&constant_1))))) {
    return false;
  }

  // Skip when the broadcast shapes and dimensions don't match.
  if (!ShapeUtil::Equal(constant_0->shape(), constant_1->shape()) ||
      broadcast_0->dimensions() != broadcast_1->dimensions()) {
    return false;
  }

  HloInstruction* new_constant_add =
      add->AddInstruction(HloInstruction::CreateBinary(
          constant_0->shape(), HloOpcode::kAdd, constant_0, constant_1));
  HloInstruction* new_bcast =
      add->AddInstruction(HloInstruction::CreateBroadcast(
          broadcast_0->shape(), new_constant_add, broadcast_0->dimensions()));
  TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
      add, HloInstruction::CreateBinary(add->shape(), HloOpcode::kAdd,
                                        new_bcast, conv)));
  return true;
}

absl::Status GpuAlgebraicSimplifierVisitor::HandleAdd(HloInstruction* add) {
  TF_ASSIGN_OR_RETURN(bool replaced,
                      TryToSinkBroadcastOperandsOfChainedAdds(add));
  if (replaced) {
    return absl::OkStatus();
  }

  return AlgebraicSimplifierVisitor::HandleAdd(add);
}

bool GpuAlgebraicSimplifierVisitor::SupportedDotPrecisionConfig(
    const PrecisionConfig& config, bool has_contracting_dim) {
  if (!has_contracting_dim) {
    return config.algorithm() == PrecisionConfig::ALG_UNSET ||
           config.algorithm() == PrecisionConfig::ALG_DOT_F32_F32_F32;
  }
  return config.algorithm() == PrecisionConfig::ALG_UNSET ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_TF32_TF32_F32 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3 ||
         config.algorithm() == PrecisionConfig::ALG_DOT_F32_F32_F32;
}

absl::StatusOr<HloInstruction*>
GpuAlgebraicSimplifierVisitor::MakeMultiplyForPrecisionAlgorithm(
    HloInstruction* dot, HloInstruction* lhs, HloInstruction* rhs) {
  return MakeMultiplyForDotPrecisionAlgorithm(
      lhs, rhs, dot->precision_config().algorithm());
}

absl::StatusOr<bool> GpuAlgebraicSimplifier::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, "GpuAlgebraicSimplifier::RunImpl(), before:\n" + module->ToString());
  bool changed = false;
  GpuAlgebraicSimplifierVisitor visitor(options_, compute_capability_, this);
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    if (visitor.Run(comp, options_, this)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(
      2, "GpuAlgebraicSimplifier::RunImpl(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla::gpu
