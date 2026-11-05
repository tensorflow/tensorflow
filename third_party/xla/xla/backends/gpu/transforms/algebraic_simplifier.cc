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

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/conv_rewriter.h"
#include "xla/backends/gpu/transforms/one_hot_gather_utils.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pass_registry.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier_options.pb.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace m = ::xla::match;

namespace {

bool IsScalarConstantZero(const HloInstruction* hlo) {
  if (!hlo->IsConstant() || !ShapeUtil::IsScalar(hlo->shape())) {
    return false;
  }
  return primitive_util::PrimitiveTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          return hlo->literal().GetFirstElement<NativeT>() ==
                 static_cast<NativeT>(0);
        }
        return false;
      },
      hlo->shape().element_type());
}

// Peels at most one widening (or equal-width) byte-aligned non-PRED array
// `kConvert` so that narrow inputs (e.g. bf16/f16 converted to f32) are
// gathered in their narrow type before converting the 1-element result to
// `out_type`. Peeling at most one `kConvert` across the broadcast boundary
// ensures intermediate rounding conversions (e.g. `f32 -> bf16 -> f32`) are
// preserved.
HloInstruction* PeelSingleSafeConvert(HloInstruction* instr,
                                      PrimitiveType out_type) {
  if (instr->opcode() != HloOpcode::kConvert) {
    return instr;
  }
  HloInstruction* src = instr->mutable_operand(0);
  const PrimitiveType src_type = src->shape().element_type();
  if (!primitive_util::IsArrayType(src_type) || src_type == PRED ||
      primitive_util::BitWidth(src_type) < 8 ||
      primitive_util::BitWidth(src_type) > primitive_util::BitWidth(out_type)) {
    return instr;
  }
  return src;
}

// Matched components of a one-hot masked sum reduction:
//   reduce_sum(select(compare(broadcast(indices), iota, EQ),
//                     on_true, broadcast(0)), init_value=0)
struct OneHotReduceMatch {
  HloInstruction* init_value;
  HloInstruction* indices_broadcast;
  HloInstruction* indices;
  HloInstruction* on_true;
  int64_t reduce_dim;
  int64_t reduce_dim_size;
};

// Phase 1 (early rejection): checks whether `reduce` is a compatible one-hot
// masked sum reduction that can be rewritten into a 1-element gather. Returns
// the matched HLO components, or `std::nullopt` if incompatible.
std::optional<OneHotReduceMatch> MatchOneHotReduceToGather(
    HloInstruction* reduce, const AlgebraicSimplifierOptions& options) {
  if (options.is_layout_sensitive() || reduce->shape().IsTuple() ||
      !reduce->shape().is_static() ||
      ShapeUtil::IsZeroElementArray(reduce->shape()) ||
      reduce->dimensions().size() != 1) {
    return std::nullopt;
  }
  HloInstruction* arg = reduce->mutable_operand(0);
  HloInstruction* init_value = reduce->mutable_operand(1);
  if (!arg->shape().is_static()) {
    return std::nullopt;
  }
  const int64_t reduce_dim = reduce->dimensions(0);
  const int64_t reduce_dim_size = arg->shape().dimensions(reduce_dim);
  if (reduce_dim_size <= 1 ||
      reduce_dim_size <= options.very_small_gather_size()) {
    return std::nullopt;
  }
  if (!IsScalarConstantZero(init_value) ||
      !Match(reduce->to_apply()->root_instruction(),
             m::AddAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    return std::nullopt;
  }

  HloInstruction* pred = nullptr;
  HloInstruction* on_true = nullptr;
  HloInstruction* zero_scalar = nullptr;
  if (!Match(arg, m::Select(m::Op(&pred), m::Op(&on_true),
                            m::Broadcast(m::Op(&zero_scalar)))) ||
      !arg->operand(2)->dimensions().empty() ||
      !IsScalarConstantZero(zero_scalar)) {
    return std::nullopt;
  }

  HloInstruction* compare_lhs = nullptr;
  HloInstruction* compare_rhs = nullptr;
  if (!Match(pred, m::Compare(m::Op(&compare_lhs), m::Op(&compare_rhs))) ||
      pred->comparison_direction() != ComparisonDirection::kEq) {
    return std::nullopt;
  }

  HloInstruction* indices_broadcast = nullptr;
  if (compare_lhs->opcode() == HloOpcode::kIota &&
      Cast<HloIotaInstruction>(compare_lhs)->iota_dimension() == reduce_dim) {
    indices_broadcast = compare_rhs;
  } else if (compare_rhs->opcode() == HloOpcode::kIota &&
             Cast<HloIotaInstruction>(compare_rhs)->iota_dimension() ==
                 reduce_dim) {
    indices_broadcast = compare_lhs;
  } else {
    return std::nullopt;
  }

  if (indices_broadcast->opcode() != HloOpcode::kBroadcast ||
      absl::c_linear_search(indices_broadcast->dimensions(), reduce_dim) ||
      !absl::c_is_sorted(indices_broadcast->dimensions())) {
    return std::nullopt;
  }

  HloInstruction* indices = indices_broadcast->mutable_operand(0);
  const PrimitiveType indices_type = indices->shape().element_type();
  if (!IsValidOneHotIndexType(indices_type, reduce_dim_size)) {
    return std::nullopt;
  }

  return OneHotReduceMatch{
      /*init_value=*/init_value,
      /*indices_broadcast=*/indices_broadcast,
      /*indices=*/indices,
      /*on_true=*/on_true,
      /*reduce_dim=*/reduce_dim,
      /*reduce_dim_size=*/reduce_dim_size,
  };
}

// Phase 2 (lowering): emits the 1-element gather and in-bounds select
// instructions into `reduce`'s parent computation and returns the replacement
// `select` instruction.
absl::StatusOr<HloInstruction*> CreateOneHotReduceGatherReplacement(
    HloInstruction* reduce, const OneHotReduceMatch& match) {
  HloComputation* computation = reduce->parent();
  const OpMetadata* metadata = &reduce->metadata();
  const int64_t out_rank = reduce->shape().dimensions().size();
  const PrimitiveType out_type = reduce->shape().element_type();
  const PrimitiveType indices_type = match.indices->shape().element_type();
  const Shape indices_full_shape =
      ShapeUtil::ChangeElementType(reduce->shape(), indices_type);
  HloInstruction* indices_full = match.indices;
  if (match.indices_broadcast->dimensions().size() != out_rank) {
    std::vector<int64_t> mapped_indices_dims;
    mapped_indices_dims.reserve(match.indices_broadcast->dimensions().size());
    for (int64_t d : match.indices_broadcast->dimensions()) {
      mapped_indices_dims.push_back(d < match.reduce_dim ? d : d - 1);
    }
    indices_full = computation->AddInstruction(
        HloInstruction::CreateBroadcast(indices_full_shape, match.indices,
                                        mapped_indices_dims),
        metadata);
  }

  HloInstruction* base_values = PeelSingleSafeConvert(match.on_true, out_type);
  const bool peeled_outer_convert = (base_values != match.on_true);
  std::vector<int64_t> value_dims;
  if (base_values->opcode() == HloOpcode::kBroadcast &&
      absl::c_is_sorted(base_values->dimensions()) &&
      absl::c_linear_search(base_values->dimensions(), match.reduce_dim)) {
    value_dims.assign(base_values->dimensions().begin(),
                      base_values->dimensions().end());
    base_values = base_values->mutable_operand(0);
    if (!peeled_outer_convert) {
      base_values = PeelSingleSafeConvert(base_values, out_type);
    }
  } else {
    value_dims.resize(base_values->shape().dimensions().size());
    absl::c_iota(value_dims, 0);
  }

  OneHotGatherSpec spec;
  spec.indices = indices_full;
  spec.num_batch_dims = out_rank;
  spec.gather_source = base_values;
  spec.lhs_is_one_hot = true;
  spec.out_type = out_type;
  spec.zero_scalar = match.init_value;
  for (int64_t sub_dim = 0; sub_dim < value_dims.size(); ++sub_dim) {
    const int64_t full_dim = value_dims[sub_dim];
    if (full_dim == match.reduce_dim) {
      spec.source_contract_dim = sub_dim;
    } else {
      const int64_t out_dim =
          full_dim < match.reduce_dim ? full_dim : full_dim - 1;
      spec.source_batch_to_indices_dim.push_back({sub_dim, out_dim});
    }
  }
  return BuildOneHotGatherAndSelect(computation, spec, metadata);
}

}  // namespace

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
  ABSL_RETURN_IF_ERROR(ReplaceWithNewInstruction(
      add, HloInstruction::CreateBinary(add->shape(), HloOpcode::kAdd,
                                        new_bcast, conv)));
  return true;
}

absl::Status GpuAlgebraicSimplifierVisitor::HandleAdd(HloInstruction* add) {
  ABSL_ASSIGN_OR_RETURN(bool replaced, TryToSinkBroadcastOperandsOfChainedAdds(add));
  if (replaced) {
    return absl::OkStatus();
  }

  return AlgebraicSimplifierVisitor::HandleAdd(add);
}

bool GpuAlgebraicSimplifierVisitor::ConvIsLowerable(
    HloInstruction* conv) const {
  return ConvRewriter::ConvIsLowerable(conv);
}

absl::StatusOr<bool>
GpuAlgebraicSimplifierVisitor::TryToSimplifyOneHotReduceToGather(
    HloInstruction* reduce) {
  std::optional<OneHotReduceMatch> match =
      MatchOneHotReduceToGather(reduce, options_);
  if (!match.has_value()) {
    return false;
  }
  ABSL_ASSIGN_OR_RETURN(HloInstruction * replacement,
                   CreateOneHotReduceGatherReplacement(reduce, *match));
  ABSL_RETURN_IF_ERROR(ReplaceInstruction(reduce, replacement));
  return true;
}

absl::Status GpuAlgebraicSimplifierVisitor::HandleReduce(
    HloInstruction* reduce) {
  ABSL_ASSIGN_OR_RETURN(bool replaced, TryToSimplifyOneHotReduceToGather(reduce));
  if (replaced) {
    return absl::OkStatus();
  }

  return AlgebraicSimplifierVisitor::HandleReduce(reduce);
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

namespace {

class GpuAlgebraicSimplifierFactory : public HloPassFactory {
 public:
  absl::string_view name() const override { return "gpu-algebraic-simplifier"; }

  absl::StatusOr<std::unique_ptr<HloPassInterface>> Create(
      const HloPassContext& context,
      const HloPassOptions& options) const override {
    ABSL_ASSIGN_OR_RETURN(const se::DeviceDescription& device_description,
                     context.Get<const se::DeviceDescription>());
    AlgebraicSimplifierOptions simplifier_options;
    options.MergeInto(simplifier_options);
    return std::make_unique<GpuAlgebraicSimplifier>(
        simplifier_options, device_description.gpu_compute_capability());
  }
};

const HloPassFactory* const kGpuAlgebraicSimplifierRegistration =
    RegisterHloPassFactory<GpuAlgebraicSimplifierFactory>();

}  // namespace

}  // namespace xla::gpu
