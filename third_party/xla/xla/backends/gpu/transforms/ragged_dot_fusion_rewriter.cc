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

#include "xla/backends/gpu/transforms/ragged_dot_fusion_rewriter.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

std::unique_ptr<HloComputation> CreateScalarAddComputation(PrimitiveType type) {
  auto embedded_builder = HloComputation::Builder("add");
  auto lhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(type, {}), "lhs"));
  auto rhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(type, {}), "rhs"));
  embedded_builder.AddInstruction(
      HloInstruction::CreateBinary(lhs->shape(), HloOpcode::kAdd, lhs, rhs));
  return embedded_builder.Build();
}

std::unique_ptr<HloInstruction> Zero(PrimitiveType type) {
  return HloInstruction::CreateConstant(LiteralUtil::Zero(type));
}

// Takes an array of shape [batch_dims..., num_groups] and returns an array of
// the same shape with the elements of the array along the last dimension
// now representing the cumulative sum of all elements in the input array up to
// the current group.
std::unique_ptr<HloInstruction> CreateCumulativeSum(
    HloInstruction* group_sizes) {
  int64_t batch_dims = group_sizes->shape().dimensions().size() - 1;
  int64_t num_groups = group_sizes->shape().dimensions(batch_dims);

  Window cumsum_window;
  // Add batch dimensions.
  for (int i = 0; i < batch_dims; ++i) {
    WindowDimension* dim = cumsum_window.add_dimensions();
    dim->set_size(1);
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_stride(1);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  // Add group dimension.
  WindowDimension* dim = cumsum_window.add_dimensions();
  dim->set_size(num_groups);
  dim->set_padding_low(num_groups - 1);
  dim->set_padding_high(0);
  dim->set_stride(1);
  dim->set_window_dilation(1);
  dim->set_base_dilation(1);

  auto type = group_sizes->shape().element_type();
  HloComputation* add = group_sizes->GetModule()->AddEmbeddedComputation(
      CreateScalarAddComputation(type));
  auto zero = group_sizes->parent()->AddInstruction(Zero(type));
  return HloInstruction::CreateReduceWindow(group_sizes->shape(), group_sizes,
                                            zero, cumsum_window, add);
}

// Pads the given dimension of `operand` up to `new_size` with zeros. Returns
// `operand` unchanged if it is already `new_size`.
HloInstruction* PadDimTo(HloInstruction* operand, int dim, int64_t new_size) {
  int64_t old_size = operand->shape().dimensions(dim);
  if (old_size == new_size) {
    return operand;
  }
  auto computation = operand->parent();
  Shape new_shape = operand->shape();
  new_shape.set_dimensions(dim, new_size);
  PaddingConfig padding_config;
  for (int i = 0; i < operand->shape().dimensions().size(); ++i) {
    auto* padding_dim = padding_config.add_dimensions();
    padding_dim->set_edge_padding_low(0);
    padding_dim->set_edge_padding_high(i == dim ? new_size - old_size : 0);
    padding_dim->set_interior_padding(0);
  }
  auto* zero =
      computation->AddInstruction(Zero(operand->shape().element_type()));
  return computation->AddInstruction(
      HloInstruction::CreatePad(new_shape, operand, zero, padding_config));
}

// Slices the given dimension of `operand` down to `new_size`, starting at 0.
// Returns `operand` unchanged if it is already `new_size`.
HloInstruction* SliceDimTo(HloInstruction* operand, int dim, int64_t new_size) {
  int64_t old_size = operand->shape().dimensions(dim);
  if (old_size == new_size) {
    return operand;
  }
  auto computation = operand->parent();
  Shape new_shape = operand->shape();
  new_shape.set_dimensions(dim, new_size);
  llvm::SmallVector<int64_t> starts(operand->shape().dimensions().size(), 0);
  llvm::SmallVector<int64_t> limits(operand->shape().dimensions().begin(),
                                    operand->shape().dimensions().end());
  limits[dim] = new_size;
  llvm::SmallVector<int64_t> strides(operand->shape().dimensions().size(), 1);
  return computation->AddInstruction(
      HloInstruction::CreateSlice(new_shape, operand, starts, limits, strides));
}

// Returns true if `dim` is `shape`'s fastest-moving (minor-most) dimension,
// using its explicit layout if one is set, or XLA's default layout
// (descending dimension order, i.e. the last dimension is minor-most)
// otherwise.
bool IsFastestMovingDimension(const Shape& shape, int dim) {
  if (shape.has_layout()) {
    return shape.layout().minor_to_major(0) == dim;
  }
  return dim == shape.dimensions().size() - 1;
}

// Swaps the two dimensions of a rank-2 `operand`.
HloInstruction* SwapDims2D(HloInstruction* operand) {
  auto computation = operand->parent();
  Shape new_shape = ShapeUtil::MakeShape(
      operand->shape().element_type(),
      {operand->shape().dimensions(1), operand->shape().dimensions(0)});
  return computation->AddInstruction(
      HloInstruction::CreateTranspose(new_shape, operand, {1, 0}));
}

// cuDNN's ragged-dot wgrad path (kRaggedContracting mode) lowers to a
// cuBLASLt grouped GEMM that relies on TMA and requires 16-byte alignment on
// the fastest-moving (minor-most) dimension of the lhs/rhs operands. If that
// dimension is the ragged M dimension, the alignment requirement falls on
// the per-group sizes -- which are only known at runtime and can't be
// padded at compile time. To avoid that, operands with M as the
// fastest-moving dimension are first transposed so that K/N becomes the
// fastest-moving dimension instead; K and N are static, so they can then be
// padded up to the required alignment ahead of time. The ragged-dot is kept
// (so it is still recognized and handled by the cuDNN fusion compiler), and
// the [G, K, N] result is sliced back down to the original K, N.
//
// Only the simple, non-batched wgrad shapes (2D lhs/rhs) are supported;
// other shapes are returned unchanged. Returns the ragged-dot instruction
// that downstream steps (padding-tail masking, cuDNN fusion conversion)
// should continue operating on: either `ragged_dot` unchanged, or the new,
// larger ragged-dot that replaced it in the graph.
absl::StatusOr<HloRaggedDotInstruction*> PadWgradForCuDNNAlignment(
    HloRaggedDotInstruction* ragged_dot) {
  const auto& ragged_dims = ragged_dot->ragged_dot_dimension_numbers();
  if (!IsRaggedDotWgrad(ragged_dims)) {
    return ragged_dot;
  }
  const auto& dot_dims = ragged_dims.dot_dimension_numbers();
  int lhs_ragged_dim = ragged_dims.lhs_ragged_dimensions(0);

  HloInstruction* lhs = ragged_dot->mutable_operand(0);
  HloInstruction* rhs = ragged_dot->mutable_operand(1);
  if (lhs->shape().dimensions().size() != 2 ||
      rhs->shape().dimensions().size() != 2) {
    return ragged_dot;
  }
  // Ragged-dots reaching this point are canonicalized to a single
  // contracting dimension (guaranteed by the rank-2 check above), so the
  // rhs ragged/contracting dimension is simply rhs_contracting_dimensions(0).
  int rhs_ragged_dim = dot_dims.rhs_contracting_dimensions(0);

  // Only swap an operand's dimensions if M is actually the fastest-moving
  // one; if K/N is already the fastest-moving dimension, it is already
  // safe to pad at compile time and no transpose is needed.
  bool transposed = false;
  if (IsFastestMovingDimension(lhs->shape(), lhs_ragged_dim)) {
    lhs = SwapDims2D(lhs);
    lhs_ragged_dim = 1 - lhs_ragged_dim;
    transposed = true;
  }
  if (IsFastestMovingDimension(rhs->shape(), rhs_ragged_dim)) {
    rhs = SwapDims2D(rhs);
    rhs_ragged_dim = 1 - rhs_ragged_dim;
    transposed = true;
  }
  int lhs_k_dim = 1 - lhs_ragged_dim;
  int rhs_n_dim = 1 - rhs_ragged_dim;

  int64_t alignment_elements = std::max<int64_t>(
      16 / primitive_util::ByteWidth(lhs->shape().element_type()), 1);
  int64_t k = lhs->shape().dimensions(lhs_k_dim);
  int64_t n = rhs->shape().dimensions(rhs_n_dim);
  int64_t padded_k = RoundUpTo(k, alignment_elements);
  int64_t padded_n = RoundUpTo(n, alignment_elements);
  if (!transposed && padded_k == k && padded_n == n) {
    return ragged_dot;
  }

  HloInstruction* padded_lhs = PadDimTo(lhs, lhs_k_dim, padded_k);
  HloInstruction* padded_rhs = PadDimTo(rhs, rhs_n_dim, padded_n);

  // Reflect wherever M ended up (0 or 1) for each (possibly swapped)
  // operand; downstream consumers key off dnums rather than assuming a
  // fixed position.
  RaggedDotDimensionNumbers new_ragged_dims = ragged_dims;
  new_ragged_dims.set_lhs_ragged_dimensions(0, lhs_ragged_dim);
  new_ragged_dims.mutable_dot_dimension_numbers()
      ->set_lhs_contracting_dimensions(0, lhs_ragged_dim);
  new_ragged_dims.mutable_dot_dimension_numbers()
      ->set_rhs_contracting_dimensions(0, rhs_ragged_dim);

  Shape padded_shape = ragged_dot->shape();
  padded_shape.set_dimensions(1, padded_k);
  padded_shape.set_dimensions(2, padded_n);

  HloComputation* computation = ragged_dot->parent();
  HloInstruction* padded_ragged_dot =
      computation->AddInstruction(HloInstruction::CreateRaggedDot(
          padded_shape, padded_lhs, padded_rhs, ragged_dot->mutable_operand(2),
          new_ragged_dims, ragged_dot->precision_config()));
  padded_ragged_dot->set_metadata(ragged_dot->metadata());

  HloInstruction* result = SliceDimTo(padded_ragged_dot, 1, k);
  result = SliceDimTo(result, 2, n);
  ABSL_RETURN_IF_ERROR(computation->ReplaceInstruction(ragged_dot, result));
  return Cast<HloRaggedDotInstruction>(padded_ragged_dot);
}

// The weight-gradient ("wgrad") flavor of ragged-dot has the ragged
// dimension on the *contracting* dim of lhs/rhs (no rhs group dimension),
// producing an output shaped [num_groups, ...]. cuDNN's
// MOE_GROUPED_MATMUL_BWD only accepts a `FirstTokenOffset` tensor with
// exactly num_groups entries and always derives the *last* group's end
// implicitly from the token tensor's full static row count, not from any
// real boundary (see cuDNN Frontend docs: "the total token count [is]
// implicit from the token tensor dimension"). For the wgrad flavor this
// means cuDNN sums the ragged buffer's entire unused padding tail into the
// last group's output, corrupting it with whatever garbage/uninitialized
// memory happens to occupy that padding.
//
// Work around this by explicitly zeroing the padding tail (rows at or past
// the true total valid element count) of both ragged-dot operands before
// they reach the ragged-dot at all. This is done with ordinary,
// universally-supported HLO ops (reduce/iota/compare/select) so it doesn't
// touch the ragged-dot's shape or the cuDNN fusion's structure in any way.
absl::Status MaskRaggedDotPaddingTail(HloRaggedDotInstruction* ragged_dot) {
  const RaggedDotDimensionNumbers& dnums =
      ragged_dot->ragged_dot_dimension_numbers();
  if (!dnums.rhs_group_dimensions().empty()) {
    // Not the wgrad flavor. Any padding-tail garbage lands in unused
    // *output* rows that are never consumed downstream, so masking isn't
    // needed there.
    return absl::OkStatus();
  }

  HloComputation* computation = ragged_dot->parent();
  HloInstruction* group_sizes = ragged_dot->mutable_operand(2);
  PrimitiveType gs_type = group_sizes->shape().element_type();

  // total_valid = reduce_sum(group_sizes) over the group dimension -- the
  // true number of real (non-padding) rows in the ragged buffer.
  int64_t batch_dims = group_sizes->shape().dimensions().size() - 1;
  HloComputation* add = group_sizes->GetModule()->AddEmbeddedComputation(
      CreateScalarAddComputation(gs_type));
  HloInstruction* zero_scalar = computation->AddInstruction(Zero(gs_type));
  Shape total_shape =
      ShapeUtil::DeleteDimension(batch_dims, group_sizes->shape());
  HloInstruction* total_valid =
      computation->AddInstruction(HloInstruction::CreateReduce(
          total_shape, group_sizes, zero_scalar, {batch_dims}, add));
  if (gs_type != S32) {
    total_valid = computation->AddInstruction(HloInstruction::CreateConvert(
        ShapeUtil::MakeShape(S32, {}), total_valid));
  }

  int64_t lhs_token_dim = dnums.lhs_ragged_dimensions(0);
  int64_t rhs_token_dim =
      dnums.dot_dimension_numbers().rhs_contracting_dimensions(0);
  int64_t token_dims[2] = {lhs_token_dim, rhs_token_dim};

  for (int operand_idx = 0; operand_idx < 2; ++operand_idx) {
    HloInstruction* operand = ragged_dot->mutable_operand(operand_idx);
    const Shape& op_shape = operand->shape();
    int64_t token_dim = token_dims[operand_idx];
    int64_t num_rows = op_shape.dimensions(token_dim);

    Shape token_iota_shape = ShapeUtil::MakeShape(S32, {num_rows});
    HloInstruction* iota = computation->AddInstruction(
        HloInstruction::CreateIota(token_iota_shape, /*iota_dimension=*/0));
    HloInstruction* total_valid_b = computation->AddInstruction(
        HloInstruction::CreateBroadcast(token_iota_shape, total_valid, {}));
    Shape pred_1d_shape = ShapeUtil::ChangeElementType(token_iota_shape, PRED);
    HloInstruction* mask_1d =
        computation->AddInstruction(HloInstruction::CreateCompare(
            pred_1d_shape, iota, total_valid_b, Comparison::Direction::kLt));

    Shape mask_shape = ShapeUtil::ChangeElementType(op_shape, PRED);
    HloInstruction* mask = computation->AddInstruction(
        HloInstruction::CreateBroadcast(mask_shape, mask_1d, {token_dim}));

    HloInstruction* zero_elem =
        computation->AddInstruction(Zero(op_shape.element_type()));
    HloInstruction* zero_b = computation->AddInstruction(
        HloInstruction::CreateBroadcast(op_shape, zero_elem, {}));

    HloInstruction* masked =
        computation->AddInstruction(HloInstruction::CreateTernary(
            op_shape, HloOpcode::kSelect, mask, operand, zero_b));
    ABSL_RETURN_IF_ERROR(ragged_dot->ReplaceOperandWith(operand_idx, masked));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<HloInstruction>> RaggedToCuDNNFusion(
    HloRaggedDotInstruction* ragged_dot) {
  std::string fusion_name =
      absl::StrCat("ragged_dot_fusion_", ragged_dot->name());
  HloComputation::Builder builder(absl::StrCat(fusion_name, "_computation"));

  HloInstruction* fused_input =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, ragged_dot->operand(0)->shape(), ragged_dot->operand(0)->name()));
  HloInstruction* fused_weight =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, ragged_dot->operand(1)->shape(), ragged_dot->operand(1)->name()));

  auto computation = ragged_dot->parent();
  HloInstruction* group_sizes = ragged_dot->mutable_operand(2);
  // cuDNN accepts cumulative sum of group sizes
  HloInstruction* cumulative_sum =
      computation->AddInstruction(CreateCumulativeSum(group_sizes));
  HloInstruction* sub =
      computation->AddInstruction(HloInstruction::CreateBinary(
          cumulative_sum->shape(), HloOpcode::kSubtract, cumulative_sum,
          group_sizes));
  HloInstruction* fused_sub_cum_group_size = builder.AddInstruction(
      HloInstruction::CreateParameter(2, sub->shape(), sub->name()));

  HloInstruction* fused_ragged_dot =
      builder.AddInstruction(ragged_dot->CloneWithNewOperands(
          ragged_dot->shape(),
          {fused_input, fused_weight, fused_sub_cum_group_size}));

  HloComputation* new_computation =
      ragged_dot->GetModule()->AddComputationAndUnifyNamesAndIds(
          builder.Build(fused_ragged_dot), /*is_entry=*/false);
  std::vector<HloInstruction*> fusion_params = {
      ragged_dot->mutable_operand(0), ragged_dot->mutable_operand(1), sub};
  return HloInstruction::CreateFusion(ragged_dot->shape(),
                                      HloInstruction::FusionKind::kCustom,
                                      fusion_params, new_computation);
}

}  // namespace

absl::StatusOr<bool> RaggedDotFusionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloRaggedDotInstruction*> ragged_dots;
  for (auto* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kRaggedDot) {
        ragged_dots.push_back(Cast<HloRaggedDotInstruction>(instruction));
      }
    }
  }

  for (auto* ragged_dot : ragged_dots) {
    ABSL_ASSIGN_OR_RETURN(ragged_dot, PadWgradForCuDNNAlignment(ragged_dot));
    ABSL_RETURN_IF_ERROR(MaskRaggedDotPaddingTail(ragged_dot));
    ABSL_ASSIGN_OR_RETURN(auto ragged_dot_fusion, RaggedToCuDNNFusion(ragged_dot));
    gpu::GpuBackendConfig gpu_backend_config;
    gpu::FusionBackendConfig* fusion_config =
        gpu_backend_config.mutable_fusion_backend_config();
    fusion_config->set_kind(gpu::kCuDnnFusionKind);
    ABSL_RETURN_IF_ERROR(ragged_dot_fusion->set_backend_config(gpu_backend_config));
    ragged_dot_fusion->set_metadata(ragged_dot->metadata());
    ABSL_RETURN_IF_ERROR(ragged_dot->parent()->ReplaceWithNewInstruction(
        ragged_dot, std::move(ragged_dot_fusion)));
  }

  return !ragged_dots.empty();
}

}  // namespace gpu
}  // namespace xla
