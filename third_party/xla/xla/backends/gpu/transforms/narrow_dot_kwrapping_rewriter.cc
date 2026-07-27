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

#include "xla/backends/gpu/transforms/narrow_dot_kwrapping_rewriter.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

std::vector<int64_t> GetNonContractingNonBatchDims(
    const Shape& shape, const google::protobuf::RepeatedField<int64_t>& contracting_dims,
    const google::protobuf::RepeatedField<int64_t>& batch_dims) {
  std::vector<int64_t> dims;
  for (int64_t i = 0; i < shape.dimensions().size(); ++i) {
    if (!absl::c_linear_search(contracting_dims, i) &&
        !absl::c_linear_search(batch_dims, i)) {
      dims.push_back(i);
    }
  }
  return dims;
}

int64_t ChooseWrappingFactor(int64_t k, int64_t narrow_dim_size) {
  int64_t target_r = 16 / narrow_dim_size;
  if (target_r < 2) target_r = 2;

  if (k % target_r == 0) {
    return target_r;
  }
  for (int64_t r : {16, 8, 4, 2}) {
    if (k % r == 0 && r >= 16 / narrow_dim_size) {
      return r;
    }
  }
  for (int64_t r : {16, 8, 4, 2}) {
    if (k % r == 0) {
      return r;
    }
  }
  return 1;
}

HloInstruction* CreateIdentityMatrix(HloComputation* computation, int64_t size,
                                     PrimitiveType type) {
  HloInstruction* iota0 =
      computation->AddInstruction(HloInstruction::CreateIota(
          ShapeUtil::MakeShape(PrimitiveType::S32, {size, size}), 0));
  HloInstruction* iota1 =
      computation->AddInstruction(HloInstruction::CreateIota(
          ShapeUtil::MakeShape(PrimitiveType::S32, {size, size}), 1));
  HloInstruction* eq =
      computation->AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeShape(PrimitiveType::PRED, {size, size}), iota0, iota1,
          ComparisonDirection::kEq));
  return computation->AddInstruction(HloInstruction::CreateConvert(
      ShapeUtil::MakeShape(type, {size, size}), eq));
}

}  // namespace

absl::StatusOr<bool> NarrowDotKWrappingRewriter::RewriteComputation(
    HloComputation* computation) {
  bool changed = false;
  std::vector<HloInstruction*> dots_to_rewrite;
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kDot) {
      dots_to_rewrite.push_back(instruction);
    }
  }

  for (HloInstruction* dot : dots_to_rewrite) {
    const auto& dnums = dot->dot_dimension_numbers();
    if (dnums.lhs_contracting_dimensions().size() != 1 ||
        dnums.rhs_contracting_dimensions().size() != 1) {
      continue;  // Only support simple contracting for now
    }

    const Shape& lhs_shape = dot->operand(0)->shape();
    const Shape& rhs_shape = dot->operand(1)->shape();

    std::vector<int64_t> lhs_m_dims = GetNonContractingNonBatchDims(
        lhs_shape, dnums.lhs_contracting_dimensions(),
        dnums.lhs_batch_dimensions());
    std::vector<int64_t> rhs_n_dims = GetNonContractingNonBatchDims(
        rhs_shape, dnums.rhs_contracting_dimensions(),
        dnums.rhs_batch_dimensions());

    if (lhs_m_dims.size() != 1 || rhs_n_dims.size() != 1) {
      continue;  // Only support simple remaining dims
    }

    int64_t m_dim = lhs_m_dims[0];
    int64_t n_dim = rhs_n_dims[0];
    int64_t m_size = lhs_shape.dimensions(m_dim);
    int64_t n_size = rhs_shape.dimensions(n_dim);
    int64_t k_size = lhs_shape.dimensions(dnums.lhs_contracting_dimensions(0));

    bool is_m_narrow = m_size <= 4;
    bool is_n_narrow = n_size <= 4;

    if (!is_m_narrow && !is_n_narrow) {
      continue;  // Not narrow
    }

    int64_t narrow_dim_size = is_m_narrow ? m_size : n_size;
    if (is_m_narrow && is_n_narrow) {
      narrow_dim_size = std::min(m_size, n_size);
    }

    int64_t r = ChooseWrappingFactor(k_size, narrow_dim_size);
    if (r <= 1) {
      continue;  // Cannot wrap
    }

    // Assume 0 or 1 batch dim for simplicity of canonicalization for now.
    if (dnums.lhs_batch_dimensions().size() > 1) {
      continue;
    }

    int64_t b_size = 1;
    bool has_batch = dnums.lhs_batch_dimensions().size() == 1;
    if (has_batch) {
      b_size = lhs_shape.dimensions(dnums.lhs_batch_dimensions(0));
    }

    // 1. Canonicalize LHS to [B, M, K]
    std::vector<int64_t> lhs_perm;
    if (has_batch) lhs_perm.push_back(dnums.lhs_batch_dimensions(0));
    lhs_perm.push_back(m_dim);
    lhs_perm.push_back(dnums.lhs_contracting_dimensions(0));

    HloInstruction* lhs_canonical = dot->mutable_operand(0);
    std::vector<int64_t> lhs_canon_dims;
    if (has_batch) lhs_canon_dims.push_back(b_size);
    lhs_canon_dims.push_back(m_size);
    lhs_canon_dims.push_back(k_size);

    HloInstruction* lhs_canon =
        computation->AddInstruction(HloInstruction::CreateTranspose(
            ShapeUtil::MakeShape(lhs_shape.element_type(), lhs_canon_dims),
            lhs_canonical, lhs_perm));

    // 2. Canonicalize RHS to [B, K, N]
    std::vector<int64_t> rhs_perm;
    if (has_batch) rhs_perm.push_back(dnums.rhs_batch_dimensions(0));
    rhs_perm.push_back(dnums.rhs_contracting_dimensions(0));
    rhs_perm.push_back(n_dim);

    std::vector<int64_t> rhs_canon_dims;
    if (has_batch) rhs_canon_dims.push_back(b_size);
    rhs_canon_dims.push_back(k_size);
    rhs_canon_dims.push_back(n_size);

    HloInstruction* rhs_canon =
        computation->AddInstruction(HloInstruction::CreateTranspose(
            ShapeUtil::MakeShape(rhs_shape.element_type(), rhs_canon_dims),
            dot->mutable_operand(1), rhs_perm));

    // 3. Transform LHS to [B, M' = M*r, K' = K/r]
    std::vector<int64_t> lhs_split_dims;
    if (has_batch) lhs_split_dims.push_back(b_size);
    lhs_split_dims.push_back(m_size);
    lhs_split_dims.push_back(k_size / r);
    lhs_split_dims.push_back(r);

    HloInstruction* lhs_split =
        computation->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(lhs_shape.element_type(), lhs_split_dims),
            lhs_canon));

    std::vector<int64_t> lhs_wrap_perm;
    if (has_batch) lhs_wrap_perm.push_back(0);   // B
    lhs_wrap_perm.push_back(has_batch ? 1 : 0);  // M
    lhs_wrap_perm.push_back(has_batch ? 3 : 2);  // r
    lhs_wrap_perm.push_back(has_batch ? 2 : 1);  // K/r

    std::vector<int64_t> lhs_wrap_dims;
    if (has_batch) lhs_wrap_dims.push_back(b_size);
    lhs_wrap_dims.push_back(m_size);
    lhs_wrap_dims.push_back(r);
    lhs_wrap_dims.push_back(k_size / r);

    HloInstruction* lhs_wrap_trans =
        computation->AddInstruction(HloInstruction::CreateTranspose(
            ShapeUtil::MakeShape(lhs_shape.element_type(), lhs_wrap_dims),
            lhs_split, lhs_wrap_perm));

    std::vector<int64_t> lhs_final_dims;
    if (has_batch) lhs_final_dims.push_back(b_size);
    lhs_final_dims.push_back(m_size * r);
    lhs_final_dims.push_back(k_size / r);

    HloInstruction* lhs_final =
        computation->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(lhs_shape.element_type(), lhs_final_dims),
            lhs_wrap_trans));

    // 4. Transform RHS to [B, K' = K/r, N' = N*r]
    std::vector<int64_t> rhs_split_dims;
    if (has_batch) rhs_split_dims.push_back(b_size);
    rhs_split_dims.push_back(k_size / r);
    rhs_split_dims.push_back(r);
    rhs_split_dims.push_back(n_size);

    HloInstruction* rhs_split =
        computation->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(rhs_shape.element_type(), rhs_split_dims),
            rhs_canon));

    std::vector<int64_t> rhs_wrap_perm;
    if (has_batch) rhs_wrap_perm.push_back(0);   // B
    rhs_wrap_perm.push_back(has_batch ? 1 : 0);  // K/r
    rhs_wrap_perm.push_back(has_batch ? 3 : 2);  // N
    rhs_wrap_perm.push_back(has_batch ? 2 : 1);  // r

    std::vector<int64_t> rhs_wrap_dims;
    if (has_batch) rhs_wrap_dims.push_back(b_size);
    rhs_wrap_dims.push_back(k_size / r);
    rhs_wrap_dims.push_back(n_size);
    rhs_wrap_dims.push_back(r);

    HloInstruction* rhs_wrap_trans =
        computation->AddInstruction(HloInstruction::CreateTranspose(
            ShapeUtil::MakeShape(rhs_shape.element_type(), rhs_wrap_dims),
            rhs_split, rhs_wrap_perm));

    std::vector<int64_t> rhs_final_dims;
    if (has_batch) rhs_final_dims.push_back(b_size);
    rhs_final_dims.push_back(k_size / r);
    rhs_final_dims.push_back(n_size * r);

    HloInstruction* rhs_final =
        computation->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(rhs_shape.element_type(), rhs_final_dims),
            rhs_wrap_trans));

    // 5. New Dot
    DotDimensionNumbers new_dnums;
    if (has_batch) {
      new_dnums.add_lhs_batch_dimensions(0);
      new_dnums.add_rhs_batch_dimensions(0);
      new_dnums.add_lhs_contracting_dimensions(2);  // K/r
      new_dnums.add_rhs_contracting_dimensions(1);  // K/r
    } else {
      new_dnums.add_lhs_contracting_dimensions(1);  // K/r
      new_dnums.add_rhs_contracting_dimensions(0);  // K/r
    }

    std::vector<int64_t> dot_dims;
    if (has_batch) dot_dims.push_back(b_size);
    dot_dims.push_back(m_size * r);
    dot_dims.push_back(n_size * r);

    HloInstruction* new_dot =
        computation->AddInstruction(HloInstruction::CreateDot(
            ShapeUtil::MakeShape(dot->shape().element_type(), dot_dims),
            lhs_final, rhs_final, new_dnums, dot->precision_config()));

    // 6. Reshape and Reduce
    std::vector<int64_t> unwrap_dims;
    if (has_batch) unwrap_dims.push_back(b_size);
    unwrap_dims.push_back(m_size);
    unwrap_dims.push_back(r);
    unwrap_dims.push_back(n_size);
    unwrap_dims.push_back(r);

    HloInstruction* unwrap_reshape =
        computation->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(dot->shape().element_type(), unwrap_dims),
            new_dot));

    HloInstruction* identity =
        CreateIdentityMatrix(computation, r, dot->shape().element_type());

    std::vector<int64_t> broadcast_dims;
    if (has_batch) {
      broadcast_dims = {2, 4};
    } else {
      broadcast_dims = {1, 3};
    }

    HloInstruction* broadcasted_identity =
        computation->AddInstruction(HloInstruction::CreateBroadcast(
            ShapeUtil::MakeShape(dot->shape().element_type(), unwrap_dims),
            identity, broadcast_dims));

    HloInstruction* multiplied =
        computation->AddInstruction(HloInstruction::CreateBinary(
            ShapeUtil::MakeShape(dot->shape().element_type(), unwrap_dims),
            HloOpcode::kMultiply, unwrap_reshape, broadcasted_identity));

    std::vector<int64_t> reduce_dims;
    if (has_batch) {
      reduce_dims = {2, 4};
    } else {
      reduce_dims = {1, 3};
    }

    HloComputation::Builder b("sum");
    auto* x = b.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(dot->shape().element_type(), {}), "x"));
    auto* y = b.AddInstruction(HloInstruction::CreateParameter(
        1, ShapeUtil::MakeShape(dot->shape().element_type(), {}), "y"));
    b.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(dot->shape().element_type(), {}), HloOpcode::kAdd,
        x, y));
    HloComputation* sum_comp =
        dot->GetModule()->AddEmbeddedComputation(b.Build());

    HloInstruction* zero =
        computation->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::Zero(dot->shape().element_type())));

    std::vector<int64_t> final_dims;
    if (has_batch) final_dims.push_back(b_size);
    final_dims.push_back(m_size);
    final_dims.push_back(n_size);

    HloInstruction* reduced =
        computation->AddInstruction(HloInstruction::CreateReduce(
            ShapeUtil::MakeShape(dot->shape().element_type(), final_dims),
            multiplied, zero, reduce_dims, sum_comp));

    if (reduced->shape() != dot->shape()) {
      continue;
    }

    RETURN_IF_ERROR(dot->ReplaceAllUsesWith(reduced));
    RETURN_IF_ERROR(computation->RemoveInstruction(dot));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> NarrowDotKWrappingRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    ASSIGN_OR_RETURN(bool result, RewriteComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
