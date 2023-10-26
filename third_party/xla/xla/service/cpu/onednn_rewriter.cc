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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include "xla/service/cpu/onednn_rewriter.h"

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/status_macros.h"
#include "tsl/platform/cpu_info.h"

namespace xla {
namespace cpu {

namespace {
namespace m = match;

Status ValidateDotDimensionNumbers(const DotDimensionNumbers& dim_numbers) {
  // Checks some invariants that do not hold in general, but DotDecomposer
  // should have established for us.
  TF_RET_CHECK(dim_numbers.lhs_contracting_dimensions_size() == 1);
  std::vector<int64_t> batch_dim_numbers(
      dim_numbers.lhs_batch_dimensions_size());
  absl::c_iota(batch_dim_numbers, 0);
  TF_RET_CHECK(
      absl::c_equal(batch_dim_numbers, dim_numbers.lhs_batch_dimensions()));
  TF_RET_CHECK(
      absl::c_equal(batch_dim_numbers, dim_numbers.rhs_batch_dimensions()));
  return OkStatus();
}

bool IsSupportedType(xla::PrimitiveType dtype) {
  using tsl::port::TestCPUFeature;
  using tsl::port::CPUFeature;
  switch (dtype) {
    case F32:
      return true;
    case BF16:
      return TestCPUFeature(CPUFeature::AVX512_BF16) ||
             TestCPUFeature(CPUFeature::AMX_BF16);
    default:
      return false;
  }
  return false;
}

}  // namespace

class OneDnnRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  // Matches patterns for possible MatMul fusions that are supported by oneDNN
  // library. Matched hlo instruction(s) are replaced by custom call.
  Status HandleDot(HloInstruction* instr) override {
    // Currently, blocking control dependencies
    if (instr->HasControlDependencies()) return OkStatus();
    HloInstruction* dot_instr;
    auto pattern = m::Op(&dot_instr).WithOpcode(HloOpcode::kDot);
    if (!Match(instr, pattern)) return OkStatus();

    // TODO(intel-tf): The rewrite pass runs after dot-decomposition pass.
    // Adjust the rewrite condition when the rewrite pass is moved to a
    // different point in the pass-pipeline.

    // Currently, we rewrite when the data type is F32 or BF16. Note we do not
    // need to check equality of contraction dim-size of the operands. HLO
    // verifier already does the job. We, however, need to check if contraction
    // is over only 1 dimension (a.k.a. K dimension in matrix-multiplication
    // parlance). We also restrict that batch dimensions of the operands
    // matches.
    if (!IsSupportedType(dot_instr->shape().element_type())) return OkStatus();
    auto dot_dim_numbers = dot_instr->dot_dimension_numbers();
    TF_RETURN_IF_ERROR(ValidateDotDimensionNumbers(dot_dim_numbers));
    const Shape& lhs_shape = dot_instr->operand(0)->shape();
    const Shape& rhs_shape = dot_instr->operand(1)->shape();
    const Shape& output_shape = dot_instr->shape();
    bool should_rewrite = true;
    // None of the operands and result should be ZeroElementArray.
    should_rewrite &= !ShapeUtil::IsZeroElementArray(lhs_shape);
    should_rewrite &= !ShapeUtil::IsZeroElementArray(rhs_shape);
    should_rewrite &= !ShapeUtil::IsZeroElementArray(output_shape);
    // OneDNN only supports 2 <= rank <= kOneDnnMaxNDims.
    should_rewrite &= (lhs_shape.rank() == rhs_shape.rank());
    should_rewrite &= (rhs_shape.rank() == output_shape.rank());
    should_rewrite &=
        (lhs_shape.rank() >= 2 && lhs_shape.rank() <= kOneDnnMaxNDims);
    if (!should_rewrite) return OkStatus();
    // Transpose scenario needs some care and blocked for oneDNN rewrite for
    // now.
    // TODO(intel-tf): Add transpose scenarios
    should_rewrite &= LayoutUtil::IsMonotonicWithDim0Major(lhs_shape.layout());
    if (!should_rewrite) return OkStatus();
    should_rewrite &= LayoutUtil::IsMonotonicWithDim0Major(rhs_shape.layout());
    if (!should_rewrite) return OkStatus();
    should_rewrite &=
        LayoutUtil::IsMonotonicWithDim0Major(output_shape.layout());
    if (!should_rewrite) return OkStatus();

    // Check contracting dimensions: [..., M, K] x [..., K, N]
    should_rewrite &=
        (dot_dim_numbers.lhs_contracting_dimensions(0) == lhs_shape.rank() - 1);
    should_rewrite &=
        (dot_dim_numbers.rhs_contracting_dimensions(0) == rhs_shape.rank() - 2);
    if (!should_rewrite) return OkStatus();

    HloInstruction* matmul_call =
        dot_instr->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape,
            {dot_instr->mutable_operand(0), dot_instr->mutable_operand(1)},
            "__onednn$matmul"));
    // Set additional info via config, e.g., fusion info.
    BackendConfig backend_config;
    // No fusion is supported now, so nothing to add to the config.
    TF_RETURN_IF_ERROR(matmul_call->set_backend_config(backend_config));
    TF_RETURN_IF_ERROR(ReplaceInstruction(dot_instr, matmul_call));
    return OkStatus();
  }
};

StatusOr<bool> OneDnnRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  OneDnnRewriterVisitor visitor;
  return visitor.RunOnModule(module, execution_threads);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
