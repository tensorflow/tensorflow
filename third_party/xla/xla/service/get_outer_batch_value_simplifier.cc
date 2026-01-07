#include "xla/service/get_outer_batch_value_simplifier.h"

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"

namespace xla {

// Use DfsHloRewriteVisitor to rewrite
class GetOuterBatchValueRewriteVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleCustomCall(HloInstruction* instr) override {
    if (instr->custom_call_target() != "GetOuterBatchValue")
      return absl::OkStatus();

    if (instr->operand_count() != 1) return absl::OkStatus();
    const HloInstruction* input = instr->operand(0);
    const Shape& in_shape = input->shape();
    int64_t multiplier = in_shape.outer_multiplier();

    if (multiplier != -1) return absl::OkStatus();

    int64_t batch_size = in_shape.dimensions(0);
    auto const_instr =
        instr->parent()->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(static_cast<int32_t>(batch_size))));

    TF_RETURN_IF_ERROR(ReplaceInstruction(instr, const_instr));
    return absl::OkStatus();
  }
};

absl::StatusOr<bool> GetOuterBatchValueSimplifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return GetOuterBatchValueRewriteVisitor{}.RunOnModule(module,
                                                        execution_threads);
}
}  // namespace xla