// License TODO ....

#include "tensorflow/compiler/xla/service/rce_optimizer.h"

#include <stdlib.h>
#include <istream>

#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

namespace {

namespace m = match;

class RceOptimizerVisitor : public DfsHloRewriteVisitor {
 public:
  explicit RceOptimizerVisitor() {}

  bool DimsAreIndexes(HloInstruction* inst) {
    for (auto i = 0; i < inst->dimensions().size(); i++) {
      if (i != inst->dimensions(i)) {
        return false;
      }
    }
    return true;
  }

  bool IsTrivialBroadcast(HloInstruction* broadcast) {
    return DimsAreIndexes(broadcast);
  }

  bool IsTrivialTranspose(HloInstruction* transpose) {
    return DimsAreIndexes(transpose); /* transpose is subset of broadcast ... */
  }

  Status HandleBroadcast(HloInstruction* broadcast) override;
  Status HandleTranspose(HloInstruction* transpose) override;
  Status HandleReshape(HloInstruction* reshape) override;
  Status HandleReduce(HloInstruction* dot) override;
  Status HandleConvert(HloInstruction* convert) override;
  Status HandleGetTupleElement(HloInstruction *get_tuple_element) override;
};

}  // namespace

Status RceOptimizerVisitor::HandleGetTupleElement(HloInstruction* get_tuple_element) {
  CHECK(Match(get_tuple_element, m::GetTupleElement()));
  HloInstruction *op;
  if (Match(get_tuple_element, m::GetTupleElement(m::Tuple(m::Op(&op))))) {
    auto tuple = get_tuple_element->operand(0);
    if (tuple->operand_count() == 1 && tuple->user_count() == 1) {
      return ReplaceInstruction(get_tuple_element, op);
    }
  }
  return OkStatus();
}

Status RceOptimizerVisitor::HandleBroadcast(HloInstruction* broadcast) {
  HloInstruction* op;
  CHECK(Match(broadcast, m::Broadcast(m::Op(&op))));

  if (ShapeUtil::Equal(broadcast->shape(), op->shape()) &&
      IsTrivialBroadcast(broadcast)) {
    // This broadcast does nothing, remove it
    return ReplaceInstruction(broadcast, op);
  }
  return OkStatus();
}

Status RceOptimizerVisitor::HandleTranspose(HloInstruction* transpose) {
  HloInstruction* op;

  CHECK(Match(transpose, m::Transpose(m::Op(&op))));

  // We need to check the shape, since the transpose might modify the physical
  // layout, in which case we might loose information.
  if (ShapeUtil::Equal(transpose->shape(), op->shape()) &&
      IsTrivialTranspose(transpose)) {
    // This transpose does nothing, remove it
    return ReplaceInstruction(transpose, op);
  }
  return OkStatus();
}

Status RceOptimizerVisitor::HandleReshape(HloInstruction* reshape) {
  HloInstruction* reshape_op = nullptr;
  CHECK(Match(reshape, m::Reshape(m::Op(&reshape_op))));

  std::stringstream ss;

  // ss << "\n.......................";
  // ss << "\n...>> enter reshape=" << reshape->ToString();
  // ss << "\n...>> enter reshape operand=" << reshape_op->ToString();
  // ss << "\n...>> reshape->opcode()=" << reshape->opcode();
  // ss << "\n...>> reshape_op->opcode()=" << reshape_op->opcode();
  // ss << "\n...>> reshape->user_count()=" << reshape->user_count();
  // ss << "\n...>> reshape_op->user_count()=" << reshape_op->user_count();
  // ss << "\n...>> HloOpcode::kReshape=" << HloOpcode::kReshape; //
  // ss << "\n";

  // TODO: Does the physical layout matter for reshapes? I don't
  // think it does, but this might be something to investigate in
  // the future if problems arise.
  if (ShapeUtil::Equal(reshape->shape(), reshape_op->shape())) {
    // This reshape does nothing, remove it

    // ss << "\n ... check simple case ??";
    // ss << "\n...!! reshape_op=" << reshape_op->ToString();
    // ss << "\n...>< reshape=" << reshape->ToString();
    // ss << "\n...<-- replace_simple_case"; 
    // LOG(INFO) << ss.str();
    return ReplaceInstruction(reshape, reshape_op);
  }

  if (reshape_op->opcode() == HloOpcode::kBroadcast) {
    HloInstruction* broadcast_operand = reshape_op->mutable_operand(0);
    // ss << "\n ... check broadcast ??";

    auto reshape_shape = reshape->shape();
    auto broadcast_operand_shape = broadcast_operand->shape();
    if (reshape_shape == broadcast_operand_shape) {
      // ss << "\n ...<-- broadcast_reshape_case";
      // LOG(INFO) << ss.str();
      return ReplaceInstruction(reshape, broadcast_operand);
    }
  }

  // Remove a chain of unnecessary reshapes.
  if (reshape_op->opcode() == HloOpcode::kReshape) {
    // ss << "\n ... check reshape_chain ??";

    HloInstruction* next_op = reshape_op;
    HloInstruction* current_op = nullptr;
    auto chain_len = 0;
    bool is_chain = true;
    while (Match(next_op, m::Reshape(m::Op(&current_op)))) {
      if (next_op->user_count() > 1) {
        is_chain = false;
        break;
      }
      next_op = current_op;
      chain_len++;
    }

    // ss << "\n...<> next_op=" << next_op->ToString();
    // ss << "\n...>< reshape=" << reshape->ToString();

    if (is_chain && ShapeUtil::Equal(next_op->shape(), reshape->shape())) {
      // ss << "\n...!! replace_long_chain";
      // LOG(INFO) << ss.str();
      return ReplaceInstruction(reshape, next_op);
    }
  }

  // Remove "op (shape) -> reduce (shape - 1) -> reshape (shape)"
  if (reshape_op->opcode() == HloOpcode::kReduce && reshape_op->user_count() == 1) {
    // ss << "\n ... check reshape_reduce ??";
    HloInstruction* reduce_op = reshape_op->mutable_operand(0);
    if (ShapeUtil::Equal(reduce_op->shape(), reshape->shape())) {
      // ss << "\n ...!! replace_reshape_and_reduce";
      // LOG(INFO) << ss.str();
      return ReplaceInstruction(reshape, reduce_op);
    }
  }

  LOG(INFO) << ss.str();
  return OkStatus();
}

Status RceOptimizerVisitor::HandleReduce(HloInstruction* reduce) {
  CHECK(Match(reduce, m::Reduce()));
  HloInstruction* op; 
  HloInstruction* init_value;

  std::stringstream ss;

  if (Match(reduce, m::Reduce(m::Reshape(m::Op(&op)), m::Op(&init_value)))) {
    auto reshape = reduce->operand(0);
    // Case 1: op (shape) -> reshape (1, shape, 1) -> reduce added dims (shape)
    //         action: remove reshape
    if (ShapeUtil::Equal(reduce->shape(), op->shape())) {
      return ReplaceInstruction(reduce, op);
    }

    if (ShapeUtil::Equal(reshape->shape(), op->shape())) {
      return OkStatus();
    }
    // Case 2: op (shape) -> reshape (1, shape, 1) -> reduce whole shape or part of it (1, shape-1)
    //         action: 1. remove reshape,
    //                 2. reduce on original indices
    //                 3. add reshape afterwards to correct output of the new reshape

    // ss << "\n ===> reduce_op=" << reduce->ToString();
    // ss << "\n ---> reduce_op_operator=" << reduce->operand(0)->ToString();
    // ss << "\n";

    std::stringstream ss;
    auto operand_shape = op->shape();
    auto reshape_shape = reshape->shape();
    auto reduce_shape = reduce->shape();
    std::vector<int64_t> input_indices(operand_shape.dimensions_size());
    absl::c_iota(input_indices, 0);

    auto optional_reshaped_indices = ShapeUtil::ReshapeLeavesDimensionsUnmodified(
        operand_shape, reshape_shape, input_indices);
    if (optional_reshaped_indices.has_value()) {
      auto reshaped_indices = *optional_reshaped_indices;
      auto reduce_indices = reduce->dimensions();
      std::vector<int64_t> reduce_dims_array;
      for (auto i = 0; i < reshaped_indices.size(); i++) {
        if (absl::c_linear_search(reduce_indices, reshaped_indices[i])) {
          reduce_dims_array.push_back(i);
        }
      }

      auto new_reduce_shape = ShapeUtil::FilterDimensions(
          [&](const auto dim) {
            return !absl::c_linear_search(reduce_dims_array, dim);
          },
          operand_shape);

      HloComputation* comp = reduce->parent();
      auto new_reduce = comp->AddInstruction(
          reduce->CloneWithNewOperands(new_reduce_shape, {op, init_value}));
      auto* new_reduce_dims = new_reduce->mutable_dimensions();
      new_reduce_dims->clear();
      absl::c_copy(reduce_dims_array, std::back_inserter(*new_reduce_dims));
      auto new_reshape = comp->AddInstruction(
          HloInstruction::CreateReshape(reduce->shape(), new_reduce));

      // LOG(INFO) << "\n ---> reduce_reshape_op=" << op->ToString();
      // ss << "\n ==> input_indices={";
      // for (auto n : input_indices) ss << n << ",";
      // ss << "}";
      // ss << "\n ==> old_reduce_shape=" << reduce->shape();
      // ss << "\n ==> op->user_count()=" << op->user_count();
      // ss << "\n ==> new_reduce_shape=" << new_reduce->shape();
      // ss << "\n ==> new_reshape_shape=" << new_reshape->shape();
      // ss << "\n ==> new_reduce->dimensions()={";
      // for (auto n : new_reduce->dimensions()) ss << n << ",";
      // ss << "}";
      // ss << "\n ==> reshaped_indices={";
      // for (auto n : reshaped_indices) ss << n << ",";
      // ss << "}";
      // ss << "\n ==> dims_to_reduce={";
      // for (auto n : reduce_indices) ss << n << ",";
      // ss << "}";
      // ss << "\n ==> ={";
      // for (auto n : reduce_dims_array) ss << n << ",";
      // ss << "}";
      // LOG(INFO) << ss.str();
      // LOG(INFO) << "\n --> op_new_user=" << op->users()[0]->ToString();

      // TF_RETURN_IF_ERROR(comp->ReplaceInstruction(reduce, new_reshape));
      // changed_ = true;

      return ReplaceInstruction(reduce, new_reshape);
    }
  }
  return OkStatus();
}

Status RceOptimizerVisitor::HandleConvert(HloInstruction* convert) {
  HloInstruction* op;
  CHECK(Match(convert, m::Convert(m::Op(&op))));

  if (ShapeUtil::Equal(convert->shape(), op->shape())) {
    // This convert does nothing, remove it
    return ReplaceInstruction(convert, op);
  }
  return OkStatus();
}

StatusOr<bool> RceOptimizer::Run(HloModule* module) {
  RceOptimizerVisitor visitor;
  LOG(INFO) << "Running RCE optimizer for " << module->name() << "'";
  bool changed = false;
  TF_ASSIGN_OR_RETURN(auto rce_change, visitor.RunOnModule(module));
  changed |= rce_change;
  // {
  //   LOG(INFO) << "Running subpipeline in RCE optimizer for " << module->name() << "'";
  //   HloPassPipeline subpipeline("before_rce_optimization_pipeline");
  //   // subpipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/is_layout_sensitive_);
  //   AlgebraicSimplifierOptions options;
  //   options.set_enable_dot_strength_reduction(false);
  //   options.set_enable_dot_to_multiply_rewrite(false);
  //   options.set_enable_conv_simplification(false);
  //   options.set_enable_conv_operand_swap(false);
  //   options.set_enable_scalar_multiply_reduction(false);
  //   options.set_replace_transpose_with_bitcast(false);
  //   options.set_enable_dot_strength_reduction(false);
  //   options.set_enable_floats_are_real(false);

  //   options.set_enable_reduce_of_reshape(true);
  //   options.set_enable_window_reduce_to_reduce_replacement(true);
  //   options.set_enable_negative_padding_replacement(true);

  //   subpipeline.AddPass<HloDCE>();
  //   subpipeline.AddPass<AlgebraicSimplifier>(options);
  //   TF_ASSIGN_OR_RETURN(auto cleanup_changed_now, subpipeline.Run(module));
  //   changed |= cleanup_changed_now;
  // }
  return changed;
}

}  // namespace xla