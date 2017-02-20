/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"

#include <cmath>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

Status HloCostAnalysis::Preprocess(HloInstruction* hlo) {
  // Set current instruction cost values to reasonable default values. Each
  // handler can overwrite these values. In Postprocess, these value are
  // accumulated and written to the per-instruction maps.
  current_flop_count_ = 0;
  current_transcendental_count_ = 0;

  // The default element count for an instruction is the sum of elements in the
  // operands and output. The default ShapeUtil::ByteSizeOf does not handle
  // opaque types.
  current_bytes_accessed_ = shape_size_(hlo->shape());
  for (const HloInstruction* operand : hlo->operands()) {
    current_bytes_accessed_ += shape_size_(operand->shape());
  }

  return Status::OK();
}

Status HloCostAnalysis::Postprocess(HloInstruction* hlo) {
  // Accumulate cost values and write into per-instruction maps.
  flop_count_ += current_flop_count_;
  hlo_to_flop_count_[hlo] = current_flop_count_;

  transcendental_count_ += current_transcendental_count_;
  hlo_to_transcendental_count_[hlo] = current_transcendental_count_;

  bytes_accessed_ += current_bytes_accessed_;
  hlo_to_bytes_accessed_[hlo] = current_bytes_accessed_;

  return Status::OK();
}

Status HloCostAnalysis::HandleElementwiseOp(HloInstruction* hlo_instruction) {
  const auto& shape = hlo_instruction->shape();
  // For element-wise operations, the number of computations is the same as the
  // number of elements in the output shape.
  auto computation_count = ShapeUtil::ElementsIn(shape);
  auto opcode = hlo_instruction->opcode();
  // We treat the two opcodes (kExp, kPower) as transcendental operations.
  if (opcode == HloOpcode::kExp || opcode == HloOpcode::kPower) {
    current_transcendental_count_ = computation_count;
  } else {
    // Note: transcendental operations are considered a separate category from
    // FLOPs.
    current_flop_count_ = computation_count;
  }
  return Status::OK();
}

Status HloCostAnalysis::HandleElementwiseUnary(HloInstruction* hlo,
                                               HloOpcode opcode,
                                               HloInstruction* operand) {
  return HandleElementwiseOp(hlo);
}

Status HloCostAnalysis::HandleElementwiseBinary(HloInstruction* hlo,
                                                HloOpcode opcode,
                                                HloInstruction* lhs,
                                                HloInstruction* rhs) {
  return HandleElementwiseOp(hlo);
}

Status HloCostAnalysis::HandleCompare(HloInstruction* compare, HloOpcode opcode,
                                      HloInstruction* lhs,
                                      HloInstruction* rhs) {
  return HandleElementwiseOp(compare);
}

Status HloCostAnalysis::HandleClamp(HloInstruction* clamp,
                                    HloInstruction* min_instruction,
                                    HloInstruction* arg_instruction,
                                    HloInstruction* max_instruction) {
  return HandleElementwiseOp(clamp);
}

Status HloCostAnalysis::HandleParameter(HloInstruction* parameter) {
  current_bytes_accessed_ = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleConstant(HloInstruction* constant,
                                       const Literal& literal) {
  current_bytes_accessed_ = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleGetTupleElement(HloInstruction* get_tuple_element,
                                              HloInstruction* operand) {
  // GetTupleElement forwards a pointer and does not touch each element in the
  // output.
  current_bytes_accessed_ = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleSelect(HloInstruction* select,
                                     HloInstruction* pred,
                                     HloInstruction* on_true,
                                     HloInstruction* on_false) {
  return Status::OK();
}

Status HloCostAnalysis::HandleReverse(HloInstruction* reverse,
                                      HloInstruction* operand_instruction) {
  return Status::OK();
}

Status HloCostAnalysis::HandleSlice(HloInstruction* slice,
                                    HloInstruction* operand_instruction) {
  return Status::OK();
}

Status HloCostAnalysis::HandleDynamicSlice(
    HloInstruction* slice,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  return Status::OK();
}

Status HloCostAnalysis::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update, HloInstruction* operand,
    HloInstruction* update, HloInstruction* start_indices) {
  return Status::OK();
}

Status HloCostAnalysis::HandleTuple(
    HloInstruction* tuple,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  // The tuple instruction only gathers pointers from inputs (it doesn't iterate
  // through them). The memory touched is then only the size of the output
  // buffer.
  current_bytes_accessed_ = shape_size_(tuple->shape());
  return Status::OK();
}

Status HloCostAnalysis::HandleConcatenate(
    HloInstruction* concatenate,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  return Status::OK();
}

Status HloCostAnalysis::HandleConvert(HloInstruction* convert,
                                      HloInstruction* operand) {
  return HandleElementwiseOp(convert);
}

Status HloCostAnalysis::HandleCopy(HloInstruction* copy,
                                   HloInstruction* operand) {
  return Status::OK();
}

Status HloCostAnalysis::HandleDot(HloInstruction* dot,
                                  HloInstruction* lhs_instruction,
                                  HloInstruction* rhs_instruction) {
  const Shape& lhs_shape = lhs_instruction->shape();
  const Shape& rhs_shape = rhs_instruction->shape();
  // Count of elements along the reduction dimension (last dimension for the
  // rhs).
  int64 reduction_width = lhs_shape.dimensions(ShapeUtil::Rank(lhs_shape) - 1);

  // First divide by reduction width before multiplying by rhs elements to avoid
  // overflow.
  int64 fma_count;
  if (reduction_width == 0) {
    fma_count = 0;
  } else {
    fma_count = (ShapeUtil::ElementsIn(lhs_shape) / reduction_width) *
                ShapeUtil::ElementsIn(rhs_shape);
  }

  // We count an FMA operation as 2 floating point operations.
  current_flop_count_ = kFmaFlops * fma_count;
  return Status::OK();
}

Status HloCostAnalysis::HandleInfeed(HloInstruction* infeed) {
  return Status::OK();
}

Status HloCostAnalysis::HandleOutfeed(HloInstruction* outfeed) {
  return Status::OK();
}

Status HloCostAnalysis::HandleMap(
    HloInstruction* map, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    HloComputation* function,
    tensorflow::gtl::ArraySlice<HloInstruction*> /*static_operands*/) {
  // Compute the cost of the user function.
  HloInstruction* function_instruction = function->root_instruction();
  HloCostAnalysis visitor(shape_size_);
  TF_RETURN_IF_ERROR(function_instruction->Accept(&visitor));

  // Compute the cost of all elements for this Map operation.
  int64 element_count = ShapeUtil::ElementsIn(map->shape());
  current_transcendental_count_ =
      element_count * visitor.transcendental_count();
  current_flop_count_ = element_count * visitor.flop_count();
  return Status::OK();
}

Status HloCostAnalysis::HandleReduce(
    HloInstruction* reduce, HloInstruction* arg, HloInstruction* init_value,
    tensorflow::gtl::ArraySlice<int64> dimensions, HloComputation* function) {
  // Compute the cost of the user function.
  HloInstruction* function_instruction = function->root_instruction();
  HloCostAnalysis visitor(shape_size_);
  TF_RETURN_IF_ERROR(function_instruction->Accept(&visitor));

  // Compute the cost of all elements for this Reduce operation.
  int64 reduction_count = ShapeUtil::ElementsIn(arg->shape()) -
                          ShapeUtil::ElementsIn(reduce->shape());
  current_flop_count_ = reduction_count * visitor.flop_count();
  current_transcendental_count_ =
      reduction_count * visitor.transcendental_count();
  return Status::OK();
}

Status HloCostAnalysis::HandleReduceWindow(HloInstruction* reduce_window,
                                           HloInstruction* operand,
                                           const Window& window,
                                           HloComputation* function) {
  // Compute the cost of the user function.
  HloInstruction* function_instruction = function->root_instruction();
  HloCostAnalysis visitor(shape_size_);
  TF_RETURN_IF_ERROR(function_instruction->Accept(&visitor));

  // Compute the cost of all elements for this ReduceWindow operation. For each
  // output element, (window_size - 1) number of user computations are applied.
  auto output_size = ShapeUtil::ElementsIn(reduce_window->shape());
  int64 window_size = 1;
  for (const auto& dimension : window.dimensions()) {
    window_size *= dimension.size();
  }
  current_flop_count_ = output_size * (window_size - 1) * visitor.flop_count();
  current_transcendental_count_ =
      output_size * (window_size - 1) * visitor.transcendental_count();
  return Status::OK();
}

Status HloCostAnalysis::HandleSelectAndScatter(HloInstruction* instruction) {
  // Compute the cost of the select and scatter function.
  HloInstruction* select = instruction->select()->root_instruction();
  HloCostAnalysis select_visitor(shape_size_);
  TF_RETURN_IF_ERROR(select->Accept(&select_visitor));
  HloInstruction* scatter = instruction->scatter()->root_instruction();
  HloCostAnalysis scatter_visitor(shape_size_);
  TF_RETURN_IF_ERROR(scatter->Accept(&scatter_visitor));

  // Compute the cost of all elements for this operation. For each scatter
  // source element, (window_size - 1) number of select computations and 1
  // scatter computation are applied.
  const auto source = instruction->operand(1);
  const auto source_element_count = ShapeUtil::ElementsIn(source->shape());
  int64 window_size = 1;
  for (const auto& dimension : instruction->window().dimensions()) {
    window_size *= dimension.size();
  }
  current_flop_count_ =
      source_element_count * ((window_size - 1) * select_visitor.flop_count() +
                              scatter_visitor.flop_count());
  current_transcendental_count_ =
      source_element_count *
      ((window_size - 1) * select_visitor.transcendental_count() +
       scatter_visitor.transcendental_count());
  return Status::OK();
}

Status HloCostAnalysis::HandleBitcast(HloInstruction* bitcast) {
  // A bitcast does no computation and touches no memory.
  current_bytes_accessed_ = 0;
  return Status::OK();
}

Status HloCostAnalysis::HandleBroadcast(HloInstruction* broadcast) {
  return Status::OK();
}

Status HloCostAnalysis::HandlePad(HloInstruction* pad) { return Status::OK(); }

Status HloCostAnalysis::HandleSend(HloInstruction* send) {
  return Status::OK();
}

Status HloCostAnalysis::HandleRecv(HloInstruction* recv) {
  return Status::OK();
}

Status HloCostAnalysis::HandleReshape(HloInstruction* reshape) {
  return Status::OK();
}

Status HloCostAnalysis::HandleTranspose(HloInstruction* transpose) {
  return Status::OK();
}

Status HloCostAnalysis::HandleConvolution(HloInstruction* convolution,
                                          HloInstruction* lhs_instruction,
                                          HloInstruction* rhs_instruction,
                                          const Window& window) {
  const auto& dnums = convolution->convolution_dimension_numbers();
  const int64 output_features =
      convolution->shape().dimensions(dnums.feature_dimension());

  // For each output element, we do one fma per element in the
  // kernel at some given output feature index.
  const int64 fmas_per_output_element =
      ShapeUtil::ElementsIn(rhs_instruction->shape()) / output_features;
  const int64 output_elements = ShapeUtil::ElementsIn(convolution->shape());
  current_flop_count_ = output_elements * fmas_per_output_element * kFmaFlops;
  return Status::OK();
}

Status HloCostAnalysis::HandleCrossReplicaSum(HloInstruction* crs) {
  // We assume 2 replicas, so that each output element is the sum of two input
  // elements.
  //
  // TODO(b/33004697): Compute correct cost here, taking the actual number of
  // replicas into account.
  current_flop_count_ = ShapeUtil::ElementsIn(crs->shape());
  return Status::OK();
}

Status HloCostAnalysis::HandleRng(HloInstruction* random,
                                  RandomDistribution distribution) {
  // TODO(b/26346211): Implement better estimates for the RNG cost, since the
  // cost changes with the implementation and the distribution. For now, assume
  // the cost of each RNG is same as a transcendental operation.
  current_transcendental_count_ = ShapeUtil::ElementsIn(random->shape());
  return Status::OK();
}

Status HloCostAnalysis::HandleFusion(HloInstruction* fusion) {
  // Compute the cost of the fused expression.
  HloInstruction* fused_expression_root = fusion->fused_expression_root();
  HloCostAnalysis visitor(shape_size_);
  TF_RETURN_IF_ERROR(fused_expression_root->Accept(&visitor));

  // Attribute the cost of the fused expression to the fusion node.
  current_transcendental_count_ = visitor.transcendental_count();
  current_flop_count_ = visitor.flop_count();
  return Status::OK();
}

Status HloCostAnalysis::HandleCall(
    HloInstruction* call, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    HloComputation* computation) {
  return Unimplemented("call");
}

Status HloCostAnalysis::HandleCustomCall(
    HloInstruction* custom_call,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    tensorflow::StringPiece custom_call_target) {
  return Unimplemented("custom-call");
}

Status HloCostAnalysis::HandleSort(HloInstruction* sort,
                                   HloInstruction* operand_instruction) {
  // The cost of sort is implementation dependent, so cannot determine at HLO
  // level. Assume comparison based N*log(N) sorting.
  int64 elements = ShapeUtil::ElementsIn(operand_instruction->shape());
  current_flop_count_ = elements * tensorflow::Log2Ceiling(elements);
  return Status::OK();
}

Status HloCostAnalysis::HandleWhile(HloInstruction* xla_while,
                                    HloInstruction* init,
                                    HloComputation* condition,
                                    HloComputation* body) {
  // Since the number of iterations of the while node is not statically
  // determined, we cannot precisely compute the cost of a while node. For now
  // compute the cost of a single iteration.
  // TODO(b/26346211): Improve the cost analysis for while node.
  HloCostAnalysis body_visitor(shape_size_);
  TF_RETURN_IF_ERROR(body->Accept(&body_visitor));
  HloCostAnalysis condition_visitor(shape_size_);
  TF_RETURN_IF_ERROR(condition->Accept(&condition_visitor));

  current_flop_count_ =
      body_visitor.flop_count() + condition_visitor.flop_count();
  current_transcendental_count_ = body_visitor.transcendental_count() +
                                  condition_visitor.transcendental_count();
  current_bytes_accessed_ =
      body_visitor.bytes_accessed() + condition_visitor.bytes_accessed();

  return Status::OK();
}

Status HloCostAnalysis::FinishVisit(HloInstruction* root) {
  return Status::OK();
}

int64 HloCostAnalysis::flop_count(const HloInstruction& hlo) const {
  auto it = hlo_to_flop_count_.find(&hlo);
  return it == hlo_to_flop_count_.end() ? 0 : it->second;
}

int64 HloCostAnalysis::transcendental_count(const HloInstruction& hlo) const {
  auto it = hlo_to_transcendental_count_.find(&hlo);
  return it == hlo_to_transcendental_count_.end() ? 0 : it->second;
}

int64 HloCostAnalysis::bytes_accessed(const HloInstruction& hlo) const {
  auto it = hlo_to_bytes_accessed_.find(&hlo);
  return it == hlo_to_bytes_accessed_.end() ? 0 : it->second;
}

}  // namespace xla
