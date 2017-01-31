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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COST_ANALYSIS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COST_ANALYSIS_H_

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// HloCostAnalysis traverses an HLO graph and calculates the amount of
// computations required for the graph. Each HLO instruction handler provides
// the computation cost of the instruction, and the values are accumulated
// during the traversal for the entire graph. We treat normal floating point
// operations separately from transcendental operations.
class HloCostAnalysis : public DfsHloVisitor {
 public:
  HloCostAnalysis() = default;
  ~HloCostAnalysis() override = default;

  Status HandleElementwiseUnary(HloInstruction* hlo, HloOpcode opcode,
                                HloInstruction* operand) override;
  Status HandleElementwiseBinary(HloInstruction* hlo, HloOpcode opcode,
                                 HloInstruction* lhs,
                                 HloInstruction* rhs) override;
  Status HandleConstant(HloInstruction* constant,
                        const Literal& literal) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element,
                               HloInstruction* operand) override;
  Status HandleSelect(HloInstruction* select, HloInstruction* pred,
                      HloInstruction* on_true,
                      HloInstruction* on_false) override;
  Status HandleCompare(HloInstruction* compare, HloOpcode opcode,
                       HloInstruction* lhs, HloInstruction* rhs) override;
  Status HandleClamp(HloInstruction* clamp, HloInstruction* min,
                     HloInstruction* arg, HloInstruction* max) override;
  Status HandleConcatenate(
      HloInstruction* concatenate,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) override;
  Status HandleSend(HloInstruction* send) override;
  Status HandleRecv(HloInstruction* recv) override;
  Status HandleConvert(HloInstruction* convert,
                       HloInstruction* operand) override;
  Status HandleCopy(HloInstruction* copy, HloInstruction* operand) override;
  Status HandleDot(HloInstruction* dot, HloInstruction* lhs,
                   HloInstruction* rhs) override;
  Status HandleConvolution(HloInstruction* convolution, HloInstruction* lhs,
                           HloInstruction* rhs, const Window& window) override;
  Status HandleCrossReplicaSum(HloInstruction* crs) override;
  Status HandleInfeed(HloInstruction* infeed) override;
  Status HandleOutfeed(HloInstruction* outfeed) override;
  Status HandleRng(HloInstruction* random,
                   RandomDistribution distribution) override;
  Status HandleReverse(HloInstruction* reverse,
                       HloInstruction* operand) override;
  Status HandleSort(HloInstruction* sort, HloInstruction* operand) override;
  Status HandleParameter(HloInstruction* parameter) override;
  Status HandleReduce(HloInstruction* reduce, HloInstruction* arg,
                      HloInstruction* init_value,
                      tensorflow::gtl::ArraySlice<int64> dimensions,
                      HloComputation* function_handle) override;
  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleCall(HloInstruction* call,
                    tensorflow::gtl::ArraySlice<HloInstruction*> operands,
                    HloComputation* computation) override;
  Status HandleCustomCall(HloInstruction* custom_call,
                          tensorflow::gtl::ArraySlice<HloInstruction*> operands,
                          tensorflow::StringPiece custom_call_target) override;
  Status HandleSlice(HloInstruction* slice, HloInstruction* operand) override;
  Status HandleDynamicSlice(
      HloInstruction* slice,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) override;
  Status HandleDynamicUpdateSlice(HloInstruction* dynamic_update_slice,
                                  HloInstruction* operand,
                                  HloInstruction* update,
                                  HloInstruction* start_indices) override;
  Status HandleTuple(
      HloInstruction* tuple,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) override;
  Status HandleMap(
      HloInstruction* map,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      HloComputation* function,
      tensorflow::gtl::ArraySlice<HloInstruction*> static_operands) override;
  Status HandleReduceWindow(HloInstruction* reduce_window,
                            HloInstruction* operand, const Window& window,
                            HloComputation* function) override;
  Status HandleSelectAndScatter(HloInstruction* instruction) override;
  Status HandleBitcast(HloInstruction* bitcast) override;
  Status HandleBroadcast(HloInstruction* broadcast) override;
  Status HandlePad(HloInstruction* pad) override;
  Status HandleReshape(HloInstruction* reshape) override;
  Status HandleTranspose(HloInstruction* transpose) override;
  Status HandleWhile(HloInstruction* xla_while, HloInstruction* init,
                     HloComputation* condition, HloComputation* body) override;
  Status FinishVisit(HloInstruction* root) override;

  // Returns the amount of computations in the graph.
  double flop_count() { return flop_count_; }
  double transcendental_count() { return transcendental_count_; }

  // Resolves the provided HLO instruction to a flop count, or 0 if the HLO was
  // not found to have a flop count in the analysis.
  double hlo_to_flop_count(const HloInstruction& hlo) const;

 private:
  // An FMA counts as two floating point operations in these analyses.
  static constexpr int64 kFmaFlops = 2;

  // Utility function to handle all element-wise operations.
  Status HandleElementwiseOp(HloInstruction* hlo_instruction);

  // Mapping from HLO instructions to the flop count we computed for them in the
  // course of the graph analysis.
  std::map<const HloInstruction*, double> hlo_to_flop_count_;

  // The number of floating point operations in the graph.
  double flop_count_ = 0.0;

  // The number of transcendental operations in the graph.
  double transcendental_count_ = 0.0;

  TF_DISALLOW_COPY_AND_ASSIGN(HloCostAnalysis);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COST_ANALYSIS_H_
