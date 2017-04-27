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
#include "tensorflow/compiler/xla/shape_util.h"
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
  // shape_size is a function which returns the size in bytes of the top-level
  // buffer of a shape.
  using ShapeSizeFunction = std::function<int64(const Shape&)>;
  explicit HloCostAnalysis(const ShapeSizeFunction& shape_size)
      : shape_size_(shape_size) {}

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
  Status HandleCall(HloInstruction* call) override;
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
  Status HandleWhile(HloInstruction* xla_while) override;
  Status FinishVisit(HloInstruction* root) override;

  Status Preprocess(HloInstruction* hlo) override;
  Status Postprocess(HloInstruction* hlo) override;

  // Returns the amount of computations in the graph.
  int64 flop_count() const { return flop_count_; }
  int64 transcendental_count() const { return transcendental_count_; }

  // Returns the respective cost computed for a particular HLO instruction, or 0
  // if the HLO was not found to have a cost in the analysis.
  int64 flop_count(const HloInstruction& hlo) const;
  int64 transcendental_count(const HloInstruction& hlo) const;

  // Returns the number of bytes read/written.
  int64 bytes_accessed(const HloInstruction& hlo) const;
  int64 bytes_accessed() const { return bytes_accessed_; }

 private:
  // An FMA counts as two floating point operations in these analyses.
  static constexpr int64 kFmaFlops = 2;

  // Utility function to handle all element-wise operations.
  Status HandleElementwiseOp(HloInstruction* hlo_instruction);

  // Function which computes the size of the top-level of a given shape (not
  // including nested elements, if any). If null then bytes_accessed methods
  // return an error.
  const ShapeSizeFunction shape_size_;

  // The total number of floating point operations, transcendental operations,
  // and bytes accesses (read or written) in the computation.
  int64 flop_count_ = 0;
  int64 transcendental_count_ = 0;
  int64 bytes_accessed_ = 0;

  // Cost counts of the current instruction. These should be set by each
  // handlers if different from the default values computed in Preprocess.
  int64 current_flop_count_;
  int64 current_transcendental_count_;
  int64 current_bytes_accessed_;

  // Mapping from HLO instructions to the cost we computed for them in the
  // course of the graph analysis.
  std::map<const HloInstruction*, int64> hlo_to_flop_count_;
  std::map<const HloInstruction*, int64> hlo_to_transcendental_count_;
  std::map<const HloInstruction*, int64> hlo_to_bytes_accessed_;

  TF_DISALLOW_COPY_AND_ASSIGN(HloCostAnalysis);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COST_ANALYSIS_H_
