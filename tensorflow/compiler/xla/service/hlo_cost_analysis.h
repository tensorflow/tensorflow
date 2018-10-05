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

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// HloCostAnalysis traverses an HLO graph and calculates the amount of
// computations required for the graph. Each HLO instruction handler provides
// the computation cost of the instruction, and the values are accumulated
// during the traversal for the entire graph. We treat normal floating point
// operations separately from transcendental operations.
class HloCostAnalysis : public ConstDfsHloVisitor {
 public:
  // Each HLO is associated to a vector of properties with the indices given
  // below. Sub-classes can add further properties.
  typedef std::map<string, float> Properties;
  static constexpr char kFlopsKey[] = "flops";
  static constexpr char kTranscendentalsKey[] = "transcendentals";
  static constexpr char kBytesAccessedKey[] = "bytes accessed";
  static constexpr char kOptimalSecondsKey[] = "optimal_seconds";

  // shape_size is a function which returns the size in bytes of the top-level
  // buffer of a shape.
  using ShapeSizeFunction = std::function<int64(const Shape&)>;
  explicit HloCostAnalysis(const ShapeSizeFunction& shape_size);

  Status HandleElementwiseUnary(const HloInstruction* hlo) override;
  Status HandleElementwiseBinary(const HloInstruction* hlo) override;
  Status HandleConstant(const HloInstruction* constant) override;
  Status HandleIota(const HloInstruction* iota) override;
  Status HandleGetTupleElement(
      const HloInstruction* get_tuple_element) override;
  Status HandleSelect(const HloInstruction* hlo) override;
  Status HandleTupleSelect(const HloInstruction* hlo) override;
  Status HandleCompare(const HloInstruction* compare) override;
  Status HandleClamp(const HloInstruction* clamp) override;
  Status HandleReducePrecision(const HloInstruction* hlo) override;
  Status HandleConcatenate(const HloInstruction* concatenate) override;
  Status HandleSend(const HloInstruction* send) override;
  Status HandleSendDone(const HloInstruction* send_done) override;
  Status HandleRecv(const HloInstruction* recv) override;
  Status HandleRecvDone(const HloInstruction* recv_done) override;
  Status HandleConvert(const HloInstruction* convert) override;
  Status HandleCopy(const HloInstruction* copy) override;
  Status HandleDomain(const HloInstruction* domain) override;
  Status HandleDot(const HloInstruction* dot) override;
  Status HandleConvolution(const HloInstruction* convolution) override;
  Status HandleFft(const HloInstruction* fft) override;
  Status HandleCrossReplicaSum(const HloInstruction* crs) override;
  Status HandleAllToAll(const HloInstruction* hlo) override;
  Status HandleCollectivePermute(const HloInstruction* hlo) override;
  Status HandleInfeed(const HloInstruction* infeed) override;
  Status HandleOutfeed(const HloInstruction* outfeed) override;
  Status HandleRng(const HloInstruction* random) override;
  Status HandleReverse(const HloInstruction* reverse) override;
  Status HandleSort(const HloInstruction* sort) override;
  Status HandleParameter(const HloInstruction* parameter) override;
  Status HandleReduce(const HloInstruction* reduce) override;
  Status HandleBatchNormTraining(
      const HloInstruction* batch_norm_training) override;
  Status HandleBatchNormInference(
      const HloInstruction* batch_norm_inference) override;
  Status HandleBatchNormGrad(const HloInstruction* batch_norm_grad) override;
  Status HandleFusion(const HloInstruction* fusion) override;
  Status HandleCall(const HloInstruction* call) override;
  Status HandleCustomCall(const HloInstruction* custom_call) override;
  Status HandleSlice(const HloInstruction* slice) override;
  Status HandleDynamicSlice(const HloInstruction* dynamic_slice) override;
  Status HandleDynamicUpdateSlice(
      const HloInstruction* dynamic_update_slice) override;
  Status HandleTuple(const HloInstruction* tuple) override;
  Status HandleMap(const HloInstruction* map) override;
  Status HandleReduceWindow(const HloInstruction* reduce_window) override;
  Status HandleSelectAndScatter(const HloInstruction* instruction) override;
  Status HandleBitcast(const HloInstruction* bitcast) override;
  Status HandleBroadcast(const HloInstruction* broadcast) override;
  Status HandlePad(const HloInstruction* pad) override;
  Status HandleReshape(const HloInstruction* reshape) override;
  Status HandleAfterAll(const HloInstruction* token) override;
  Status HandleTranspose(const HloInstruction* transpose) override;
  Status HandleWhile(const HloInstruction* xla_while) override;
  Status HandleConditional(const HloInstruction* conditional) override;
  Status HandleGather(const HloInstruction* gather) override;
  Status HandleScatter(const HloInstruction* scatter) override;
  Status FinishVisit(const HloInstruction* root) override;

  Status Preprocess(const HloInstruction* hlo) override;
  Status Postprocess(const HloInstruction* hlo) override;

  // Set the rates used to calculate the time taken by the computation. These
  // need to be set before visiting starts.
  void set_flops_per_second(float value) {
    per_second_rates_[kFlopsKey] = value;
  }
  void set_transcendentals_per_second(float value) {
    per_second_rates_[kTranscendentalsKey] = value;
  }
  void set_bytes_per_second(float value) {
    per_second_rates_[kBytesAccessedKey] = value;
  }

  // Returns properties for the computation.
  float flop_count() const;
  float transcendental_count() const;
  float bytes_accessed() const;
  float optimal_seconds() const;

  // Returns the respective cost computed for a particular HLO instruction, or 0
  // if the HLO was not found to have a cost in the analysis.
  int64 flop_count(const HloInstruction& hlo) const;
  int64 transcendental_count(const HloInstruction& hlo) const;
  int64 bytes_accessed(const HloInstruction& hlo) const;
  float optimal_seconds(const HloInstruction& hlo) const;

  const Properties& properties() const { return properties_sum_; }
  const float property(const string& key) const {
    return GetProperty(key, properties());
  }

 protected:
  typedef std::unordered_map<const HloInstruction*, Properties> HloToProperties;

  // An FMA counts as two floating point operations in these analyzes.
  static constexpr int64 kFmaFlops = 2;

  HloCostAnalysis(const ShapeSizeFunction& shape_size,
                  const Properties& per_second_rates);

  // Returns the properties computed from visiting the computation rooted at the
  // given hlo.
  StatusOr<Properties> ProcessSubcomputation(HloComputation* computation);

  // Utility function to handle all element-wise operations.
  Status HandleElementwiseOp(const HloInstruction* hlo_instruction);

  // Returns the default value if the key is not present in the
  // properties. Otherwise, returns the value that the key maps to from the
  // properties parameter.
  static float GetProperty(const string& key, const Properties& properties,
                           float default_value = 0.0f);

  // Returns 0.0f if the hlo is not present in hlo_to_properties or if the key
  // is not present in hlo_to_properties[hlo]. Otherwise, returns the value that
  // the key maps to in the properties of the given hlo.
  static float GetPropertyForHlo(const HloInstruction& hlo, const string& key,
                                 const HloToProperties& hlo_to_properties);

  // Decorates shape_size_ by returning 0 immediately if the shape does not have
  // a layout.
  int64 GetShapeSize(const Shape& shape) const;

  // Function which computes the size of the top-level of a given shape (not
  // including nested elements, if any). If null then bytes_accessed methods
  // return an error.
  const ShapeSizeFunction shape_size_;

  HloToProperties hlo_properties_;

  // If true, the time taken will be computed from the rates for each property
  // and the total time will be the maximum time, which is the time of the
  // bottleneck.
  bool current_should_compute_bottleneck_time_;

  // The properties of the currently visited instruction. A HandleFoo method can
  // modify these to change the default values computed in Preprocess.
  Properties current_properties_;

  // The sum of the properties of all HLOs in the computation.
  Properties properties_sum_;

  // How much of each property can be processed per second. E.g. if the property
  // is bytes accessed, this is the number of bytes that can be processed per
  // second. Is empty if no rates have been set.
  Properties per_second_rates_;

  TF_DISALLOW_COPY_AND_ASSIGN(HloCostAnalysis);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_COST_ANALYSIS_H_
