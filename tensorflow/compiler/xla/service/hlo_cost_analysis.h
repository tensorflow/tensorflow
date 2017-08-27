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
  // Each HLO is associated to a vector of properties with the indices given
  // below. Sub-classes can add further properties.
  typedef std::map<string, float> Properties;
  static constexpr char kFlopsKey[] = "flops";
  static constexpr char kTranscendentalsKey[] = "transcendentals";
  static constexpr char kBytesAccessedKey[] = "bytes accessed";
  static constexpr char kSecondsKey[] = "seconds";

  // shape_size is a function which returns the size in bytes of the top-level
  // buffer of a shape.
  using ShapeSizeFunction = std::function<int64(const Shape&)>;
  explicit HloCostAnalysis(const ShapeSizeFunction& shape_size);

  Status HandleElementwiseUnary(HloInstruction* hlo, HloOpcode opcode) override;
  Status HandleElementwiseBinary(HloInstruction* hlo,
                                 HloOpcode opcode) override;
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
  Status HandleReducePrecision(HloInstruction* hlo) override;
  Status HandleConcatenate(
      HloInstruction* concatenate,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) override;
  Status HandleSend(HloInstruction* send) override;
  Status HandleRecv(HloInstruction* recv) override;
  Status HandleConvert(HloInstruction* convert) override;
  Status HandleCopy(HloInstruction* copy) override;
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
  Status HandleBatchNormTraining(HloInstruction* batchNormTraining) override;
  Status HandleBatchNormInference(HloInstruction* batchNormInference) override;
  Status HandleBatchNormGrad(HloInstruction* batchNormGrad) override;
  Status HandleFusion(HloInstruction* fusion) override;
  Status HandleCall(HloInstruction* call) override;
  Status HandleCustomCall(HloInstruction* custom_call,
                          tensorflow::gtl::ArraySlice<HloInstruction*> operands,
                          tensorflow::StringPiece custom_call_target) override;
  Status HandleSlice(HloInstruction* slice, HloInstruction* operand) override;
  Status HandleDynamicSlice(HloInstruction* dynamic_slice,
                            HloInstruction* operand,
                            HloInstruction* start_indices) override;
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
  float seconds() const;

  // Returns the respective cost computed for a particular HLO instruction, or 0
  // if the HLO was not found to have a cost in the analysis.
  int64 flop_count(const HloInstruction& hlo) const;
  int64 transcendental_count(const HloInstruction& hlo) const;
  int64 bytes_accessed(const HloInstruction& hlo) const;
  float seconds(const HloInstruction& hlo) const;

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
  // given hlo. Uses shape_size_ to calculate shape sizes if shape_size is null,
  // otherwise uses shape_size_.
  StatusOr<Properties> ProcessSubcomputation(
      HloComputation* computation,
      const ShapeSizeFunction* shape_size = nullptr);

  // Utility function to handle all element-wise operations.
  Status HandleElementwiseOp(HloInstruction* hlo_instruction);

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
