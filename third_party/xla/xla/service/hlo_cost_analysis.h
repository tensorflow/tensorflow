/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_COST_ANALYSIS_H_
#define XLA_SERVICE_HLO_COST_ANALYSIS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

// HloCostAnalysis traverses an HLO graph and calculates the amount of
// computations required for the graph. Each HLO instruction handler provides
// the computation cost of the instruction, and the values are accumulated
// during the traversal for the entire graph. We treat normal floating point
// operations separately from transcendental operations.
class HloCostAnalysis : public ConstDfsHloVisitor {
 public:
  static inline constexpr absl::string_view kFlopsKey = "flops";
  static inline constexpr absl::string_view kTranscendentalsKey =
      "transcendentals";
  static inline constexpr absl::string_view kBytesAccessedKey =
      "bytes accessed";
  static inline constexpr absl::string_view kOptimalSecondsKey =
      "optimal_seconds";
  static inline constexpr absl::string_view kUtilizationKey = "utilization";

  // Keys reserved for use by subclasses.  These get the same special "fast
  // path" treatment in Properties as the other keys above.
  static inline constexpr absl::string_view kReserved0Key = "reserved0";
  static inline constexpr absl::string_view kReserved1Key = "reserved1";

  // A data structure like hash_map<string, float> for storing info about an HLO
  // instruction or computation.
  //
  // Note that unlike a regular hashtable, there's no notion of an "unset" key.
  // All keys are logically present, with value 0.
  //
  // This data structure *could* be simply map<string, float>, and indeed it
  // was, once.  The problem is, XLA:GPU uses HloCostAnalysis during
  // compilation.  This class is used *everywhere* within cost analysis, and the
  // hashtable lookups added up to the majority (!) of its runtime.
  //
  // This is a bit silly, because the vast majority of the time, we're looking
  // up a small, fixed set of keys.  So you might be tempted to convert
  // Properties into a simple struct of floats.
  //
  // The problem with *that* is threefold.  (1) subclasses expect to be able to
  // store arbitrary keys inside Properties.  This doesn't work if it's a
  // struct.  (2) We expect to be able to store *and retrieve* values
  // representing e.g. "the utilization of operand n at shape index i", and (3)
  // the hashtable-ness of this class is part of XLA's public API and so is hard
  // to change.
  //
  // So instead we end up with this Frankenstein's monster of a class.  It
  // *acts* like a hashtable, but before falling back to the hashtable, it
  // checks whether the string matches one of a list of "known keys".  If so, it
  // returns that special value from the struct.
  //
  // Normally this would be much worse than just using a plain hashtable.  But
  // we happen to know that you're almost always doing prop[kKnownKey], in which
  // case operator[] can be inlined and the string comparison optimized away.
  //
  // Sorry for all this complexity, but this is the most impactful single
  // optimization we were able make to GPU compilation time.
  //
  class Properties {
   public:
    Properties()
        : flops_(0),
          transcendentals_(0),
          bytes_accessed_(0),
          optimal_seconds_(0),
          utilization_(0),
          operand0_utilization_(0),
          operand1_utilization_(0),
          operand0_bytes_accessed_(0),
          operand1_bytes_accessed_(0),
          output_root_bytes_accessed_(0),
          reserved0_(0),
          reserved1_(0) {
      DCHECK_EQ(kOperand0UtilizationKey, GetOperandUtilizationKey(0, {}));
      DCHECK_EQ(kOperand1UtilizationKey, GetOperandUtilizationKey(1, {}));
      DCHECK_EQ(kOperand0BytesAccessedKey, GetOperandBytesAccessedKey(0, {}));
      DCHECK_EQ(kOperand1BytesAccessedKey, GetOperandBytesAccessedKey(1, {}));
      DCHECK_EQ(kOutputRootBytesAccessedKey, GetOutputBytesAccessedKey({}));
    }

    float& operator[](absl::string_view property) {
      if (property == kFlopsKey) {
        return flops_;
      }
      if (property == kTranscendentalsKey) {
        return transcendentals_;
      }
      if (property == kBytesAccessedKey) {
        return bytes_accessed_;
      }
      if (property == kOptimalSecondsKey) {
        return optimal_seconds_;
      }
      if (property == kUtilizationKey) {
        return utilization_;
      }
      if (property == kOperand0UtilizationKey) {
        return operand0_utilization_;
      }
      if (property == kOperand1UtilizationKey) {
        return operand1_utilization_;
      }
      if (property == kOperand0BytesAccessedKey) {
        return operand0_bytes_accessed_;
      }
      if (property == kOperand1BytesAccessedKey) {
        return operand1_bytes_accessed_;
      }
      if (property == kOutputRootBytesAccessedKey) {
        return output_root_bytes_accessed_;
      }
      if (property == kReserved0Key) {
        return reserved0_;
      }
      if (property == kReserved1Key) {
        return reserved1_;
      }

      auto it = named_props_.lazy_emplace(property, [&](const auto& ctor) {
        ctor(std::string(property), 0.f);
      });
      return it->second;
    }

    float operator[](absl::string_view property) const {
      if (property == kFlopsKey) {
        return flops_;
      }
      if (property == kTranscendentalsKey) {
        return transcendentals_;
      }
      if (property == kBytesAccessedKey) {
        return bytes_accessed_;
      }
      if (property == kOptimalSecondsKey) {
        return optimal_seconds_;
      }
      if (property == kUtilizationKey) {
        return utilization_;
      }
      if (property == kOperand0UtilizationKey) {
        return operand0_utilization_;
      }
      if (property == kOperand1UtilizationKey) {
        return operand1_utilization_;
      }
      if (property == kOperand0BytesAccessedKey) {
        return operand0_bytes_accessed_;
      }
      if (property == kOperand1BytesAccessedKey) {
        return operand1_bytes_accessed_;
      }
      if (property == kOutputRootBytesAccessedKey) {
        return output_root_bytes_accessed_;
      }
      if (property == kReserved0Key) {
        return reserved0_;
      }
      if (property == kReserved1Key) {
        return reserved1_;
      }

      auto it = named_props_.find(property);
      if (it != named_props_.end()) {
        return it->second;
      }
      return 0;
    }

    template <typename Fn>
    void ForEach(Fn&& fn) const {
      if (flops_ != 0) {
        fn(kFlopsKey, flops_);
      }
      if (transcendentals_ != 0) {
        fn(kTranscendentalsKey, transcendentals_);
      }
      if (bytes_accessed_ != 0) {
        fn(kBytesAccessedKey, bytes_accessed_);
      }
      if (optimal_seconds_ != 0) {
        fn(kOptimalSecondsKey, optimal_seconds_);
      }
      if (utilization_ != 0) {
        fn(kUtilizationKey, utilization_);
      }
      if (operand0_utilization_ != 0) {
        fn(kOperand0UtilizationKey, operand0_utilization_);
      }
      if (operand1_utilization_ != 0) {
        fn(kOperand1UtilizationKey, operand1_utilization_);
      }
      if (operand0_bytes_accessed_ != 0) {
        fn(kOperand0BytesAccessedKey, operand0_bytes_accessed_);
      }
      if (operand1_bytes_accessed_ != 0) {
        fn(kOperand1BytesAccessedKey, operand1_bytes_accessed_);
      }
      if (output_root_bytes_accessed_ != 0) {
        fn(kOutputRootBytesAccessedKey, output_root_bytes_accessed_);
      }
      if (reserved0_ != 0) {
        fn(kReserved0Key, reserved0_);
      }
      if (reserved1_ != 0) {
        fn(kReserved1Key, reserved1_);
      }

      for (const auto& [k, v] : named_props_) {
        if (v != 0) {
          fn(k, v);
        }
      }
    }

    // No getters/setters for simple properties like flops().  For these,
    // props[kFlopsKey] gets optimized to `return flops_` just fine.

    // Getters/setters for more complex properties like operand utilization,
    // where we have a fastpath, e.g., operand 0/1 + shape_index {}.
    float operand_utilization(int64_t operand,
                              const ShapeIndex& shape_index = {}) {
      if (operand == 0 && shape_index.empty()) {
        return operand0_utilization_;
      }
      if (operand == 1 && shape_index.empty()) {
        return operand1_utilization_;
      }

      auto it =
          named_props_.find(GetOperandUtilizationKey(operand, shape_index));
      if (it != named_props_.end()) {
        return it->second;
      }
      return 0;
    }
    void set_operand_utilization(int64_t operand, float value) {
      set_operand_utilization(operand, /*shape_index=*/{}, value);
    }
    void set_operand_utilization(int64_t operand, const ShapeIndex& shape_index,
                                 float value) {
      if (operand == 0 && shape_index.empty()) {
        operand0_utilization_ = value;
      } else if (operand == 1 && shape_index.empty()) {
        operand1_utilization_ = value;
      } else {
        named_props_[GetOperandUtilizationKey(operand, shape_index)] = value;
      }
    }

    float operand_bytes_accessed(int64_t operand,
                                 const ShapeIndex& shape_index = {}) {
      if (operand == 0 && shape_index.empty()) {
        return operand0_bytes_accessed_;
      }
      if (operand == 1 && shape_index.empty()) {
        return operand1_bytes_accessed_;
      }

      auto it =
          named_props_.find(GetOperandBytesAccessedKey(operand, shape_index));
      if (it != named_props_.end()) {
        return it->second;
      }
      return 0;
    }
    void set_operand_bytes_accessed(int64_t operand, float value) {
      set_operand_bytes_accessed(operand, /*shape_index=*/{}, value);
    }
    void set_operand_bytes_accessed(int64_t operand,
                                    const ShapeIndex& shape_index,
                                    float value) {
      if (operand == 0 && shape_index.empty()) {
        operand0_bytes_accessed_ = value;
      } else if (operand == 1 && shape_index.empty()) {
        operand1_bytes_accessed_ = value;
      } else {
        named_props_[GetOperandBytesAccessedKey(operand, shape_index)] = value;
      }
    }

    float output_bytes_accessed(const ShapeIndex& shape_index = {}) {
      if (shape_index.empty()) {
        return output_root_bytes_accessed_;
      }
      auto it = named_props_.find(GetOutputBytesAccessedKey(shape_index));
      if (it != named_props_.end()) {
        return it->second;
      }
      return 0;
    }
    void set_output_bytes_accessed(float value) {
      set_output_bytes_accessed({}, value);
    }
    void set_output_bytes_accessed(const ShapeIndex& shape_index, float value) {
      if (shape_index.empty()) {
        output_root_bytes_accessed_ = value;
      } else {
        named_props_[GetOutputBytesAccessedKey(shape_index)] = value;
      }
    }

   private:
    // These must match GetOperandUtilizationKey(0, {}) etc.
    static inline constexpr absl::string_view kOperand0UtilizationKey =
        "utilization0{}";
    static inline constexpr absl::string_view kOperand1UtilizationKey =
        "utilization1{}";
    static inline constexpr absl::string_view kOperand0BytesAccessedKey =
        "bytes accessed0{}";
    static inline constexpr absl::string_view kOperand1BytesAccessedKey =
        "bytes accessed1{}";
    static inline constexpr absl::string_view kOutputRootBytesAccessedKey =
        "bytes accessedout{}";

    float flops_;
    float transcendentals_;
    float bytes_accessed_;
    float optimal_seconds_;
    float utilization_;

    float operand0_utilization_;
    float operand1_utilization_;

    float operand0_bytes_accessed_;
    float operand1_bytes_accessed_;

    float output_root_bytes_accessed_;

    // Fields reserved for use by subclasses.
    float reserved0_;
    float reserved1_;

    absl::flat_hash_map<std::string, float> named_props_;
  };

  // shape_size is a function which returns the size in bytes of the top-level
  // buffer of a shape.
  using ShapeSizeFunction = std::function<int64_t(const Shape&)>;

  // A struct to encapsulate hardware-related options. This includes the shape
  // size function, which is used to encode hardware-specific padding and per
  // second rates of FLOPs, bytes per second (available bandwidth), and
  // transcendentals per second.
  struct Options {
    // Function which computes the size of the top-level of a given shape (not
    // including nested elements, if any). If null then bytes_accessed methods
    // return an error.
    ShapeSizeFunction shape_size;
    // How much of each property can be processed per second. E.g. if the
    // property is bytes accessed, this is the number of bytes that can be
    // processed per second. Is empty if no rates have been set.
    Properties per_second_rates = {};
    // Operations like broadcast with reused inputs are not handled
    // efficiently on some platforms. Depending on the goal of the analysis
    // we may need to count or ignore them.
    bool count_multiple_input_accesses = false;

    // Set the rates used to calculate the time taken by the computation.
    void set_flops_per_second(float value) {
      per_second_rates[kFlopsKey] = value;
    }
    void set_transcendentals_per_second(float value) {
      per_second_rates[kTranscendentalsKey] = value;
    }
    void set_bytes_per_second(float value) {
      per_second_rates[kBytesAccessedKey] = value;
    }

    // Returns the specified per-second rate used by cost analysis.
    float per_second_rate(absl::string_view key) const {
      return per_second_rates[key];
    }
  };

  explicit HloCostAnalysis(const Options& options);
  explicit HloCostAnalysis(ShapeSizeFunction shape_size,
                           const Properties& per_second_rates = {});

  Status HandleElementwiseUnary(const HloInstruction* hlo) override;
  Status HandleElementwiseBinary(const HloInstruction* hlo) override;
  Status HandleConstant(const HloInstruction* constant) override;
  Status HandleIota(const HloInstruction* iota) override;
  Status HandleGetTupleElement(
      const HloInstruction* get_tuple_element) override;
  Status HandleSelect(const HloInstruction* hlo) override;
  Status HandleCompare(const HloInstruction* compare) override;
  Status HandleClamp(const HloInstruction* clamp) override;
  Status HandleReducePrecision(const HloInstruction* hlo) override;
  Status HandleConcatenate(const HloInstruction* concatenate) override;
  Status HandleAsyncStart(const HloInstruction* async_start) override;
  Status HandleAsyncUpdate(const HloInstruction* async_update) override;
  Status HandleAsyncDone(const HloInstruction* async_done) override;
  Status HandleCopyStart(const HloInstruction* send) override;
  Status HandleCopyDone(const HloInstruction* send_done) override;
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
  Status HandleTriangularSolve(const HloInstruction* hlo) override;
  Status HandleCholesky(const HloInstruction* hlo) override;
  Status HandleOptimizationBarrier(const HloInstruction* hlo) override;
  Status HandleAllGather(const HloInstruction* hlo) override;
  Status HandleAllGatherStart(const HloInstruction* hlo) override;
  Status HandleAllGatherDone(const HloInstruction* hlo) override;
  Status HandleAllReduce(const HloInstruction* crs) override;
  Status HandleReduceScatter(const HloInstruction* hlo) override;
  Status HandleAllReduceStart(const HloInstruction* hlo) override;
  Status HandleAllReduceDone(const HloInstruction* hlo) override;
  Status HandleAllToAll(const HloInstruction* hlo) override;
  Status HandleCollectiveBroadcast(const HloInstruction* hlo) override;
  Status HandleCollectivePermute(const HloInstruction* hlo) override;
  Status HandleCollectivePermuteStart(const HloInstruction* hlo) override;
  Status HandleCollectivePermuteDone(const HloInstruction* hlo) override;
  Status HandleReplicaId(const HloInstruction* hlo) override;
  Status HandlePartitionId(const HloInstruction* hlo) override;
  Status HandleInfeed(const HloInstruction* infeed) override;
  Status HandleOutfeed(const HloInstruction* outfeed) override;
  Status HandleRng(const HloInstruction* random) override;
  Status HandleRngBitGenerator(const HloInstruction* random) override;
  Status HandleRngGetAndUpdateState(const HloInstruction* random) override;
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
  Status HandleDynamicReshape(const HloInstruction* reshape) override;
  Status HandleAddDependency(const HloInstruction* add_dependency) override;
  Status HandleAfterAll(const HloInstruction* token) override;
  Status HandleTranspose(const HloInstruction* transpose) override;
  Status HandleWhile(const HloInstruction* xla_while) override;
  Status HandleConditional(const HloInstruction* conditional) override;
  Status HandleGather(const HloInstruction* gather) override;
  Status HandleScatter(const HloInstruction* hlo) override;
  Status HandleGetDimensionSize(const HloInstruction* get_size) override;
  Status HandleSetDimensionSize(const HloInstruction* set_size) override;
  Status HandleTopK(const HloInstruction* topk) override;
  Status FinishVisit(const HloInstruction* root) override;

  Status Preprocess(const HloInstruction* hlo) override;
  Status Postprocess(const HloInstruction* hlo) override;

  // Enable efficient updates if a known small set of instructions within an
  // HLO graph was modified.
  // Updates the cost analysis by removing one instruction.
  Status RemoveInstruction(HloInstruction* instruction);
  // Updates the cost analysis by re-doing the analysis of one instruction.
  Status RevisitInstruction(HloInstruction* instruction);

  // Decorates shape_size_ by returning 0 immediately if the shape does not have
  // a layout.
  int64_t GetShapeSize(const Shape& shape) const;

  // Returns properties for the computation.
  float flop_count() const;
  float transcendental_count() const;
  float bytes_accessed() const;
  float optimal_seconds() const;

  // Returns the respective cost computed for a particular HLO instruction, or 0
  // if the HLO was not found to have a cost in the analysis.
  //
  // Note that the cost for sub HLO instructions are also returned if asked. For
  // example, body and condition of a while, fused instructions within a
  // fusion, or the add instruction of a reduce.
  int64_t flop_count(const HloInstruction& hlo) const;
  int64_t transcendental_count(const HloInstruction& hlo) const;
  int64_t bytes_accessed(const HloInstruction& hlo) const;
  int64_t operand_bytes_accessed(const HloInstruction& hlo, int64_t operand_num,
                                 ShapeIndex index = {}) const;
  // Value indicating how much each input of the instruction
  // is used assuming its output is fully used.
  // This is 1.0 for most cases except operations involving slicing (<1)
  // and on some backends in addition reuse of inputs (>1).
  float operand_utilization(const HloInstruction& hlo, int64_t operand_num,
                            ShapeIndex index = {}) const;
  int64_t output_bytes_accessed(const HloInstruction& hlo,
                                ShapeIndex index = {}) const;
  float optimal_seconds(const HloInstruction& hlo) const;

  // Get bytes read/written by this HLO. If memory_space is provided, it returns
  // the bytes read/written from/to the given memory space only.
  int64_t GetBytesRead(
      const HloInstruction& hlo,
      std::optional<int64_t> memory_space = std::nullopt) const;
  int64_t GetBytesWritten(
      const HloInstruction& hlo,
      std::optional<int64_t> memory_space = std::nullopt) const;

  const Properties& properties() const { return properties_sum_; }
  float property(absl::string_view key) { return properties_sum_[key]; }

  // Returns the specified per-second rate used by cost analysis.
  float per_second_rate(absl::string_view key) const {
    return options_.per_second_rate(key);
  }

  // Return the key that is used to index into Properties for the specified
  // input/output at the shape index.
  static std::string GetOperandBytesAccessedKey(int64_t operand_num,
                                                const ShapeIndex& index = {});
  static std::string GetOperandUtilizationKey(int64_t operand_num,
                                              const ShapeIndex& index = {});
  static std::string GetOutputBytesAccessedKey(const ShapeIndex& index = {});

  // Returns the estimated convolution flops.
  virtual int64_t GetConvolutionFlops(const HloInstruction* convolution);
  // Same as above but with parameters for shapes to allow for backends to
  // refine these.
  static int64_t GetConvolutionFlops(const HloInstruction* convolutions,
                                     const Shape& lhs_shape,
                                     const Shape& rhs_shape,
                                     const Shape& result_shape);

  // Returns the estimated dot flops.
  static int64_t GetDotFlops(const Shape& lhs_shape, const Shape& result_shape,
                             const DotDimensionNumbers& dnums);

 protected:
  // Computes the bytes accessed based on the outputs produced by the fusion
  // instruction.
  virtual Status FusionProcessOutputBytesAccessed(const HloInstruction* fusion);

  // Computes the bytes accessed (read) based on the inputs consumed by the
  // fusion instruction.
  virtual Status FusionProcessOperandBytesRead(const HloInstruction* fusion);

  // Computes memory access to all larger constants in the fusion instruction.
  virtual Status FusionCountConstantsMemoryAccess(const HloInstruction* fusion);

  // Allows exclusion of certain types of inputs from bytes accessed during
  // FusionProcessOperandBytesRead.
  virtual bool ShouldFilterFusionInput(const HloInstruction* fusion,
                                       int64_t input_index) {
    return false;
  }

  // Allows exclusion of certain instructions from FusionCalculateUtilizations.
  virtual bool ShouldFilterFusionInstruction(
      const HloInstruction* fusion, const HloInstruction* instruction) {
    return false;
  }

  // Allows exclusion of certain types of output from bytes written during
  // FusionProcessOutputBytesAccessed.
  virtual bool ShouldFilterFusionOutputIndex(const HloInstruction* fusion,
                                             const ShapeIndex& output_index) {
    return false;
  }

  typedef absl::flat_hash_map<const HloInstruction*, Properties>
      HloToProperties;

  // An FMA counts as two floating point operations in these analyzes.
  static constexpr int64_t kFmaFlops = 2;

  // Small constants can be embedded in the assembly and not require
  // memory access.
  virtual size_t immediate_constant_max_elements() const { return 1; }

  // Creates a nested instance of HloCostAnalysis using the same Options.
  virtual std::unique_ptr<HloCostAnalysis> CreateNestedCostAnalysis();

  // Returns the properties computed from visiting the computation rooted at the
  // given hlo. The cost of visited sub HLO instructions is saved to
  // hlo_properties_, which will be used by functions such as
  // flop_count(hlo_instruction) to return cost of a particular HLO instruction.
  virtual StatusOr<Properties> ProcessSubcomputation(
      HloComputation* computation);

  // Utility function to handle all element-wise operations.
  Status HandleElementwiseOp(const HloInstruction* hlo_instruction);

  // Returns 0.0f if the hlo is not present in hlo_to_properties or if the key
  // is not present in hlo_to_properties[hlo]. Otherwise, returns the value that
  // the key maps to in the properties of the given hlo.
  static float GetPropertyForHlo(const HloInstruction& hlo,
                                 absl::string_view key,
                                 const HloToProperties& hlo_to_properties);

  // Traverses a fusion operand to find the actual bytes accessed by the fusion
  // node.
  virtual int64_t FusionParameterReadBytes(const HloInstruction* hlo) const;

  // Traverses a fusion counting total utilization of every instruction inside.
  // Currently implemented non-trivially only in the GPU cost analysis.
  virtual Status FusionCalculateUtilizations(const HloInstruction* fusion);

  HloToProperties hlo_properties_;

  // If true, the time taken will be computed from the rates for each property
  // and the total time will be the maximum time, which is the time of the
  // bottleneck.
  bool current_should_compute_bottleneck_time_;

  // The properties of the currently visited instruction. A HandleFoo method
  // modify these to change the default values computed in Preprocess.
  Properties current_properties_;

  // The sum of the properties of all HLOs in the computation.
  Properties properties_sum_;

  // The hardware-specific options that contains things like the shape size
  // function and per-second rates.
  Options options_;

  // Determines which properties propagate from subcomputations to parents.
  virtual bool KeyToCopyFromSubcomputation(absl::string_view key) const;

  HloCostAnalysis(const HloCostAnalysis&) = delete;
  HloCostAnalysis& operator=(const HloCostAnalysis&) = delete;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_COST_ANALYSIS_H_
