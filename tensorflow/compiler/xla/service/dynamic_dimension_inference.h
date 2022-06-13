/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DYNAMIC_DIMENSION_INFERENCE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DYNAMIC_DIMENSION_INFERENCE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// DynamicDimensionInference analyzes each HLO instruction in a graph and
// inferences which dimensions are dynamic and which scalar instructions
// represent the runtime real size of those dynamic dimensions.
class DynamicDimensionInference {
 public:
  enum ShapeCheckMode {
    kInvalid = 0,
    // At compile time, pessimisticly assumes runtime shape checks may fail and
    // returns a compile-time error.
    kCompileTime,
    // Insert runtime checks as Hlo ops.
    kRuntime,
    // Ignore shape check.
    kIgnore,
  };
  using CustomCallInferenceHandler =
      std::function<Status(HloInstruction*, DynamicDimensionInference*)>;

  // Generate an assertion which fails the execution if the instruction value is
  // false.
  using AssertionGenerator = std::function<void(HloInstruction*)>;

  static StatusOr<DynamicDimensionInference> Run(
      HloModule* module,
      CustomCallInferenceHandler custom_call_handler = nullptr,
      ShapeCheckMode shape_check_mode = ShapeCheckMode::kIgnore,
      const AssertionGenerator& assertion_generator = nullptr);

  std::string ToString() const;

  // If the dimension `dim` of instruction `inst` at `index` has a dynamic size,
  // returns a scalar HloInstruction that represents the runtime size of that
  // dimension. Otherwise returns nullptr.
  HloInstruction* GetDynamicSize(HloInstruction* inst, const ShapeIndex& index,
                                 int64_t dim) const;

  // Returns dynamic sizes of all dimensions of `inst`'s leaf node at `index`.
  // Static sizes are represented by nullptr.
  std::vector<HloInstruction*> GetDynamicSizes(HloInstruction* inst,
                                               const ShapeIndex& index) const;

  // Returns if `index` at `inst` contains any dynamic dimension.
  // Recursively go into tuples.
  bool HasDynamicDimension(HloInstruction* inst,
                           ShapeIndexView index = {}) const;

  // Forward dynamic dimension size at `dim` from `inst` to `new_inst`.
  Status ForwardDynamicSize(HloInstruction* inst, HloInstruction* new_inst,
                            const ShapeIndex& index);

  // Update the dynamic mapping so that we know dimension `dim` of instruction
  // `inst` at `index` has a dynamic size, and its runtime size is represented
  // by a scalar instruction `size`.
  void SetDynamicSize(HloInstruction* inst, const ShapeIndex& index,
                      int64_t dim, HloInstruction* size);

  // For all tensors whose dynamic dimension is `replace`, replace them with
  // `with`.
  void ReplaceAllDynamicDimensionUsesWith(HloInstruction* replace,
                                          HloInstruction* with);

  // Update dynamic dimension inference to analyze `inst`. Useful to
  // incrementally track new instructions added after initial run.
  Status Update(HloInstruction* inst);

  friend class DynamicDimensionInferenceVisitor;

 private:
  explicit DynamicDimensionInference(
      HloModule* module, CustomCallInferenceHandler custom_call_handler,
      ShapeCheckMode shape_check_mode, AssertionGenerator assertion_generator);

  // DynamicDimension is used as a key in the dynamic key-value mapping. It
  // unambiguously represents a dynamic dimension of a instruction at a given
  // index.
  struct DynamicDimension {
    // HloInstruction that holds the dimension.
    HloInstruction* inst;
    // Subshape of the instruction that holds the dimension.
    ShapeIndex index;
    // The dimension number of the dynamic dimension at given index of a given
    // instruction.
    int64_t dim;

    // Artifacts needed to make this struct able to be used as a `key` in absl
    // maps. "friend" keywords are added so these functions can be found through
    // ADL.
    template <typename H>
    friend H AbslHashValue(H h, const DynamicDimension& m) {
      return H::combine(std::move(h), m.inst, m.index, m.dim);
    }

    friend bool operator==(const DynamicDimension& lhs,
                           const DynamicDimension& rhs) {
      return lhs.inst == rhs.inst && lhs.index == rhs.index &&
             lhs.dim == rhs.dim;
    }

    std::tuple<int, int, std::string, int64_t> ToTuple() const {
      return std::make_tuple(
          inst && inst->GetModule() ? inst->GetModule()->unique_id() : -1,
          inst ? inst->unique_id() : -1, index.ToString(), dim);
    }

    friend bool operator<(const DynamicDimension& lhs,
                          const DynamicDimension& rhs) {
      return lhs.ToTuple() < rhs.ToTuple();
    }
  };

  // Copies the internal mapping from instruction `from` to instruction `to`.
  // This is useful when an instruction is replaced by the other during the
  // inferencing process.
  void CopyMapping(HloInstruction* from, HloInstruction* to);

  // AnalyzeDynamicDimensions starts the analysis of the dynamic dimensions in
  // module_.
  Status AnalyzeDynamicDimensions();

  // HloModule being analyzed.
  HloModule* module_;

  // dynamic_mapping_ holds the result of the analysis. It maps a dynamic
  // dimension to a scalar HloInstruction that represents the real dynamic size
  // of the dynamic dimension.
  using DynamicMapping = std::map<DynamicDimension, HloInstruction*>;
  DynamicMapping dynamic_mapping_;

  // A convenient mapping from an hlo to the set of dynamic dimensions that it
  // holds.
  using PerHloDynamicDimensions =
      ConstHloInstructionMap<std::set<DynamicDimension>>;
  PerHloDynamicDimensions per_hlo_dynamic_dimensions_;

  // A handler for custom calls.
  CustomCallInferenceHandler custom_call_handler_;

  // Indicates what to do at places where shape check is needed.
  ShapeCheckMode shape_check_mode_;

  AssertionGenerator assertion_generator_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DYNAMIC_DIMENSION_INFERENCE_H_
