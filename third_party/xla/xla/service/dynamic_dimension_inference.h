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

#ifndef XLA_SERVICE_DYNAMIC_DIMENSION_INFERENCE_H_
#define XLA_SERVICE_DYNAMIC_DIMENSION_INFERENCE_H_

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"

namespace xla {

// Each instruction can have one of the three modes in supporting dynamic
// lowering.
enum OpDynamismSupport : uint8_t {
  // There is no support for dynamic lowering -- dynamic padder will make sure
  // the input to that op has static bound by rewriting the op (e.g, extra space
  // in reduce_sum will be padded with 0).
  kNoSupport = 0,
  // The op can take either dynamic input or static input.
  kOptional,
  // The op only has a dynamic lowering, dynamic padder will make sure the input
  // to this op is in dynamic form.
  kRequired,
};

// Returns true if given instruction supports native dynamic lowering. If
// so, dynamic padder will not attempt to pad it.
using OpSupportsDynamismHandler =
    std::function<OpDynamismSupport(HloInstruction*)>;

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
      OpSupportsDynamismHandler op_supports_dynamism_handler = nullptr,
      CustomCallInferenceHandler custom_call_handler = nullptr,
      ShapeCheckMode shape_check_mode = ShapeCheckMode::kIgnore,
      const AssertionGenerator& assertion_generator = nullptr,
      const absl::flat_hash_set<absl::string_view>& execution_threads_ = {});

  std::string ToString() const;

  // If the dimension `dim` of instruction `inst` at `index` has a dynamic size,
  // returns a scalar HloInstruction that represents the runtime size of that
  // dimension. Otherwise returns nullptr.
  HloInstruction* GetDynamicSize(HloInstruction* inst, const ShapeIndex& index,
                                 int64_t dim) const;

  const HloInstruction* GetDynamicSize(const HloInstruction* inst,
                                       const ShapeIndex& index,
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

  // Get the original dynamic shape of the given instruction.
  Shape GetDynamicShape(HloInstruction* inst);

  // Returns true iff all dynamic dimensions on the operands of the given
  // instruction have inferred dynamic sizes.
  bool CanInfer(HloInstruction* hlo);

  // Returns true iff DynamicDimensionInferenceVisitor made changes to the
  // module.
  bool changed() const { return changed_; }

  friend class DynamicDimensionInferenceVisitor;

 private:
  explicit DynamicDimensionInference(
      HloModule* module, OpSupportsDynamismHandler op_supports_dynamism_handler,
      CustomCallInferenceHandler custom_call_handler,
      ShapeCheckMode shape_check_mode, AssertionGenerator assertion_generator,
      const absl::flat_hash_set<absl::string_view>& execution_threads_);

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
  // For cases where the `from` and `to` instructions are in different
  // computations, a `dynamic_size_map` can be provided which maps the dynamic
  // size instructions in the `from` computation into the corresponding
  // instruction in the `to` computation.
  void CopyMapping(HloInstruction* from, HloInstruction* to,
                   const absl::flat_hash_map<HloInstruction*, HloInstruction*>*
                       dynamic_size_map = nullptr);

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

  OpSupportsDynamismHandler op_supports_dynamism_handler_;

  // A handler for custom calls.
  CustomCallInferenceHandler custom_call_handler_;

  // Indicates what to do at places where shape check is needed.
  ShapeCheckMode shape_check_mode_;

  AssertionGenerator assertion_generator_;

  bool changed_ = false;

  const absl::flat_hash_set<absl::string_view>& execution_threads_;
};

}  // namespace xla

#endif  // XLA_SERVICE_DYNAMIC_DIMENSION_INFERENCE_H_
