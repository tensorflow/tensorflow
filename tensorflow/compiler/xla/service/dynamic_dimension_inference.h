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

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// DynamicDimensionInference analyzes each HLO instruction in a graph and
// inferences which dimensions are dynamic and which scalar instructions
// represent the runtime real size of those dynamic dimensions.
class DynamicDimensionInference {
 public:
  static StatusOr<DynamicDimensionInference> Run(HloModule* module);

  string ToString() const;

  // If the dimension `dim` of instruction `inst` at `index` has a dynamic size,
  // returns a scalar HloInstruction that represents the runtime size of that
  // dimension. Otherwise returns nullptr.
  HloInstruction* GetDynamicSize(HloInstruction* inst, const ShapeIndex& index,
                                 int64 dim) const;

  friend class DynamicDimensionInferenceVisitor;

 private:
  explicit DynamicDimensionInference(HloModule* module);

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
    int64 dim;

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
  };

  // Update the dynamic mapping so that we know dimension `dim` of instruction
  // `inst` at `index` has a dynamic size, and its runtime size is represented
  // by a scalar instruction `size`.
  void SetDynamicSize(HloInstruction* inst, const ShapeIndex& index, int64 dim,
                      HloInstruction* size) {
    dynamic_mapping_.try_emplace(DynamicDimension{inst, index, dim}, size);
    auto iter = per_hlo_dynamic_dimensions_.try_emplace(inst);
    iter.first->second.emplace(DynamicDimension{inst, index, dim});
  }

  // AnalyzeDynamicDimensions starts the analysis of the dynamic dimensions in
  // module_.
  Status AnalyzeDynamicDimensions();

  // HloModule being analyzed.
  HloModule* module_;

  // dynamic_mapping_ holds the result of the analysis. It maps a dynamic
  // dimension to a scalar HloInstruction that represents the real dynamic size
  // of the dynamic dimension.
  using DynamicMapping = absl::flat_hash_map<DynamicDimension, HloInstruction*>;
  DynamicMapping dynamic_mapping_;

  using PerHloDynamicDimensions =
      absl::flat_hash_map<HloInstruction*,
                          absl::flat_hash_set<DynamicDimension>>;
  PerHloDynamicDimensions per_hlo_dynamic_dimensions_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DYNAMIC_DIMENSION_INFERENCE_H_
