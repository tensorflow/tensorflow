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

  // Forward dynamic dimension size at `dim` and its constraint from `inst` to
  // `new_inst`.
  Status ForwardDynamicSize(HloInstruction* inst, HloInstruction* new_inst,
                            const ShapeIndex& index);

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

  // DimensionConstraint is attached to each dynamic dimension and describe the
  // constraint of each dimension. This is used to disambiguate the index of
  // dynamic dimension for reshapes that "splits" a dimension into two.
  //
  // As an example, consider the following reshapes:
  // [<=3, 3]   <- Assume first dimension is dynamic.
  //   |
  // Reshape.1
  //   |
  //  [<=9]     <- Dimension 9 is dynamic
  //   |
  // Reshape.2
  //   |
  // [3, 3]   <- Ambiguous dimension after splitting 9 into [3, 3]
  //
  // There is no way to know which dimension is dynamic by looking at the second
  // reshape locally.
  //
  // However, if we look at the dynamic dimension 9, since it comes from
  // collapsing a major dynamic dimension of 3 (the dynamic size can be 0, 1, 2,
  // 3, denoted as i in the diagram below) and a minor static dimension of 3, we
  // know it has certain constraints that the reshape can only be one of the 4
  // forms:
  //
  // o: Padded Data
  // x: Effective Data
  //
  //     [<=3, 3] to [9]
  //
  //     +---+            +---+            +---+            +---+
  //     |ooo|            |ooo|            |ooo|            |xxx|
  //     |ooo|            |ooo|            |xxx|            |xxx|
  //     |ooo|            |xxx|            |xxx|            |xxx|
  //     +---+            +---+            +---+            +---+
  //
  //    Reshape          Reshape          Reshape          Reshape
  //
  // +-----------+    +-----------+    +-----------+    +-----------+
  // |ooo|ooo|ooo| or |xxx|ooo|ooo| or |xxx|xxx|ooo| or |xxx|xxx|xxx|  stride=1
  // +-----------+    +-----------+    +-----------+    +-----------+
  //     i = 0             i = 1            i = 2            i = 3
  //
  // On the other hand, if the minor dimension 3 is dynamic and major dimension
  // is static, we will have the following form:
  //
  //     [3, <=3] to [9]
  //
  //     +---+            +---+            +---+            +---+
  //     |ooo|            |xoo|            |xxo|            |xxx|
  //     |ooo|            |xoo|            |xxo|            |xxx|
  //     |ooo|            |xoo|            |xxo|            |xxx|
  //     +---+            +---+            +---+            +---+
  //
  //    Reshape          Reshape          Reshape          Reshape
  //
  // +-----------+    +-----------+    +-----------+    +-----------+
  // |ooo|ooo|ooo| or |xoo|xoo|xoo| or |xxo|xxo|xxo| or |xxo|xxo|xxo|  stride=3
  // +-----------+    +-----------+    +-----------+    +-----------+
  //     i = 0             i = 1            i = 2            i = 3
  //
  // By encoding constraint as a stride of elements we can recover this
  // information later when we reshape from [9] to [3, 3]. We know which form
  // ([3, i] or [i,3]) we should reshape the [9] into.
  //
  //
  struct DimensionConstraint {
    explicit DimensionConstraint(int64 s, int64 m)
        : stride(s), multiple_of(m) {}
    DimensionConstraint() : stride(1), multiple_of(1) {}
    // Stride represents the distance of a newly placed element and the previous
    // placed element on this dynamic dimension.
    int64 stride;

    // multiple_of represents the constraints that
    //
    // `dynamic_size` % `multiple_of` == 0
    int64 multiple_of;
  };

  using ConstraintMapping =
      absl::flat_hash_map<DynamicDimension, DimensionConstraint>;

  ConstraintMapping constraint_mapping_;

  // Update the dynamic mapping so that we know dimension `dim` of instruction
  // `inst` at `index` has a dynamic size, and its runtime size is represented
  // by a scalar instruction `size`.
  void SetDynamicSize(HloInstruction* inst, const ShapeIndex& index, int64 dim,
                      HloInstruction* size, DimensionConstraint constraint) {
    VLOG(1) << "Set dimension inst " << inst->ToString() << " index "
            << index.ToString() << "@" << dim << " to " << size->ToShortString()
            << " constraint: " << constraint.multiple_of;
    Shape subshape = ShapeUtil::GetSubshape(inst->shape(), index);
    CHECK(!subshape.IsTuple())
        << "Can't set a tuple shape to dynamic dimension";
    CHECK(dim < subshape.rank() && dim >= 0)
        << "Asked to set invalid dynamic dimension. Shape: "
        << subshape.ToString() << ", Dimension: " << dim;
    DynamicDimension dynamic_dimension{inst, index, dim};
    // Updating a dynamic dimension twice overwrites the previous one.
    dynamic_mapping_[dynamic_dimension] = size;
    if (constraint_mapping_.count(dynamic_dimension) != 0) {
      CHECK_EQ(constraint_mapping_[dynamic_dimension].stride,
               constraint.stride);
    }
    constraint_mapping_[dynamic_dimension] = constraint;
    auto iter = per_hlo_dynamic_dimensions_.try_emplace(inst);
    iter.first->second.emplace(dynamic_dimension);
  }

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
  using DynamicMapping = absl::flat_hash_map<DynamicDimension, HloInstruction*>;
  DynamicMapping dynamic_mapping_;

  // A convenient mapping from an hlo to the set of dynamic dimensions that it
  // holds.
  using PerHloDynamicDimensions =
      absl::flat_hash_map<HloInstruction*,
                          absl::flat_hash_set<DynamicDimension>>;
  PerHloDynamicDimensions per_hlo_dynamic_dimensions_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DYNAMIC_DIMENSION_INFERENCE_H_
