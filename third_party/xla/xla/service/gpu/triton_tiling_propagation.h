/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRITON_TILING_PROPAGATION_H_
#define XLA_SERVICE_GPU_TRITON_TILING_PROPAGATION_H_

// This file contains the logic of the Triton Tiling Propagation in a functional
// paradigm. Stateful operations belong in triton_fusion_analysis.

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Illustration explaining why slice_start for concatenations is negative:

// Slice
// =====
//            input
// [--------------------------]
// .      .        .
// . offset        .
// |------> output .
//        [--------]
//
// output[x] = input[x + offset]

// Concatenation
// =============
//
//          input_n
// [......][--------][........]
//         .        .
//  offset .        .
// <-------|        .
// .       .        .
// .       . output .
// [--------------------------]
//
// output[x] = input_n[x - offset]

class TensorIterationSpec {
 public:
  // Description of basic iteration: `count` elements separated by `stride`
  // with initial offset of `slice_start` and only `sliced_count` elements used.
  struct IterationSpecFragment {
    int64_t stride;
    int64_t count;
    int64_t slice_start;
    int64_t sliced_count;
    // Logical subfragments:
    // These are the sizes of the HLO dimensions which make up this basic
    // iteration.
    std::vector<int64_t> subfragments;

    bool is_sliced() const { return count != sliced_count; }

    auto ToTuple() const {
      return std::make_tuple(stride, count, slice_start, sliced_count,
                             subfragments);
    }

    bool operator==(const IterationSpecFragment& other) const {
      return ToTuple() == other.ToTuple();
    }
    template <typename H>
    friend H AbslHashValue(H h, const IterationSpecFragment& fragment) {
      return H::combine(std::move(h), fragment.ToTuple());
    }

    bool IsPhysicallyEquivalent(const IterationSpecFragment& other) const {
      // Subfragments don't change the physical layout.
      return stride == other.stride && count == other.count &&
             slice_start == other.slice_start &&
             sliced_count == other.sliced_count;
    }

    std::string ToString() const;
  };
  // Description of complex iteration over a sequence of several strides.
  // Describes a logically contiguous dimension of a tensor physically
  // separated into multiple fragments by other dimensions.
  using DimIterationSpec = std::vector<IterationSpecFragment>;

  const DimIterationSpec& operator[](const int dimension) const {
    return dim_iteration_specs_.at(dimension);
  }
  DimIterationSpec& operator[](const int dimension) {
    return dim_iteration_specs_[dimension];
  }
  // Returns nullptr if not found.
  const DimIterationSpec* Find(int dimension) const;

  std::vector<int> GetDimensions() const;

  void RemoveEmptyDimensions() {
    absl::erase_if(dim_iteration_specs_,
                   [](const auto& it) { return it.second.empty(); });
  }

  bool operator==(const TensorIterationSpec& other) const {
    return dim_iteration_specs_ == other.dim_iteration_specs_;
  }

  template <typename H>
  friend H AbslHashValue(H h, const TensorIterationSpec& spec) {
    return H::combine(std::move(h), spec.dim_iteration_specs_);
  }

  // Compares physical layouts of tensors ignoring subfragments of dimensions.
  // Checking with this, instead of "==" allows a few more edge cases to be
  // fused.
  bool IsPhysicallyEquivalent(const TensorIterationSpec& other) const;

  std::string ToString() const;

 private:
  // Maps dimensions to DimIterationSpecs.
  absl::flat_hash_map<int, DimIterationSpec> dim_iteration_specs_;
};

// The details of the Triton fusion / tiling propagation are in a separate
// namespace to avoid littering the xla::gpu namespace.
namespace triton_fusion {

// Handles numbers of dimensions of an HLO instruction
// projected onto another one.
// Used to calculate cumulative index transformations done by non-elementwise
// instructions between source and target.
class DimensionOrder {
 public:
  // Softmax fusions have a fixed tiling scheme. These numbers are chosen to
  // reflect that reductions in softmax fusions currently happen on the minor-
  // most dimension (dimensions_minor(0)) and the rest (1+) is treated as a
  // single non-tiled batch dimension. The numbers have to match those the
  // emitter uses in the queries to the analysis.
  static constexpr int kSoftmaxReductionDimension = 0;
  static constexpr int kSoftmaxBatchDimension = 1;

  static DimensionOrder FromDotOperandOrOutput(
      const HloInstruction& hlo, int split_k_dimension_index = -1);

  static DimensionOrder FromSoftmaxRoot(const HloInstruction& hlo);

  // Description of a continuous fragment of one dimension of a tensor.
  class Fragment {
   public:
    explicit Fragment(int dst_dim_number, int64_t count)
        : dst_dim_number_(dst_dim_number),
          count_(count),
          slice_start_(0),
          sliced_count_(count) {}

    std::string ToString() const;

    // Label carrying the dimension number of an defining operation.
    int dst_dim_number() const { return dst_dim_number_; }
    // Total number of elements in the fragment ignoring slicing.
    int64_t full_count() const { return count_; }
    // First used element.
    int64_t slice_start() const { return slice_start_; }
    // Number of used elements.
    int64_t sliced_count() const { return sliced_count_; }
    bool is_sliced() const { return count_ != sliced_count_; }
    void set_slice(int64_t start, int64_t count) {
      slice_start_ = start;
      sliced_count_ = count;
    }
    void set_count(int64_t count) { count_ = count; }

   private:
    const int dst_dim_number_;
    int64_t count_;
    int64_t slice_start_;
    int64_t sliced_count_;
  };
  using Fragments = std::vector<Fragment>;
  using FragmentOrders = absl::flat_hash_map<int, std::vector<int>>;

  const Fragments& TensorFragmentsOrder() const {
    return tensor_fragments_order_;
  }
  Fragments& TensorFragmentsOrder() { return tensor_fragments_order_; }

  const FragmentOrders& DimFragmentsOrders() const {
    return dim_fragments_orders_;
  }
  FragmentOrders& DimFragmentsOrders() { return dim_fragments_orders_; }

  std::string ToString() const;

  TensorIterationSpec ToTensorIterationSpec() const;

  // Tells that two dimension orders describe the same tensor physical layout.
  bool IsPhysicallyEquivalent(const DimensionOrder& other) const {
    return ToTensorIterationSpec().IsPhysicallyEquivalent(
        other.ToTensorIterationSpec());
  }

 private:
  // Sequence of all fragments of dimensions of tensor's shape
  // in layout minor-to-major (physical) order.
  Fragments tensor_fragments_order_;
  // Iteration orders of fragments of each dimension of the defining operation
  // (fragments can be physically unordered and disconnected within
  // the shape due to reshapes and transposes).
  FragmentOrders dim_fragments_orders_;
};

// This represents an invalid dimension index.
inline constexpr int kNoDimensionIndex = -1;
struct DotProperties {
  const int noncontracting_dimension;
  // Index of dot dimension that can be split.
  // Currently typically LHS non-contracting one.
  const int splittable_dimension_index;
};
struct SoftmaxProperties {
  const int softmax_reduction_dimension;
  const int softmax_batch_dimension;
};
// HeroProperties depend only on the hero op and they don't change as we
// change the fusion.
using HeroProperties = std::variant<DotProperties, SoftmaxProperties>;

// A special value for splittable_dimension_major_part_size.
inline constexpr int kNoSplitRequirement = 1;
struct DotRequirements {
  explicit DotRequirements(int64_t splittable_dimension_major_part_size)
      : splittable_dimension_major_part_size(
            splittable_dimension_major_part_size) {
    CHECK_GE(splittable_dimension_major_part_size, 1);
  }
  // If not kNoSplitRequirement, then the major part size of the splittable
  // dimension must be the given value.
  int64_t splittable_dimension_major_part_size;
};
struct SoftmaxRequirements {};
// Requirements can change depending on what we fuse.
using Requirements = std::variant<DotRequirements, SoftmaxRequirements>;
using RequirementsOrError = std::variant<Requirements, FusionDecision>;

RequirementsOrError CombineRequirements(Requirements a,
                                        RequirementsOrError b_or_error);

enum class TransformDirection { kInputToOutput, kOutputToInput };
using DimOrderMap = absl::flat_hash_map<const HloInstruction*, DimensionOrder>;
using DimOrderMapOrError = std::variant<DimOrderMap, FusionDecision>;

// The dimension orders and requirements resulting from propagating the
// dimension orders through an HLO.
struct DimOrdersAndReqs {
  DimOrderMap dim_orders;
  Requirements requirements;
};
using DimOrdersAndReqsOrError = std::variant<DimOrdersAndReqs, FusionDecision>;

// If fusing the instruction is possible then it propagates
// the `src_dim_order` (describing one side of `hlo`) to the other side and
// returns those dim orders and the requirements that they impose on the
// fusion.
DimOrdersAndReqsOrError GetPropagatedDimOrdersAndRequirements(
    const HloInstruction& hlo, const DimensionOrder& src_dim_order,
    TransformDirection direction, const HeroProperties& properties);
// If fusing the instruction is possible *and profitable* then it propagates
// the `src_dim_order` (describing one side of `hlo`) to the other side and
// returns those dim orders and the requirements that they impose on the
// fusion.
//
// `src_operand_index` must be set iff `transform_direction` is
// kInputToOutput.
DimOrdersAndReqsOrError
GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
    const HloInstruction& hlo, TransformDirection transform_direction,
    const std::optional<int>& src_operand_index,
    const DimensionOrder& src_dim_order,
    const se::GpuComputeCapability& gpu_version,
    const HeroProperties& properties);

}  // namespace triton_fusion
}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRITON_TILING_PROPAGATION_H_
