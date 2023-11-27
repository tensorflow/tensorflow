/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/gemm_rewriter_triton.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <list>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/permutation_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_padding_requirements.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/tensor_float_32_utils.h"

namespace xla {
namespace gpu {

bool TensorIterationSpec::operator==(const TensorIterationSpec& other) const {
  VLOG(9) << this->ToString();
  VLOG(9) << other.ToString();
  auto it_this = dim_iteration_specs_.cbegin();
  while (it_this != dim_iteration_specs_.cend()) {
    auto it_other = other.dim_iteration_specs_.find(it_this->first);
    if (it_other == other.dim_iteration_specs_.cend()) {
      return false;
    }
    if (it_this->second.size() != it_other->second.size()) {
      return false;
    }
    for (int fragment = 0; fragment < it_this->second.size(); ++fragment) {
      if (it_this->second[fragment] != it_other->second[fragment]) {
        return false;
      }
    }
    ++it_this;
  }
  return true;
}

bool IsDistributiveOverAddition(const HloInstruction& hlo) {
  // The list is most likely incomplete.
  // For example division can be added too but only for operand #0.
  if (hlo.opcode() == HloOpcode::kMultiply ||
      hlo.opcode() == HloOpcode::kNegate ||
      hlo.opcode() == HloOpcode::kBitcast ||
      hlo.opcode() == HloOpcode::kReshape || hlo.opcode() == HloOpcode::kCopy ||
      hlo.opcode() == HloOpcode::kTranspose ||
      hlo.opcode() == HloOpcode::kConvert ||
      hlo.opcode() == HloOpcode::kBroadcast ||
      hlo.opcode() == HloOpcode::kSlice) {
    return true;
  }
  return false;
}

namespace {

FusionDecision IsConversionWorthFusing(const HloInstruction& input,
                                       se::GpuComputeCapability gpu_version) {
  // TODO(b/266862494): Can pick up almost any
  // convert, but if it's reducing the data volume it should rather be fused
  // to the output of the producer kernel. However not all operations support
  // output fusion - then it should be fused here anyway!
  if (ShapeUtil::ByteSizeOf(input.operand(0)->shape()) >
      ShapeUtil::ByteSizeOf(input.shape())) {
    return "Narrowing conversion.";
  }
  return FusionDecision{};
}

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
      const HloInstruction& hlo, const int split_k_dimension_index = -1) {
    DimensionOrder dim_order;
    dim_order.tensor_fragments_order_.reserve(hlo.shape().rank());
    for (const int i : hlo.shape().layout().minor_to_major()) {
      int target_dim_number = i;
      if (i == split_k_dimension_index) {
        CHECK(!dim_order.tensor_fragments_order_.empty())
            << "The split-K batch dimension has be preceded by the contracting "
               "dimension it originates from by construction.";
        target_dim_number =
            dim_order.tensor_fragments_order_.back().dst_dim_number();
      }
      dim_order.dim_fragments_orders_[target_dim_number].push_back(
          dim_order.tensor_fragments_order_.size());
      dim_order.tensor_fragments_order_.push_back(
          Fragment{target_dim_number, hlo.shape().dimensions(i)});
    }
    return dim_order;
  }

  static DimensionOrder FromSoftmaxRoot(const HloInstruction& hlo) {
    DimensionOrder dim_order;
    dim_order.tensor_fragments_order_.reserve(hlo.shape().rank());
    dim_order.dim_fragments_orders_[kSoftmaxReductionDimension].push_back(
        dim_order.tensor_fragments_order_.size());
    dim_order.tensor_fragments_order_.push_back(
        Fragment{kSoftmaxReductionDimension, hlo.shape().dimensions_minor(0)});
    for (int i = 1; i < hlo.shape().rank(); ++i) {
      dim_order.dim_fragments_orders_[kSoftmaxBatchDimension].push_back(
          dim_order.tensor_fragments_order_.size());
      dim_order.tensor_fragments_order_.push_back(
          Fragment{kSoftmaxBatchDimension, hlo.shape().dimensions_minor(i)});
    }
    return dim_order;
  }

  // Description of a continuous fragment of one dimension of a tensor.
  class Fragment {
   public:
    explicit Fragment(int dst_dim_number, int64_t size)
        : dst_dim_number_(dst_dim_number),
          size_(size),
          slice_start_(0),
          slice_limit_(size) {}

    std::string ToString() const {
      return absl::StrCat(dst_dim_number_, ":", size_, ":", slice_start_, "-",
                          slice_limit_);
    }
    // Label carrying the dimension number of an defining operation.
    int dst_dim_number() const { return dst_dim_number_; }
    // Total number of elements in the fragment ignoring slicing.
    int64_t full_size() const { return size_; }
    // First used element.
    int64_t slice_start() const { return slice_start_; }
    // Last used element.
    int64_t slice_limit() const { return slice_limit_; }
    int64_t sliced_size() const { return slice_limit_ - slice_start_; }
    bool is_sliced() const { return full_size() != sliced_size(); }
    void set_slice(int64_t start, int64_t limit) {
      slice_start_ = start;
      slice_limit_ = limit;
    }
    void set_size(int64_t size) { size_ = size; }

   private:
    const int dst_dim_number_;
    int64_t size_;
    int64_t slice_start_;
    int64_t slice_limit_;
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

  // Tells that two dimension orders describe the same tensor physical layout.
  bool IsPhysicallyEquivalent(const DimensionOrder& other) const;

  std::string ToString() const {
    std::string ret = absl::StrJoin(tensor_fragments_order_, " - ",
                                    [](std::string* out, const Fragment& f) {
                                      absl::StrAppend(out, f.ToString(), " ");
                                    });
    absl::StrAppend(&ret, "|");
    for (const auto& [dim, fragments] : dim_fragments_orders_) {
      absl::StrAppend(&ret, dim, ":", absl::StrJoin(fragments, ","), " ");
    }
    return ret;
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

using DimIterationSpec = TensorIterationSpec::DimIterationSpec;
using Fragment = DimensionOrder::Fragment;
using Fragments = DimensionOrder::Fragments;
using FragmentOrders = DimensionOrder::FragmentOrders;
using DimOrderMap = absl::flat_hash_map<const HloInstruction*, DimensionOrder>;
using DimOrderMapOrError = std::variant<DimOrderMap, FusionDecision>;

// This represents an invalid dimension index.
constexpr int kNoDimensionIndex = -1;
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
constexpr int kNoSplitRequirement = 1;
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

// The dimension orders and requirements resulting from propagating the
// dimension orders through an HLO.
struct DimOrdersAndReqs {
  DimOrderMap dim_orders;
  Requirements requirements;
};
using DimOrdersAndReqsOrError = std::variant<DimOrdersAndReqs, FusionDecision>;

using Int64OrError = std::variant<int64_t, FusionDecision>;
Int64OrError CombineSplitDimMajorPartSizeReqs(int64_t a, int64_t b) {
  if (a == b || b == kNoSplitRequirement) {
    return a;
  }
  if (a == kNoSplitRequirement) {
    return b;
  }
  return FusionDecision("Conflicting splits of splittable dimension");
}

RequirementsOrError CombineDotRequirements(DotRequirements a,
                                           DotRequirements b) {
  Int64OrError combined_size_req =
      CombineSplitDimMajorPartSizeReqs(a.splittable_dimension_major_part_size,
                                       b.splittable_dimension_major_part_size);
  if (std::holds_alternative<FusionDecision>(combined_size_req)) {
    return std::get<FusionDecision>(combined_size_req);
  }
  return DotRequirements(std::get<int64_t>(combined_size_req));
}

RequirementsOrError CombineSoftmaxRequirements(SoftmaxRequirements a,
                                               SoftmaxRequirements b) {
  // SoftmaxRequirements is an empty class for now.
  return a;
}

RequirementsOrError CombineRequirements(Requirements a,
                                        RequirementsOrError b_or_error) {
  if (std::holds_alternative<FusionDecision>(b_or_error)) {
    return b_or_error;
  }
  const Requirements& b = std::get<Requirements>(b_or_error);
  if (std::holds_alternative<DotRequirements>(b)) {
    return CombineDotRequirements(std::get<DotRequirements>(a),
                                  std::get<DotRequirements>(b));
  }
  return CombineSoftmaxRequirements(std::get<SoftmaxRequirements>(a),
                                    std::get<SoftmaxRequirements>(b));
}

TensorIterationSpec DimensionOrderToTensorIterationSpec(
    const DimensionOrder& order) {
  const Fragments& dim_fragments = order.TensorFragmentsOrder();
  TensorIterationSpec tensor_spec;
  int64_t accumulated_stride = 1;
  int last_dim = -1;
  auto remove_last_fragment_if_degenerate = [&tensor_spec](const int dim_idx) {
    if (dim_idx >= 0 && !tensor_spec[dim_idx].empty() &&
        tensor_spec[dim_idx].back().count == 1) {
      tensor_spec[dim_idx].pop_back();
    }
  };
  for (int dim_order_index = 0; dim_order_index < dim_fragments.size();
       ++dim_order_index) {
    const DimensionOrder::Fragment& fragment = dim_fragments[dim_order_index];
    VLOG(6) << fragment.ToString();

    DimIterationSpec& dim_spec = tensor_spec[fragment.dst_dim_number()];
    if (last_dim == fragment.dst_dim_number()) {
      // Remove previous 1-sized subfragment if present.
      if (!dim_spec.empty() && !dim_spec.back().subfragments.empty() &&
          dim_spec.back().subfragments.back() == 1) {
        dim_spec.back().subfragments.pop_back();
      }
      // Contiguous dimension, split only logically. Merge it back.
      if (fragment.full_size() > 1) {
        CHECK(!dim_spec.empty());
        CHECK(!dim_spec.back().is_sliced())
            << "Only the major-most fragment can have an offset.";
        dim_spec.back().slice_start =
            fragment.slice_start() * dim_spec.back().count;
        dim_spec.back().slice_limit =
            fragment.slice_limit() * dim_spec.back().count;
        dim_spec.back().count *= fragment.full_size();
        dim_spec.back().subfragments.push_back(fragment.sliced_size());
      }
    } else {
      remove_last_fragment_if_degenerate(last_dim);
      // Add part of the dimension.
      dim_spec.push_back(
          TensorIterationSpec::IterationSpecFragment{accumulated_stride,
                                                     fragment.full_size(),
                                                     fragment.slice_start(),
                                                     fragment.slice_limit(),
                                                     {fragment.sliced_size()}});
    }

    accumulated_stride *= fragment.full_size();
    last_dim = fragment.dst_dim_number();
  }
  remove_last_fragment_if_degenerate(last_dim);
  tensor_spec.RemoveEmptyDimensions();
  return tensor_spec;
}

bool DimensionOrder::IsPhysicallyEquivalent(const DimensionOrder& other) const {
  return DimensionOrderToTensorIterationSpec(*this) ==
         DimensionOrderToTensorIterationSpec(other);
}

// Logical index of a dimension in `shape` labeled with `label` in the
// `dim_order` describing the shape.
std::optional<int> LogicalIndexOfLabeledDimension(
    const Shape& shape, const DimensionOrder& dim_order, const int label) {
  auto fragment_it = dim_order.TensorFragmentsOrder().cbegin();
  for (int dim : shape.layout().minor_to_major()) {
    const int64_t dim_size = shape.dimensions()[dim];
    int64_t fragments_size = 1;
    while (fragments_size < dim_size) {
      fragments_size *= fragment_it->full_size();
      if (fragment_it->dst_dim_number() == label) {
        return dim;
      }
      ++fragment_it;
    }
  }
  return std::nullopt;
}

enum class TransformDirection { kInputToOutput, kOutputToInput };

using OldToNewHloMap =
    absl::flat_hash_map<const HloInstruction*, HloInstruction*>;

class FusionContext {
  FusionContext(HeroProperties properties, Requirements requirements)
      : properties_(properties), requirements_(requirements) {}

 public:
  // Create fusion context from a dot operand according to
  // the currently supported configurations.
  static FusionContext FromDotOperand(const HloInstruction& dot,
                                      int operand_number, int split_k = 1);

  // Create fusion context from dot's output.
  static FusionContext FromDotOutput(
      const HloInstruction& dot, int split_k,
      int64_t splittable_dimension_major_part_size);

  static FusionContext FromSoftmaxRoot(const HloInstruction&);

  // If possible, propagates `src_dim_order` (describing one side of `hlo`) to
  // the other side and returns those dim orders.
  static DimOrderMapOrError GetPropagatedDimOrders(
      const HloInstruction& hlo, TransformDirection direction,
      const DimensionOrder& src_dim_order, const HeroProperties& properties);

  // If the dimension order is supported by the triton emitters, this returns
  // which requirements does this order impose on the fusion.
  //
  // All subdimensions within a dimension have to be ordered.
  static RequirementsOrError GetRequirementsIfSupportedOrder(
      const DimensionOrder& order, const HeroProperties& properties);
  // Apply GetRequirementsIfSupportedOrder() to all known
  // dimension orders around `hlo` and combine the result.
  static RequirementsOrError GetRequirementsIfSupportedOrders(
      const HloInstruction& hlo, const DimOrderMap& dim_orders,
      const HeroProperties& properties);
  // If fusing the instruction is possible then it propagates
  // the `src_dim_order` (describing one side of `hlo`) to the other side and
  // returns those dim orders and the requirements that they impose on the
  // fusion.
  static DimOrdersAndReqsOrError GetPropagatedDimOrdersAndRequirements(
      const HloInstruction& hlo, const DimensionOrder& src_dim_order,
      TransformDirection direction, const HeroProperties& properties);
  // If fusing the instruction is possible *and profitable* then it propagates
  // the `src_dim_order` (describing one side of `hlo`) to the other side and
  // returns those dim orders and the requirements that they impose on the
  // fusion.
  //
  // `src_operand_index` must be set iff `transform_direction` is
  // kInputToOutput.
  static DimOrdersAndReqsOrError
  GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
      const HloInstruction& hlo, TransformDirection transform_direction,
      const std::optional<int>& src_operand_index,
      const DimensionOrder& src_dim_order,
      const se::GpuComputeCapability& gpu_version,
      const HeroProperties& properties);

  // Add dimension orders from `update` to `dim_orders_` and update
  // `requirements_` if all of them are compatible.
  bool CombineDimOrdersAndReqs(const DimOrdersAndReqs& update);
  // Fuse an instruction with all its fusible inputs.
  // If an input is not fusible stop there and make a parameter of the new
  // fusion, otherwise put it onto stack and check its own inputs first.
  void TryToFuseWithInputsRecursively(
      HloInstruction& root, se::GpuComputeCapability gpu_version,
      OldToNewHloMap& old_to_new_map,
      std::vector<HloInstruction*>& fusion_inputs,
      HloComputation::Builder& builder);
  // Propagate dimension orders in consumer->producer direction starting at
  // `origin` with output `origin_dim_order` till parameters of the computation.
  // Store the found parameters and their iteration specs.
  Status PropagateDimensionOrdersToParameters(
      const HloInstruction& origin, ConstHloInstructionSet& parameters,
      ConstHloInstructionMap<TensorIterationSpec>& iter_specs);

  int64_t splittable_dimension_major_part_size() const {
    CHECK(std::holds_alternative<DotRequirements>(requirements_));
    return std::get<DotRequirements>(requirements_)
        .splittable_dimension_major_part_size;
  }
  const HeroProperties& hero_properties() const { return properties_; }
  const DimOrderMap& dim_orders() const { return dim_orders_; }

 private:
  static DimOrderMap GetPropagatedDimOrdersForElementwise(
      const HloInstruction& hlo, TransformDirection direction,
      const DimensionOrder& src_dim_order);
  static DimOrderMapOrError GetPropagatedDimOrdersForBitcast(
      const HloInstruction& hlo, TransformDirection direction,
      const DimensionOrder& src_dim_order, const HeroProperties& properties);
  static DimOrderMapOrError GetPropagatedDimOrdersForDimAlteringOp(
      const HloInstruction& hlo, TransformDirection direction,
      const DimensionOrder& src_dim_order, const HeroProperties& properties);

  const HeroProperties properties_;
  Requirements requirements_;
  DimOrderMap dim_orders_;
};

FusionContext FusionContext::FromDotOperand(const HloInstruction& dot,
                                            const int operand_number,
                                            const int split_k) {
  // There can be either none or one split-K batch dimension.
  const int num_split_k_batch_dims = split_k > 1;
  int split_k_dimension_index = kNoDimensionIndex;
  if (split_k > 1) {
    split_k_dimension_index =
        ContractingDimensionIndex(dot, operand_number) - 1;
  }
  int splittable_dimension_index = kNoDimensionIndex;
  // LHS non-contracting dimension can be split if non-splitK batch is absent.
  if (operand_number == 0 &&
      dot.dot_dimension_numbers().lhs_batch_dimensions_size() -
              num_split_k_batch_dims ==
          0) {
    splittable_dimension_index =
        NonContractingDimensionIndex(dot, operand_number);
  }
  FusionContext context(
      DotProperties{
          static_cast<int>(NonContractingDimensionIndex(dot, operand_number)),
          splittable_dimension_index},
      DotRequirements(kNoSplitRequirement));
  context.dim_orders_[dot.operand(operand_number)] =
      DimensionOrder::FromDotOperandOrOutput(*dot.operand(operand_number),
                                             split_k_dimension_index);
  return context;
}

FusionContext FusionContext::FromDotOutput(
    const HloInstruction& dot, const int split_k,
    const int64_t splittable_dimension_major_part_size) {
  // Allow non-contracting dimension originating from LHS to split if
  // this dimension is split at the output at the same ratio as
  // at the input.
  int splittable_dimension_index = kNoDimensionIndex;
  if (splittable_dimension_major_part_size > 1) {
    // Split-K dimension is the first one in the output if present;
    // LHS non-contracting follows (batch is absent in this case).
    splittable_dimension_index = (split_k > 1) ? 1 : 0;
  }
  FusionContext context(DotProperties{/*noncontracting_dimension=*/-1,
                                      splittable_dimension_index},
                        DotRequirements(splittable_dimension_major_part_size));
  context.dim_orders_[&dot] = DimensionOrder::FromDotOperandOrOutput(dot);
  return context;
}

FusionContext FusionContext::FromSoftmaxRoot(const HloInstruction& root) {
  FusionContext context(
      SoftmaxProperties{DimensionOrder::kSoftmaxReductionDimension,
                        DimensionOrder::kSoftmaxBatchDimension},
      SoftmaxRequirements{});
  context.dim_orders_[&root] = DimensionOrder::FromSoftmaxRoot(root);
  return context;
}

/*static*/ RequirementsOrError FusionContext::GetRequirementsIfSupportedOrder(
    const DimensionOrder& order, const HeroProperties& properties) {
  VLOG(8) << order.ToString();
  int64_t split_dim_major_part = kNoSplitRequirement;
  const Fragments& tensor_dim_fragments = order.TensorFragmentsOrder();
  for (const auto& [dim_index, dim_fragments] : order.DimFragmentsOrders()) {
    CHECK(!dim_fragments.empty());
    for (int i = 0; i < dim_fragments.size() - 1; ++i) {
      if (tensor_dim_fragments[dim_fragments[i]].is_sliced()) {
        return "Sliced non-major-most fragment.";
      }
    }
    int group_counter = 0;
    int last_seen_group_last_fragment_index = -1;
    auto fragment_it = dim_fragments.cbegin();
    while (true) {
      if (fragment_it == dim_fragments.cend()) {
        break;
      }
      int64_t grouped_size = tensor_dim_fragments[*fragment_it].full_size();
      // Gather contiguous fragments: they have consecutive indices.
      while ((fragment_it + 1) != dim_fragments.cend() &&
             *(fragment_it + 1) == *fragment_it + 1) {
        ++fragment_it;
        grouped_size *= tensor_dim_fragments[*fragment_it].full_size();
      }
      // Ignore 1-sized groups of fragments.
      if (grouped_size == 1) {
        ++fragment_it;
        continue;
      }

      if (last_seen_group_last_fragment_index > *fragment_it) {
        return "Transpose within a dimension.";
      }

      ++group_counter;
      if (group_counter > 1) {
        if (!std::holds_alternative<DotProperties>(properties)) {
          return "Splitting a dimension is not supported for Softmax.";
        }
        // Only the dimension indicated by `splittable_dimension_index` (if any)
        // can be split physically once by other dimensions. Other ones can be
        // only split logically.
        const int splittable_dimension_index =
            std::get<DotProperties>(properties).splittable_dimension_index;
        if (dim_index == splittable_dimension_index) {
          if (group_counter == 2) {
            if (split_dim_major_part != kNoSplitRequirement &&
                split_dim_major_part != grouped_size) {
              return "Conflicting splits of splittable dimension";
            }
            split_dim_major_part = grouped_size;
          } else if (group_counter > 2) {
            return "2nd split of a splittable dimension.";
          }
        } else {
          return "Unsupported split of a dimension.";
        }
      }

      last_seen_group_last_fragment_index = *fragment_it;
      ++fragment_it;
    }
  }

  if (std::holds_alternative<DotProperties>(properties)) {
    return DotRequirements(split_dim_major_part);
  }
  return SoftmaxRequirements{};
}

/*static*/ RequirementsOrError FusionContext::GetRequirementsIfSupportedOrders(
    const HloInstruction& hlo, const DimOrderMap& dim_orders,
    const HeroProperties& properties) {
  const Requirements empty_requirements =
      std::holds_alternative<DotProperties>(properties)
          ? Requirements(DotRequirements(kNoSplitRequirement))
          : Requirements(SoftmaxRequirements{});
  auto get_requirements =
      [&](const HloInstruction& instr) -> RequirementsOrError {
    if (auto it = dim_orders.find(&instr); it != dim_orders.end()) {
      return GetRequirementsIfSupportedOrder(it->second, properties);
    }
    return empty_requirements;
  };

  Requirements requirements = empty_requirements;
  for (const HloInstruction* operand : hlo.operands()) {
    RequirementsOrError requirements_or_error =
        CombineRequirements(requirements, get_requirements(*operand));
    if (std::holds_alternative<FusionDecision>(requirements_or_error)) {
      return requirements_or_error;
    }
    requirements = std::get<Requirements>(requirements_or_error);
  }

  return CombineRequirements(requirements, get_requirements(hlo));
}

/*static*/ DimOrderMap FusionContext::GetPropagatedDimOrdersForElementwise(
    const HloInstruction& hlo, TransformDirection direction,
    const DimensionOrder& src_dim_order) {
  if (direction == TransformDirection::kOutputToInput) {
    DimOrderMap map;
    for (const HloInstruction* operand : hlo.operands()) {
      map.insert({operand, src_dim_order});
    }
    return map;
  }

  DimOrderMap map;
  map.insert({&hlo, src_dim_order});
  // TODO(tdanyluk): For now, the "input to output" direction of this function
  // also returns the dim orders for the operands, not just the output. This is
  // needed to propagate the dim order of one input to the other(s) when fusing
  // elementwise ops to the output. Perhaps we can separate the "input to
  // output" and "output to input" directions of that in a later CL.
  for (const HloInstruction* operand : hlo.operands()) {
    map.insert({operand, src_dim_order});
  }
  return map;
}

const HloInstruction& GetSourceHlo(const HloInstruction& hlo,
                                   TransformDirection direction) {
  CHECK_GE(hlo.operand_count(), 1);

  if (direction == TransformDirection::kOutputToInput) {
    return hlo;
  }
  return *hlo.operand(0);
}

using ConstInstructionVector = absl::InlinedVector<const HloInstruction*, 2>;
ConstInstructionVector GetDestHlos(const HloInstruction& hlo,
                                   TransformDirection direction) {
  if (direction == TransformDirection::kInputToOutput) {
    return {&hlo};
  }

  ConstInstructionVector hlos;
  hlos.reserve(hlo.operands().size());
  for (const HloInstruction* operand : hlo.operands()) {
    hlos.push_back(operand);
  }
  return hlos;
}

const HloInstruction& GetDestHlo(const HloInstruction& hlo,
                                 TransformDirection direction) {
  CHECK_EQ(hlo.operand_count(), 1);

  if (direction == TransformDirection::kInputToOutput) {
    return hlo;
  }

  return *hlo.operand(0);
}

/*static*/ DimOrderMapOrError FusionContext::GetPropagatedDimOrdersForBitcast(
    const HloInstruction& hlo, const TransformDirection direction,
    const DimensionOrder& src_dim_order, const HeroProperties& properties) {
  const HloInstruction& dst = GetDestHlo(hlo, direction);
  const Shape& dst_shape = dst.shape();
  const Fragments& src_fragments_order = src_dim_order.TensorFragmentsOrder();
  DimOrderMap dst_dim_orders;
  DimensionOrder& dst_dim_order =
      dst_dim_orders.insert({&dst, DimensionOrder()}).first->second;
  Fragments& dst_fragments_order = dst_dim_order.TensorFragmentsOrder();
  // Size of not yet assigned part of current target dimension.
  int64_t dst_remaining_size = 1;
  // Track destination fragments created from a source one.
  absl::flat_hash_map<const Fragment*, std::vector<int>> src_to_dst;
  // Iterate in parallel over source dimension order and target dimensions
  // in minor_to_major order. Find groups of dimensions of equal size
  // and project the source dimension order onto the destination.
  auto dst_dim_it = dst_shape.layout().minor_to_major().cbegin();
  const auto dst_dim_end = dst_shape.layout().minor_to_major().cend();
  for (auto src_dim = src_fragments_order.cbegin();
       src_dim != src_fragments_order.cend(); ++src_dim) {
    auto add_new_fragment = [&](const Fragment& fragment) {
      dst_fragments_order.push_back(fragment);
      src_to_dst[&*src_dim].push_back(dst_fragments_order.size() - 1);
    };
    if (std::holds_alternative<SoftmaxProperties>(properties) &&
        src_dim->dst_dim_number() ==
            std::get<SoftmaxProperties>(properties).softmax_batch_dimension) {
      // Special handling for softmax batch dimension: allow arbitrary reshapes
      // on it because it's guaranteed by the construction of the fusion to have
      // no physical alterations like transposes.
      // Find a continuous group of fragments corresponding to this dimension in
      // the source and assign the corresponding size in fragments of the
      // destination ignoring the source ones.
      dst_remaining_size = src_dim->full_size();
      while (src_dim + 1 != src_fragments_order.cend() &&
             (src_dim + 1)->dst_dim_number() == src_dim->dst_dim_number()) {
        ++src_dim;
        dst_remaining_size *= src_dim->full_size();
      }
      while (dst_remaining_size > 1) {
        CHECK(dst_dim_it != dst_dim_end);
        add_new_fragment(Fragment{src_dim->dst_dim_number(),
                                  dst_shape.dimensions(*dst_dim_it)});
        dst_remaining_size /= dst_shape.dimensions(*dst_dim_it);
        ++dst_dim_it;
      }
      continue;
    }
    if (dst_remaining_size >= src_dim->full_size()) {
      if (dst_remaining_size % src_dim->full_size()) {
        return "Unsupported bitcast";
      }
      // Source dimension fragment completely fits into the destination one:
      // just copy it as is.
      add_new_fragment(*src_dim);
      // Update the size of the remaining part of the destination that is
      // carried over to next source dimensions.
      dst_remaining_size /= src_dim->full_size();
    } else {
      // Source is larger than destination.
      // Assign further destination dimensions.
      // Size of the not yet assigned part of the source dimension.
      int64_t src_remaining_size = src_dim->full_size();
      // Handle dimension splits.
      if (dst_remaining_size > 1) {
        // If there is a remaining fragment of a previous destination dimension
        // assign it first.
        if (src_remaining_size % dst_remaining_size || (src_dim->is_sliced())) {
          return "Unsupported bitcast";
        }
        add_new_fragment(
            Fragment{src_dim->dst_dim_number(), dst_remaining_size});
        // Update the size of the fragment remaining to assign.
        src_remaining_size /= dst_remaining_size;
        dst_remaining_size = 1;
      }
      while (src_remaining_size > 1) {
        // Assign destination dimensions until the source remainder is covered.
        CHECK(dst_dim_it != dst_dim_end);
        int64_t dst_dim_size = dst_shape.dimensions(*dst_dim_it);
        int64_t new_fragment_size = dst_dim_size;
        if (dst_dim_size > src_remaining_size) {
          // If adding the next destination dimension exceeds source fragment
          // size assign the remainder of the source and carry over the
          // remainder of the destination.
          if (dst_dim_size % src_remaining_size) {
            return "Unsupported bitcast";
          }
          dst_remaining_size = dst_dim_size / src_remaining_size;
          new_fragment_size = src_remaining_size;
        }
        if (src_dim->is_sliced()) {
          return "Unsupported bitcast";
        }
        add_new_fragment(
            Fragment{src_dim->dst_dim_number(), new_fragment_size});
        src_remaining_size /= new_fragment_size;
        ++dst_dim_it;
      }
    }
  }
  CHECK_EQ(dst_remaining_size, 1);

  // Handle remaining major dimensions of the destination. Call all degenerate
  // ones subdimensions of the most-major non-degenerate one. Otherwise
  // give up.
  while (dst_dim_it != dst_dim_end) {
    if (dst_shape.dimensions(*dst_dim_it) != 1) {
      return "Unsupported bitcast";
    }
    if (!dst_fragments_order.empty()) {
      dst_fragments_order.push_back(
          Fragment{dst_fragments_order.back().dst_dim_number(), 1});
      src_to_dst[&src_fragments_order.back()].push_back(
          dst_fragments_order.size() - 1);
    }
    ++dst_dim_it;
  }

  FragmentOrders& dst_dim_fragment_orders = dst_dim_order.DimFragmentsOrders();
  for (const auto& [dim_index, dim_sequence] :
       src_dim_order.DimFragmentsOrders()) {
    std::vector<int>& dst = dst_dim_fragment_orders[dim_index];
    dst.reserve(dim_sequence.size());
    for (const int src : dim_sequence) {
      std::copy(src_to_dst[&src_fragments_order[src]].cbegin(),
                src_to_dst[&src_fragments_order[src]].cend(),
                std::back_inserter(dst));
    }
  }

  return dst_dim_orders;
}

// Handle copy, transpose, broadcast or reduce.
// Common between them is that they alter the tensor dimensions or their order
// and the way to handle layouts.
/*static*/ DimOrderMapOrError
FusionContext::GetPropagatedDimOrdersForDimAlteringOp(
    const HloInstruction& hlo, const TransformDirection direction,
    const DimensionOrder& src_dim_order, const HeroProperties& properties) {
  // Temporary storage for new fragments local to this function.
  // Please keep this as the first local variable of this function, with type
  // std::list to make sure that all pointers to elements of this remain valid
  // throughout the entire function. std::deque would also work but it is
  // unnecessarily big for a typical size of 1.
  std::list<Fragment> new_fragments;

  const HloInstruction& src = GetSourceHlo(hlo, direction);
  // Note: copying instead of using a const reference because
  // some operations (slice) will modify fragment properties in-place.
  Fragments src_fragments_order = src_dim_order.TensorFragmentsOrder();
  if (hlo.opcode() == HloOpcode::kSlice &&
      ShapeUtil::IsEffectiveScalar(hlo.shape())) {
    return FusionDecision("Slice to scalar is not implemented yet.");
  }
  // Every HLO dimension can correspond to a group of subdimensions in
  // dim_order_. For the easier handling of permutations: group dim_order_ by
  // dimension, apply permutations, then finally remove the grouping.
  // Group subdimensions by iterating over them in the same order as over
  // full dimensions and matching by total size.
  std::vector<std::vector<Fragment*>> src_physical;
  src_physical.reserve(src.shape().rank());
  auto src_fragment_it = src_fragments_order.begin();
  for (int64_t dim_index : src.shape().layout().minor_to_major()) {
    const int64_t dim_size = src.shape().dimensions(dim_index);
    int64_t subdim_size_accumulator = 1;
    std::vector<Fragment*> subdim_group;
    do {
      CHECK(src_fragment_it != src_fragments_order.end());
      subdim_size_accumulator *= src_fragment_it->full_size();
      subdim_group.push_back(&*src_fragment_it);
      ++src_fragment_it;
    } while (subdim_size_accumulator < dim_size);
    CHECK_EQ(subdim_size_accumulator, dim_size);
    src_physical.push_back(subdim_group);
  }

  // Source physical -> source logical.
  std::vector<std::vector<Fragment*>> src_logical;
  src_logical.resize(src_physical.size());
  for (int i = 0; i < src_physical.size(); ++i) {
    src_logical[src.shape().layout().minor_to_major(i)] = src_physical[i];
  }

  DimOrderMap dst_dim_orders;
  for (const HloInstruction* dst : GetDestHlos(hlo, direction)) {
    DimensionOrder& dst_dim_order =
        dst_dim_orders.insert({dst, DimensionOrder()}).first->second;
    // Source logical -> destination logical.
    std::vector<std::vector<Fragment*>> dst_logical;
    if (hlo.opcode() == HloOpcode::kTranspose) {
      const auto* transpose = Cast<HloTransposeInstruction>(&hlo);
      std::vector<int64_t> permutation(transpose->dimensions().cbegin(),
                                       transpose->dimensions().cend());
      if (direction == TransformDirection::kInputToOutput) {
        permutation = InversePermutation(permutation);
      }
      dst_logical.resize(permutation.size());
      for (int i = 0; i < permutation.size(); ++i) {
        dst_logical[permutation[i]] = src_logical[i];
      }
    } else if (hlo.opcode() == HloOpcode::kBroadcast) {
      const auto* broadcast = Cast<HloBroadcastInstruction>(&hlo);
      dst_logical.resize(broadcast->dimensions().size());
      for (int i = 0; i < broadcast->dimensions().size(); ++i) {
        dst_logical[i] = src_logical[broadcast->dimensions()[i]];
      }
    } else if (hlo.opcode() == HloOpcode::kReduce) {
      // Operand 1 (the neutral value) has to be a scalar.
      if (dst != &hlo && hlo.operand_index(dst) == 1) {
        continue;
      }
      const auto* reduce = Cast<HloReduceInstruction>(&hlo);
      dst_logical.resize(src_logical.size() + reduce->dimensions().size());
      if (reduce->dimensions().size() != 1) {
        return FusionDecision("Unsupported reduction.");
      }
      for (int i = 0; i < dst_logical.size(); ++i) {
        if (i == reduce->dimensions().front()) {
          // This way to assign the reduction dimension will only work for
          // softmax fusions with known patterns for now. Generally a reduction
          // should create a new tiled dimension.
          dst_logical[i] = {&new_fragments.emplace_back(
              std::get<SoftmaxProperties>(properties)
                  .softmax_reduction_dimension,
              reduce->operand(0)->shape().dimensions(i))};
        } else {
          dst_logical[i] = src_logical[i];
        }
      }
    } else if (hlo.opcode() == HloOpcode::kConcatenate) {
      dst_logical.resize(src_logical.size());
      for (int i = 0; i < src_logical.size(); ++i) {
        dst_logical[i] = src_logical[i];
        if (i == hlo.concatenate_dimension()) {
          if (src_logical[i].size() != 1 || src_logical[i][0]->is_sliced()) {
            return FusionDecision("Unsupported concatenation.");
          }
          dst_logical[i][0]->set_size(dst->shape().dimensions(i));
          dst_logical[i][0]->set_slice(0, dst->shape().dimensions(i));
        }
      }
    } else if (hlo.opcode() == HloOpcode::kCopy) {
      // Copy preserves the logical shape, just permutes the layout.
      CHECK(ShapeUtil::SameDimensions(src.shape(), dst->shape()));
      dst_logical = src_logical;
    } else if (hlo.opcode() == HloOpcode::kPad) {
      // Operand 1 (the padding value) has to be a scalar.
      if (dst != &hlo && hlo.operand_index(dst) == 1) {
        continue;
      }
      const auto* pad = Cast<HloPadInstruction>(&hlo);
      dst_logical.resize(src_logical.size());
      for (int i = 0; i < src_logical.size(); ++i) {
        // This only handles the padding added by
        // PadDotOperandsIfNeededForSplitK, which sets only edge_padding_high.
        const int padding =
            pad->padding_config().dimensions(i).edge_padding_high();
        CHECK_EQ(pad->padding_config().dimensions(i).edge_padding_low(), 0);
        CHECK_EQ(pad->padding_config().dimensions(i).interior_padding(), 0);
        if (padding == 0) {
          dst_logical[i] = src_logical[i];
        } else {
          // This case is executed for the contracting dimension when we run the
          // TritonFusionAnalysis after the padding and the split-k transform
          // are applied.
          const std::vector<Fragment*>& fragments = src_logical[i];
          // We must have 2 fragments at this point.
          CHECK_EQ(fragments.size(), 2);
          // The dst_dim_numbers must be the same for the 2 fragments of the
          // contracting dimension after applying split-k.
          CHECK_EQ(fragments[0]->dst_dim_number(),
                   fragments[1]->dst_dim_number());

          new_fragments.emplace_back(
              fragments[0]->dst_dim_number(),
              fragments[0]->full_size() * fragments[1]->full_size() - padding);
          dst_logical[i] = {&new_fragments.back()};
        }
      }
    } else if (hlo.opcode() == HloOpcode::kSlice) {
      const auto slice = Cast<HloSliceInstruction>(&hlo);
      dst_logical.resize(src_logical.size());
      for (int dim = 0; dim < src_logical.size(); ++dim) {
        dst_logical[dim] = src_logical[dim];
        if (slice->slice_limits(dim) - slice->slice_starts(dim) !=
            dst->shape().dimensions(dim)) {
          if (dst_logical[dim].size() > 1) {
            return FusionDecision("Slicing of fragmented dimension.");
          }
          auto fragment = dst_logical[dim].front();
          fragment->set_size(dst->shape().dimensions(dim));
          // Slicing of an already sliced dimension means adding offsets.
          fragment->set_slice(
              fragment->slice_start() + slice->slice_starts(dim),
              fragment->slice_start() + slice->slice_starts(dim) +
                  fragment->sliced_size());
        }
      }
    } else {
      return FusionDecision("Function called on a wrong instruction.");
    }
    // Destination logical -> destination physical and ungroup subdimensions.
    // Map original fragments to the resulting ones to derive their new
    // logical ordering within each dimension.
    absl::flat_hash_map<const Fragment*, int> src_to_dst;
    Fragments& dst_fragments_order = dst_dim_order.TensorFragmentsOrder();
    FragmentOrders& dst_dim_fragments_order =
        dst_dim_order.DimFragmentsOrders();
    // Remember which dimensions are present before a broadcast;
    // skip cases when already present dimension is being expanded.
    absl::flat_hash_set<int> dim_numbers_present_in_dst;
    for (const int64_t dim_idx : dst->shape().layout().minor_to_major()) {
      for (const Fragment* subdim : dst_logical[dim_idx]) {
        dst_fragments_order.push_back(*subdim);
        src_to_dst[subdim] = dst_fragments_order.size() - 1;
        dim_numbers_present_in_dst.insert(subdim->dst_dim_number());
      }
    }
    for (const auto& [dim_index, dim_sequence] :
         src_dim_order.DimFragmentsOrders()) {
      for (const int fragment_number : dim_sequence) {
        const auto it = src_to_dst.find(&src_fragments_order[fragment_number]);
        if (it == src_to_dst.cend()) {
          if (hlo.opcode() == HloOpcode::kBroadcast &&
              src_fragments_order[fragment_number].full_size() > 1 &&
              dim_numbers_present_in_dst.contains(dim_index)) {
            return FusionDecision("Unsupported broadcast");
          }
          continue;
        }
        dst_dim_fragments_order[dim_index].push_back(it->second);
      }
    }
  }
  return dst_dim_orders;
}

// Infers DimensionOrders of all unknown sides (output, operands)
// of `hlo` from the known ones.
/*static*/ DimOrderMapOrError FusionContext::GetPropagatedDimOrders(
    const HloInstruction& hlo, const TransformDirection direction,
    const DimensionOrder& src_dim_order, const HeroProperties& properties) {
  VLOG(7) << "Analyzing " << hlo.ToString();
  if (hlo.opcode() != HloOpcode::kParameter &&
      direction == TransformDirection::kOutputToInput &&
      absl::c_any_of(hlo.users(), [](const HloInstruction* user) {
        return user->opcode() == HloOpcode::kConcatenate;
      })) {
    return "No fusion into concatenations";
  }
  if (hlo.opcode() == HloOpcode::kParameter ||
      hlo_query::IsScalarConstant(&hlo)) {
    CHECK(direction == TransformDirection::kOutputToInput);
    return DimOrderMap{};
  } else if (hlo.opcode() == HloOpcode::kTranspose ||
             hlo.opcode() == HloOpcode::kCopy) {
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.opcode() == HloOpcode::kBroadcast) {
    if (direction != TransformDirection::kOutputToInput) {
      return "Unsupported broadcast direction.";
    }
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.opcode() == HloOpcode::kReduce) {
    if (!std::holds_alternative<SoftmaxProperties>(properties)) {
      return "Reductions are not supported in GEMM fusions yet.";
    }
    if (direction != TransformDirection::kOutputToInput) {
      return "Unsupported direction of reduction.";
    }
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.opcode() == HloOpcode::kPad) {
    if (direction != TransformDirection::kOutputToInput) {
      return "Unsupported pad direction.";
    }
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.operand_count() > 0 &&
             IsTritonSupportedElementwise(
                 hlo.opcode(), hlo.operand(0)->shape().element_type())) {
    return GetPropagatedDimOrdersForElementwise(hlo, direction, src_dim_order);
  } else if (hlo.opcode() == HloOpcode::kBitcast) {
    return GetPropagatedDimOrdersForBitcast(hlo, direction, src_dim_order,
                                            properties);
  } else if (hlo.opcode() == HloOpcode::kSlice) {
    if (direction != TransformDirection::kOutputToInput) {
      return "Unsupported slice direction.";
    }
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  } else if (hlo.opcode() == HloOpcode::kReshape) {
    if (!ShapeUtil::ReshapeIsBitcast(hlo.operand(0)->shape(), hlo.shape())) {
      return "Non-bitcast reshape.";
    }
    return GetPropagatedDimOrdersForBitcast(hlo, direction, src_dim_order,
                                            properties);
  } else if (hlo.opcode() == HloOpcode::kConcatenate &&
             direction == TransformDirection::kOutputToInput) {
    if (!std::holds_alternative<DotProperties>(properties)) {
      return "Concatenations for now are only supported in GEMM fusions.";
    }
    auto dim = LogicalIndexOfLabeledDimension(
        hlo.shape(), src_dim_order,
        std::get<DotProperties>(properties).noncontracting_dimension);
    if (!dim.has_value() || dim.value() != hlo.concatenate_dimension()) {
      return "Unsupported concatenation.";
    }
    if (absl::c_any_of(hlo.operands(), [](const HloInstruction* operand) {
          return operand->user_count() > 1;
        })) {
      return FusionDecision(
          "Concatenation has to be the only user of its inputs.");
    }
    if (absl::c_any_of(hlo.operands(), [&hlo](const HloInstruction* operand) {
          // In the current simple implementation of concatenation the size of
          // each of its inputs along the concatenated dimension has to be
          // divisible by the tile size used for this dimension. Concatenations
          // with any operand not divisible by kMinConcatFragmentSize will not
          // be fused; tiling configurations with tile size for this dimension
          // larger than kMinConcatFragmentSize will not be emitted.
          constexpr int kMinConcatFragmentSize = 128;
          return operand->shape().dimensions(hlo.concatenate_dimension()) %
                     kMinConcatFragmentSize !=
                 0;
        })) {
      return FusionDecision(
          "One or more operands of concatenation can not be perfectly tiled.");
    }
    return GetPropagatedDimOrdersForDimAlteringOp(hlo, direction, src_dim_order,
                                                  properties);
  }
  return "Unimplemented instruction.";
}

// Difference of input and output data volumes of an instruction.
int64_t InputMinusOutputBytes(const HloInstruction& hlo) {
  CHECK(!hlo.shape().IsTuple());
  int64_t input_size = 0;
  for (const HloInstruction* operand : hlo.operands()) {
    CHECK(!operand->shape().IsTuple());
    input_size += ShapeUtil::ByteSizeOf(operand->shape());
  }
  return input_size - ShapeUtil::ByteSizeOf(hlo.shape());
}

// Tells if an instruction has no user into which it could be fused.
// More cases should be added here.
bool CanNotBeFusedIntoAUser(const HloInstruction& hlo) {
  return hlo.IsRoot() || (hlo.user_count() == 1 && hlo.users()[0]->IsRoot() &&
                          hlo.users()[0]->opcode() == HloOpcode::kTuple);
}

// Let input and output data volumes of a fusion grow by small amounts.
constexpr int kIoToleranceBytes = 1024;

// Tells that fusing an instruction as an input is efficient.
bool IsInputWorthFusing(const HloInstruction& hlo) {
  if (InputMinusOutputBytes(hlo) <= kIoToleranceBytes) {
    return true;
  }
  if (hlo.user_count() > 1) {
    return false;
  }
  if (hlo.opcode() == HloOpcode::kSlice &&
      hlo_query::AllOperandsAreParametersOrConstants(hlo)) {
    return true;
  }
  return hlo_query::AllOperandsAreParametersOrConstantsWithSingleUser(hlo);
}

// Tells that fusing an instruction as an output is efficient.
bool IsOutputWorthFusing(const HloInstruction& hlo) {
  return CanNotBeFusedIntoAUser(hlo) ||
         InputMinusOutputBytes(hlo) >= -kIoToleranceBytes;
}

/*static*/ DimOrdersAndReqsOrError
FusionContext::GetPropagatedDimOrdersAndRequirements(
    const HloInstruction& hlo, const DimensionOrder& src_dim_order,
    TransformDirection direction, const HeroProperties& properties) {
  DimOrderMapOrError propagated_dim_orders_or_error =
      GetPropagatedDimOrders(hlo, direction, src_dim_order, properties);
  if (std::holds_alternative<FusionDecision>(propagated_dim_orders_or_error)) {
    return std::get<FusionDecision>(propagated_dim_orders_or_error);
  }
  DimOrderMap propagated_dim_orders =
      std::move(std::get<DimOrderMap>(propagated_dim_orders_or_error));
  RequirementsOrError requirements_or_error =
      GetRequirementsIfSupportedOrders(hlo, propagated_dim_orders, properties);
  if (std::holds_alternative<FusionDecision>(requirements_or_error)) {
    return std::get<FusionDecision>(requirements_or_error);
  }
  return DimOrdersAndReqs{propagated_dim_orders,
                          std::get<Requirements>(requirements_or_error)};
}

/*static*/ DimOrdersAndReqsOrError
FusionContext::GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
    const HloInstruction& hlo, TransformDirection transform_direction,
    const std::optional<int>& src_operand_index,
    const DimensionOrder& src_dim_order,
    const se::GpuComputeCapability& gpu_version,
    const HeroProperties& properties) {
  CHECK_EQ(transform_direction == TransformDirection::kInputToOutput,
           src_operand_index.has_value());

  if (hlo.opcode() == HloOpcode::kTuple ||
      hlo.opcode() == HloOpcode::kGetTupleElement) {
    return "Unsupported instruction.";
  }
  if (hlo.opcode() == HloOpcode::kReduce) {
    return "Reductions are not fused yet.";
  }
  if (hlo.opcode() == HloOpcode::kPad) {
    return "Pads are not fused yet.";
  }
  for (const HloInstruction* operand : hlo.operands()) {
    if (!IsTritonSupportedDataType(operand->shape().element_type(),
                                   gpu_version)) {
      return "Unsupported input data type.";
    }
  }
  if (!IsTritonSupportedDataType(hlo.shape().element_type(), gpu_version)) {
    return "Unsupported output data type.";
  }
  DimOrdersAndReqsOrError result_or_error =
      GetPropagatedDimOrdersAndRequirements(hlo, src_dim_order,
                                            transform_direction, properties);
  if (!std::holds_alternative<DimOrdersAndReqs>(result_or_error)) {
    return result_or_error;
  }
  DimOrdersAndReqs dim_orders_and_requirements =
      std::move(std::get<DimOrdersAndReqs>(result_or_error));
  int fusion_level =
      hlo.GetModule()->config().debug_options().xla_gpu_triton_fusion_level();
  if (!std::get<se::CudaComputeCapability>(gpu_version)
           .IsAtLeast(se::CudaComputeCapability::AMPERE)) {
    fusion_level = std::min(fusion_level, 1);
  }
  if (transform_direction == TransformDirection::kOutputToInput) {
    if (fusion_level < 2) {
      if (hlo.opcode() == HloOpcode::kConvert) {
        if (FusionDecision decision = IsConversionWorthFusing(hlo, gpu_version);
            !decision) {
          return decision;
        }
      } else if (hlo.IsElementwise() && hlo.opcode() != HloOpcode::kCopy) {
        return "Ignored elementwise operation";
      }
    } else {
      // Exception for binary elementwise operations: in most cases these are
      // not trivial to fuse because they increase DRAM traffic but if one
      // of the inputs is for example a broadcast that can be fused too it
      // becomes worth fusing. Look ahead and analyze operands here.
      bool accepted = false;
      if (hlo.IsElementwise() && hlo.operand_count() == 2) {
        for (const HloInstruction* operand : hlo.operands()) {
          if (operand->opcode() == HloOpcode::kBroadcast &&
              (operand->operand(0)->opcode() == HloOpcode::kParameter ||
               operand->operand(0)->opcode() == HloOpcode::kConstant) &&
              std::holds_alternative<DimOrdersAndReqs>(
                  GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
                      *operand, TransformDirection::kOutputToInput,
                      /*src_operand_index=*/std::nullopt,
                      /*src_dim_order=*/
                      dim_orders_and_requirements.dim_orders.at(operand),
                      gpu_version, properties))) {
            accepted = true;
            break;
          }
        }
      }
      if (!accepted && !IsInputWorthFusing(hlo)) {
        return "Not obviously profitable to fuse as input.";
      }
    }
  } else {
    if (fusion_level < 2) {
      return "Skipping fusing outputs at low fusion levels.";
    }
    for (int i = 0; i < hlo.operand_count(); ++i) {
      const HloInstruction* operand = hlo.operand(i);
      // Skip source operand.
      if (i == *src_operand_index) {
        continue;
      }
      // Currently only broadcasts of scalar constants or parameters
      // are accepted as other inputs of non-unary operations
      // in the output fusion.
      if (hlo_query::IsBroadcastOfScalarConstant(*operand) ||
          operand->opcode() == HloOpcode::kParameter) {
        continue;
      }
      return "Has multiple inputs - not properly analyzed yet.";
    }
    if (!IsOutputWorthFusing(hlo)) {
      return "Not obviously profitable to fuse as output.";
    }
  }
  return dim_orders_and_requirements;
}

// Gets the fused HLO corresponding to `hlo` or adds a new parameter if not
// found.
HloInstruction* GetFusedHloOrAddParameter(
    HloInstruction& hlo, OldToNewHloMap& old_to_new_map,
    std::vector<HloInstruction*>& fusion_inputs,
    HloComputation::Builder& builder) {
  if (auto it = old_to_new_map.find(&hlo); it != old_to_new_map.end()) {
    return it->second;
  }
  fusion_inputs.push_back(&hlo);
  return old_to_new_map
      .insert(
          {&hlo, builder.AddInstruction(HloInstruction::CreateParameter(
                     fusion_inputs.size() - 1, hlo.shape(),
                     absl::StrCat("parameter_", fusion_inputs.size() - 1)))})
      .first->second;
}

// Clone an instruction into the fusion.
//
// For the hero dot operation in the dot fusion, please use FuseDotOnly.
void Fuse(HloInstruction& hlo, OldToNewHloMap& old_to_new_map,
          std::vector<HloInstruction*>& fusion_inputs,
          HloComputation::Builder& builder) {
  if (old_to_new_map.contains(&hlo)) {
    return;
  }
  VLOG(3) << "Fusing " << hlo.ToString();
  if (hlo.opcode() == HloOpcode::kParameter ||
      hlo.opcode() == HloOpcode::kGetTupleElement) {
    GetFusedHloOrAddParameter(hlo, old_to_new_map, fusion_inputs, builder);
  } else {
    std::vector<HloInstruction*> hlo_new_operands;
    for (HloInstruction* operand : hlo.operands()) {
      hlo_new_operands.push_back(GetFusedHloOrAddParameter(
          *operand, old_to_new_map, fusion_inputs, builder));
    }
    old_to_new_map[&hlo] = builder.AddInstruction(
        hlo.CloneWithNewOperands(hlo.shape(), hlo_new_operands));
  }
}

// Clones the hero kDot operation into the fusion.
void FuseDotOnly(HloInstruction& hlo, OldToNewHloMap& output_old_to_new_map,
                 OldToNewHloMap& lhs_old_to_new_map,
                 OldToNewHloMap& rhs_old_to_new_map,
                 std::vector<HloInstruction*>& fusion_inputs,
                 HloComputation::Builder& builder) {
  CHECK_EQ(hlo.opcode(), HloOpcode::kDot);
  CHECK_EQ(hlo.operand_count(), 2);
  VLOG(3) << "Fusing " << hlo.ToString();

  std::array<HloInstruction*, 2> hlo_new_operands = {
      GetFusedHloOrAddParameter(*hlo.mutable_operand(0), lhs_old_to_new_map,
                                fusion_inputs, builder),
      GetFusedHloOrAddParameter(*hlo.mutable_operand(1), rhs_old_to_new_map,
                                fusion_inputs, builder)};
  output_old_to_new_map[&hlo] = builder.AddInstruction(
      hlo.CloneWithNewOperands(hlo.shape(), hlo_new_operands));
}

// Tells how many new parameters does a fusion gain by fusing the operation as
// an input.
int64_t NumAddedParameters(const HloInstruction& hlo) {
  // Non-scalar constant is equivalent to a parameter: one input, one output.
  if (hlo.opcode() == HloOpcode::kConstant &&
      !ShapeUtil::IsScalar(hlo.shape())) {
    return 0;
  }
  // All other instructions add all own inputs and remove own single output.
  return hlo.operand_count() - 1;
}

bool FusionContext::CombineDimOrdersAndReqs(const DimOrdersAndReqs& update) {
  // First check that all updates to insert are compatible to avoid
  // incomplete merges.
  for (const auto& [key, value] : update.dim_orders) {
    auto it = dim_orders_.find(key);
    if (it != dim_orders_.cend() && !it->second.IsPhysicallyEquivalent(value)) {
      return false;
    }
  }

  RequirementsOrError requirements_or_error =
      CombineRequirements(requirements_, update.requirements);
  if (std::holds_alternative<FusionDecision>(requirements_or_error)) {
    return false;
  }

  requirements_ = std::move(std::get<Requirements>(requirements_or_error));
  dim_orders_.insert(update.dim_orders.begin(), update.dim_orders.end());
  return true;
}

void FusionContext::TryToFuseWithInputsRecursively(
    HloInstruction& root, const se::GpuComputeCapability gpu_version,
    OldToNewHloMap& old_to_new_map, std::vector<HloInstruction*>& fusion_inputs,
    HloComputation::Builder& builder) {
  // Instructions at the fusion edge that can either get fused too or
  // become parameters of the fusion. Used to track the number of parameters.
  absl::flat_hash_set<const HloInstruction*> inputs;
  // Traverse all connected instructions that could be fused, analyze them and
  // collect ones that will be fused.
  absl::flat_hash_set<const HloInstruction*> to_fuse_set;
  std::list<HloInstruction*> to_fuse_list;
  absl::flat_hash_set<const HloInstruction*> enqueued;
  std::queue<HloInstruction*> to_visit;
  to_visit.push(&root);
  int num_requeued = 0;
  while (to_visit.size() > num_requeued) {
    HloInstruction* hlo = to_visit.front();
    to_visit.pop();
    // Watch the total number of fusion parameters.
    if (inputs.size() + NumAddedParameters(*hlo) >
        TritonFusionAnalysis::kMaxParameterPerDotScope) {
      // Re-queue: the number of parameters may go down when other instructions
      // are processed.
      to_visit.push(hlo);
      // Prevent infinite loops.
      ++num_requeued;
      continue;
    }
    num_requeued = 0;
    const DimOrdersAndReqsOrError result =
        GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
            *hlo, TransformDirection::kOutputToInput,
            /*src_operand_index=*/std::nullopt, dim_orders_.at(hlo),
            gpu_version, properties_);
    if (!std::holds_alternative<DimOrdersAndReqs>(result) ||
        !CombineDimOrdersAndReqs(std::get<DimOrdersAndReqs>(result))) {
      continue;
    }
    if (hlo->opcode() != HloOpcode::kParameter) {
      inputs.erase(hlo);
    }
    inputs.insert(hlo->operands().cbegin(), hlo->operands().cend());
    to_fuse_set.insert(hlo);
    to_fuse_list.push_back(hlo);
    for (HloInstruction* operand : hlo->operands()) {
      if (enqueued.insert(operand).second) {
        VLOG(6) << "Enqueueing " << operand->ToString();
        to_visit.push(operand);
      }
    }
  }
  // Find one by one instructions that have no operands queued to be fused and
  // fuse them.
  while (!to_fuse_list.empty()) {
    for (auto it = to_fuse_list.begin(); it != to_fuse_list.end();) {
      bool ready_to_fuse = true;
      for (const HloInstruction* operand : (*it)->operands()) {
        if (to_fuse_set.contains(operand)) {
          ready_to_fuse = false;
          break;
        }
      }
      if (ready_to_fuse) {
        Fuse(**it, old_to_new_map, fusion_inputs, builder);
        to_fuse_set.erase(*it);
        it = to_fuse_list.erase(it);
      } else {
        ++it;
      }
    }
  }
}

// Fuses dot and the compatible and profitable to fuse operations around it
// into a new fusion computation constructed using the builder. fusion_inputs
// get populated with the non-fused instructions that become operands of the
// call to this fusion. fusion_output_ptr (if not nullptr) gets assigned the
// original instruction that has to be replaced by the call to the fusion.
StatusOr<FusionDecision> FuseDot(HloInstruction& dot,
                                 const se::GpuComputeCapability gpu_version,
                                 HloComputation::Builder& builder,
                                 std::vector<HloInstruction*>& fusion_inputs,
                                 HloInstruction** fusion_output_ptr) {
  VLOG(5) << dot.ToString();
  if (FusionDecision can_handle = CanTritonHandleGEMM(dot, gpu_version);
      !can_handle) {
    VLOG(3) << can_handle.Explain();
    return can_handle;
  }

  // Separate traversal from LHS and RHS inputs of the dot: they use
  // differently shaped tiles but may go through same HLO graph nodes.
  // Direct dot inputs have well defined dimension orders.

  auto fuse_inputs =
      [&](int operand_number,
          OldToNewHloMap& old_to_new_map) -> StatusOr<FusionContext> {
    const int operand_count_before = fusion_inputs.size();
    // Direct dot inputs have well defined dimension orders.
    auto context = FusionContext::FromDotOperand(dot, operand_number);
    context.TryToFuseWithInputsRecursively(*dot.mutable_operand(operand_number),
                                           gpu_version, old_to_new_map,
                                           fusion_inputs, builder);
    const int new_parameters = fusion_inputs.size() - operand_count_before;
    TF_RET_CHECK(new_parameters <=
                 TritonFusionAnalysis::kMaxParameterPerDotScope)
        << "Too many new parameters: " << new_parameters << " > "
        << TritonFusionAnalysis::kMaxParameterPerDotScope;
    return context;
  };

  // Original instruction -> fused one. Separate for each scope.
  OldToNewHloMap lhs_old_to_new_map;
  TF_ASSIGN_OR_RETURN(const FusionContext lhs_context,
                      fuse_inputs(0, lhs_old_to_new_map));

  OldToNewHloMap rhs_old_to_new_map;
  if (auto result = fuse_inputs(1, rhs_old_to_new_map); !result.ok()) {
    return result.status();
  }

  OldToNewHloMap output_old_to_new_map;
  // Fuse the dot into output_old_to_new_map and use lhs_old_to_new_map and
  // rhs_old_to_new_map to generate / determine its operands.
  FuseDotOnly(dot, output_old_to_new_map, lhs_old_to_new_map,
              rhs_old_to_new_map, fusion_inputs, builder);

  // Fusion at dot's output.

  // These describe _outputs_ of corresponding HLOs.
  auto context = FusionContext::FromDotOutput(
      dot, /*split_k=*/1, lhs_context.splittable_dimension_major_part_size());
  HloInstruction* fusion_output = &dot;
  bool output_changed = true;
  while (output_changed) {
    output_changed = false;
    if (fusion_output->user_count() != 1) {
      break;
    }
    HloInstruction* user = fusion_output->users()[0];
    if (!IsDistributiveOverAddition(*user)) {
      break;
    }
    DimOrdersAndReqsOrError result =
        FusionContext::GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
            *user, TransformDirection::kInputToOutput,
            user->operand_index(fusion_output),
            context.dim_orders().at(fusion_output), gpu_version,
            context.hero_properties());
    if (!std::holds_alternative<DimOrdersAndReqs>(result) ||
        !context.CombineDimOrdersAndReqs(std::get<DimOrdersAndReqs>(result))) {
      break;
    }
    for (HloInstruction* operand : user->operands()) {
      if (!output_old_to_new_map.contains(operand)) {
        context.TryToFuseWithInputsRecursively(*operand, gpu_version,
                                               output_old_to_new_map,
                                               fusion_inputs, builder);
      }
    }
    Fuse(*user, output_old_to_new_map, fusion_inputs, builder);
    fusion_output = user;
    output_changed = true;
  }
  if (fusion_output_ptr != nullptr) {
    *fusion_output_ptr = fusion_output;
  }
  if (dot.GetModule()->config().debug_options().xla_gpu_triton_gemm_any()) {
    return FusionDecision{};
  }

  for (auto* old_to_new_map : std::array<const OldToNewHloMap*, 3>{
           &lhs_old_to_new_map, &rhs_old_to_new_map, &output_old_to_new_map}) {
    for (auto [_, new_hlo] : *old_to_new_map) {
      static constexpr std::array<HloOpcode, 4> kPureOpcodes = {
          HloOpcode::kBitcast, HloOpcode::kDot, HloOpcode::kParameter,
          HloOpcode::kReshape};
      // Fuse if this is not a "pure" matmul.
      if (absl::c_find(kPureOpcodes, new_hlo->opcode()) == kPureOpcodes.end()) {
        return FusionDecision{};
      }
    }
  }
  return "No profitable operations to fuse.";
}

// Extracts into fused computations parts of HLO graph including dot()
// operations that can target the triton GEMM emitter.
class GemmRewriterTritonVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmRewriterTritonVisitor(const se::GpuComputeCapability gpu_version)
      : gpu_version_(gpu_version) {}
  // Checks that a dot() should be targeting the triton GEMM emitter;
  // if so - fuses all its compatible inputs and outputs as a new computation
  // and replaces the original dot() with a call to the computation.
  Status HandleDot(HloInstruction* dot) override {
    std::string fusion_name = absl::StrCat("triton_gemm_", dot->name());
    HloComputation::Builder builder(absl::StrCat(fusion_name, "_computation"));
    std::vector<HloInstruction*> fusion_inputs;
    HloInstruction* fusion_output = nullptr;
    TF_ASSIGN_OR_RETURN(
        const FusionDecision should_fuse,
        FuseDot(*dot, gpu_version_, builder, fusion_inputs, &fusion_output));
    if (builder.last_added_instruction() == nullptr) {
      return OkStatus();
    }
    // If a GEMM requiring padding for cuBLAS is encountered here this
    // happened because earlier ShouldTritonHandleGEMM() accepted it and padding
    // was skipped. Accept it ignoring profitability checks.
    if (!CublasRequiresPadding(
            *Cast<HloDotInstruction>(dot),
            std::get<se::CudaComputeCapability>(gpu_version_)) &&
        !should_fuse) {
      return OkStatus();
    }

    HloComputation* computation =
        dot->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                            /*is_entry=*/false);
    HloInstruction* dot_fusion =
        dot->parent()->AddInstruction(HloInstruction::CreateFusion(
            computation->root_instruction()->shape(),
            HloInstruction::FusionKind::kCustom, fusion_inputs, computation));
    dot_fusion->GetModule()->SetAndUniquifyInstrName(dot_fusion, fusion_name);

    TF_ASSIGN_OR_RETURN(auto backend_config,
                        dot_fusion->backend_config<FusionBackendConfig>());
    backend_config.set_kind(std::string(kTritonGemmFusionKind));
    TF_RETURN_IF_ERROR(dot_fusion->set_backend_config(backend_config));

    if (fusion_output->IsRoot()) {
      fusion_output->parent()->set_root_instruction(dot_fusion);
      TF_RETURN_IF_ERROR(
          fusion_output->parent()->RemoveInstructionAndUnusedOperands(
              fusion_output));
      MarkAsChanged();
    } else {
      TF_RETURN_IF_ERROR(ReplaceInstruction(fusion_output, dot_fusion));
    }
    XLA_VLOG_LINES(5, computation->ToString(HloPrintOptions::ShortParsable()));
    return OkStatus();
  }

 private:
  se::GpuComputeCapability gpu_version_;
};

StatusOr<bool> RunOnComputation(HloComputation* computation,
                                se::GpuComputeCapability gpu_version) {
  GemmRewriterTritonVisitor visitor(gpu_version);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

Status FusionContext::PropagateDimensionOrdersToParameters(
    const HloInstruction& origin, ConstHloInstructionSet& parameters,
    ConstHloInstructionMap<TensorIterationSpec>& iter_specs) {
  absl::flat_hash_set<const HloInstruction*> visited;
  std::queue<const HloInstruction*> to_process;
  // Dimension orders describing outputs of corresponding instructions.
  visited.insert(&origin);
  to_process.push(&origin);
  while (!to_process.empty()) {
    const HloInstruction* hlo = to_process.front();
    to_process.pop();
    if (hlo->opcode() == HloOpcode::kParameter) {
      // One parameter corresponds to one iteration spec in the results of the
      // analysis. This describes well situations when a parameter has one or
      // more elementwise users - they share the same tiling. Situations when
      // one instruction is read differently by different users in the same
      // scope of the dot are currently prevented during the fusion.
      TF_RET_CHECK(parameters.insert(hlo).second);
      VLOG(5) << hlo->ToString();
    }
    DimOrdersAndReqsOrError result = GetPropagatedDimOrdersAndRequirements(
        *hlo, dim_orders_.at(hlo), TransformDirection::kOutputToInput,
        properties_);
    if (std::holds_alternative<FusionDecision>(result)) {
      LOG(FATAL) << std::get<FusionDecision>(result).Explain();
    }
    TF_RET_CHECK(CombineDimOrdersAndReqs(std::get<DimOrdersAndReqs>(result)));
    iter_specs[hlo] = DimensionOrderToTensorIterationSpec(dim_orders_.at(hlo));
    for (const HloInstruction* operand : hlo->operands()) {
      if (!visited.insert(operand).second) {
        continue;
      }
      if (operand->opcode() == HloOpcode::kDot) {
        // Encountering the dot itself happens during the processing of the
        // output fusion. The propagation should stop at it.
        continue;
      }
      to_process.push(operand);
    }
  }
  return OkStatus();
}

}  // namespace

// Data types that are supported by the Triton emitters.
bool IsTritonSupportedDataType(PrimitiveType type,
                               se::GpuComputeCapability gpu_version) {
  auto cuda_compute_capability =
      std::get<se::CudaComputeCapability>(gpu_version);
  switch (type) {
    case PRED:
    case S8:
    case S16:
    case S32:
    case F16:
    case F32:
      return true;
    case BF16:
      return cuda_compute_capability.IsAtLeast(
          stream_executor::CudaComputeCapability::AMPERE);
    default:
      return false;
  }
}

// BF16 is supported in a sense that all operations on it are implemented
// through F32 and converts have to be inserted into the HLO graph, but
// they can be missing during fusion.

std::vector<HloOpcode> TritonSupportedUnaryElementwise(
    PrimitiveType element_type) {
  std::vector<HloOpcode> ret = {HloOpcode::kConvert};
  if (element_type == PrimitiveType::PRED) {
    ret.push_back(HloOpcode::kNot);
    return ret;
  }
  ret.push_back(HloOpcode::kAbs);
  ret.push_back(HloOpcode::kNegate);
  if (element_type == PrimitiveType::F32 ||
      element_type == PrimitiveType::BF16 ||
      element_type == PrimitiveType::F64) {
    absl::c_copy(std::vector<HloOpcode>{HloOpcode::kCos, HloOpcode::kExp,
                                        HloOpcode::kExpm1, HloOpcode::kLog,
                                        HloOpcode::kLog1p, HloOpcode::kRsqrt,
                                        HloOpcode::kSin, HloOpcode::kSqrt,
                                        HloOpcode::kCbrt, HloOpcode::kTan,
                                        HloOpcode::kTanh},
                 std::back_inserter(ret));
  }
  return ret;
}

std::vector<HloOpcode> TritonSupportedBinaryElementwise(
    PrimitiveType element_type) {
  if (element_type == PrimitiveType::PRED) {
    return {HloOpcode::kAnd, HloOpcode::kOr, HloOpcode::kXor,
            HloOpcode::kCompare};
  }
  std::vector<HloOpcode> ret = {HloOpcode::kAdd,      HloOpcode::kCompare,
                                HloOpcode::kMaximum,  HloOpcode::kMinimum,
                                HloOpcode::kMultiply, HloOpcode::kSubtract};
  if (element_type == PrimitiveType::F32 ||
      element_type == PrimitiveType::BF16 ||
      element_type == PrimitiveType::F64) {
    ret.push_back(HloOpcode::kAtan2);
    ret.push_back(HloOpcode::kDivide);
    ret.push_back(HloOpcode::kPower);
  }
  return ret;
}

std::vector<HloOpcode> TritonSupportedTernaryElementwise(
    PrimitiveType element_type) {
  return {HloOpcode::kSelect};
}

bool IsTritonSupportedElementwise(HloOpcode opcode,
                                  PrimitiveType element_type) {
  return absl::c_linear_search(TritonSupportedUnaryElementwise(element_type),
                               opcode) ||
         absl::c_linear_search(TritonSupportedBinaryElementwise(element_type),
                               opcode) ||
         absl::c_linear_search(TritonSupportedTernaryElementwise(element_type),
                               opcode);
}

StatusOr<TritonFusionAnalysis> TritonFusionAnalysis::Execute(
    const HloComputation& computation, const int split_k) {
  VLOG(5) << computation.ToString(HloPrintOptions::ShortParsable());
  TritonFusionAnalysis analysis;
  const HloInstruction* dot =
      hlo_query::GetFirstInstructionWithOpcode(computation, HloOpcode::kDot);
  if (dot != nullptr) {
    TF_RETURN_IF_ERROR(analysis.ExecuteForDotFusion(*dot, split_k));
  } else {
    TF_RETURN_IF_ERROR(
        analysis.ExecuteForSoftmaxFusion(*computation.root_instruction()));
  }
  return analysis;
}

Status TritonFusionAnalysis::ExecuteForSoftmaxFusion(
    const HloInstruction& root) {
  auto context = FusionContext::FromSoftmaxRoot(root);
  // Softmax fusion uses one tiled scope.
  TF_RETURN_IF_ERROR(context.PropagateDimensionOrdersToParameters(
      root, parameters_[Scope::OUTPUT], iter_specs_[Scope::OUTPUT]));
  iter_specs_[Scope::LHS] = {};
  iter_specs_[Scope::RHS] = {};
  return OkStatus();
}

Status TritonFusionAnalysis::ExecuteForDotFusion(const HloInstruction& dot,
                                                 const int split_k) {
  int64_t lhs_nc_split_major_part_size = kNoSplitRequirement;
  for (const Scope scope : {Scope::LHS, Scope::RHS}) {
    const int operand_number = static_cast<int>(scope);
    auto context = FusionContext::FromDotOperand(dot, operand_number, split_k);
    TF_RETURN_IF_ERROR(context.PropagateDimensionOrdersToParameters(
        *dot.operand(operand_number), parameters_[scope], iter_specs_[scope]));
    if (scope == Scope::LHS) {
      lhs_nc_split_major_part_size =
          context.splittable_dimension_major_part_size();
    }
  }

  auto context =
      FusionContext::FromDotOutput(dot, split_k, lhs_nc_split_major_part_size);
  const HloInstruction* output = &dot;
  // Currently supported is one fusion output and one path from dot to it.
  // Propagate dimension order from dot to root.
  while (!output->IsRoot()) {
    TF_RET_CHECK(output->user_count() == 1);
    const HloInstruction* input = output;
    output = output->users()[0];
    DimOrdersAndReqsOrError result =
        context.GetPropagatedDimOrdersAndRequirements(
            *output, context.dim_orders().at(input),
            TransformDirection::kInputToOutput, context.hero_properties());
    TF_RET_CHECK(std::holds_alternative<DimOrdersAndReqs>(result));
    TF_RET_CHECK(
        context.CombineDimOrdersAndReqs(std::get<DimOrdersAndReqs>(result)));
  }
  TF_RET_CHECK(iter_specs_[Scope::OUTPUT]
                   .insert({output, DimensionOrderToTensorIterationSpec(
                                        context.dim_orders().at(output))})
                   .second);
  if (output != &dot) {
    // Propagate back to parameters of the output fusion.
    TF_RETURN_IF_ERROR(context.PropagateDimensionOrdersToParameters(
        *output, parameters_[Scope::OUTPUT], iter_specs_[Scope::OUTPUT]));
  }
  return OkStatus();
}

const DimIterationSpec* TritonFusionAnalysis::IterSpec(
    const TritonFusionAnalysis::Scope scope, const HloInstruction* hlo,
    const int dimension) const {
  auto hlo_spec = iter_specs_.at(scope).find(hlo);
  if (hlo_spec != iter_specs_.at(scope).cend()) {
    auto dim_spec = hlo_spec->second.Storage().find(dimension);
    if (dim_spec != hlo_spec->second.Storage().cend()) {
      return &dim_spec->second;
    }
  }
  return nullptr;
}

FusionDecision CanTritonHandleGEMM(const HloInstruction& dot,
                                   const se::GpuComputeCapability gpu_version) {
  if (dot.opcode() != HloOpcode::kDot ||
      !tsl::tensor_float_32_execution_enabled() ||
      absl::c_any_of(dot.precision_config().operand_precision(),
                     [](int x) { return x != PrecisionConfig::DEFAULT; })) {
    return "Non-default precision.";
  }

  auto supported_output_type = [&](const PrimitiveType t) {
    const auto cuda_compute_capability =
        std::get<se::CudaComputeCapability>(gpu_version);
    switch (t) {
      case F16:
      case F32:
        return true;
      case BF16:
        return cuda_compute_capability.IsAtLeast(
            stream_executor::CudaComputeCapability::AMPERE);
      default:
        return false;
    }
  };

  // TODO(b/266862493): Support more output types.
  if (!supported_output_type(dot.shape().element_type())) {
    return "Unsupported output data type.";
  }

  if (!IsTritonSupportedDataType(dot.operand(0)->shape().element_type(),
                                 gpu_version) ||
      !IsTritonSupportedDataType(dot.operand(1)->shape().element_type(),
                                 gpu_version)) {
    return "Unsupported input data type.";
  }

  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  // TODO(b/269580541): support multiple batch dimensions.
  if (dim_numbers.lhs_batch_dimensions().size() > 1) {
    return "Multiple batch dimensions.";
  }

  // Cases where lhs or rhs have no non-contracting dims are not handled.
  if (dim_numbers.lhs_batch_dimensions().size() +
              dim_numbers.lhs_contracting_dimensions().size() ==
          dot.operand(0)->shape().rank() ||
      dim_numbers.rhs_batch_dimensions().size() +
              dim_numbers.rhs_contracting_dimensions().size() ==
          dot.operand(1)->shape().rank()) {
    return "No non-contracting dimensions.";
  }

  for (int operand_number = 0; operand_number <= 1; ++operand_number) {
    // This pass relies on dot decomposer which ensures that all non-contracting
    // dimensions are merged into one. Using NonContractingDimensionIndex is
    // sufficient.
    const int64_t nc_size =
        dot.operand(operand_number)
            ->shape()
            .dimensions(NonContractingDimensionIndex(dot, operand_number));
    if (nc_size <= 1) {
      return "Trivial non-contracting dimensions.";
    }
  }

  return FusionDecision{};
}

bool ShouldTritonHandleGEMM(HloInstruction& dot,
                            const se::GpuComputeCapability gpu_version) {
  std::vector<HloInstruction*> fusion_inputs;
  HloComputation::Builder builder("disposable");
  return FuseDot(dot, gpu_version, builder, fusion_inputs,
                 /*fusion_output_ptr=*/nullptr)
      ->CanFuse();
}

StatusOr<bool> GemmRewriterTriton::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result,
                        RunOnComputation(computation, gpu_version_));
    changed |= result;
  }
  return changed;
}

static std::string IterationSpecByInstructionMapToString(  // NOLINT
    const TritonFusionAnalysis::IterationSpecByInstructionMap& m) {
  return absl::StrCat("IterSpec{",
                      absl::StrJoin(m, ", ",
                                    [&](std::string* s, const auto& kv) {
                                      absl::StrAppend(s, kv.first->name(), ": ",
                                                      kv.second.ToString());
                                    }),
                      "}");
}

static std::string ScopeToString(TritonFusionAnalysis::Scope s) {  // NOLINT
  switch (s) {
    case TritonFusionAnalysis::Scope::LHS:
      return "LHS";
    case TritonFusionAnalysis::Scope::RHS:
      return "RHS";
    case TritonFusionAnalysis::Scope::OUTPUT:
      return "OUTPUT";
  }
}

std::string TensorIterationSpec::IterationSpecFragment::ToString() const {
  return absl::StrCat("{stride=", stride, ", count=", count,
                      ", slice_start=", slice_start, ", subfragments=[",
                      absl::StrJoin(subfragments, ", "), "]}");
}

bool TensorIterationSpec::IterationSpecFragment::operator!=(
    const IterationSpecFragment& other) const {
  return stride != other.stride || count != other.count ||
         slice_start != other.slice_start || slice_limit != other.slice_limit;
}

std::string TensorIterationSpec::ToString() const {
  return absl::StrCat(
      "{",
      absl::StrJoin(dim_iteration_specs_, ", ",
                    [&](std::string* s, const auto& kv) {
                      absl::StrAppend(
                          s, kv.first, ": ", "[",
                          absl::StrJoin(kv.second, ", ",
                                        [&](std::string* ss, const auto& v) {
                                          absl::StrAppend(ss, v.ToString());
                                        }),
                          "]");
                    }),
      "}");
}

std::string TritonFusionAnalysis::ToString() const {
  return absl::StrCat(
      "TritonFusionAnalysis{\n",
      absl::StrJoin(iter_specs_, ",\n",
                    [&](std::string* s, const auto& kv) {
                      absl::StrAppend(
                          s, ScopeToString(kv.first), ": ",
                          IterationSpecByInstructionMapToString(kv.second));
                    }),
      "\n}");
}

}  // namespace gpu
}  // namespace xla
