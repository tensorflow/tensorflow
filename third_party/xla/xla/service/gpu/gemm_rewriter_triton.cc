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
#include <queue>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
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

FusionDecision RequireTritonFusibleConvert(
    const HloInstruction* input, se::GpuComputeCapability gpu_version) {
  // TODO(b/266862494): Can pick up almost any
  // convert, but if it's reducing the data volume it should rather be fused
  // to the output of the producer kernel. However not all operations support
  // output fusion - then it should be fused here anyway!
  if (ShapeUtil::ByteSizeOf(input->operand(0)->shape()) >
      ShapeUtil::ByteSizeOf(input->shape())) {
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

struct DimOrderUpdates {
  DimOrderMap map;
  int64_t splittable_dimension_major_part_size = 0;
};

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

enum class TransformDirection { kInputToOutput, kOutputToInput };

using DimOrderUpdatesOrError = std::variant<FusionDecision, DimOrderUpdates>;

class FusionContext {
  struct DotProperties {
    int splittable_dimension;
    int64_t splittable_dimension_supported_major_part_size;
  };
  struct SoftmaxProperties {
    int softmax_reduction_dimension;
    int softmax_batch_dimension;
  };

  explicit FusionContext(DotProperties properties) : properties_(properties) {}

  explicit FusionContext(SoftmaxProperties properties)
      : properties_(properties) {}

  DimOrderUpdatesOrError HandleElementwise(const HloInstruction* hlo,
                                           const DimOrderMap& dim_orders) const;
  DimOrderUpdatesOrError HandleBitcast(const HloInstruction* hlo,
                                       const DimOrderMap& dim_orders,
                                       TransformDirection direction) const;
  DimOrderUpdatesOrError HandleDimensionAlteringOp(
      const HloInstruction* hlo, const DimOrderMap& dim_orders,
      TransformDirection direction) const;

 public:
  // Create fusion context from a dot operand according to
  // the currently supported configurations.
  static FusionContext FromDotOperand(const HloInstruction& dot,
                                      int operand_number, int split_k = 1);

  // Create fusion context from dot's output.
  static FusionContext FromDotOutput(
      const HloInstruction& dot, int split_k,
      int64_t splittable_dimension_supported_major_part_size);

  static FusionContext FromSoftmaxRoot(const HloInstruction&);

  DimOrderUpdatesOrError HandleInstruction(const HloInstruction* hlo,
                                           const DimOrderMap& dim_orders,
                                           TransformDirection direction) const;

  // Tells if the dimension order is supported by the triton emitters.
  // Only the dimension indicated by SplittableDimensionIndex() can be split
  // physically once by other dimensions. Other ones can be only split
  // logically. All subdimensions within a dimension have to be ordered.
  // Return major part of splittable dimension in split_dim_major_part if a
  // supported split is detected.
  FusionDecision RequireSupportedDimOrder(const DimensionOrder& order,
                                          int64_t& split_dim_major_part) const;
  // Apply RequireSupportedDimOrder() to all known dimension orders
  // around `hlo`.
  FusionDecision RequireSupportedDimOrders(const HloInstruction& hlo,
                                           DimOrderUpdates& updates) const;
  // Try to calculate transformations of dimensions defined by the
  // instruction, then check that the resulting dimension orders are supported.
  DimOrderUpdatesOrError RequireSupportedInstruction(
      const HloInstruction& hlo, const DimOrderMap& dim_orders,
      TransformDirection direction) const;
  // Checks if the instruction is possible and profitable to fuse.
  // If so tries to transform dim_order describing one side of `hlo` into
  // description(s) of its other side if it is supported.
  DimOrderUpdatesOrError AnalyzeForFusion(
      const HloInstruction& hlo, TransformDirection transform_direction,
      absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
          old_to_new_mapping,
      se::GpuComputeCapability gpu_version) const;
  // Add dimension orders from `updates` to `dim_orders_` and update the
  // splittable dimension ratio if all of them are compatible.
  bool MergeUpdates(const DimOrderUpdates& updates);
  // Fuse an instruction with all its fusible inputs.
  // If an input is not fusible stop there and make a parameter of the new
  // fusion, otherwise put it onto stack and check its own inputs first.
  void TryToFuseWithInputsRecursively(
      HloInstruction& root, se::GpuComputeCapability gpu_version,
      absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
          old_to_new_mapping,
      std::vector<HloInstruction*>& fusion_inputs,
      HloComputation::Builder& builder);
  // Propagate dimension orders in consumer->producer direction starting at
  // `origin` with output `origin_dim_order` till parameters of the computation.
  // Store the found parameters and their iteration specs.
  Status PropagateDimensionOrdersToParameters(
      const HloInstruction& origin, ConstHloInstructionSet& parameters,
      ConstHloInstructionMap<TensorIterationSpec>& iter_specs);

  // Index of dot dimension that can be split.
  // Currently typically LHS non-contracting one.
  int64_t SplittableDimensionIndex() const {
    CHECK(std::holds_alternative<DotProperties>(properties_));
    return std::get<DotProperties>(properties_).splittable_dimension;
  }
  // Tells whether `size` major part of a dimension can be physically split.
  bool IsSupportedSplittableDimensionMajorPartSize(const int64_t size) const {
    CHECK_NE(size, 0);
    CHECK(std::holds_alternative<DotProperties>(properties_));
    // 0 means no specific size requirement.
    return std::get<DotProperties>(properties_)
                   .splittable_dimension_supported_major_part_size == 0 ||
           std::get<DotProperties>(properties_)
                   .splittable_dimension_supported_major_part_size == size;
  }
  int SplittableDimensionMajorPartSize() const {
    CHECK(std::holds_alternative<DotProperties>(properties_));
    return std::get<DotProperties>(properties_)
        .splittable_dimension_supported_major_part_size;
  }
  const DimOrderMap& DimOrders() const { return dim_orders_; }

 private:
  DimOrderUpdatesOrError AnalyzeForFusionImpl(
      const HloInstruction& hlo, TransformDirection transform_direction,
      absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
          old_to_new_mapping,
      const DimOrderMap& dim_orders,
      se::GpuComputeCapability gpu_version) const;
  bool SetSplittableDimensionMajorPartSize(const int64_t size) {
    if (IsSupportedSplittableDimensionMajorPartSize(size)) {
      std::get<DotProperties>(properties_)
          .splittable_dimension_supported_major_part_size = size;
      return true;
    }
    return false;
  }

  std::variant<DotProperties, SoftmaxProperties> properties_;
  DimOrderMap dim_orders_;
};

FusionContext FusionContext::FromDotOperand(const HloInstruction& dot,
                                            const int operand_number,
                                            const int split_k) {
  // There can be either none or one split-K batch dimension.
  const int num_split_k_batch_dims = split_k > 1;
  int split_k_dimension_index = -1;
  if (split_k > 1) {
    split_k_dimension_index =
        ContractingDimensionIndex(dot, operand_number) - 1;
  }
  int splittable_dimension_index = -1;
  // LHS non-contracting dimension can be split if non-splitK batch is absent.
  if (operand_number == 0 &&
      dot.dot_dimension_numbers().lhs_batch_dimensions_size() -
              num_split_k_batch_dims ==
          0) {
    splittable_dimension_index =
        NonContractingDimensionIndex(dot, operand_number);
  }
  FusionContext context(FusionContext::DotProperties{
      splittable_dimension_index,
      /*splittable_dimension_supported_major_size=*/0});
  context.dim_orders_[dot.operand(operand_number)] =
      DimensionOrder::FromDotOperandOrOutput(*dot.operand(operand_number),
                                             split_k_dimension_index);
  return context;
}

FusionContext FusionContext::FromDotOutput(
    const HloInstruction& dot, const int split_k,
    const int64_t splittable_dimension_supported_major_part_size) {
  // Allow non-contracting dimension originating from LHS to split if
  // this dimension is split at the output at the same ratio as
  // at the input.
  int splittable_dimension_index = -1;
  if (splittable_dimension_supported_major_part_size > 1) {
    // Split-K dimension is the first one in the output if present;
    // LHS non-contracting follows (batch is absent in this case).
    splittable_dimension_index = (split_k > 1) ? 1 : 0;
  }
  FusionContext context(FusionContext::DotProperties{
      splittable_dimension_index,
      splittable_dimension_supported_major_part_size});
  context.dim_orders_[&dot] = DimensionOrder::FromDotOperandOrOutput(dot);
  return context;
}

FusionContext FusionContext::FromSoftmaxRoot(const HloInstruction& root) {
  FusionContext context(FusionContext::SoftmaxProperties{
      DimensionOrder::kSoftmaxReductionDimension,
      DimensionOrder::kSoftmaxBatchDimension});
  context.dim_orders_[&root] = DimensionOrder::FromSoftmaxRoot(root);
  return context;
}

FusionDecision FusionContext::RequireSupportedDimOrder(
    const DimensionOrder& order, int64_t& split_dim_major_part) const {
  VLOG(8) << order.ToString();
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
        if (dim_index == SplittableDimensionIndex() &&
            IsSupportedSplittableDimensionMajorPartSize(grouped_size)) {
          if (group_counter == 2) {
            if (split_dim_major_part != 0 &&
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
  return FusionDecision{};
}

FusionDecision FusionContext::RequireSupportedDimOrders(
    const HloInstruction& hlo, DimOrderUpdates& updates) const {
  auto check_if_present = [&](const HloInstruction* instr) {
    if (auto it = updates.map.find(instr); it != updates.map.end()) {
      return RequireSupportedDimOrder(
          it->second, updates.splittable_dimension_major_part_size);
    }
    return FusionDecision{};
  };
  for (const HloInstruction* operand : hlo.operands()) {
    if (auto result = check_if_present(operand); !result) {
      return result;
    }
  }
  return check_if_present(&hlo);
}

DimOrderUpdatesOrError FusionContext::HandleElementwise(
    const HloInstruction* hlo, const DimOrderMap& dim_orders) const {
  // The output and all the input dimension orders of `hlo` have to be the same.
  const HloInstruction* src = nullptr;
  const DimensionOrder* src_dim_order;
  // Try using the output as a reference if it's already described, otherwise
  // scan through all operands.
  if (auto it = dim_orders.find(hlo); it != dim_orders.cend()) {
    src = it->first;
    src_dim_order = &it->second;
  } else {
    for (const HloInstruction* operand : hlo->operands()) {
      if (auto it = dim_orders.find(operand); it != dim_orders.cend()) {
        src = it->first;
        src_dim_order = &it->second;
        break;
      }
    }
    CHECK_NE(src, nullptr);
  }

  DimOrderUpdates result;
  result.map.insert({hlo, DimensionOrder(*src_dim_order)});
  for (const HloInstruction* operand : hlo->operands()) {
    result.map.insert({operand, DimensionOrder(dim_orders.at(src))});
  }
  return result;
}

DimOrderUpdatesOrError FusionContext::HandleBitcast(
    const HloInstruction* hlo, const DimOrderMap& dim_orders,
    const TransformDirection direction) const {
  const HloInstruction* src =
      (direction == TransformDirection::kOutputToInput) ? hlo : hlo->operand(0);
  const HloInstruction* dst =
      (direction == TransformDirection::kOutputToInput) ? hlo->operand(0) : hlo;
  const Shape& dst_shape = dst->shape();
  const Fragments& src_fragments_order =
      dim_orders.at(src).TensorFragmentsOrder();
  DimOrderUpdates result;
  DimensionOrder& dst_dim_order =
      result.map.insert({dst, DimensionOrder()}).first->second;
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
    if (std::holds_alternative<SoftmaxProperties>(properties_) &&
        src_dim->dst_dim_number() ==
            std::get<SoftmaxProperties>(properties_).softmax_batch_dimension) {
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
       dim_orders.at(src).DimFragmentsOrders()) {
    std::vector<int>& dst = dst_dim_fragment_orders[dim_index];
    dst.reserve(dim_sequence.size());
    for (const int src : dim_sequence) {
      std::copy(src_to_dst[&src_fragments_order[src]].cbegin(),
                src_to_dst[&src_fragments_order[src]].cend(),
                std::back_inserter(dst));
    }
  }

  return result;
}

// Handle copy, transpose, broadcast or reduce.
// Common between them is that they alter the tensor dimensions or their order
// and the way to handle layouts.
DimOrderUpdatesOrError FusionContext::HandleDimensionAlteringOp(
    const HloInstruction* hlo, const DimOrderMap& dim_orders,
    const TransformDirection direction) const {
  // Temporary storage for new fragments local to this function.
  // Please keep this as the first local variable of this function, with type
  // std::list to make sure that all pointers to elements of this remain valid
  // throughout the entire function. std::deque would also work but it is
  // unnecessarily big for a typical size of 1.
  std::list<Fragment> new_fragments;

  const HloInstruction* src =
      (direction == TransformDirection::kOutputToInput) ? hlo : hlo->operand(0);
  const HloInstruction* dst =
      (direction == TransformDirection::kOutputToInput) ? hlo->operand(0) : hlo;
  // Note: copying instead of using a const reference because
  // some operations (slice) will modify fragment properties in-place.
  Fragments src_fragments_order = dim_orders.at(src).TensorFragmentsOrder();
  if (hlo->opcode() == HloOpcode::kSlice &&
      ShapeUtil::IsEffectiveScalar(hlo->shape())) {
    return FusionDecision("Slice to scalar is not implemented yet.");
  }
  DimOrderUpdates result;
  if (hlo->opcode() == HloOpcode::kReduce || hlo->opcode() == HloOpcode::kPad) {
    // Operand 1 (the neutral value or padding value) has to be a scalar.
    result.map.insert({hlo->operand(1), DimensionOrder()});
  }
  DimensionOrder& dst_dim_order =
      result.map.insert({dst, DimensionOrder()}).first->second;
  Fragments& dst_fragments_order = dst_dim_order.TensorFragmentsOrder();
  // Every HLO dimension can correspond to a group of subdimensions in
  // dim_order_. For the easier handling of permutations: group dim_order_ by
  // dimension, apply permutations, then finally remove the grouping.
  // Group subdimensions by iterating over them in the same order as over
  // full dimensions and matching by total size.
  std::vector<std::vector<Fragment*>> src_physical;
  src_physical.reserve(src->shape().rank());
  auto src_fragment_it = src_fragments_order.begin();
  for (int64_t dim_index : src->shape().layout().minor_to_major()) {
    const int64_t dim_size = src->shape().dimensions(dim_index);
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
    src_logical[src->shape().layout().minor_to_major(i)] = src_physical[i];
  }
  // Source logical -> destination logical.
  std::vector<std::vector<Fragment*>> dst_logical;
  if (hlo->opcode() == HloOpcode::kTranspose) {
    const auto* transpose = Cast<HloTransposeInstruction>(hlo);
    std::vector<int64_t> permutation(transpose->dimensions().cbegin(),
                                     transpose->dimensions().cend());
    if (direction == TransformDirection::kInputToOutput) {
      permutation = InversePermutation(permutation);
    }
    dst_logical.resize(permutation.size());
    for (int i = 0; i < permutation.size(); ++i) {
      dst_logical[permutation[i]] = src_logical[i];
    }
  } else if (hlo->opcode() == HloOpcode::kBroadcast) {
    const auto* broadcast = Cast<HloBroadcastInstruction>(hlo);
    dst_logical.resize(broadcast->dimensions().size());
    for (int i = 0; i < broadcast->dimensions().size(); ++i) {
      dst_logical[i] = src_logical[broadcast->dimensions()[i]];
    }
  } else if (hlo->opcode() == HloOpcode::kReduce) {
    const auto* reduce = Cast<HloReduceInstruction>(hlo);
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
            std::get<SoftmaxProperties>(properties_)
                .softmax_reduction_dimension,
            reduce->operand(0)->shape().dimensions(i))};
      } else {
        dst_logical[i] = src_logical[i];
      }
    }
  } else if (hlo->opcode() == HloOpcode::kCopy) {
    // Copy preserves the logical shape, just permutes the layout.
    CHECK(ShapeUtil::SameDimensions(src->shape(), dst->shape()));
    dst_logical = src_logical;
  } else if (hlo->opcode() == HloOpcode::kPad) {
    const auto* pad = Cast<HloPadInstruction>(hlo);
    dst_logical.resize(src_logical.size());
    for (int i = 0; i < src_logical.size(); ++i) {
      // This only handles the padding added by PadDotOperandsIfNeededForSplitK,
      // which sets only edge_padding_high.
      const int padding =
          pad->padding_config().dimensions(i).edge_padding_high();
      CHECK_EQ(pad->padding_config().dimensions(i).edge_padding_low(), 0);
      CHECK_EQ(pad->padding_config().dimensions(i).interior_padding(), 0);
      if (padding == 0) {
        dst_logical[i] = src_logical[i];
      } else {
        // This case is executed for the contracting dimension when we run the
        // TritonFusionAnalysis after the padding and the split-k transform are
        // applied.
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
  } else if (hlo->opcode() == HloOpcode::kSlice) {
    const auto slice = Cast<HloSliceInstruction>(hlo);
    dst_logical.resize(src_logical.size());
    for (int dim = 0; dim < src_logical.size(); ++dim) {
      dst_logical[dim] = src_logical[dim];
      if (slice->slice_limits(dim) - slice->slice_starts(dim) !=
          dst->shape().dimensions(dim)) {
        if (dst_logical[dim].size() > 1) {
          return FusionDecision("Slicing of fragmented dimension.");
        }
        dst_logical[dim].front()->set_size(dst->shape().dimensions(dim));
        dst_logical[dim].front()->set_slice(slice->slice_starts(dim),
                                            slice->slice_limits(dim));
      }
    }
  } else {
    return FusionDecision("Function called on a wrong instruction.");
  }
  // Destination logical -> destination physical and ungroup subdimensions.
  // Map original fragments to the resulting ones to derive their new
  // logical ordering within each dimension.
  absl::flat_hash_map<const Fragment*, int> src_to_dst;
  FragmentOrders& dst_dim_fragments_order = dst_dim_order.DimFragmentsOrders();
  // Remember which dimensions are present before a broadcast;
  // skip cases when already present dimension is being expanded.
  absl::flat_hash_set<int> dim_numbers_present_in_dst;
  for (const int64_t dim_idx : dst->shape().layout().minor_to_major()) {
    for (const Fragment* subdim : dst_logical[dim_idx]) {
      dst_fragments_order.push_back(*subdim);
      src_to_dst[subdim] = dst_fragments_order.size() - 1;
      dim_numbers_present_in_dst.insert(subdim->dst_dim_number());
      if (std::holds_alternative<SoftmaxProperties>(properties_) &&
          subdim->dst_dim_number() == std::get<SoftmaxProperties>(properties_)
                                          .softmax_reduction_dimension) {
        dst_dim_fragments_order[subdim->dst_dim_number()].push_back(
            dst_fragments_order.size() - 1);
      }
    }
  }
  for (const auto& [dim_index, dim_sequence] :
       dim_orders.at(src).DimFragmentsOrders()) {
    for (const int fragment_number : dim_sequence) {
      const auto it = src_to_dst.find(&src_fragments_order[fragment_number]);
      if (it == src_to_dst.cend()) {
        if (hlo->opcode() == HloOpcode::kBroadcast &&
            src_fragments_order[fragment_number].full_size() > 1 &&
            dim_numbers_present_in_dst.contains(dim_index)) {
          return FusionDecision("Unsupported broadcast");
        }
        continue;
      }
      dst_dim_fragments_order[dim_index].push_back(it->second);
    }
  }
  return result;
}

// Infers DimensionOrders of all unknown sides (output, operands)
// of `hlo` from the known ones.
DimOrderUpdatesOrError FusionContext::HandleInstruction(
    const HloInstruction* hlo, const DimOrderMap& dim_orders,
    const TransformDirection direction) const {
  VLOG(7) << "Analyzing " << hlo->ToString();
  if (hlo->opcode() == HloOpcode::kParameter ||
      hlo_query::IsScalarConstant(hlo)) {
    return DimOrderUpdates{};
  } else if (hlo->opcode() == HloOpcode::kTranspose ||
             hlo->opcode() == HloOpcode::kCopy) {
    return HandleDimensionAlteringOp(hlo, dim_orders, direction);
  } else if (hlo->opcode() == HloOpcode::kBroadcast) {
    if (direction != TransformDirection::kOutputToInput) {
      return "Unsupported broadcast direction.";
    }
    return HandleDimensionAlteringOp(hlo, dim_orders, direction);
  } else if (hlo->opcode() == HloOpcode::kReduce) {
    if (!std::holds_alternative<SoftmaxProperties>(properties_)) {
      return "Reductions are not supported in GEMM fusions yet.";
    }
    if (direction != TransformDirection::kOutputToInput) {
      return "Unsupported direction of reduction.";
    }
    return HandleDimensionAlteringOp(hlo, dim_orders, direction);
  } else if (hlo->opcode() == HloOpcode::kPad) {
    if (direction != TransformDirection::kOutputToInput) {
      return "Unsupported pad direction.";
    }
    return HandleDimensionAlteringOp(hlo, dim_orders, direction);
  } else if (hlo->operand_count() > 0 &&
             IsTritonSupportedElementwise(
                 hlo->opcode(), hlo->operand(0)->shape().element_type())) {
    return HandleElementwise(hlo, dim_orders);
  } else if (hlo->opcode() == HloOpcode::kBitcast) {
    return HandleBitcast(hlo, dim_orders, direction);
  } else if (hlo->opcode() == HloOpcode::kSlice) {
    if (direction != TransformDirection::kOutputToInput) {
      return "Unsupported slice direction.";
    }
    return HandleDimensionAlteringOp(hlo, dim_orders, direction);
  } else if (hlo->opcode() == HloOpcode::kReshape) {
    if (!ShapeUtil::ReshapeIsBitcast(hlo->operand(0)->shape(), hlo->shape())) {
      return "Non-bitcast reshape.";
    }
    return HandleBitcast(hlo, dim_orders, direction);
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

DimOrderUpdatesOrError FusionContext::RequireSupportedInstruction(
    const HloInstruction& hlo, const DimOrderMap& dim_orders,
    const TransformDirection transform_direction) const {
  auto result = HandleInstruction(&hlo, dim_orders, transform_direction);
  if (!std::holds_alternative<DimOrderUpdates>(result)) {
    return std::get<FusionDecision>(result);
  }

  if (FusionDecision supported =
          RequireSupportedDimOrders(hlo, std::get<DimOrderUpdates>(result));
      !supported) {
    return supported;
  }
  return std::get<DimOrderUpdates>(result);
}

DimOrderUpdatesOrError FusionContext::AnalyzeForFusion(
    const HloInstruction& hlo, const TransformDirection transform_direction,
    absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
        old_to_new_mapping,
    const se::GpuComputeCapability gpu_version) const {
  return AnalyzeForFusionImpl(hlo, transform_direction, old_to_new_mapping,
                              dim_orders_, gpu_version);
}

DimOrderUpdatesOrError FusionContext::AnalyzeForFusionImpl(
    const HloInstruction& hlo, const TransformDirection transform_direction,
    absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
        old_to_new_mapping,
    const DimOrderMap& dim_orders,
    const se::GpuComputeCapability gpu_version) const {
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
  DimOrderUpdatesOrError result =
      RequireSupportedInstruction(hlo, dim_orders, transform_direction);
  if (!std::holds_alternative<DimOrderUpdates>(result)) {
    return result;
  }
  int fusion_level =
      hlo.GetModule()->config().debug_options().xla_gpu_triton_fusion_level();
  if (!std::get<se::CudaComputeCapability>(gpu_version)
           .IsAtLeast(se::CudaComputeCapability::AMPERE)) {
    fusion_level = std::min(fusion_level, 1);
  }
  if (transform_direction == TransformDirection::kOutputToInput) {
    if (fusion_level < 2) {
      if (hlo.opcode() == HloOpcode::kConvert) {
        if (FusionDecision decision =
                RequireTritonFusibleConvert(&hlo, gpu_version);
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
              std::holds_alternative<DimOrderUpdates>(AnalyzeForFusionImpl(
                  *operand, transform_direction, old_to_new_mapping,
                  std::get<DimOrderUpdates>(result).map, gpu_version))) {
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
    for (const HloInstruction* operand : hlo.operands()) {
      // Skip already fused operands.
      if (old_to_new_mapping.contains(operand)) {
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
  return std::get<DimOrderUpdates>(result);
}

// Clone an instruction into the fusion.
void Fuse(HloInstruction& hlo,
          absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
              old_to_new_mapping,
          std::vector<HloInstruction*>& fusion_inputs,
          HloComputation::Builder& builder) {
  if (old_to_new_mapping.contains(&hlo)) {
    return;
  }
  VLOG(3) << "Fusing " << hlo.ToString();
  auto get_or_add_parameter = [&](HloInstruction& instr) {
    if (auto it = old_to_new_mapping.find(&instr);
        it != old_to_new_mapping.end()) {
      return it->second;
    }
    fusion_inputs.push_back(&instr);
    return old_to_new_mapping
        .insert({&instr,
                 builder.AddInstruction(HloInstruction::CreateParameter(
                     fusion_inputs.size() - 1, instr.shape(),
                     absl::StrCat("parameter_", fusion_inputs.size() - 1)))})
        .first->second;
  };
  if (hlo.opcode() == HloOpcode::kParameter ||
      hlo.opcode() == HloOpcode::kGetTupleElement) {
    get_or_add_parameter(hlo);
  } else {
    std::vector<HloInstruction*> hlo_new_operands;
    for (HloInstruction* operand : hlo.operands()) {
      hlo_new_operands.push_back(get_or_add_parameter(*operand));
    }
    old_to_new_mapping[&hlo] = builder.AddInstruction(
        hlo.CloneWithNewOperands(hlo.shape(), hlo_new_operands));
  }
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

bool FusionContext::MergeUpdates(const DimOrderUpdates& updates) {
  // First check that all updates to insert are compatible to avoid
  // incomplete merges.
  for (const auto& [key, value] : updates.map) {
    auto it = dim_orders_.find(key);
    if (it != dim_orders_.cend() && !it->second.IsPhysicallyEquivalent(value)) {
      return false;
    }
  }
  if (updates.splittable_dimension_major_part_size > 1 &&
      !SetSplittableDimensionMajorPartSize(
          updates.splittable_dimension_major_part_size)) {
    return false;
  }
  dim_orders_.insert(updates.map.begin(), updates.map.end());
  return true;
}

void FusionContext::TryToFuseWithInputsRecursively(
    HloInstruction& root, const se::GpuComputeCapability gpu_version,
    absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
        old_to_new_mapping,
    std::vector<HloInstruction*>& fusion_inputs,
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
    if (inputs.size() >= TritonFusionAnalysis::kMaxParameterPerScope &&
        NumAddedParameters(*hlo) > 0) {
      // Re-queue: the number of parameters may go down when other instructions
      // are processed.
      to_visit.push(hlo);
      // Prevent infinite loops.
      ++num_requeued;
      continue;
    }
    num_requeued = 0;
    const DimOrderUpdatesOrError result =
        AnalyzeForFusion(*hlo, TransformDirection::kOutputToInput,
                         old_to_new_mapping, gpu_version);
    if (!std::holds_alternative<DimOrderUpdates>(result) ||
        !MergeUpdates(std::get<DimOrderUpdates>(result))) {
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
        Fuse(**it, old_to_new_mapping, fusion_inputs, builder);
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

  // Original instruction -> fused one.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      old_to_new_mapping;

  // Separate traversal from LHS and RHS inputs of the dot: they use
  // differently shaped tiles but may go through same HLO graph nodes.
  // Direct dot inputs have well defined dimension orders.

  auto fuse_inputs = [&](int operand_number) -> StatusOr<FusionContext> {
    const int operand_count_before = fusion_inputs.size();
    // Direct dot inputs have well defined dimension orders.
    auto context = FusionContext::FromDotOperand(dot, operand_number);
    context.TryToFuseWithInputsRecursively(*dot.mutable_operand(operand_number),
                                           gpu_version, old_to_new_mapping,
                                           fusion_inputs, builder);
    TF_RET_CHECK(fusion_inputs.size() - operand_count_before <=
                 TritonFusionAnalysis::kMaxParameterPerScope);
    return context;
  };

  TF_ASSIGN_OR_RETURN(const FusionContext lhs_context, fuse_inputs(0));
  if (auto result = fuse_inputs(1); !result.ok()) {
    return result.status();
  }

  Fuse(dot, old_to_new_mapping, fusion_inputs, builder);

  // Fusion at dot's output.

  // These describe _outputs_ of corresponding HLOs.
  auto context = FusionContext::FromDotOutput(
      dot, /*split_k=*/1, lhs_context.SplittableDimensionMajorPartSize());
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
    auto result =
        context.AnalyzeForFusion(*user, TransformDirection::kInputToOutput,
                                 old_to_new_mapping, gpu_version);
    if (!std::holds_alternative<DimOrderUpdates>(result)) {
      continue;
    }
    TF_RET_CHECK(context.MergeUpdates(std::get<DimOrderUpdates>(result)));
    for (HloInstruction* operand : user->operands()) {
      if (!old_to_new_mapping.contains(operand)) {
        context.TryToFuseWithInputsRecursively(
            *operand, gpu_version, old_to_new_mapping, fusion_inputs, builder);
      }
    }
    Fuse(*user, old_to_new_mapping, fusion_inputs, builder);
    fusion_output = user;
    output_changed = true;
  }
  if (fusion_output_ptr != nullptr) {
    *fusion_output_ptr = fusion_output;
  }
  if (dot.GetModule()->config().debug_options().xla_gpu_triton_gemm_any()) {
    return FusionDecision{};
  }
  for (const auto& iter : old_to_new_mapping) {
    if (iter.second->opcode() == HloOpcode::kConvert ||
        iter.second->opcode() == HloOpcode::kSlice ||
        iter.second->opcode() == HloOpcode::kTranspose) {
      return FusionDecision{};
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
    auto result =
        HandleInstruction(hlo, dim_orders_, TransformDirection::kOutputToInput);
    TF_RET_CHECK(std::holds_alternative<DimOrderUpdates>(result));
    TF_RET_CHECK(
        RequireSupportedDimOrders(*hlo, std::get<DimOrderUpdates>(result)));
    TF_RET_CHECK(MergeUpdates(std::get<DimOrderUpdates>(result)));
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

}  // anonymous namespace

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
  int64_t lhs_nc_split_major_part_size = -1;
  for (const Scope scope : {Scope::LHS, Scope::RHS}) {
    const int operand_number = static_cast<int>(scope);
    auto context = FusionContext::FromDotOperand(dot, operand_number, split_k);
    TF_RETURN_IF_ERROR(context.PropagateDimensionOrdersToParameters(
        *dot.operand(operand_number), parameters_[scope], iter_specs_[scope]));
    if (scope == Scope::LHS && context.SplittableDimensionMajorPartSize() > 1) {
      lhs_nc_split_major_part_size = context.SplittableDimensionMajorPartSize();
    }
  }

  auto context =
      FusionContext::FromDotOutput(dot, split_k, lhs_nc_split_major_part_size);
  const HloInstruction* output = &dot;
  // Currently supported is one fusion output and one path from dot to it.
  // Propagate dimension order from dot to root.
  while (!output->IsRoot()) {
    TF_RET_CHECK(output->user_count() == 1);
    output = output->users()[0];
    auto result = context.HandleInstruction(output, context.DimOrders(),
                                            TransformDirection::kInputToOutput);
    TF_RET_CHECK(std::holds_alternative<DimOrderUpdates>(result));
    TF_RET_CHECK(context.RequireSupportedDimOrders(
        *output, std::get<DimOrderUpdates>(result)));
    TF_RET_CHECK(context.MergeUpdates(std::get<DimOrderUpdates>(result)));
  }
  TF_RET_CHECK(iter_specs_[Scope::OUTPUT]
                   .insert({output, DimensionOrderToTensorIterationSpec(
                                        context.DimOrders().at(output))})
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
