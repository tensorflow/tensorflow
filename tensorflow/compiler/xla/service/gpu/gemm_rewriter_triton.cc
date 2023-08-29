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

#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/autotuning.pb.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_schedule.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_query.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_padding_requirements.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

bool TensorIterationSpec::operator==(const TensorIterationSpec& other) const {
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
      if (it_this->second.size() != it_other->second.size()) {
        return false;
      }
      if (it_this->second[fragment].stride !=
              it_other->second[fragment].stride ||
          it_this->second[fragment].count != it_other->second[fragment].count) {
        return false;
      }
    }
    ++it_this;
  }
  return true;
}

namespace {

// Batch dimensions of an operand of a dot instruction.
// Just an unified accessor to lhs_batch_dimensions and rhs_batch_dimensions.
const tsl::protobuf::RepeatedField<int64_t>& BatchDimensionsForOperand(
    const HloInstruction& dot, const int operand_number) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  if (operand_number == 0) {
    return dimension_numbers.lhs_batch_dimensions();
  }
  return dimension_numbers.rhs_batch_dimensions();
}

// Index of the only contracting dimension of dot instruction operand.
int64_t ContractingDimensionIndex(const HloInstruction& dot,
                                  const int operand_number) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  if (operand_number == 0) {
    CHECK_EQ(dimension_numbers.lhs_contracting_dimensions().size(), 1);
    return dimension_numbers.lhs_contracting_dimensions(0);
  }
  CHECK_EQ(dimension_numbers.rhs_contracting_dimensions().size(), 1);
  return dimension_numbers.rhs_contracting_dimensions(0);
}

// Index of the only non-contracting dimension of dot instruction operand.
int64_t NonContractingDimensionIndex(const HloInstruction& dot,
                                     const int operand_number) {
  StatusOr<std::vector<int64_t>> non_contracting_dims =
      GetNonContractingDims(dot.operand(operand_number)->shape(),
                            BatchDimensionsForOperand(dot, operand_number),
                            {ContractingDimensionIndex(dot, operand_number)});
  TF_CHECK_OK(non_contracting_dims.status());
  CHECK_EQ(non_contracting_dims->size(), 1);
  return non_contracting_dims->front();
}

// Tells if f(a+b) == f(a) + f(b).
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

FusionDecision RequireTritonFusibleConvert(const HloInstruction* input,
                                           GpuVersion gpu_version) {
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
            dim_order.tensor_fragments_order_.back().dst_dim_number;
      }
      dim_order.dim_fragments_orders_[target_dim_number].push_back(
          dim_order.tensor_fragments_order_.size());
      dim_order.tensor_fragments_order_.push_back(
          {target_dim_number, hlo.shape().dimensions(i)});
    }
    return dim_order;
  }

  // Description of a continuous fragment of one dimension of a tensor.
  struct Fragment {
    // Label carrying the dimension number of an defining operation.
    int dst_dim_number;
    // Number of elements in the fragment.
    int64_t size;
    std::string ToString() const {
      return absl::StrCat(dst_dim_number, ":", size);
    }
    Fragment(int dst_dim_number, int64_t size)
        : dst_dim_number(dst_dim_number), size(size) {}
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
    VLOG(6) << fragment.dst_dim_number << "\t" << fragment.size;

    DimIterationSpec& dim_spec = tensor_spec[fragment.dst_dim_number];
    if (last_dim == fragment.dst_dim_number) {
      // Contiguous dimension, split only logically. Merge it back.
      if (!dim_spec.empty() && !dim_spec.back().subfragments.empty() &&
          dim_spec.back().subfragments.back() == 1) {
        // Remove previous 1-sized subfragment.
        dim_spec.back().subfragments.pop_back();
      }
      if (fragment.size > 1) {
        CHECK(!dim_spec.empty());
        dim_spec.back().count *= fragment.size;
        dim_spec.back().subfragments.push_back(fragment.size);
      }
    } else {
      remove_last_fragment_if_degenerate(last_dim);
      // Add part of the dimension.
      dim_spec.push_back({accumulated_stride, fragment.size, {fragment.size}});
    }

    accumulated_stride *= fragment.size;
    last_dim = fragment.dst_dim_number;
  }
  remove_last_fragment_if_degenerate(last_dim);
  // Create all absent dimensions as degenerate ones to simplify later queries.
  for (auto& [dim_idx, dim_spec] : tensor_spec) {
    if (dim_spec.empty()) {
      dim_spec.push_back({/*stride=*/0, /*count=*/1, /*subfragments=*/{1}});
    }
  }
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

  explicit FusionContext(DotProperties properties) : properties_(properties) {}

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
  // Checks if the instruction is possible and profitable to fuse.
  // If so tries to transform dim_order describing one side of `hlo` into
  // description(s) of its other side if it is supported.
  DimOrderUpdatesOrError AnalyzeForFusion(
      const HloInstruction& hlo, bool as_input,
      absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
          old_to_new_mapping,
      GpuVersion gpu_version) const;
  // Add dimension orders from `updates` to `dim_orders_` and update the
  // splittable dimension ratio if all of them are compatible.
  bool MergeUpdates(const DimOrderUpdates& updates);
  // Fuse an instruction with all its fusible inputs.
  // If an input is not fusible stop there and make a parameter of the new
  // fusion, otherwise put it onto stack and check its own inputs first.
  void TryToFuseWithInputsRecursively(
      HloInstruction& root, GpuVersion gpu_version,
      absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
          old_to_new_mapping,
      std::vector<HloInstruction*>& fusion_inputs,
      HloComputation::Builder& builder);
  // Propagate dimension orders in consumer->producer direction starting at
  // `origin` with output `origin_dim_order` till parameters of the computation.
  // Store the found parameters and their iteration specs.
  Status PropagateDimensionOrdersToParameters(
      const HloInstruction& origin,
      absl::flat_hash_set<const HloInstruction*>& parameters,
      absl::flat_hash_map<const HloInstruction*, TensorIterationSpec>&
          iter_specs);

  // Index of dot dimension that can be split.
  // Currently typically LHS non-contracting one.
  int64_t SplittableDimensionIndex() const {
    return properties_.splittable_dimension;
  }
  // Tells whether `size` major part of a dimension can be physically split.
  bool IsSupportedSplittableDimensionMajorPartSize(const int64_t size) const {
    CHECK_NE(size, 0);
    // 0 means no specific size requirement.
    return properties_.splittable_dimension_supported_major_part_size == 0 ||
           properties_.splittable_dimension_supported_major_part_size == size;
  }
  int SplittableDimensionMajorPartSize() const {
    return properties_.splittable_dimension_supported_major_part_size;
  }
  const DimOrderMap& DimOrders() const { return dim_orders_; }

 private:
  bool SetSplittableDimensionMajorPartSize(const int64_t size) {
    if (IsSupportedSplittableDimensionMajorPartSize(size)) {
      properties_.splittable_dimension_supported_major_part_size = size;
      return true;
    }
    return false;
  }

  DotProperties properties_;
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

FusionDecision FusionContext::RequireSupportedDimOrder(
    const DimensionOrder& order, int64_t& split_dim_major_part) const {
  VLOG(8) << order.ToString();
  const Fragments& tensor_dim_fragments = order.TensorFragmentsOrder();
  for (const auto& [dim_index, dim_fragments] : order.DimFragmentsOrders()) {
    int split_counter = -1;
    auto fragment = dim_fragments.cbegin();
    while (true) {
      if (fragment == dim_fragments.cend()) {
        break;
      }
      int64_t grouped_size = tensor_dim_fragments[*fragment].size;
      // Gather contiguous fragments.
      while ((fragment + 1) != dim_fragments.cend() &&
             *(fragment + 1) == *fragment + 1) {
        ++fragment;
        grouped_size *= tensor_dim_fragments[*fragment].size;
      }

      if (grouped_size == 1) {
        ++fragment;
        continue;
      }

      if (fragment != dim_fragments.cbegin() && *fragment < *(fragment - 1)) {
        return "Transpose within a dimension.";
      }

      ++split_counter;
      if (split_counter > 0) {
        if (dim_index == SplittableDimensionIndex() &&
            IsSupportedSplittableDimensionMajorPartSize(grouped_size)) {
          if (split_counter == 1) {
            if (split_dim_major_part != 0 &&
                split_dim_major_part != grouped_size) {
              return "Conflicting splits of splittable dimension";
            }
            split_dim_major_part = grouped_size;
          } else if (split_counter > 1) {
            return "2nd split of a splittable dimension.";
          }
        } else {
          return "Unsupported split of a dimension.";
        }
      }

      ++fragment;
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
  auto dst_dim_iter = dst_shape.layout().minor_to_major().cbegin();
  for (auto src_dim = src_fragments_order.cbegin();
       src_dim != src_fragments_order.cend(); ++src_dim) {
    auto add = [&](const Fragment& fragment) {
      dst_fragments_order.push_back(fragment);
      src_to_dst[&*src_dim].push_back(dst_fragments_order.size() - 1);
    };
    if (dst_remaining_size >= src_dim->size) {
      if (dst_remaining_size % src_dim->size) {
        return "Unsupported bitcast";
      }
      // Source dimension fragment completely fits into the destination one:
      // just copy it as is.
      add(*src_dim);
      // Update the size of the remaining part of the destination that is
      // carried over to next source dimensions.
      dst_remaining_size /= src_dim->size;
    } else {
      // Source is larger than destination.
      // Assign further destination dimensions.
      // Size of the not yet assigned part of the source dimension.
      int64_t src_remaining_size = src_dim->size;
      // Handle dimension splits.
      if (dst_remaining_size > 1) {
        // If there is a remaining fragment of a previous destination dimension
        // assign it first.
        if (src_remaining_size % dst_remaining_size) {
          return "Unsupported bitcast";
        }
        add({src_dim->dst_dim_number, dst_remaining_size});
        // Update the size of the fragment remaining to assign.
        src_remaining_size /= dst_remaining_size;
        dst_remaining_size = 1;
      }
      while (src_remaining_size > 1) {
        // Assign destination dimensions until the source remainder is covered.
        int64_t dst_dim_size = dst_shape.dimensions(*dst_dim_iter);
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
        add({src_dim->dst_dim_number, new_fragment_size});
        src_remaining_size /= new_fragment_size;
        ++dst_dim_iter;
      }
    }
  }
  CHECK_EQ(dst_remaining_size, 1);

  // Handle remaining major dimensions of the destination. Call all degenerate
  // ones subdimensions of the most-major non-degenerate one. Otherwise
  // give up.
  while (dst_dim_iter != dst_shape.layout().minor_to_major().cend()) {
    if (dst_shape.dimensions(*dst_dim_iter) != 1) {
      return "Unsupported bitcast";
    }
    dst_fragments_order.push_back(
        {dst_fragments_order.back().dst_dim_number, 1});
    src_to_dst[&src_fragments_order.back()].push_back(
        dst_fragments_order.size() - 1);
    ++dst_dim_iter;
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

// Handle copy, transpose or broadcast.
// Common between them is that they alter the tensor dimensions or their order
// and the way to handle layouts.
DimOrderUpdatesOrError FusionContext::HandleDimensionAlteringOp(
    const HloInstruction* hlo, const DimOrderMap& dim_orders,
    const TransformDirection direction) const {
  const HloInstruction* src =
      (direction == TransformDirection::kOutputToInput) ? hlo : hlo->operand(0);
  const HloInstruction* dst =
      (direction == TransformDirection::kOutputToInput) ? hlo->operand(0) : hlo;
  const Fragments& src_fragments_order =
      dim_orders.at(src).TensorFragmentsOrder();
  DimOrderUpdates result;
  DimensionOrder& dst_dim_order =
      result.map.insert({dst, DimensionOrder()}).first->second;
  Fragments& dst_fragments_order = dst_dim_order.TensorFragmentsOrder();
  // Every HLO dimension can correspond to a group of subdimensions in
  // dim_order_. For the easier handling of permutations: group dim_order_ by
  // dimension, apply permutations, then finally remove the grouping.
  // Group subdimensions by iterating over them in the same order as over
  // full dimensions and matching by total size.
  std::vector<std::vector<const Fragment*>> src_physical;
  src_physical.reserve(src->shape().rank());
  auto dim_order_it = src_fragments_order.cbegin();
  for (int64_t dim_index : src->shape().layout().minor_to_major()) {
    const int64_t dim_size = src->shape().dimensions(dim_index);
    int64_t subdim_size_accumulator = 1;
    std::vector<const Fragment*> subdim_group;
    do {
      subdim_size_accumulator *= dim_order_it->size;
      subdim_group.push_back(&*dim_order_it);
      ++dim_order_it;
    } while (subdim_size_accumulator < dim_size);
    CHECK_EQ(subdim_size_accumulator, dim_size);
    src_physical.push_back(subdim_group);
  }
  // Source physical -> source logical.
  std::vector<std::vector<const Fragment*>> src_logical;
  src_logical.resize(src_physical.size());
  for (int i = 0; i < src_physical.size(); ++i) {
    src_logical[src->shape().layout().minor_to_major(i)] = src_physical[i];
  }
  // Source logical -> destination logical.
  std::vector<std::vector<const Fragment*>> dst_logical;
  if (hlo->opcode() == HloOpcode::kTranspose) {
    const auto transpose = Cast<HloTransposeInstruction>(hlo);
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
    const auto broadcast = Cast<HloBroadcastInstruction>(hlo);
    dst_logical.resize(broadcast->dimensions().size());
    for (int i = 0; i < broadcast->dimensions().size(); ++i) {
      dst_logical[i] = src_logical[broadcast->dimensions()[i]];
    }
  } else {
    // Copy preserves the logical shape, just permutes the layout.
    CHECK(ShapeUtil::SameDimensions(src->shape(), dst->shape()));
    dst_logical = src_logical;
  }
  // Destination logical -> destination physical and ungroup subdimensions.
  // Map original fragments to the resulting ones to derive their new
  // logical ordering within each dimension.
  absl::flat_hash_map<const Fragment*, int> src_to_dst;
  for (const int64_t dim_idx : dst->shape().layout().minor_to_major()) {
    for (const Fragment* subdim : dst_logical[dim_idx]) {
      dst_fragments_order.push_back(*subdim);
      src_to_dst[subdim] = dst_fragments_order.size() - 1;
    }
  }
  FragmentOrders& dst_dim_fragments_order = dst_dim_order.DimFragmentsOrders();
  for (const auto& [dim_index, dim_sequence] :
       dim_orders.at(src).DimFragmentsOrders()) {
    for (const int fragment_number : dim_sequence) {
      const auto it = src_to_dst.find(&src_fragments_order[fragment_number]);
      if (it == src_to_dst.cend()) {
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
  VLOG(7) << hlo->ToString();
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
  } else if (hlo->operand_count() > 0 &&
             IsTritonSupportedElementwise(
                 hlo->opcode(), hlo->operand(0)->shape().element_type())) {
    return HandleElementwise(hlo, dim_orders);
  } else if (hlo->opcode() == HloOpcode::kBitcast) {
    return HandleBitcast(hlo, dim_orders, direction);
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
  return hlo.user_count() == 1 &&
         hlo_query::AllOperandsAreParametersOrConstantsWithSingleUser(hlo);
}

// Tells that fusing an instruction as an output is efficient.
bool IsOutputWorthFusing(const HloInstruction& hlo) {
  return CanNotBeFusedIntoAUser(hlo) ||
         InputMinusOutputBytes(hlo) >= -kIoToleranceBytes;
}

DimOrderUpdatesOrError FusionContext::AnalyzeForFusion(
    const HloInstruction& hlo, bool as_input,
    absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
        old_to_new_mapping,
    const GpuVersion gpu_version) const {
  int fusion_level =
      hlo.GetModule()->config().debug_options().xla_gpu_triton_fusion_level();
  if (!std::get<se::CudaComputeCapability>(gpu_version)
           .IsAtLeast(se::CudaComputeCapability::AMPERE)) {
    fusion_level = std::min(fusion_level, 1);
  }
  if (hlo.opcode() == HloOpcode::kTuple ||
      hlo.opcode() == HloOpcode::kGetTupleElement) {
    return "Unsupported instruction.";
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
  if (hlo.opcode() == HloOpcode::kBroadcast &&
      !hlo_query::IsScalarConstant(hlo.operand(0))) {
    return "Skipping unsupported broadcast.";
  }
  if (as_input) {
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
      if (!IsInputWorthFusing(hlo)) {
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

  auto result =
      HandleInstruction(&hlo, dim_orders_,
                        as_input ? TransformDirection::kOutputToInput
                                 : TransformDirection::kInputToOutput);
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
    HloInstruction& root, const GpuVersion gpu_version,
    absl::flat_hash_map<const HloInstruction*, HloInstruction*>&
        old_to_new_mapping,
    std::vector<HloInstruction*>& fusion_inputs,
    HloComputation::Builder& builder) {
  absl::flat_hash_set<const HloInstruction*> visited;
  std::stack<HloInstruction*> to_fuse;
  // Instructions at the edge of 'to_fuse' that can either get fused too or
  // become parameters of the fusion. Used to track the number of parameters
  // of the fusion.
  absl::flat_hash_set<const HloInstruction*> inputs;
  // Currently only one physically unique dim order per scope is supported.
  // Let it change while the scope has one input; afterwards require all
  // of them to be physically compatible.
  const HloInstruction* reference_dim_order_hlo = nullptr;
  auto try_fuse_one = [&](HloInstruction& hlo) {
    const DimOrderUpdatesOrError result = AnalyzeForFusion(
        hlo, /*as_input=*/true, old_to_new_mapping, gpu_version);
    if (!std::holds_alternative<DimOrderUpdates>(result)) {
      return false;
    }
    for (const HloInstruction* operand : hlo.operands()) {
      const DimensionOrder& dim_order =
          std::get<DimOrderUpdates>(result).map.at(operand);
      if (reference_dim_order_hlo != nullptr &&
          !dim_order.IsPhysicallyEquivalent(
              dim_orders_.at(reference_dim_order_hlo))) {
        return false;
      }
    }
    if (!MergeUpdates(std::get<DimOrderUpdates>(result))) {
      return false;
    }
    to_fuse.push(&hlo);
    if (hlo.opcode() != HloOpcode::kParameter) {
      inputs.erase(&hlo);
    }
    inputs.insert(hlo.operands().cbegin(), hlo.operands().cend());
    return true;
  };
  try_fuse_one(root);
  visited.insert(&root);
  while (!to_fuse.empty()) {
    bool top_is_ready_to_fuse = true;
    HloInstruction* hlo = to_fuse.top();
    if (reference_dim_order_hlo == nullptr && hlo->operand_count() > 1) {
      reference_dim_order_hlo = hlo;
    }
    for (HloInstruction* operand : hlo->mutable_operands()) {
      if (visited.insert(operand).second) {
        // Stop adding new parameters.
        if (inputs.size() >= TritonFusionAnalysis::kMaxParameterPerScope &&
            NumAddedParameters(*operand) > 0) {
          continue;
        }
        if (try_fuse_one(*operand)) {
          top_is_ready_to_fuse = false;
        }
      }
    }
    if (top_is_ready_to_fuse) {
      Fuse(*hlo, old_to_new_mapping, fusion_inputs, builder);
      to_fuse.pop();
    }
  }
}

// Fuses dot and the compatible and profitable to fuse operations around it
// into a new fusion computation constructed using the builder. fusion_inputs
// get populated with the non-fused instructions that become operands of the
// call to this fusion. fusion_output_ptr (if not nullptr) gets assigned the
// original instruction that has to be replaced by the call to the fusion.
StatusOr<FusionDecision> FuseDot(HloInstruction& dot,
                                 const GpuVersion gpu_version,
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
    auto result = context.AnalyzeForFusion(*user, /*as_input=*/false,
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
  explicit GemmRewriterTritonVisitor(const GpuVersion gpu_version)
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
  GpuVersion gpu_version_;
};

StatusOr<bool> RunOnComputation(HloComputation* computation,
                                GpuVersion gpu_version) {
  GemmRewriterTritonVisitor visitor(gpu_version);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

// Copy source values into destination incrementing those >= threshold by 1.
void CopyIncrementingAboveThreshold(
    const tsl::protobuf::RepeatedField<int64_t>& source,
    tsl::protobuf::RepeatedField<int64_t>& destination, const int threshold) {
  destination.Reserve(source.size());
  for (int64_t x : source) {
    if (x >= threshold) {
      ++x;
    }
    destination.Add(x);
  }
}

// Copy source values into destination incrementing those >= threshold by 1.
void CopyIncrementingAboveThreshold(absl::Span<const int64_t> source,
                                    DimensionVector& destination,
                                    const int threshold) {
  destination.reserve(source.size());
  for (int64_t x : source) {
    if (x >= threshold) {
      ++x;
    }
    destination.push_back(x);
  }
}

Status UncompilableMatmul(absl::string_view explanation) {
  Status s = absl::CancelledError(explanation);
  s.SetPayload(kUncompilableFusion, absl::Cord(explanation));
  return s;
}

StatusOr<HloInstruction*> MakeSplitKOperand(
    HloInstruction& dot, const TritonFusionAnalysis& analysis,
    const AutotuneResult::TritonGemmKey& tiling,
    const int64_t contracting_dim_idx, const int operand_number) {
  const Shape& shape = dot.operand(operand_number)->shape();
  Shape new_shape(shape.element_type(), {}, {}, {});

  // TODO(b/274775195): implement split-K with padding.
  if (tiling.split_k() > shape.dimensions(contracting_dim_idx)) {
    return UncompilableMatmul("Too small total contracting dimension size.");
  }
  TritonFusionAnalysis::Scope scope = (operand_number == 0)
                                          ? TritonFusionAnalysis::Scope::LHS
                                          : TritonFusionAnalysis::Scope::RHS;
  for (const HloInstruction* param : analysis.ScopeParameters(scope)) {
    // If an operand of dot does not read any parameters its K dimension
    // does not need analysis for fragmentation.
    const DimIterationSpec* spec =
        analysis.IterSpec(scope, param, contracting_dim_idx);
    if (spec == nullptr) {
      // No contracting dimension in the parameter - no checks needed.
      continue;
    }
    if (spec->size() != 1) {
      return UncompilableMatmul("Unsupported case.");
    }
    auto fragment = spec->at(0).subfragments.crbegin();
    int64_t size_to_split = tiling.split_k();
    while (size_to_split > *fragment) {
      if (size_to_split % *fragment) {
        return UncompilableMatmul("Contracting dimension is too fragmented.");
      }
      size_to_split /= *fragment;
      ++fragment;
    }
    if (*fragment % size_to_split) {
      return UncompilableMatmul("Contracting dimension is too fragmented.");
    }
    if (tiling.split_k() > ceil(1.0 * spec->at(0).count / tiling.block_k())) {
      return UncompilableMatmul(
          "Too small divisible part of the contracting dimension.");
    }
  }

  for (int i = 0; i < shape.rank(); ++i) {
    const int64_t dimension_size = shape.dimensions(i);
    if (i == contracting_dim_idx) {
      new_shape.add_dimensions(tiling.split_k());
      new_shape.add_dimensions(dimension_size / tiling.split_k());
    } else {
      new_shape.add_dimensions(dimension_size);
    }
  }

  Layout* new_layout = new_shape.mutable_layout();
  // Iterate through the logical dimension numbers in their physical order;
  // copy them into the new layout incrementing by one those that get shifted
  // by the insertion of the new batch dimension.
  for (int64_t logical_dim_idx : shape.layout().minor_to_major()) {
    // When 'logical_dim_idx' == 'contracting_dim_idx' add both
    // 'logical_dim_idx'+1 and 'logical_dim_idx' because it gets split into two.
    if (logical_dim_idx >= contracting_dim_idx) {
      new_layout->add_minor_to_major(logical_dim_idx + 1);
    }
    if (logical_dim_idx <= contracting_dim_idx) {
      new_layout->add_minor_to_major(logical_dim_idx);
    }
  }
  return MakeBitcastHlo(dot.mutable_operand(operand_number), new_shape);
}

// Apply split K configuration from the tiling to the fused dot() computation:
// bitcast the operands, change the output shape and the dot dimensions.
Status MakeDotComputationSplitKBatch(
    HloComputation* computation, const AutotuneResult::TritonGemmKey& tiling,
    bool disable_reduced_precision_reduction) {
  HloInstruction* dot =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  TF_ASSIGN_OR_RETURN(const auto analysis,
                      TritonFusionAnalysis::Execute(*computation));
  const DotDimensionNumbers& old_dim_numbers = dot->dot_dimension_numbers();
  DotDimensionNumbers new_dim_numbers;

  const int64_t lhs_contracting_idx = ContractingDimensionIndex(*dot, 0);
  CopyIncrementingAboveThreshold(
      old_dim_numbers.lhs_contracting_dimensions(),
      *new_dim_numbers.mutable_lhs_contracting_dimensions(),
      lhs_contracting_idx);
  new_dim_numbers.mutable_lhs_batch_dimensions()->Add(lhs_contracting_idx);
  CopyIncrementingAboveThreshold(
      old_dim_numbers.lhs_batch_dimensions(),
      *new_dim_numbers.mutable_lhs_batch_dimensions(), lhs_contracting_idx);

  const int64_t rhs_contracting_idx = ContractingDimensionIndex(*dot, 1);
  CopyIncrementingAboveThreshold(
      old_dim_numbers.rhs_contracting_dimensions(),
      *new_dim_numbers.mutable_rhs_contracting_dimensions(),
      rhs_contracting_idx);
  new_dim_numbers.mutable_rhs_batch_dimensions()->Add(rhs_contracting_idx);
  CopyIncrementingAboveThreshold(
      old_dim_numbers.rhs_batch_dimensions(),
      *new_dim_numbers.mutable_rhs_batch_dimensions(), rhs_contracting_idx);

  // Collect HLOs to transform between dot output and root. These will
  // get a new major most batch dimension sized as split K factor. Other inputs
  // of these HLOs will get broadcasted.
  std::stack<HloInstruction*> to_process;
  // Store the same HLOs also in a hash set for quick lookups.
  absl::flat_hash_set<HloInstruction*> to_process_set;
  HloInstruction* current = dot;
  do {
    to_process.push(current);
    CHECK(to_process_set.insert(current).second);
    if (current->users().empty()) {
      break;
    }
    CHECK_EQ(current->user_count(), 1);
    current = current->users()[0];
    if (!IsDistributiveOverAddition(*current)) {
      return Cancelled("Operation non-distributive over addition after dot.");
    }
  } while (true);

  // Process the collected HLOs from computation root to dot.
  while (!to_process.empty()) {
    HloInstruction* current = to_process.top();
    to_process.pop();
    // Add split-K dimension to `current`.
    HloInstruction* expanded;
    if (current == dot) {
      TF_ASSIGN_OR_RETURN(
          HloInstruction * lhs,
          MakeSplitKOperand(*dot, analysis, tiling, lhs_contracting_idx, 0));
      TF_ASSIGN_OR_RETURN(
          HloInstruction * rhs,
          MakeSplitKOperand(*dot, analysis, tiling, rhs_contracting_idx, 1));
      expanded = MakeDotHlo(lhs, rhs, new_dim_numbers, dot->precision_config(),
                            dot->shape().element_type())
                     .value();
      // Make the added batch dimension the major-most, keep the order of the
      // original dimensions.
      expanded->mutable_shape()->mutable_layout()->clear_minor_to_major();
      CopyIncrementingAboveThreshold(dot->shape().layout().minor_to_major(),
                                     *expanded->mutable_shape()
                                          ->mutable_layout()
                                          ->mutable_minor_to_major(),
                                     0);
      expanded->mutable_shape()->mutable_layout()->add_minor_to_major(0);
      dot->SetupDerivedInstruction(expanded);
    } else {
      expanded = computation->AddInstruction(
          current->CloneWithNewShape(ShapeUtil::PrependMajorDimension(
              tiling.split_k(), current->shape())));
    }
    TF_RETURN_IF_ERROR(current->ReplaceAllUsesWithDifferentShape(expanded));
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(current));
    // Broadcast operands.
    if (current == dot) {
      continue;
    }
    for (int i = 0; i < expanded->operands().size(); ++i) {
      HloInstruction* operand = expanded->mutable_operand(i);
      if (!to_process_set.contains(operand)) {
        std::vector<int64_t> broadcast_dimensions(operand->shape().rank());
        absl::c_iota(broadcast_dimensions, 1);
        TF_RETURN_IF_ERROR(expanded->ReplaceOperandWithDifferentShape(
            i, MakeBroadcastHlo(operand, broadcast_dimensions,
                                ShapeUtil::PrependMajorDimension(
                                    tiling.split_k(), operand->shape()))));
      }
    }
  }

  if (disable_reduced_precision_reduction) {
    PrimitiveType output_type =
        computation->root_instruction()->shape().element_type();
    PrimitiveType accumulator_type = output_type == PrimitiveType::F64
                                         ? PrimitiveType::F64
                                         : PrimitiveType::F32;

    computation->root_instruction()->mutable_shape()->set_element_type(
        accumulator_type);
  }
  return OkStatus();
}

Status FusionContext::PropagateDimensionOrdersToParameters(
    const HloInstruction& origin,
    absl::flat_hash_set<const HloInstruction*>& parameters,
    absl::flat_hash_map<const HloInstruction*, TensorIterationSpec>&
        iter_specs) {
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
  // For now all parameters of one scope have to use the same tiling.
  for (const HloInstruction* parameter : parameters) {
    TF_RET_CHECK(dim_orders_.at(parameter).IsPhysicallyEquivalent(
        dim_orders_.at(*parameters.cbegin())));
    iter_specs[parameter] =
        DimensionOrderToTensorIterationSpec(dim_orders_.at(parameter));
  }
  return OkStatus();
}

}  // anonymous namespace

// Data types that are supported by the Triton emitters.
bool IsTritonSupportedDataType(PrimitiveType type, GpuVersion gpu_version) {
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

Status MakeDotSplitKBatch(HloInstruction* dot_fusion,
                          const AutotuneResult::TritonGemmKey& tiling) {
  CHECK_EQ(dot_fusion->opcode(), HloOpcode::kFusion);

  if (dot_fusion->shape().IsTuple()) {
    return Unimplemented("Tuple output is not supported with split-K yet.");
  }

  const bool disable_reduced_precision_reduction =
      dot_fusion->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_triton_gemm_disable_reduced_precision_reduction();
  const PrimitiveType output_type = dot_fusion->shape().element_type();
  const Layout output_layout = dot_fusion->shape().layout();

  TF_RETURN_IF_ERROR(MakeDotComputationSplitKBatch(
      dot_fusion->fused_instructions_computation(), tiling,
      disable_reduced_precision_reduction));
  const HloInstruction* root = dot_fusion->fused_expression_root();

  *dot_fusion->mutable_shape() = root->shape();
  HloInstruction* zero =
      dot_fusion->parent()->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(root->shape().element_type())));
  // The batch dimension to reduce is the first one by construction.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * reduce,
      MakeReduceHlo(dot_fusion, zero, /*dimensions=*/{0}, HloOpcode::kAdd));

  // The output of the reduce has to have the layout of the original dot.
  *reduce->mutable_shape()->mutable_layout() = output_layout;

  if (dot_fusion->IsRoot()) {
    dot_fusion->parent()->set_root_instruction(reduce,
                                               /*accept_different_shape=*/true);
  } else {
    TF_RETURN_IF_ERROR(dot_fusion->ReplaceAllUsesWithDifferentShape(reduce));
  }

  if (disable_reduced_precision_reduction) {
    HloInstruction* convert = MakeConvertToHlo(reduce, output_type);
    if (reduce->IsRoot()) {
      reduce->parent()->set_root_instruction(convert,
                                             /*accept_different_shape=*/true);
    } else {
      TF_RETURN_IF_ERROR(reduce->ReplaceAllUsesWithDifferentShape(convert));
    }
  }

  return OkStatus();
}

StatusOr<TritonFusionAnalysis> TritonFusionAnalysis::Execute(
    const HloComputation& computation, const int split_k) {
  VLOG(5) << computation.ToString(HloPrintOptions::ShortParsable());
  TritonFusionAnalysis analysis;
  const HloInstruction* dot =
      hlo_query::GetFirstInstructionWithOpcode(computation, HloOpcode::kDot);
  TF_RETURN_IF_ERROR(analysis.ExecuteForDotFusion(*dot, split_k));
  return analysis;
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
                                   const GpuVersion gpu_version) {
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

  return FusionDecision{};
}

bool ShouldTritonHandleGEMM(HloInstruction& dot, const GpuVersion gpu_version) {
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

}  // namespace gpu
}  // namespace xla
