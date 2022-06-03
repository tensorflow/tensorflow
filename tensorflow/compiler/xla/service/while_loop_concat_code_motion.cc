/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/while_loop_concat_code_motion.h"

#include <map>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {

namespace {

// This algorithm tries to group HLO instructions into concat candidates. Each
// instruction can only belong to a single group.
//
// For simplicity, after finding the groups, it in-place updates the first group
// member to the full shape, and replaces non-grouped uses with slices of it.
// Then it relies on TupleSimplifier, WhileLoopSimplifier, and DCE passes to
// remove other elements.

// Represents a group of elements and how to concat them.
struct ConcatGroup {
  ConcatGroup(std::vector<HloInstruction*> elements, int64_t concat_dim,
              bool inserted_concat_dim)
      : elements(std::move(elements)),
        element_sizes(this->elements.size(), 1),
        element_offsets(this->elements.size(), 0),
        concat_dim(concat_dim),
        inserted_concat_dim(inserted_concat_dim) {
    if (inserted_concat_dim) {
      absl::c_iota(element_offsets, 0);
    } else {
      for (int64_t i = 0; i < element_sizes.size(); ++i) {
        element_sizes[i] = this->elements[i]->shape().dimensions(concat_dim);
        if (i > 0) {
          element_offsets[i] = element_offsets[i - 1] + element_sizes[i - 1];
        }
      }
    }
  }

  Shape GetConcatShape() const {
    if (inserted_concat_dim) {
      std::vector<int64_t> dims;
      const Shape& element_shape = elements.back()->shape();
      dims.reserve(element_shape.rank() + 1);
      for (int64_t i = 0; i < element_shape.rank(); ++i) {
        if (i == concat_dim) {
          dims.push_back(elements.size());
        }
        dims.push_back(element_shape.dimensions(i));
      }
      if (dims.size() == concat_dim) {
        dims.push_back(elements.size());
      }
      return ShapeUtil::MakeShape(element_shape.element_type(), dims);
    } else {
      int64_t dim_size = 0;
      for (int64_t size : element_sizes) {
        dim_size += size;
      }
      Shape shape = elements.back()->shape();
      shape.set_dimensions(concat_dim, dim_size);
      return shape;
    }
  }

  HloInstruction* CreateSlice(HloInstruction* full_data, int64_t element_index,
                              HloComputation* comp) const {
    Shape shape = full_data->shape();
    shape.set_dimensions(concat_dim, element_sizes[element_index]);
    std::vector<int64_t> starts(shape.rank(), 0);
    std::vector<int64_t> limits(shape.dimensions().begin(),
                                shape.dimensions().end());
    starts[concat_dim] = element_offsets[element_index];
    limits[concat_dim] += starts[concat_dim];
    auto slice = comp->AddInstruction(
        HloInstruction::CreateSlice(shape, full_data, starts, limits,
                                    std::vector<int64_t>(shape.rank(), 1)));
    if (!inserted_concat_dim) {
      return slice;
    }
    std::vector<int64_t> element_shape;
    element_shape.reserve(shape.rank() - 1);
    for (int64_t i = 0; i < shape.rank(); ++i) {
      if (i != concat_dim) {
        element_shape.push_back(shape.dimensions(i));
      }
    }
    return comp->AddInstruction(HloInstruction::CreateReshape(
        ShapeUtil::MakeShape(shape.element_type(), element_shape), slice));
  }

  HloInstruction* CreateConcat(std::vector<HloInstruction*> input_elements,
                               HloComputation* comp) const {
    if (inserted_concat_dim) {
      for (int64_t i = 0; i < input_elements.size(); ++i) {
        std::vector<int64_t> element_shape;
        element_shape.reserve(input_elements[i]->shape().rank() + 1);
        for (int64_t j = 0; j < input_elements[i]->shape().rank(); ++j) {
          if (j == concat_dim) {
            element_shape.push_back(1);
          }
          element_shape.push_back(input_elements[i]->shape().dimensions(j));
        }
        if (element_shape.size() == concat_dim) {
          element_shape.push_back(1);
        }
        input_elements[i] = comp->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(input_elements[i]->shape().element_type(),
                                 element_shape),
            input_elements[i]));
      }
    }

    return comp->AddInstruction(HloInstruction::CreateConcatenate(
        GetConcatShape(), input_elements, concat_dim));
  }

  std::vector<HloInstruction*> elements;
  std::vector<int64_t> element_sizes;
  std::vector<int64_t> element_offsets;
  int64_t concat_dim;
  // Whether the concat dim is an inserted new dimension.
  bool inserted_concat_dim;
};

// A collection of ConcatGroup's where each HLO can only belong to a single
// group.
class ConcatGroups {
 public:
  // Returns the group index and element index in group for an HLO, if it
  // belongs to a group.
  absl::optional<std::pair<int64_t, int64_t>> GetGroupIndex(
      const HloInstruction* hlo) const {
    auto it = element_to_group_.find(hlo);
    if (it == element_to_group_.end()) {
      return absl::nullopt;
    }
    return it->second;
  }

  const ConcatGroup& GetGroup(int64_t index) const { return groups_[index]; }

  // Creates a new group and returns the index if it doesn't exist, or returns
  // existing group index. If the new group doesn't match exactly with an
  // existing group but shared some of the elements, returns -1 as the index.
  // It also returns whether a new group is created. So the return value is a
  // pair of {whether created, group index}.
  std::pair<bool, int64_t> MaybeCreateNewGroup(ConcatGroup group) {
    int64_t group_id = -1;
    absl::flat_hash_set<HloInstruction*> elements_dedup;
    for (int64_t i = 0; i < group.elements.size(); ++i) {
      if (!elements_dedup.insert(group.elements[i]).second) {
        VLOG(2) << "Duplicates in group. Element: "
                << group.elements[i]->ToString();
      }
      if (concat_disallowed_.contains(group.elements[i])) {
        VLOG(2) << "Failed creating group. Grouping disallowed on "
                << group.elements[i]->ToString();
        return std::pair<bool, int64_t>(false, -1);
      }
      auto existing = GetGroupIndex(group.elements[i]);
      if (existing.has_value() &&
          (i != existing->second ||
           groups_[existing->first].concat_dim != group.concat_dim)) {
        // We allow mismatched inserted_concat_dim, since that only requires a
        // trivial reshape.
        VLOG(2)
            << "Failed creating group. Different than existing group. Element: "
            << group.elements[i]->ToString();
        return std::pair<bool, int64_t>(false, -1);
      }
      if (i == 0 && existing.has_value()) {
        group_id = existing->first;
      }
      if (i > 0) {
        if (existing.has_value() && existing->first != group_id) {
          VLOG(2) << "Failed creating group. Different than existing group. "
                     "Element: "
                  << group.elements[i]->ToString();
          return std::pair<bool, int64_t>(false, -1);
        }
        if (!existing.has_value() && group_id >= 0) {
          VLOG(2) << "Failed creating group. Different than existing group. "
                     "Element: "
                  << group.elements[i]->ToString();
          return std::pair<bool, int64_t>(false, -1);
        }
      }
    }
    if (group_id >= 0) {
      VLOG(2) << "Group already exists at " << group_id << " for "
              << group.elements[0]->ToString();
      return std::pair<bool, int64_t>(false, group_id);
    }
    int64_t index = groups_.size();
    for (int64_t i = 0; i < group.elements.size(); ++i) {
      element_to_group_[group.elements[i]] =
          std::pair<int64_t, int64_t>(index, i);
    }
    VLOG(2) << "Created new group at " << index << " for "
            << group.elements[0]->ToString()
            << ", concat_dim: " << group.concat_dim
            << ", inserted: " << group.inserted_concat_dim;
    groups_.push_back(std::move(group));
    return std::pair<bool, int64_t>(true, index);
  }

  const std::vector<ConcatGroup>& Groups() const { return groups_; }

  int64_t NextGroupIndex() const { return groups_.size(); }

  void RemoveTailingGroups(int64_t start_index) {
    while (groups_.size() > start_index) {
      for (auto element : groups_.back().elements) {
        element_to_group_.erase(element);
      }
      groups_.pop_back();
    }
  }

  void DisallowGroupingOn(const HloInstruction* hlo) {
    VLOG(2) << "Disallow grouping on " << hlo->ToString();
    concat_disallowed_.insert(hlo);
  }

 private:
  // element -> {group index in groups_, element index in group}.
  absl::flat_hash_map<const HloInstruction*, std::pair<int64_t, int64_t>>
      element_to_group_;
  std::vector<ConcatGroup> groups_;
  absl::flat_hash_set<const HloInstruction*> concat_disallowed_;
};

// Infers an operand's concat dim and whether it's an inserted dim. For example,
// if hlo is f32[2,4,2] broadcast(f32[2,4]), dimensions={0,1} concatenated on
// dim 2, then this function will return {2, true}.
//
// If the operand is already transformed to the combined shape, specify its
// group in combined_operand_group. (Only required for kReshape.)
absl::optional<std::pair<int64_t, bool>> GetOperandConcatDim(
    const HloInstruction* hlo, int64_t operand_index, int64_t hlo_concat_dim,
    bool hlo_inserted_concat_dim,
    const ConcatGroup* combined_operand_group = nullptr) {
  if (hlo->IsElementwise() || hlo->opcode() == HloOpcode::kAllReduce) {
    return std::pair<int64_t, bool>(hlo_concat_dim, hlo_inserted_concat_dim);
  }
  int64_t operand_concat_dim = -1;
  bool operand_inserted_concat_dim = false;
  const Shape& operand_shape =
      combined_operand_group == nullptr
          ? hlo->operand(operand_index)->shape()
          : combined_operand_group->elements.back()->shape();
  if (hlo->opcode() == HloOpcode::kBroadcast) {
    operand_concat_dim = 0;
    operand_inserted_concat_dim = true;
    // Try to place operand_concat_dim adjacent to dims the same way as the
    // output, if it does not exist in the operand..
    int64_t min_dist_to_concat_dim = hlo->shape().rank();
    for (int64_t i = 0; i < operand_shape.rank(); ++i) {
      if (hlo->dimensions(i) == hlo_concat_dim) {
        operand_concat_dim = i;
        operand_inserted_concat_dim = hlo_inserted_concat_dim;
        break;
      }
      if (hlo->dimensions(i) < hlo_concat_dim &&
          min_dist_to_concat_dim > hlo_concat_dim - hlo->dimensions(i)) {
        operand_concat_dim = i + 1;
        min_dist_to_concat_dim = hlo_concat_dim - hlo->dimensions(i);
      }
      if (hlo->dimensions(i) > hlo_concat_dim &&
          min_dist_to_concat_dim > hlo->dimensions(i) - hlo_concat_dim) {
        operand_concat_dim = i;
        min_dist_to_concat_dim = hlo->dimensions(i) - hlo_concat_dim;
      }
    }
  } else if (hlo->opcode() == HloOpcode::kReduce) {
    if (operand_index != 0) {
      return absl::nullopt;
    }
    operand_concat_dim = hlo_concat_dim;
    operand_inserted_concat_dim = hlo_inserted_concat_dim;
    std::set<int64_t> sorted_reduce_dims;
    for (int64_t dim : hlo->dimensions()) {
      sorted_reduce_dims.insert(dim);
    }
    for (int64_t dim : sorted_reduce_dims) {
      if ((hlo_inserted_concat_dim && dim < operand_concat_dim) ||
          (!hlo_inserted_concat_dim && dim <= operand_concat_dim)) {
        operand_concat_dim++;
      }
    }
  } else if (hlo->opcode() == HloOpcode::kReshape) {
    int64_t i = 0;
    int64_t j = 0;
    operand_inserted_concat_dim = false;
    // Only support adding/removing trivial dims.
    while (i < operand_shape.rank() || j <= hlo_concat_dim) {
      if (i < operand_shape.rank() && j < hlo->shape().rank() &&
          operand_shape.dimensions(i) == hlo->shape().dimensions(j)) {
        if (j == hlo_concat_dim) {
          operand_inserted_concat_dim =
              hlo_inserted_concat_dim && operand_shape.dimensions(i) != 1;
          operand_concat_dim = i;
          break;
        }
        i++;
        j++;
        continue;
      }
      if (i < operand_shape.rank() && operand_shape.dimensions(i) == 1) {
        if (j == hlo_concat_dim && hlo_inserted_concat_dim) {
          operand_concat_dim = i;
          break;
        }
        i++;
        continue;
      }
      if (j == hlo_concat_dim) {
        operand_concat_dim = i;
        operand_inserted_concat_dim = true;
        break;
      }
      if (j < hlo->shape().rank() && hlo->shape().dimensions(j) == 1) {
        j++;
        continue;
      }
      return absl::nullopt;
    }
  } else {
    return absl::nullopt;
  }
  CHECK_GE(operand_concat_dim, 0);
  return std::pair<int64_t, bool>(operand_concat_dim,
                                  operand_inserted_concat_dim);
}

void ModifyHloPropertiesForConcatShape(const ConcatGroup& group,
                                       HloInstruction* hlo) {
  *hlo->mutable_shape() = group.GetConcatShape();
  if (hlo->opcode() == HloOpcode::kBroadcast) {
    // Use the last element to infer the operand concat dim, since the first
    // element's operand might have been rewriten.
    auto operand_dim = GetOperandConcatDim(
        group.elements.back(), 0, group.concat_dim, group.inserted_concat_dim);
    CHECK(operand_dim.has_value());
    int64_t operand_concat_dim = operand_dim->first;
    bool operand_inserted_concat_dim = operand_dim->second;
    if (operand_inserted_concat_dim) {
      // We should have added an dimension on the operand.
      CHECK_EQ(hlo->operand(0)->shape().rank(), hlo->dimensions().size() + 1)
          << hlo->ToString();
    } else {
      CHECK_EQ(hlo->operand(0)->shape().rank(), hlo->dimensions().size());
    }
    std::vector<int64_t> dims;
    const int64_t rank = hlo->operand(0)->shape().rank();
    dims.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
      if (i == operand_concat_dim && operand_inserted_concat_dim) {
        dims.push_back(group.concat_dim);
      } else {
        if (i > operand_concat_dim && operand_inserted_concat_dim) {
          dims.push_back(hlo->dimensions(i - 1));
        } else {
          dims.push_back(hlo->dimensions(i));
        }
        if (group.inserted_concat_dim && dims.back() >= group.concat_dim) {
          dims.back()++;
        }
      }
    }
    *hlo->mutable_dimensions() = std::move(dims);
  } else if (hlo->opcode() == HloOpcode::kReduce) {
    auto operand_dim = GetOperandConcatDim(
        group.elements.back(), 0, group.concat_dim, group.inserted_concat_dim);
    int64_t operand_concat_dim = operand_dim->first;
    bool operand_inserted_concat_dim = operand_dim->second;
    CHECK(operand_dim.has_value());
    if (operand_inserted_concat_dim) {
      auto dims = hlo->mutable_dimensions();
      for (int64_t i = 0; i < dims->size(); ++i) {
        if ((*dims)[i] >= operand_concat_dim) {
          (*dims)[i]++;
        }
      }
    }
  }
}

// Main method to assign groups to HLOs, based on a concat.
bool GroupHlosForConcat(
    HloComputation* body, HloInstruction* concat,
    absl::flat_hash_map<const HloInstruction*, int64_t> topological_order,
    ConcatGroups* groups) {
  const int64_t group_size = concat->operand_count();
  absl::flat_hash_set<int64_t> used_groups;
  auto root_tuple = body->root_instruction();
  CHECK_EQ(root_tuple->opcode(), HloOpcode::kTuple);
  absl::flat_hash_map<HloInstruction*, int64_t> root_tuple_element_use_count;
  for (auto operand : root_tuple->operands()) {
    root_tuple_element_use_count.emplace(operand, 0).first->second++;
  }
  // Priority Queue sorted by topological order. Users come before operands, so
  // it uses -topological_order[element0] as the key. We start with the concat
  // operands.
  std::multimap<int64_t, ConcatGroup> pq;
  const int64_t first_group_id_to_create = groups->NextGroupIndex();
  auto fail_and_cleanup = [&] {
    VLOG(1) << "Failed to get the subcomputation to optimize for "
            << concat->ToString() << ", clear groups starting at "
            << first_group_id_to_create;
    groups->RemoveTailingGroups(first_group_id_to_create);
    return false;
  };
  struct GroupUse {
    int64_t group_id;
    bool newly_created;
    bool already_used_by_subcomp;
  };
  auto maybe_create_group = [&](ConcatGroup group) {
    auto res = groups->MaybeCreateNewGroup(std::move(group));
    GroupUse use{res.second, false, false};
    if (res.second < 0) {
      return use;
    }
    use.newly_created = res.first;
    use.already_used_by_subcomp = !used_groups.insert(res.second).second;
    return use;
  };
  std::vector<HloInstruction*> concat_operands(concat->operands().begin(),
                                               concat->operands().end());
  int64_t concat_operand_order = -topological_order[concat_operands[0]];
  pq.emplace(concat_operand_order,
             ConcatGroup(std::move(concat_operands),
                         concat->concatenate_dimension(), false));

  // Find the subcomputation on elements to combine, in order to move `concat`
  // out of the loop without adding new concats. We start from the concat's
  // operands, and the priority queue is ordered in reverse topological order
  // so we process outputs before inputs. Each entry in the queue is a group of
  // elements to combine. A legitimate group consists of identical ops, except
  // that they each operate on one element. When a group of loop inputs are
  // processed, we also enqueue the corresponding loop outputs to keep them
  // match in shape.
  while (!pq.empty()) {
    auto group = std::move(pq.begin()->second);
    pq.erase(pq.begin());
    const auto& hlos = group.elements;
    VLOG(2) << "GroupHlosForConcat dequeued " << hlos[0]->ToString();
    bool group_is_param_gtes = false;
    if (absl::c_all_of(hlos, [&](const HloInstruction* element) {
          return element == hlos[0];
        })) {
      // Shared operand.
      if (groups->GetGroupIndex(hlos[0]).has_value()) {
        VLOG(1) << "We do not support the case if a shared operand also part "
                   "of a group: "
                << hlos[0]->ToString();
        return fail_and_cleanup();
      }
      groups->DisallowGroupingOn(hlos[0]);
      continue;
    }
    if (absl::c_all_of(hlos, [&](const HloInstruction* element) {
          return element->opcode() == HloOpcode::kGetTupleElement &&
                 element->operand(0) == body->parameter_instruction(0);
        })) {
      group_is_param_gtes = true;
    } else if (((hlos[0]->IsElementwise() ||
                 hlos[0]->opcode() == HloOpcode::kAllReduce) &&
                !hlos[0]->HasSideEffect()) ||
               hlos[0]->opcode() == HloOpcode::kBroadcast ||
               hlos[0]->opcode() == HloOpcode::kReduce ||
               hlos[0]->opcode() == HloOpcode::kReshape ||
               hlos[0]->IsCustomCall("Sharding")) {
      if (hlos[0]->opcode() == HloOpcode::kAllReduce &&
          (!hlos[0]->shape().IsArray() || hlos[0]->IsCrossModuleAllReduce())) {
        VLOG(2) << "Unsupported allreduce: " << hlos[0]->ToString();
        return fail_and_cleanup();
      }
      // Check if these elements can be concatenated.
      if (absl::c_any_of(hlos, [&](const HloInstruction* element) {
            auto eq_operand = [](const HloInstruction* a,
                                 const HloInstruction* b) {
              return ShapeUtil::Compatible(a->shape(), b->shape());
            };
            auto eq_computations = [](const HloComputation* lhs,
                                      const HloComputation* rhs) {
              return lhs->Equal(*rhs, /*is_layout_sensitive=*/false);
            };
            if (!hlos[0]->Identical(*element, eq_operand, eq_computations,
                                    /*layout_sensitive=*/false)) {
              return true;
            }
            if (element->opcode() == HloOpcode::kReduce &&
                (element->operand_count() != 2 ||
                 element->operand(1) != hlos[0]->operand(1))) {
              return true;
            }
            return false;
          })) {
        VLOG(2) << "Different types of elements. First element: "
                << hlos[0]->ToString();
        return fail_and_cleanup();
      }
      // Now enqueue the inputs.
      int64_t input_count = hlos[0]->operand_count();
      if (hlos[0]->opcode() == HloOpcode::kReduce) {
        CHECK_EQ(input_count, 2);
        // Exclude the init value that we have checked to be the same.
        input_count = 1;
      }
      for (int64_t i = 0; i < input_count; ++i) {
        std::vector<HloInstruction*> elements(group_size);
        for (int64_t j = 0; j < group_size; ++j) {
          elements[j] = hlos[j]->mutable_operand(i);
        }
        auto maybe_new_concat_dim = GetOperandConcatDim(
            hlos[0], i, group.concat_dim, group.inserted_concat_dim);
        if (!maybe_new_concat_dim.has_value()) {
          VLOG(2) << "Cannot find operand concat dimension for operand " << i
                  << " of " << hlos[0]->ToString();
          return fail_and_cleanup();
        }
        int64_t new_group_concat_dim = maybe_new_concat_dim->first;
        bool inserted_concat_dim = maybe_new_concat_dim->second;
        // Enqueue the input group.
        int64_t element_order = -topological_order[elements[0]];
        pq.emplace(element_order,
                   ConcatGroup(std::move(elements), new_group_concat_dim,
                               inserted_concat_dim));
      }
    } else if (hlos[0]->opcode() == HloOpcode::kSlice) {
      int64_t offset = 0;
      auto operand = hlos[0]->operand(0);
      if (group.inserted_concat_dim) {
        VLOG(2) << "Slices cannot be grouped on new dimension.";
        return fail_and_cleanup();
      }
      if (groups->GetGroupIndex(operand).has_value()) {
        // Should not slice an operand to be grouped.
        return fail_and_cleanup();
      }
      groups->DisallowGroupingOn(operand);
      for (int64_t i = 0; i < group_size; ++i) {
        if (hlos[i]->operand(0) != operand) {
          VLOG(2) << "Slices of different operands.";
          return fail_and_cleanup();
        }
        for (int64_t j = 0; j < hlos[i]->shape().rank(); ++j) {
          if (hlos[i]->slice_strides(j) != 1) {
            VLOG(2) << "Slices with strides.";
            return fail_and_cleanup();
          }
          if (j == group.concat_dim) {
            if (hlos[i]->slice_starts(j) != offset) {
              VLOG(2) << "Slices with unsupported offsets.";
              return fail_and_cleanup();
            }
            offset += hlos[i]->shape().dimensions(j);
          } else {
            if (hlos[i]->slice_starts(j) != 0 ||
                hlos[i]->slice_limits(j) != operand->shape().dimensions(j)) {
              VLOG(2) << "Slice with unsupported offsets at dimension " << j
                      << ", " << hlos[i]->ToString();
              return fail_and_cleanup();
            }
          }
        }
      }
      if (offset != operand->shape().dimensions(group.concat_dim)) {
        VLOG(2) << "Slices with unsupported sizes.";
        return fail_and_cleanup();
      }
    } else {
      VLOG(2) << "Unsupported opcode: " << hlos[0]->ToString();
      return fail_and_cleanup();
    }
    auto guse = maybe_create_group(std::move(group));
    if (guse.group_id < 0) {
      VLOG(2) << "Failed to create group.";
      return fail_and_cleanup();
    }
    const auto& registered_group = groups->GetGroup(guse.group_id);
    if (!guse.already_used_by_subcomp && group_is_param_gtes) {
      // When we processed a group of parameter GTEs, we should also enqueue the
      // corresponding root tuple operands, so that they have matching shapes.
      std::vector<HloInstruction*> new_outputs(group_size);
      for (int64_t i = 0; i < group_size; ++i) {
        new_outputs[i] = root_tuple->mutable_operand(
            registered_group.elements[i]->tuple_index());
      }
      int64_t new_output_order = -topological_order[new_outputs[0]];
      pq.emplace(
          new_output_order,
          ConcatGroup(std::move(new_outputs), registered_group.concat_dim,
                      registered_group.inserted_concat_dim));
    }
  }
  return groups->Groups().size() > first_group_id_to_create;
}

std::vector<bool> TupleElementsUsedInCond(HloInstruction* loop) {
  std::vector<bool> result(loop->shape().tuple_shapes_size(), false);
  for (auto user : loop->while_condition()->parameter_instruction(0)->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      absl::c_fill(result, true);
      return result;
    }
    result[user->tuple_index()] = true;
  }
  return result;
}

// Adds copies to returned values to keep RewriteLoopWithConcatGroups simple:
// the copies do not have other users and only appear once in the root tuple.
Status AddCopiesToRoot(HloComputation* body,
                       absl::Span<HloInstruction* const> param_gtes,
                       ConcatGroups* groups) {
  auto root = body->root_instruction();
  CHECK_EQ(root->opcode(), HloOpcode::kTuple);
  std::vector<HloInstruction*> copies(root->operand_count(), nullptr);
  for (int64_t i = 0; i < copies.size(); ++i) {
    auto element = root->mutable_operand(i);
    if (!element->shape().IsArray()) {
      continue;
    }
    copies[i] = body->AddInstruction(HloInstruction::CreateUnary(
        element->shape(), HloOpcode::kCopy, element));
    TF_RETURN_IF_ERROR(root->ReplaceOperandWith(i, copies[i]));
  }
  for (int64_t i = 0; i < copies.size(); ++i) {
    auto copy = copies[i];
    if (groups->GetGroupIndex(copy).has_value()) {
      // Already handled by earlier group members.
      continue;
    }
    auto param_group_index = groups->GetGroupIndex(param_gtes[i]);
    if (!param_group_index.has_value()) {
      continue;
    }
    const auto& param_group = groups->GetGroup(param_group_index->first);
    std::vector<HloInstruction*> copy_group(param_group.elements.size());
    for (int64_t j = 0; j < copy_group.size(); ++j) {
      copy_group[j] = copies[param_group.elements[j]->tuple_index()];
    }
    CHECK(groups
              ->MaybeCreateNewGroup(
                  ConcatGroup(std::move(copy_group), param_group.concat_dim,
                              param_group.inserted_concat_dim))
              .first);
  }
  return OkStatus();
}

Status RemoveCopiesFromRoot(HloComputation* body) {
  auto root = body->root_instruction();
  CHECK_EQ(root->opcode(), HloOpcode::kTuple);
  for (int64_t i = 0; i < root->operand_count(); ++i) {
    auto copy = root->mutable_operand(i);
    if (copy->opcode() == HloOpcode::kCopy) {
      TF_RETURN_IF_ERROR(root->ReplaceOperandWith(i, copy->mutable_operand(0)));
    }
  }
  return OkStatus();
}

Status RewriteLoopWithConcatGroups(HloInstruction* loop,
                                   absl::Span<HloInstruction* const> param_gtes,
                                   ConcatGroups& groups) {
  VLOG(1) << "RewriteLoopWithConcatGroups with " << groups.Groups().size()
          << " groups.";
  // For simplicity, for each group, we rewrite the first element into full
  // shape, and leave the other elements unchagned. Non-grouped users will be
  // have slices of the expanded first element as the new input. Later
  // simplification and DCE passes can remove the other elements.
  absl::flat_hash_set<int64_t> processed_groups;
  auto body = loop->while_body();
  auto param = body->parameter_instruction(0);
  auto cond_param = loop->while_condition()->parameter_instruction(0);

  // First, modify loop signature and operands/users.
  std::vector<HloInstruction*> init_elements(loop->shape().tuple_shapes_size());
  for (int64_t i = 0; i < param_gtes.size(); ++i) {
    init_elements[i] =
        loop->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
            loop->shape().tuple_shapes(i), loop->mutable_operand(0), i));
  }
  for (int64_t i = 0; i < param_gtes.size(); ++i) {
    const auto& group_and_index = groups.GetGroupIndex(param_gtes[i]);
    if (!group_and_index.has_value() || group_and_index->second != 0) {
      continue;
    }
    const auto& group = groups.GetGroup(group_and_index->first);
    // Change body parameter shape.
    *param_gtes[i]->mutable_shape() = group.GetConcatShape();
    *param->mutable_shape()->mutable_tuple_shapes(i) = param_gtes[i]->shape();
    *body->root_instruction()->mutable_shape()->mutable_tuple_shapes(i) =
        param_gtes[i]->shape();
    *cond_param->mutable_shape()->mutable_tuple_shapes(i) =
        param_gtes[i]->shape();
    *loop->mutable_shape()->mutable_tuple_shapes(i) = param_gtes[i]->shape();
    processed_groups.insert(group_and_index->first);
    std::vector<HloInstruction*> input_concat_elements;
    input_concat_elements.reserve(group.elements.size());
    for (auto param_gte : group.elements) {
      input_concat_elements.push_back(init_elements[param_gte->tuple_index()]);
    }
    init_elements[i] =
        group.CreateConcat(std::move(input_concat_elements), loop->parent());
  }
  TF_RETURN_IF_ERROR(loop->ReplaceOperandWithDifferentShape(
      0, loop->parent()->AddInstruction(
             HloInstruction::CreateTuple(init_elements))));
  // Adjust loop users.
  auto original_loop_users = loop->users();
  const bool loop_is_root = loop == loop->parent()->root_instruction();
  std::vector<HloInstruction*> output_elements(
      loop->shape().tuple_shapes_size());
  for (int64_t i = 0; i < param_gtes.size(); ++i) {
    output_elements[i] =
        loop->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
            init_elements[i]->shape(), loop, i));
  }
  for (int64_t i = 0; i < param_gtes.size(); ++i) {
    const auto& group_and_index = groups.GetGroupIndex(param_gtes[i]);
    if (!group_and_index.has_value() || group_and_index->second != 0) {
      continue;
    }
    const auto& group = groups.GetGroup(group_and_index->first);
    auto concat_output = output_elements[group.elements[0]->tuple_index()];
    for (int64_t j = 0; j < group.elements.size(); ++j) {
      const auto param_gte = group.elements[j];
      output_elements[param_gte->tuple_index()] =
          group.CreateSlice(concat_output, j, loop->parent());
    }
  }
  auto new_output_tuple = loop->parent()->AddInstruction(
      HloInstruction::CreateTuple(output_elements));
  for (auto user : original_loop_users) {
    TF_RETURN_IF_ERROR(
        loop->ReplaceUseWithDifferentShape(user, new_output_tuple));
  }
  if (loop_is_root) {
    loop->parent()->set_root_instruction(new_output_tuple,
                                         /*accept_different_shape=*/true);
  }

  // Now rewrite the loop body.
  std::vector<HloInstruction*> slices_to_remove;
  absl::flat_hash_set<HloInstruction*> new_reshapes;
  for (auto hlo : body->MakeInstructionPostOrder()) {
    const auto& group_and_index = groups.GetGroupIndex(hlo);
    if (!group_and_index.has_value() || group_and_index->second != 0) {
      continue;
    }

    if (!processed_groups.insert(group_and_index->first).second) {
      // Already processed the group at the first element.
      continue;
    }
    const auto& group = groups.GetGroup(group_and_index->first);
    if (hlo->opcode() == HloOpcode::kSlice) {
      // We could just replace hlo with its operand; however, to follow the
      // practice of using the first element as full data, we defer that
      // replacement.
      slices_to_remove.push_back(hlo);
    } else {
      int64_t operand_count_to_adjust = hlo->operand_count();
      if (hlo->opcode() == HloOpcode::kReduce) {
        CHECK_EQ(operand_count_to_adjust, 2);
        operand_count_to_adjust = 1;
      }
      for (int64_t i = 0; i < operand_count_to_adjust; ++i) {
        auto operand_group_index = groups.GetGroupIndex(hlo->operand(i));
        const ConcatGroup* operand_group =
            operand_group_index.has_value()
                ? &groups.GetGroup(operand_group_index->first)
                : nullptr;
        auto maybe_operand_concat_dim = GetOperandConcatDim(
            hlo, i, group.concat_dim, group.inserted_concat_dim, operand_group);
        CHECK(maybe_operand_concat_dim.has_value())
            << "Operand " << i << " of " << hlo->ToString();
        int64_t operand_concat_dim = maybe_operand_concat_dim->first;
        bool operand_inserted_concat_dim = maybe_operand_concat_dim->second;
        if (operand_group != nullptr) {
          CHECK_EQ(operand_concat_dim, operand_group->concat_dim);
          if (operand_inserted_concat_dim !=
              operand_group->inserted_concat_dim) {
            // The operand's actual inserted_concat_dim doesn't match the
            // expected operand_inserted_concat_dim. Need a reshape.
            std::vector<int64_t> new_dims;
            int64_t d = 0;
            for (; d < operand_concat_dim; ++d) {
              new_dims.push_back(hlo->operand(i)->shape().dimensions(d));
            }
            if (operand_inserted_concat_dim) {
              // Split operand concat dim.
              new_dims.push_back(group.elements.size());
              new_dims.push_back(
                  hlo->operand(i)->shape().dimensions(operand_concat_dim) /
                  group.elements.size());
              d = operand_concat_dim + 1;
            } else {
              // Combine operand concat dim with the next.
              new_dims.push_back(
                  group.elements.size() *
                  hlo->operand(i)->shape().dimensions(operand_concat_dim + 1));
              d = operand_concat_dim + 2;
            }
            for (; d < hlo->operand(i)->shape().rank(); ++d) {
              new_dims.push_back(hlo->operand(i)->shape().dimensions(d));
            }
            auto reshape = body->AddInstruction(HloInstruction::CreateReshape(
                ShapeUtil::MakeShape(hlo->operand(i)->shape().element_type(),
                                     new_dims),
                hlo->mutable_operand(i)));
            new_reshapes.insert(reshape);
            TF_RETURN_IF_ERROR(
                hlo->ReplaceOperandWithDifferentShape(i, reshape));
          }
          continue;
        }
        // This is a shared operand, we need to broadcast it.
        CHECK(
            absl::c_all_of(group.elements, [&](const HloInstruction* element) {
              return element->operand(i) == hlo->operand(i);
            }));
        VLOG(2) << "Broadcasting shared operand "
                << hlo->operand(i)->ToString();
        Shape data_shape = hlo->operand(i)->shape();
        std::vector<int64_t> broadcast_dims;
        std::vector<int64_t> broadcast_shape;
        const int64_t data_shape_rank = data_shape.rank();
        broadcast_dims.reserve(data_shape_rank);
        broadcast_shape.reserve(data_shape_rank + 1);
        for (int64_t j = 0; j < data_shape_rank; ++j) {
          if (j < operand_concat_dim) {
            broadcast_dims.push_back(j);
          } else {
            broadcast_dims.push_back(j + 1);
          }
          if (j == operand_concat_dim) {
            broadcast_shape.push_back(group.elements.size());
          }
          broadcast_shape.push_back(data_shape.dimensions(j));
        }
        if (broadcast_shape.size() == data_shape.rank()) {
          // New dim at the end.
          broadcast_shape.push_back(group.elements.size());
        }
        auto broadcast = body->AddInstruction(HloInstruction::CreateBroadcast(
            ShapeUtil::MakeShape(data_shape.element_type(), broadcast_shape),
            hlo->mutable_operand(i), broadcast_dims));

        if (!operand_inserted_concat_dim) {
          // Concat on existing dim. Reshape to merge the broadcast dim.
          data_shape.set_dimensions(
              operand_concat_dim,
              data_shape.dimensions(operand_inserted_concat_dim) *
                  group.elements.size());
          broadcast = body->AddInstruction(
              HloInstruction::CreateReshape(data_shape, broadcast));
        }
        TF_RETURN_IF_ERROR(hlo->ReplaceOperandWithDifferentShape(i, broadcast));
      }
    }
    VLOG(2) << "Modifying HLO to full shape " << hlo->ToString();
    ModifyHloPropertiesForConcatShape(group, hlo);
    VLOG(2) << "Modified HLO to full shape " << hlo->ToString();
  }

  // For non-grouped HLOs, replace grouped inputs with slices. Also inlcude
  // grouped reduce HLOs because their init values are not grouped.
  for (auto hlo : body->MakeInstructionPostOrder()) {
    if (new_reshapes.contains(hlo)) {
      continue;
    }
    const auto& group_and_index = groups.GetGroupIndex(hlo);
    if ((!group_and_index.has_value() || hlo->opcode() == HloOpcode::kReduce) &&
        hlo != body->root_instruction()) {
      auto operands = hlo->operands();
      if (group_and_index.has_value()) {
        // Only handle reduce init value.
        CHECK_EQ(operands.size(), 2);
        CHECK_EQ(hlo->opcode(), HloOpcode::kReduce);
        operands.erase(operands.begin());
      }
      for (int64_t i = 0; i < operands.size(); ++i) {
        auto operand = operands[i];
        auto operand_group_index = groups.GetGroupIndex(operand);
        if (!operand_group_index.has_value()) {
          continue;
        }
        const auto& operand_group = groups.GetGroup(operand_group_index->first);
        auto slice = operand_group.CreateSlice(
            operand_group.elements[0], operand_group_index->second, body);
        TF_RETURN_IF_ERROR(hlo->ReplaceOperandWithDifferentShape(i, slice));
      }
    }
  }
  for (auto slice : slices_to_remove) {
    TF_RETURN_IF_ERROR(slice->ReplaceAllUsesWith(slice->mutable_operand(0)));
    TF_RETURN_IF_ERROR(body->RemoveInstruction(slice));
  }
  return OkStatus();
}

StatusOr<bool> RunOnLoop(HloInstruction* loop,
                         int64_t min_operand_count_to_optimize) {
  auto body = loop->while_body();
  auto param = body->parameter_instruction(0);
  auto root = body->root_instruction();
  if (!param->shape().IsTuple() || root->opcode() != HloOpcode::kTuple) {
    return false;
  }
  std::vector<HloInstruction*> gtes(param->shape().tuple_shapes_size(),
                                    nullptr);
  ConcatGroups groups;
  auto indices_used_in_cond = TupleElementsUsedInCond(loop);
  for (auto user : param->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      // Unhandled user opcode.
      return false;
    }
    int64_t idx = user->tuple_index();
    if (gtes[idx] != nullptr) {
      // Seen this index before.
      return false;
    }
    gtes[idx] = user;
    if (indices_used_in_cond[idx]) {
      groups.DisallowGroupingOn(user);
    }
  }
  std::vector<HloInstruction*> concats;
  auto body_instructions = body->MakeInstructionPostOrder();
  absl::flat_hash_map<const HloInstruction*, int64_t> topological_order;
  for (int64_t i = 0; i < body_instructions.size(); ++i) {
    auto hlo = body_instructions[i];
    topological_order[hlo] = i;
    if (hlo->opcode() == HloOpcode::kConcatenate &&
        hlo->operand_count() >= min_operand_count_to_optimize) {
      concats.push_back(hlo);
    }
  }

  for (auto& concat : concats) {
    if (!GroupHlosForConcat(body, concat, topological_order, &groups)) {
      concat = nullptr;
    }
  }
  if (groups.Groups().empty()) {
    return false;
  }

  TF_RETURN_IF_ERROR(AddCopiesToRoot(body, gtes, &groups));
  TF_RETURN_IF_ERROR(RewriteLoopWithConcatGroups(loop, gtes, groups));
  for (auto concat : concats) {
    if (concat == nullptr) {
      continue;
    }
    // We have repalced the operands of the concat with slices of full data.
    auto new_slice = concat->mutable_operand(0);
    CHECK_EQ(new_slice->opcode(), HloOpcode::kSlice);
    TF_RETURN_IF_ERROR(
        concat->ReplaceAllUsesWith(new_slice->mutable_operand(0)));
    TF_RETURN_IF_ERROR(body->RemoveInstruction(concat));
  }
  TF_RETURN_IF_ERROR(RemoveCopiesFromRoot(body));
  // Finally pass-through replaced elements from parameter to root, so that
  // while loop simplifier can get rid of them.
  for (auto gte : gtes) {
    auto group_index = groups.GetGroupIndex(gte);
    if (group_index.has_value() && group_index->second > 0) {
      TF_RETURN_IF_ERROR(root->ReplaceOperandWith(gte->tuple_index(), gte));
    }
  }
  return true;
}

}  // namespace

StatusOr<bool> WhileLoopConcatCodeMotion::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    for (HloInstruction* hlo : comp->MakeInstructionPostOrder()) {
      if (hlo->opcode() == HloOpcode::kWhile) {
        TF_ASSIGN_OR_RETURN(bool loop_changed,
                            RunOnLoop(hlo, min_operand_count_to_optimize_));
        changed |= loop_changed;
      }
    }
  }
  if (changed) {
    HloPassPipeline pipeline("loop-concat-motion-cleanup");
    pipeline.AddPass<TupleSimplifier>();
    pipeline.AddPass<HloDCE>();
    pipeline.AddPass<WhileLoopSimplifier>();
    pipeline.AddPass<TupleSimplifier>();
    pipeline.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(pipeline.Run(module).status());
  }
  return changed;
}

}  // namespace xla
