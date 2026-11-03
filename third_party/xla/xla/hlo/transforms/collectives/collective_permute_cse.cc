/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/collective_permute_cse.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_registry.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

static bool IsContained(const HloInstruction* large_slice,
                        const HloInstruction* small_slice) {
  for (int64_t i = 0; i < large_slice->slice_starts().size(); ++i) {
    if (large_slice->slice_strides()[i] != small_slice->slice_strides()[i]) {
      return false;
    }
    int64_t l_start = large_slice->slice_starts()[i];
    int64_t s_start = small_slice->slice_starts()[i];
    int64_t stride = large_slice->slice_strides()[i];
    if (s_start < l_start || (s_start - l_start) % stride != 0) {
      return false;
    }
    int64_t new_start = (s_start - l_start) / stride;
    int64_t n_l = large_slice->shape().dimensions(i);
    int64_t n_s = small_slice->shape().dimensions(i);
    if (new_start + n_s > n_l) {
      return false;
    }
  }
  return true;
}

static absl::Status RemoveControlDependencies(
    HloInstruction* small, HloInstruction* large,
    HloReachabilityMap* reachability) {
  CHECK(reachability != nullptr);
  std::vector<HloInstruction*> preds_to_remove;
  for (HloInstruction* pred : large->control_predecessors()) {
    if (pred == small || reachability->IsReachable(small, pred)) {
      preds_to_remove.push_back(pred);
    }
  }
  for (HloInstruction* pred : preds_to_remove) {
    ABSL_RETURN_IF_ERROR(pred->RemoveControlDependencyTo(large));
  }
  return absl::OkStatus();
}

// The window of a root array that a permute's operand reads: the operand
// itself, or a chain of unit-stride slices below it composed into one window
// on the array at the bottom of the chain.
struct PermuteWindow {
  HloInstruction* root;
  std::vector<int64_t> starts;
  std::vector<int64_t> limits;

  int64_t rank() const { return starts.size(); }
  int64_t size(int64_t dim) const { return limits[dim] - starts[dim]; }
  bool IsWholeRoot() const {
    for (int64_t i = 0; i < rank(); ++i) {
      if (starts[i] != 0 || limits[i] != root->shape().dimensions(i)) {
        return false;
      }
    }
    return true;
  }
};

static PermuteWindow ResolveWindow(HloInstruction* operand) {
  PermuteWindow w;
  w.root = operand;
  w.starts.assign(operand->shape().dimensions().size(), 0);
  w.limits.assign(operand->shape().dimensions().begin(),
                  operand->shape().dimensions().end());
  while (w.root->opcode() == HloOpcode::kSlice &&
         absl::c_all_of(w.root->slice_strides(),
                        [](int64_t s) { return s == 1; })) {
    for (int64_t i = 0; i < w.rank(); ++i) {
      w.starts[i] += w.root->slice_starts()[i];
      w.limits[i] += w.root->slice_starts()[i];
    }
    w.root = w.root->mutable_operand(0);
  }
  return w;
}

static bool Contains(const PermuteWindow& large, const PermuteWindow& small) {
  for (int64_t i = 0; i < large.rank(); ++i) {
    if (small.starts[i] < large.starts[i] ||
        small.limits[i] > large.limits[i]) {
      return false;
    }
  }
  return true;
}

// The single dimension in which two windows of one root differ, when their
// intervals in it touch or overlap, so that the union of the two windows is
// again a window and moves no element that neither of them moved.
static std::optional<int64_t> UnionDim(const PermuteWindow& a,
                                       const PermuteWindow& b) {
  std::optional<int64_t> dim;
  for (int64_t i = 0; i < a.rank(); ++i) {
    if (a.starts[i] == b.starts[i] && a.limits[i] == b.limits[i]) {
      continue;
    }
    if (dim.has_value()) {
      return std::nullopt;
    }
    if (a.starts[i] > b.limits[i] || b.starts[i] > a.limits[i]) {
      return std::nullopt;
    }
    dim = i;
  }
  return dim;
}

static Shape WindowShape(const PermuteWindow& w) {
  std::vector<int64_t> dims(w.rank());
  for (int64_t i = 0; i < w.rank(); ++i) {
    dims[i] = w.size(i);
  }
  Shape shape = ShapeUtil::MakeShape(w.root->shape().element_type(), dims);
  if (w.root->shape().has_layout()) {
    *shape.mutable_layout() = w.root->shape().layout();
  }
  return shape;
}

// `inner`, a window of the root, as a slice of `value`, which holds the
// window `outer` of the same root.
static HloInstruction* SliceOfWindow(HloComputation* computation,
                                     HloInstruction* value,
                                     const PermuteWindow& outer,
                                     const PermuteWindow& inner,
                                     const Shape& shape) {
  std::vector<int64_t> starts(outer.rank()), limits(outer.rank()),
      strides(outer.rank(), 1);
  for (int64_t i = 0; i < outer.rank(); ++i) {
    starts[i] = inner.starts[i] - outer.starts[i];
    limits[i] = starts[i] + inner.size(i);
  }
  return computation->AddInstruction(
      HloInstruction::CreateSlice(shape, value, starts, limits, strides));
}

static bool IsPlainPermute(const HloInstruction* permute) {
  return permute->operand_count() == 1 && permute->shape().IsArray() &&
         ShapeUtil::Compatible(permute->shape(), permute->operand(0)->shape());
}

absl::StatusOr<bool> CollectivePermuteCSE::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    if (computation->IsFusionComputation()) {
      continue;
    }

    std::unique_ptr<HloReachabilityMap> reachability;

    std::vector<HloInstruction*> permutes;
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kCollectivePermute) {
        permutes.push_back(inst);
      }
    }

    if (permutes.size() < 2) {
      continue;
    }

    for (size_t i = 0; i < permutes.size(); ++i) {
      HloInstruction* a = permutes[i];
      if (!a || a->user_count() == 0) {
        continue;
      }
      auto a_pairs = a->source_target_pairs();
      absl::c_sort(a_pairs);
      for (size_t j = i + 1; j < permutes.size(); ++j) {
        HloInstruction* b = permutes[j];
        if (!b || b->user_count() == 0) {
          continue;
        }

        auto b_pairs = b->source_target_pairs();
        absl::c_sort(b_pairs);

        if (a_pairs != b_pairs) {
          continue;
        }

        HloInstruction* large = nullptr;
        HloInstruction* small = nullptr;
        bool is_slice_cse = false;

        if (a->operand(0) == b->operand(0)) {
          large = a;
          small = b;
        } else if (a->operand(0)->opcode() == HloOpcode::kSlice &&
                   a->operand(0)->operand(0) == b->operand(0)) {
          small = a;
          large = b;
          is_slice_cse = true;
        } else if (b->operand(0)->opcode() == HloOpcode::kSlice &&
                   b->operand(0)->operand(0) == a->operand(0)) {
          small = b;
          large = a;
          is_slice_cse = true;
        } else if (a->operand(0)->opcode() == HloOpcode::kSlice &&
                   b->operand(0)->opcode() == HloOpcode::kSlice &&
                   a->operand(0)->operand(0) == b->operand(0)->operand(0)) {
          if (IsContained(b->operand(0), a->operand(0))) {
            large = b;
            small = a;
            is_slice_cse = true;
          } else if (IsContained(a->operand(0), b->operand(0))) {
            large = a;
            small = b;
            is_slice_cse = true;
          }
        }

        // The operands are not one slice of the same value, but may read
        // windows of the same array through chains of slices: one window
        // inside the other reuses the larger permute, and two windows that
        // differ in one dimension with touching or overlapping intervals
        // are sent by one permute of their union, which moves no element
        // that neither of them moved.
        bool is_window_cse = false;
        PermuteWindow window_large, window_small;
        if (!large && !small && IsPlainPermute(a) && IsPlainPermute(b)) {
          PermuteWindow wa = ResolveWindow(a->mutable_operand(0));
          PermuteWindow wb = ResolveWindow(b->mutable_operand(0));
          if (wa.root == wb.root && wa.root->shape().IsArray()) {
            if (Contains(wa, wb)) {
              large = a;
              small = b;
              window_large = wa;
              window_small = wb;
              is_window_cse = true;
            } else if (Contains(wb, wa)) {
              large = b;
              small = a;
              window_large = wb;
              window_small = wa;
              is_window_cse = true;
            } else if (std::optional<int64_t> dim = UnionDim(wa, wb)) {
              PermuteWindow wu = wa;
              wu.starts[*dim] = std::min(wa.starts[*dim], wb.starts[*dim]);
              wu.limits[*dim] = std::max(wa.limits[*dim], wb.limits[*dim]);
              HloInstruction* union_operand = wu.root;
              Shape union_shape = WindowShape(wu);
              if (!wu.IsWholeRoot()) {
                union_operand =
                    SliceOfWindow(computation, wu.root, ResolveWindow(wu.root),
                                  wu, union_shape);
              }
              HloInstruction* union_permute = computation->AddInstruction(
                  a->CloneWithNewOperands(union_shape, {union_operand}));
              union_permute->set_metadata(a->metadata());
              if (!reachability) {
                reachability = HloReachabilityMap::Build(computation);
              }
              for (HloInstruction* old : {a, b}) {
                for (HloInstruction* pred : old->control_predecessors()) {
                  // A predecessor that itself depends on a or b would form a
                  // cycle through the slices that replace them.
                  if (pred != a && pred != b &&
                      !reachability->IsReachable(a, pred) &&
                      !reachability->IsReachable(b, pred)) {
                    ABSL_RETURN_IF_ERROR(
                        pred->AddControlDependencyTo(union_permute));
                  }
                }
                for (HloInstruction* succ : old->control_successors()) {
                  if (succ != a && succ != b) {
                    ABSL_RETURN_IF_ERROR(
                        union_permute->AddControlDependencyTo(succ));
                  }
                }
              }
              for (auto [old, window] : {std::pair(a, wa), std::pair(b, wb)}) {
                // A unit-stride slice of the old permute (such as the slices
                // an earlier union left) becomes a slice of the union directly.
                for (HloInstruction* user : std::vector<HloInstruction*>(
                         old->users().begin(), old->users().end())) {
                  if (user->opcode() != HloOpcode::kSlice ||
                      !absl::c_all_of(user->slice_strides(),
                                      [](int64_t s) { return s == 1; }) ||
                      !user->control_predecessors().empty() ||
                      !user->control_successors().empty()) {
                    continue;
                  }
                  PermuteWindow sub = window;
                  for (int64_t i = 0; i < sub.rank(); ++i) {
                    sub.starts[i] = window.starts[i] + user->slice_starts()[i];
                    sub.limits[i] = window.starts[i] + user->slice_limits()[i];
                  }
                  HloInstruction* replacement = SliceOfWindow(
                      computation, union_permute, wu, sub, user->shape());
                  ABSL_RETURN_IF_ERROR(user->ReplaceAllUsesWith(replacement));
                  ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(user));
                }
                HloInstruction* old_operand = old->mutable_operand(0);
                ABSL_RETURN_IF_ERROR(old->DropAllControlDeps());
                if (old->user_count() != 0) {
                  HloInstruction* replacement = SliceOfWindow(
                      computation, union_permute, wu, window, old->shape());
                  ABSL_RETURN_IF_ERROR(old->ReplaceAllUsesWith(replacement));
                }
                ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(old));
                if (old_operand->opcode() == HloOpcode::kSlice &&
                    old_operand->user_count() == 0 &&
                    old_operand->control_predecessors().empty() &&
                    old_operand->control_successors().empty()) {
                  ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(old_operand));
                }
              }
              // The union takes a's slot and may absorb further permutes;
              // the map no longer describes the computation.
              reachability.reset();
              permutes[i] = a = union_permute;
              permutes[j] = nullptr;
              changed = true;
              continue;
            }
          }
        }

        if (large && small) {
          if (!reachability) {
            reachability = HloReachabilityMap::Build(computation);
          }
          CHECK(reachability != nullptr);
          ABSL_RETURN_IF_ERROR(
              RemoveControlDependencies(small, large, reachability.get()));

          HloInstruction* replacement = large;
          if (is_window_cse) {
            if (window_small.starts != window_large.starts ||
                !ShapeUtil::Compatible(small->shape(), large->shape())) {
              replacement = SliceOfWindow(computation, large, window_large,
                                          window_small, small->shape());
            }
          } else if (is_slice_cse) {
            const HloInstruction* small_slice = small->operand(0);
            if (large->operand(0)->opcode() == HloOpcode::kSlice) {
              const HloInstruction* large_slice = large->operand(0);
              std::vector<int64_t> new_starts, new_limits, new_strides;
              for (int64_t i = 0; i < small_slice->slice_starts().size(); ++i) {
                new_starts.push_back((small_slice->slice_starts()[i] -
                                      large_slice->slice_starts()[i]) /
                                     large_slice->slice_strides()[i]);
                new_limits.push_back(new_starts.back() +
                                     small_slice->shape().dimensions(i));
                new_strides.push_back(1);
              }
              replacement = computation->AddInstruction(
                  HloInstruction::CreateSlice(small->shape(), large, new_starts,
                                              new_limits, new_strides));
            } else {
              replacement = computation->AddInstruction(
                  small_slice->CloneWithNewOperands(small->shape(), {large}));
            }
          }

          // Force large to dominate small's location. By default, creating a
          // slice of large and replacing it means slice executes after large
          // and replaces small. Is it possible small was before large? Yes. If
          // small was before large, slice(large) will be computed where large
          // is, replacing the outputs that used small. We might affect memory
          // size if large is delayed. Replacement handles dependencies.
          ABSL_RETURN_IF_ERROR(small->ReplaceAllUsesWith(replacement));
          reachability->Replace(small, replacement);
          if (small->user_count() == 0) {
            ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(small));
          }
          if (small == a) {
            permutes[i] = nullptr;
          }
          if (small == b) {
            permutes[j] = nullptr;
          }
          changed = true;
          if (small == a) {
            break;
          }
        }
      }
    }
  }
  return changed;
}

namespace {

const HloPassFactory* const kCollectivePermuteCSERegistration =
    RegisterHloPassFactory<DefaultHloPassFactory<CollectivePermuteCSE>>(
        "collective-permute-cse");

}  // namespace

}  // namespace xla
