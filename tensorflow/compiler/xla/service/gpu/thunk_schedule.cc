/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/thunk_schedule.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace gpu {

void ThunkSchedule::AddDependenciesOnTransitiveOperands(
    const Thunk& thunk, const HloInstruction& operand,
    const std::unordered_map<const HloInstruction*, Thunk*>& hlo_to_thunk) {
  if (hlo_to_thunk.count(&operand)) {
    // If `operand` is mapped to a thunk, adds `operand` to `thunk`'s dependency
    // list if `operand` is assigned to a different stream. As an optimization,
    // we skip `operand`'s operands because `operand` depends on them already.
    if (stream_assignment_->StreamNumberForHlo(operand) !=
        stream_assignment_->StreamNumberForHlo(*thunk.hlo_instruction())) {
      depends_on_[&thunk].push_back(FindOrDie(hlo_to_thunk, &operand));
    }
  } else {
    // If `operand` doesn't need a thunk (e.g. bitcast), continue with its
    // operands.
    for (const auto* operand_of_operand : operand.operands()) {
      AddDependenciesOnTransitiveOperands(thunk, *operand_of_operand,
                                          hlo_to_thunk);
    }
  }
}

ThunkSchedule::ThunkSchedule(
    std::unique_ptr<ThunkSequence> thunks,
    std::unique_ptr<StreamAssignment> stream_assignment,
    const std::vector<const HloInstruction*>& hlo_total_order)
    : thunks_(std::move(thunks)),
      stream_assignment_(std::move(stream_assignment)) {
  std::unordered_map<const HloInstruction*, Thunk*> hlo_to_thunk;
  for (const auto& thunk : *thunks_) {
    InsertOrDie(&hlo_to_thunk, thunk->hlo_instruction(), thunk.get());
  }

  for (const HloInstruction* hlo : hlo_total_order) {
    if (hlo_to_thunk.count(hlo)) {
      thunk_total_order_.push_back(FindOrDie(hlo_to_thunk, hlo));
    }
  }

  for (const Thunk* thunk : thunk_total_order_) {
    const auto* dst = thunk->hlo_instruction();
    CHECK(stream_assignment_->HasStreamAssigned(*dst));
    for (const auto* src : dst->operands()) {
      AddDependenciesOnTransitiveOperands(*thunk, *src, hlo_to_thunk);
    }
  }

  RemoveRedundantDependencyEdges();

  // Compute `depended_by_`, the inverse of `depends_on_`.
  for (const auto& dependency : depends_on_) {
    for (const auto* depended : dependency.second) {
      depended_by_.insert(depended);
    }
  }
}

void ThunkSchedule::RemoveRedundantDependencyEdges() {
  std::unordered_map<const Thunk*, int> thunk_to_total_order;
  for (int i = 0; i < thunk_total_order_.size(); ++i) {
    InsertOrDie(&thunk_to_total_order, thunk_total_order_[i], i);
  }

  int stream_count = stream_assignment_->StreamCount();
  // S1  S2
  //
  // T1<----+
  //        |
  // T3<--+ |
  //      | | depends on
  //     T4 |
  //        |
  //     T2-+
  //
  // Suppose thunk T1 and T3 are scheduled on stream S1, and T2 and T4 are on
  // stream S2. If T2 depends on T1 and T4 depends on T3, and
  // order(T1)<order(T3)<order(T4)<order(T2), the dependency of T2 on T1 is
  // redundant.
  //
  // To efficiently detect such redundancy, we leverage array `last_dependency`.
  // last_dependency[S1][S2] indicates the last thunk (with the maximum order
  // number) on stream S2 that thunks on S1 depends on. Therefore, if a future
  // S1 thunk depends on a S2 thunk ordered <=last_dependency[S1][S2], that is a
  // redundant dependency edge.
  Array2D<int> last_dependency(stream_count, stream_count, -1);
  for (const Thunk* dst : thunk_total_order_) {
    if (!depends_on_.count(dst)) {
      continue;
    }

    int dst_stream =
        stream_assignment_->StreamNumberForHlo(*dst->hlo_instruction());
    std::list<const Thunk*>& sources = FindOrDie(depends_on_, dst);
    for (auto iter = sources.begin(); iter != sources.end();) {
      const Thunk* src = *iter;
      // `dst` depends on `src`.
      int src_stream =
          stream_assignment_->StreamNumberForHlo(*src->hlo_instruction());
      int src_order = FindOrDie(thunk_to_total_order, src);
      if (src_order <= last_dependency(dst_stream, src_stream)) {
        iter = sources.erase(iter);
      } else {
        last_dependency(dst_stream, src_stream) = src_order;
        ++iter;
      }
    }
    if (sources.empty()) {
      depends_on_.erase(dst);
    }
  }
}

const std::list<const Thunk*>& ThunkSchedule::DependsOn(
    const Thunk* thunk) const {
  if (depends_on_.count(thunk)) {
    return FindOrDie(depends_on_, thunk);
  } else {
    return empty_thunk_list_;
  }
}

string ThunkSchedule::ToString() const {
  string result = "Total order:\n";
  for (Thunk* thunk : thunk_total_order_) {
    absl::StrAppend(&result, "\t", thunk->hlo_instruction()->ToString(), "\n");
  }
  absl::StrAppend(&result, "Dependencies:\n");
  for (const auto& entry : depends_on_) {
    const Thunk* dependent = entry.first;
    for (const Thunk* dependency : entry.second) {
      absl::StrAppend(&result, "\t", dependent->hlo_instruction()->name(),
                      " depends on ", dependency->hlo_instruction()->name(),
                      "\n");
    }
  }
  return result;
}

}  // namespace gpu
}  // namespace xla
