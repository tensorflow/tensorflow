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

#ifndef XLA_SERVICE_SCHEDULING_ANNOTATIONS_UTIL_H_
#define XLA_SERVICE_SCHEDULING_ANNOTATIONS_UTIL_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/collective_pipeliner_utils.h"

namespace xla {

// Defines utility functions for scheduling group annotations. A valid
// annotation should be in the format of "group_id:iteration_id", where group_id
// is a positive integer, and iteration_id is also a positive integer, and
// either of them can be omitted.
//
// Examples:
// example 1. "123:-1" means the instruction has group id 123, and iteration id
// -1.
// example 2. "123" means the instruction has group id 123, and no iteration id.
// example 3. ":-1" means the instruction has no group id, and iteration id
// -1.

struct AnnotationIterationId {
  // TODO(b/399444332): extend this to be a pair of ids marking the relative
  // iteration index for the async-start and async-done instructions.
  int64_t iteration_id;
  friend bool operator==(const AnnotationIterationId& a,
                         const AnnotationIterationId& b) {
    return a.iteration_id == b.iteration_id;
  }
  friend bool operator!=(const AnnotationIterationId& a,
                         const AnnotationIterationId& b) {
    return !(a == b);
  }
  friend bool operator<(const AnnotationIterationId& a,
                        const AnnotationIterationId& b) {
    if (a == b) {
      return false;
    }
    return a.iteration_id < b.iteration_id;
  }
};

using AnnotationGroupId = int64_t;

// Data structure to hold the group id and iteration id of an annotated
// instruction.
struct Annotation {
  std::optional<AnnotationGroupId> group_id;
  std::optional<AnnotationIterationId> iteration_id;

  explicit Annotation(
      std::optional<AnnotationGroupId> group_id = std::nullopt,
      std::optional<AnnotationIterationId> iteration_id = std::nullopt)
      : group_id(group_id), iteration_id(iteration_id) {}

  std::string ToString() const {
    if (group_id.has_value() && iteration_id.has_value()) {
      return absl::StrCat(*group_id, ":", iteration_id->iteration_id);
    }
    if (group_id.has_value()) {
      return absl::StrCat(*group_id);
    }
    if (iteration_id.has_value()) {
      return absl::StrCat(":", iteration_id->iteration_id);
    }
    return "";
  }
  friend bool operator==(const Annotation& a, const Annotation& b) {
    return ((a.group_id == b.group_id) && (a.iteration_id == b.iteration_id));
  }
  friend bool operator!=(const Annotation& a, const Annotation& b) {
    return !(a == b);
  }
  friend bool operator<(const Annotation& a, const Annotation& b) {
    if (a.group_id == b.group_id) {
      if (a.iteration_id == b.iteration_id) {
        return false;
      }
      return a.iteration_id && *a.iteration_id < *b.iteration_id;
    }
    return a.group_id && *a.group_id < *b.group_id;
  }

  template <typename H>
  friend H AbslHashValue(H h, const Annotation& key) {
    return H::combine(std::move(h), key.ToString());
  }
};

// Returns true if the instruction has a scheduling annotation.
bool HasSchedulingAnnotation(const HloInstruction* instr);

// Sets the scheduling annotation for the given instruction.
absl::StatusOr<std::optional<Annotation>> GetSchedulingAnnotation(
    const HloInstruction* instr);

// Sets the scheduling annotation for the given instruction.
absl::Status SetSchedulingAnnotation(HloInstruction* instr,
                                     std::string annotation);

// Same as above.
absl::Status SetSchedulingAnnotation(HloInstruction* instr,
                                     Annotation annotation);

// Removes the scheduling annotation for the given instruction, and returns
// true if the instruction has a scheduling annotation removed.
bool RemoveSchedulingAnnotation(HloInstruction* instr);

// Returns the scheduling annotation iteration id for the given instruction. If
// the instruction does not have a scheduling annotation, or the annotation is
// not an integer returns std::nullopt.
absl::StatusOr<std::optional<AnnotationIterationId>>
GetSchedulingAnnotationIterationId(const HloInstruction* instr);

// Removes the scheduling annotation iteration id for the given instruction,
// and returns true if the instruction has a scheduling annotation iteration id
// removed.
absl::StatusOr<bool> RemoveSchedulingAnnotationIterationId(
    HloInstruction* instr);

// Returns the scheduling annotation group id for the given instruction. If the
// instruction does not have a scheduling annotation, or the annotation is not
// an integer returns std::nullopt.
absl::StatusOr<std::optional<AnnotationGroupId>> GetSchedulingAnnotationGroupId(
    const HloInstruction* instr);

// Sets the scheduling annotation group id for the given instruction.
absl::Status SetSchedulingAnnotationGroupId(HloInstruction* instr,
                                            AnnotationGroupId id);

// Returns the next available scheduling group id for the given module. The next
// available group id is the maximum scheduling group id in the module plus one.
absl::StatusOr<AnnotationGroupId> NextSchedulingGroupId(
    const HloModule& module);

bool IsIterationIdConstentWithPipeliningDirection(
    const AnnotationIterationId& iteration_id,
    collective_pipeliner_utils::PipeliningDirection pipeline_direction);

}  // namespace xla

#endif  // XLA_SERVICE_SCHEDULING_ANNOTATIONS_UTIL_H_
