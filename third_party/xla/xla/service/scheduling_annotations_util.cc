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

#include "xla/service/scheduling_annotations_util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/collective_pipeliner_utils.h"
#include "xla/side_effect_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

constexpr absl::string_view delimiter = ":";

absl::Status VerifyAnnotation(const HloInstruction* instr,
                              absl::string_view annotation) {
  auto verify_integer_or_empty =
      [instr, annotation](
          absl::string_view str, absl::string_view field_name,
          bool verify_non_negative_integer = false) -> absl::Status {
    if (str.empty()) {
      return absl::OkStatus();
    }
    int64_t integer;
    if (!absl::SimpleAtoi(str, &integer)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Instruction has a non-integer scheduling annotation ", field_name,
          ", inst: ", instr->name(), ", annotation: ", annotation));
    }
    if (verify_non_negative_integer && integer < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Instruction has a negative scheduling annotation ", field_name,
          ", inst: ", instr->name(), ", annotation: ", annotation));
    }
    return absl::OkStatus();
  };
  std::vector<absl::string_view> annotation_fields =
      absl::StrSplit(annotation, delimiter);
  CHECK_GE(annotation_fields.size(), 1);
  if (annotation_fields.size() > 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Instruction has more than 2 scheduling annotation fields, inst: ",
        instr->name(), ", annotation: ", annotation));
  }
  TF_RETURN_IF_ERROR(verify_integer_or_empty(
      annotation_fields[0], "group id", /*verify_non_negative_integer=*/true));
  if (annotation_fields.size() == 2) {
    TF_RETURN_IF_ERROR(
        verify_integer_or_empty(annotation_fields[1], "iteration id"));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<Annotation>> ParseAnnotation(
    const HloInstruction* instr) {
  const auto& attrs = instr->frontend_attributes().map();
  if (!attrs.contains(kXlaSchedulingGroupIdAttr)) {
    return std::nullopt;
  }
  absl::string_view annotation_str = attrs.at(kXlaSchedulingGroupIdAttr);
  VLOG(2) << "Annotated instruction: " << instr->name() << " "
          << annotation_str;
  TF_RETURN_IF_ERROR(VerifyAnnotation(instr, annotation_str));
  std::vector<absl::string_view> annotation_fields =
      absl::StrSplit(annotation_str, delimiter);

  auto parse_integer = [](absl::string_view str) -> std::optional<int64_t> {
    if (str.empty()) {
      return std::nullopt;
    }
    int64_t integer;
    CHECK(absl::SimpleAtoi(str, &integer));
    return integer;
  };

  Annotation annotation;
  annotation.group_id = parse_integer(annotation_fields[0]);
  if (annotation_fields.size() == 2 &&
      parse_integer(annotation_fields[1]).has_value()) {
    annotation.iteration_id = AnnotationIterationId{
        .iteration_id = *parse_integer(annotation_fields[1])};
  }

  return annotation;
}

}  // namespace

bool HasSchedulingAnnotation(const HloInstruction* instr) {
  return instr->frontend_attributes().map().contains(kXlaSchedulingGroupIdAttr);
}

absl::StatusOr<std::optional<Annotation>> GetSchedulingAnnotation(
    const HloInstruction* instr) {
  return ParseAnnotation(instr);
}

absl::Status SetSchedulingAnnotation(HloInstruction* instr,
                                     std::string annotation) {
  TF_RETURN_IF_ERROR(VerifyAnnotation(instr, annotation));
  FrontendAttributes frontend_attributes = instr->frontend_attributes();
  if (frontend_attributes.map().contains(kXlaSchedulingGroupIdAttr)) {
    frontend_attributes.mutable_map()->find(kXlaSchedulingGroupIdAttr)->second =
        annotation;
  } else {
    frontend_attributes.mutable_map()->insert(
        {kXlaSchedulingGroupIdAttr, annotation});
  }
  instr->set_frontend_attributes(frontend_attributes);
  return absl::OkStatus();
}

absl::Status SetSchedulingAnnotation(HloInstruction* instr,
                                     Annotation annotation) {
  return SetSchedulingAnnotation(instr, annotation.ToString());
}

bool RemoveSchedulingAnnotation(HloInstruction* instr) {
  FrontendAttributes frontend_attributes = instr->frontend_attributes();
  if (!frontend_attributes.map().contains(kXlaSchedulingGroupIdAttr)) {
    return false;
  }
  frontend_attributes.mutable_map()->erase(kXlaSchedulingGroupIdAttr);
  instr->set_frontend_attributes(frontend_attributes);
  return true;
}

absl::StatusOr<std::optional<AnnotationIterationId>>
GetSchedulingAnnotationIterationId(const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto annotation, ParseAnnotation(instr));
  if (!annotation.has_value()) {
    return std::nullopt;
  }
  return annotation->iteration_id;
}

absl::StatusOr<bool> RemoveSchedulingAnnotationIterationId(
    HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(std::optional<Annotation> annotation,
                      GetSchedulingAnnotation(instr));
  if (!annotation || !annotation->iteration_id) {
    return false;
  }
  if (!annotation->group_id) {
    // If the annotation has no group id, we remove the annotation entirely.
    return RemoveSchedulingAnnotation(instr);
  }
  annotation->iteration_id = std::nullopt;
  TF_RETURN_IF_ERROR(SetSchedulingAnnotation(instr, *annotation));
  return true;
}

absl::StatusOr<std::optional<int64_t>> GetSchedulingAnnotationGroupId(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto annotation, ParseAnnotation(instr));
  if (!annotation.has_value()) {
    return std::nullopt;
  }
  return annotation->group_id;
}

absl::Status SetSchedulingAnnotationGroupId(HloInstruction* instr, int64_t id) {
  return SetSchedulingAnnotation(instr, absl::StrCat(id));
}

absl::StatusOr<AnnotationGroupId> NextSchedulingGroupId(
    const HloModule& module) {
  int64_t next_scheduling_id = 1;
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* hlo : comp->instructions()) {
      TF_ASSIGN_OR_RETURN(std::optional<int64_t> scheduling_id,
                          GetSchedulingAnnotationGroupId(hlo));
      if (scheduling_id.has_value()) {
        next_scheduling_id =
            std::max(next_scheduling_id, scheduling_id.value() + 1);
      }
    }
  }
  return next_scheduling_id;
}

bool IsIterationIdConstentWithPipeliningDirection(
    const AnnotationIterationId& iteration_id,
    collective_pipeliner_utils::PipeliningDirection pipeline_direction) {
  if (pipeline_direction ==
      collective_pipeliner_utils::PipeliningDirection::kForward) {
    return iteration_id.iteration_id == 1;
  }
  if (pipeline_direction ==
      collective_pipeliner_utils::PipeliningDirection::kBackward) {
    return iteration_id.iteration_id == -1;
  }
  return false;
}

}  // namespace xla
