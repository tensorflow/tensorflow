/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/ir/stack_frames.h"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/service/hlo.pb.h"

namespace xla {

absl::StatusOr<StackFrames> StackFrames::FromProto(StackFrameIndexProto proto) {
  for (int i = 0; i < proto.file_locations_size(); ++i) {
    const auto& loc = proto.file_locations(i);
    if (loc.file_name_id() < 1 ||
        loc.file_name_id() > proto.file_names_size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid file_name_id ", loc.file_name_id()));
    }
    if (loc.function_name_id() < 1 ||
        loc.function_name_id() > proto.function_names_size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid function_name_id ", loc.function_name_id()));
    }
  }
  for (int i = 0; i < proto.stack_frames_size(); ++i) {
    const auto& frame = proto.stack_frames(i);
    if (frame.file_location_id() < 1 ||
        frame.file_location_id() > proto.file_locations_size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid file_location_id ", frame.file_location_id()));
    }
    if (frame.parent_frame_id() != 0) {
      if (frame.parent_frame_id() < 1 ||
          frame.parent_frame_id() > proto.stack_frames_size()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid parent_frame_id ", frame.parent_frame_id()));
      }
    }
  }

  StackFrames dag;
  dag.proto_ = std::move(proto);
  for (int i = 0; i < dag.proto_.file_names_size(); ++i) {
    if (!dag.file_name_to_id_.emplace(dag.proto_.file_names(i), i + 1).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate file name: ", dag.proto_.file_names(i)));
    }
  }
  for (int i = 0; i < dag.proto_.function_names_size(); ++i) {
    if (!dag.function_name_to_id_.emplace(dag.proto_.function_names(i), i + 1)
             .second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Duplicate function name: ", dag.proto_.function_names(i)));
    }
  }
  for (int i = 0; i < dag.proto_.file_locations_size(); ++i) {
    const auto& loc = dag.proto_.file_locations(i);
    if (!dag.location_to_id_
             .emplace(
                 std::make_tuple(loc.file_name_id(), loc.function_name_id(),
                                 loc.line(), loc.column(), loc.end_line(),
                                 loc.end_column()),
                 i + 1)
             .second) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate file location at index ", i));
    }
  }
  for (int i = 0; i < dag.proto_.stack_frames_size(); ++i) {
    const auto& frame = dag.proto_.stack_frames(i);
    if (!dag.frame_to_id_
             .emplace(std::make_pair(frame.file_location_id(),
                                     StackFrameId{frame.parent_frame_id()}),
                      StackFrameId{i + 1})
             .second) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate stack frame at index ", i));
    }
  }
  return dag;
}

HloStackFrame StackFrames::GetStackFrame(StackFrameId id) const {
  if (!id.valid() || id.value > proto_.stack_frames_size()) {
    return {};
  }
  const auto& frame_proto = proto_.stack_frames(id.value - 1);
  const auto& loc_proto =
      proto_.file_locations(frame_proto.file_location_id() - 1);
  return {proto_.file_names(loc_proto.file_name_id() - 1),
          proto_.function_names(loc_proto.function_name_id() - 1),
          loc_proto.line(),
          loc_proto.column(),
          loc_proto.end_line(),
          loc_proto.end_column(),
          StackFrameId{frame_proto.parent_frame_id()}};
}

StackFrameId StackFrames::AddStackFrame(const HloStackFrame& frame) {
  auto [file_it, file_inserted] = file_name_to_id_.try_emplace(
      std::string(frame.file_name), proto_.file_names_size() + 1);
  if (file_inserted) {
    proto_.add_file_names(std::string(frame.file_name));
  }
  FileNameId file_id = file_it->second;

  auto [func_it, func_inserted] = function_name_to_id_.try_emplace(
      std::string(frame.function_name), proto_.function_names_size() + 1);
  if (func_inserted) {
    proto_.add_function_names(std::string(frame.function_name));
  }
  FunctionNameId func_id = func_it->second;

  FileLocationKey loc_key = {file_id,      func_id,        frame.line,
                             frame.column, frame.end_line, frame.end_column};
  auto [loc_it, loc_inserted] =
      location_to_id_.try_emplace(loc_key, proto_.file_locations_size() + 1);
  if (loc_inserted) {
    auto* loc_proto = proto_.add_file_locations();
    loc_proto->set_file_name_id(file_id);
    loc_proto->set_function_name_id(func_id);
    loc_proto->set_line(frame.line);
    loc_proto->set_column(frame.column);
    loc_proto->set_end_line(frame.end_line);
    loc_proto->set_end_column(frame.end_column);
  }
  FileLocationId loc_id = loc_it->second;

  FrameKey frame_key = {loc_id, frame.parent_frame_id};
  auto [frame_it, frame_inserted] = frame_to_id_.try_emplace(
      frame_key, StackFrameId{proto_.stack_frames_size() + 1});
  if (frame_inserted) {
    auto* frame_proto = proto_.add_stack_frames();
    frame_proto->set_file_location_id(loc_id);
    frame_proto->set_parent_frame_id(frame.parent_frame_id.value);
  }
  return frame_it->second;
}

bool StackFrames::IsPrefix(StackFrameId prefix, StackFrameId full) const {
  if (!prefix.valid()) {
    return true;
  }
  while (full.valid()) {
    if (full == prefix) {
      return true;
    }
    full = GetStackFrame(full).parent_frame_id;
  }
  return false;
}

StackFrameId StackFrames::Concatenate(StackFrameId prefix,
                                      StackFrameId suffix) {
  std::vector<HloStackFrame> frames;
  while (suffix.valid()) {
    frames.push_back(GetStackFrame(suffix));
    suffix = frames.back().parent_frame_id;
  }

  StackFrameId current_id = prefix;
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    HloStackFrame frame = *it;
    frame.parent_frame_id = current_id;
    current_id = AddStackFrame(frame);
  }
  return current_id;
}

}  // namespace xla
