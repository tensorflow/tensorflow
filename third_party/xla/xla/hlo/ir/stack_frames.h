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

#ifndef XLA_HLO_IR_STACK_FRAMES_H_
#define XLA_HLO_IR_STACK_FRAMES_H_

#include <string>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/service/hlo.pb.h"

namespace xla {

// Wrapper class for StackFrameIndexProto that bundles the stack frame
// representation together with the operations on that representation.
class StackFrames {
 public:
  StackFrames() = default;
  static absl::StatusOr<StackFrames> FromProto(StackFrameIndexProto proto);

  // Internal helper to add a frame and return its new ID.
  StackFrameId AddStackFrame(const HloStackFrame& frame);

  // Returns the stack frame with the given ID.
  // Returns an empty HloStackFrame if the id is invalid.
  HloStackFrame GetStackFrame(StackFrameId id) const;

  const StackFrameIndexProto& proto() const { return proto_; }

  // Returns true if 'prefix' is a prefix of 'full'.
  bool IsPrefix(StackFrameId prefix, StackFrameId full) const;

  // Concatenates 'prefix' and 'suffix' and returns the new stack frame ID.
  StackFrameId Concatenate(StackFrameId prefix, StackFrameId suffix);

  bool empty() const { return proto_.stack_frames().empty(); }

 private:
  using FileNameId = int;
  using FunctionNameId = int;
  using FileLocationId = int;

  StackFrameIndexProto proto_;

  absl::flat_hash_map<std::string, FileNameId> file_name_to_id_;
  absl::flat_hash_map<std::string, FunctionNameId> function_name_to_id_;

  using FileLocationKey =
      std::tuple<FileNameId, FunctionNameId, int, int, int, int>;
  absl::flat_hash_map<FileLocationKey, FileLocationId> location_to_id_;

  using FrameKey = std::pair<FileLocationId, StackFrameId>;
  absl::flat_hash_map<FrameKey, StackFrameId> frame_to_id_;
};

}  // namespace xla

#endif  // XLA_HLO_IR_STACK_FRAMES_H_
