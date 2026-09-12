/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/hlo/translate/mhlo_to_hlo/stack_frame_index_builder.h"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/hlo.pb.h"

namespace mlir {
namespace {

int FindId(absl::string_view key,
           const absl::flat_hash_map<absl::string_view, int>& index) {
  auto entry_iterator = index.find(key);
  if (entry_iterator == index.end()) {
    return 0;
  }
  return entry_iterator->second;
}

absl::string_view ToStringView(llvm::StringRef ref) {
  return absl::string_view(ref.data(), ref.size());
}

bool IsFrameNameLocation(mlir::Location location) {
  return isa<mlir::NameLoc>(location) &&
         isa<mlir::FileLineColLoc>(cast<mlir::NameLoc>(location).getChildLoc());
}

std::vector<mlir::NameLoc> CollectFrames(const mlir::Location& loc) {
  std::vector<mlir::NameLoc> frames;
  std::vector<mlir::Location> stack;
  stack.push_back(loc);
  while (!stack.empty()) {
    mlir::Location curr = stack.back();
    stack.pop_back();
    if (auto call_site = dyn_cast<mlir::CallSiteLoc>(curr)) {
      stack.push_back(call_site.getCaller());
      stack.push_back(call_site.getCallee());
    } else if (IsFrameNameLocation(curr)) {
      frames.push_back(cast<mlir::NameLoc>(curr));
    }
  }
  return frames;
}

}  // namespace

int StackFrameIndexBuilder::AddStackFrameLocation(
    const mlir::NameLoc& name_location, int parent_frame_id) {
  mlir::FileLineColLoc file_line_location =
      cast<mlir::FileLineColLoc>(name_location.getChildLoc());

  int line = file_line_location.getLine();
  int end_line = file_line_location.getEndLine();
  int column = file_line_location.getColumn();
  int end_column = file_line_location.getEndColumn();
  absl::string_view filename =
      ToStringView(file_line_location.getFilename().getValue());
  absl::string_view function_name =
      ToStringView(name_location.getName().getValue());

  int filename_id = FindId(filename, file_name_to_id_);
  if (filename_id == 0) {
    indexes_.add_file_names(filename);
    filename_id = indexes_.file_names_size();
    file_name_to_id_[indexes_.file_names(filename_id - 1)] = filename_id;
  }

  int function_name_id = FindId(function_name, function_name_to_id_);
  if (function_name_id == 0) {
    indexes_.add_function_names(function_name);
    function_name_id = indexes_.function_names_size();
    function_name_to_id_[indexes_.function_names(function_name_id - 1)] =
        function_name_id;
  }

  std::tuple<int, int, int, int> location_tuple(filename_id, function_name_id,
                                                line, column);
  auto file_location_iterator = file_location_to_id_.find(location_tuple);
  int file_location_id = 0;
  if (file_location_iterator == file_location_to_id_.end()) {
    auto file_location = indexes_.add_file_locations();
    file_location->set_file_name_id(filename_id);
    file_location->set_function_name_id(function_name_id);
    file_location->set_line(line);
    file_location->set_end_line(end_line);
    file_location->set_column(column);
    file_location->set_end_column(end_column);

    file_location_id = indexes_.file_locations_size();
    file_location_to_id_[location_tuple] = file_location_id;
  } else {
    file_location_id = file_location_iterator->second;
  }

  std::pair<int, int> frame_key(file_location_id, parent_frame_id);
  auto stack_frame_iterator = frame_to_id_.find(frame_key);
  int stack_frame_id = 0;
  if (stack_frame_iterator == frame_to_id_.end()) {
    auto frame = indexes_.add_stack_frames();
    frame->set_file_location_id(file_location_id);
    frame->set_parent_frame_id(parent_frame_id);

    stack_frame_id = indexes_.stack_frames_size();
    frame_to_id_[frame_key] = stack_frame_id;
  } else {
    stack_frame_id = stack_frame_iterator->second;
  }

  return stack_frame_id;
}

int StackFrameIndexBuilder::AddFrames(const mlir::Location& loc,
                                      int parent_frame_id) {
  std::vector<mlir::NameLoc> frames = CollectFrames(loc);
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    parent_frame_id = AddStackFrameLocation(*it, parent_frame_id);
  }
  return parent_frame_id;
}

StackFrameIndexBuilder::AddStackFrameResult
StackFrameIndexBuilder::AddCallStackAndGetFirstFrameId(
    const mlir::Location& root_loc) {
  // Follow the callers down to the first location indexed before, whose frames
  // are all in place, or to the leaf, which is indexed here. The callees above
  // it are then added outermost first, the order a walk of the whole chain
  // would use.
  llvm::SmallVector<mlir::CallSiteLoc> pending_call_sites;
  mlir::Location loc = root_loc;
  int frame_id = StackFrameIndexBuilder::kInvalidIndex;
  while (true) {
    auto call_stack_iterator = call_stack_to_frame_id_.find(loc);
    if (call_stack_iterator != call_stack_to_frame_id_.end()) {
      frame_id = call_stack_iterator->second;
      break;
    }
    auto call_site = dyn_cast<mlir::CallSiteLoc>(loc);
    if (!call_site) {
      frame_id = AddFrames(loc, StackFrameIndexBuilder::kInvalidIndex);
      call_stack_to_frame_id_[loc] = frame_id;
      break;
    }
    pending_call_sites.push_back(call_site);
    loc = call_site.getCaller();
  }
  for (mlir::CallSiteLoc call_site : llvm::reverse(pending_call_sites)) {
    frame_id = AddFrames(call_site.getCallee(), frame_id);
    call_stack_to_frame_id_[mlir::Location(call_site)] = frame_id;
  }

  if (frame_id == StackFrameIndexBuilder::kInvalidIndex) {
    return {StackFrameIndexBuilder::kInvalidIndex, "", 0};
  }

  const xla::StackFrameIndexProto::StackFrame& stack_frame =
      indexes_.stack_frames(frame_id - 1);
  const xla::StackFrameIndexProto::FileLocation& file_location =
      indexes_.file_locations(stack_frame.file_location_id() - 1);
  return {frame_id,
          indexes_.file_names(file_location.file_name_id() - 1),
          file_location.line(),
          file_location.end_line(),
          file_location.column(),
          file_location.end_column()};
}

xla::StackFrameIndexProto StackFrameIndexBuilder::Build() const {
  return std::move(indexes_);
}
}  // namespace mlir
