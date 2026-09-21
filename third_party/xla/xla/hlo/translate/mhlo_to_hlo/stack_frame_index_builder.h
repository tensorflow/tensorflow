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

#ifndef XLA_HLO_TRANSLATE_MHLO_TO_HLO_STACK_FRAME_INDEX_BUILDER_H_
#define XLA_HLO_TRANSLATE_MHLO_TO_HLO_STACK_FRAME_INDEX_BUILDER_H_

#include <string>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Location.h"
#include "xla/service/hlo.pb.h"

namespace mlir {
class StackFrameIndexBuilder {
 public:
  constexpr static int kInvalidIndex = 0;

  xla::StackFrameIndexProto Build() const;

  struct AddStackFrameResult {
    int last_frame_id;
    std::string last_frame_file;
    int last_frame_line;
    int last_frame_end_line;
    int last_frame_column;
    int last_frame_end_column;
  };

  AddStackFrameResult AddCallStackAndGetFirstFrameId(
      const mlir::Location &root_loc);

 private:
  int AddStackFrameLocation(const mlir::NameLoc &name_location,
                            int parent_frame_id);

  // Adds every frame of `loc` on top of `parent_frame_id`, outermost first,
  // and returns the id of the innermost one (`parent_frame_id` if there is
  // none).
  int AddFrames(const mlir::Location& loc, int parent_frame_id);

  xla::StackFrameIndexProto indexes_;

  // The name keys point into the strings owned by `indexes_`.
  absl::flat_hash_map<absl::string_view, int> function_name_to_id_;
  absl::flat_hash_map<absl::string_view, int> file_name_to_id_;
  absl::flat_hash_map<std::tuple<int, int, int, int>, int> file_location_to_id_;
  absl::flat_hash_map<std::pair<int, int>, int> frame_to_id_;

  // Innermost frame id of every root location and every caller reached from
  // one. Ops share their locations and locations share their callers, so each
  // is indexed once.
  llvm::DenseMap<mlir::Location, int> call_stack_to_frame_id_;
};
}  // namespace mlir

#endif  // XLA_HLO_TRANSLATE_MHLO_TO_HLO_STACK_FRAME_INDEX_BUILDER_H_
