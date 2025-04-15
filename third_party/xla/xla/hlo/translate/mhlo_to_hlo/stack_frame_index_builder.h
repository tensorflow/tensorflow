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

#include <map>
#include <string>
#include <tuple>

#include "absl/strings/string_view.h"
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
  };

  AddStackFrameResult AddCallStackAndGetFirstFrameId(
      const mlir::Location &root_loc);

 private:
  int AddStackFrameLocation(const mlir::NameLoc &name_location,
                            int parent_frame_id);

  xla::StackFrameIndexProto indexes_;

  std::map<absl::string_view, int> function_name_to_id_;
  std::map<absl::string_view, int> file_name_to_id_;
  std::map<std::tuple<int, int, int, int>, int> file_location_to_id_;
  std::map<std::tuple<int, int>, int> frame_to_id_;
};
}  // namespace mlir

#endif  // XLA_HLO_TRANSLATE_MHLO_TO_HLO_STACK_FRAME_INDEX_BUILDER_H_
