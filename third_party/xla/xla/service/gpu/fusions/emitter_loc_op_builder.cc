/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/fusions/emitter_loc_op_builder.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"

namespace xla::gpu {

// Aligns the annotations to the Nth character of the lines.
constexpr size_t kAnnotationPadding = 100ul;

/* static */ std::string EmitterLocOpBuilder::FormatTritonIrWithAnnotations(
    absl::string_view mlir_ir) {
  auto triton_with_annotations = absl::StrSplit(mlir_ir, '\n');
  std::vector<std::string> formatted_lines;
  for (auto& line : triton_with_annotations) {
    std::vector<std::string> line_and_annotation = absl::StrSplit(line, '"');
    constexpr int kInstructionLineFragments = 3;
    if (line_and_annotation.size() != kInstructionLineFragments) {
      // The line does not matches with the pattern:
      // x = instruction(y, z) "annotation"
      // So we just add it to the output as is.
      formatted_lines.emplace_back(line);
      continue;
    }
    auto text_size =
        std::min(line_and_annotation[0].size(), kAnnotationPadding);
    auto new_line =
        absl::StrCat(line_and_annotation[0],
                     std::string(kAnnotationPadding - text_size, ' '), "\"",
                     line_and_annotation[1], "\"", line_and_annotation[2]);
    formatted_lines.emplace_back(new_line);
  }
  return absl::StrJoin(formatted_lines, "\n");
}

mlir::Location EmitterLocOpBuilder::Loc(
    EmitterLocOpBuilder::SourceLocation location) const {
  if (!annotate_loc_ || location.line() == 0) {
    return current_loc_;
  }
  std::vector<std::string> file_name =
      absl::StrSplit(location.file_name(), '/');
  std::string previous_loc;
  if (mlir::isa<mlir::NameLoc>(current_loc_)) {
    auto name_loc = mlir::cast<mlir::NameLoc>(current_loc_);
    previous_loc = name_loc.getName().str();
  }

  const std::string text = absl::StrCat(previous_loc, " -> ", file_name.back(),
                                        ":", location.line());
  return mlir::NameLoc::get(mlir::StringAttr::get(getContext(), text));
}

}  // namespace xla::gpu
