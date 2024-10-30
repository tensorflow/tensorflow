// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/tools/tool_display.h"

#include <optional>
#include <ostream>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

namespace litert::tools {

ToolDisplay::ToolDisplay(OptOstreamRefT display_stream,
                         const absl::string_view tool_label)
    : display_(display_stream) {
  label_ = absl::StrFormat(
      "[LITERT_TOOLS%s] ",
      tool_label.empty() ? tool_label : absl::StrFormat(":%s", tool_label));
}

std::ostream& ToolDisplay::Display() {
  return display_.has_value() ? display_.value().get() : null_display_;
}

std::ostream& ToolDisplay::Labeled() {
  Display() << label_;
  return Display();
}

std::ostream& ToolDisplay::Indented() {
  Display() << "\t";
  return Display();
}

void ToolDisplay::Start(const absl::string_view start_label) {
  Labeled() << absl::StreamFormat("Starting %s...\n", start_label);
}

void ToolDisplay::Done() {
  Labeled();
  Indented() << "Done!\n";
}

void ToolDisplay::Fail() {
  Labeled();
  Indented() << "Failed\n";
}

}  // namespace litert::tools
