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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_TOOLS_TOOL_DISPLAY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_TOOLS_TOOL_DISPLAY_H_

#include <functional>
#include <optional>
#include <ostream>

#include "absl/strings/string_view.h"

namespace litert::tools {

// Utility class for interactive logging for usage in command line tools only.
// Allows user to explicitly set target stream.
class ToolDisplay {
  using OptOstreamRefT = std::optional<std::reference_wrapper<std::ostream>>;

 public:
  // Construct configured ToolDisplay. Label is used for prefixing dumps
  // in "LabeledStream". If "dump" is null, all printing through this class
  // is silenced.
  explicit ToolDisplay(OptOstreamRefT display_stream = std::nullopt,
                       absl::string_view tool_label = "");

  // Get out stream.
  std::ostream& Display();

  // Get Display with label prefix.
  std::ostream& Labeled();

  // Get Display with indent.
  std::ostream& Indented();

  // Log string indicating a sub rountine is beginning.
  void Start(absl::string_view start_label);

  // Log string indicating a sub rountine is done and succeeded.
  void Done();

  // Log string indicating a sub rountine is done and failed.
  void Fail();

 private:
  std::string label_;
  std::ostream null_display_ = std::ostream(nullptr);
  OptOstreamRefT display_;
};

}  // namespace litert::tools

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_TOOLS_TOOL_DISPLAY_H_
