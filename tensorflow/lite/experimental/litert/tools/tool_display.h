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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_TOOL_DISPLAY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_TOOL_DISPLAY_H_

#include <memory>
#include <ostream>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/tools/outstream.h"

namespace litert::tools {

// Utility class for interactive logging for usage in command line tools only.
// Allows user to explicitly set target stream.
class ToolDisplay {
 public:
  using Ptr = std::unique_ptr<ToolDisplay>;
  // Construct configured ToolDisplay. Label is used for prefixing dumps
  // in "LabeledStream".
  explicit ToolDisplay(UserStream&& ostream, absl::string_view tool_label = "")
      : label_(MakeLabel(tool_label)),
        ostream_(std::forward<UserStream>(ostream)) {}
  explicit ToolDisplay(OutStream ostream, absl::string_view tool_label = "")
      : label_(MakeLabel(tool_label)), ostream_(UserStream(ostream)) {}

  ToolDisplay(const ToolDisplay&) = delete;
  ToolDisplay& operator=(const ToolDisplay&) = delete;
  ToolDisplay(ToolDisplay&&) = delete;
  ToolDisplay& operator=(ToolDisplay&&) = delete;

  // Get out stream.
  std::ostream& Display();

  // Get Display with label prefix.
  std::ostream& Labeled();

  // Get Display with indent.
  std::ostream& Indented();

  // Log string indicating a sub rountine is beginning.
  void Start(absl::string_view scope_name);

  // Log string indicating a sub rountine is done and succeeded.
  void Done(absl::string_view scope_name = "");

  // Log string indicating a sub rountine is done and failed.
  void Fail();

  // Logs "start/finish" messages automatically.
  class LoggedScope {
    friend class ToolDisplay;

   public:
    LoggedScope(const LoggedScope&) = delete;
    LoggedScope& operator=(const LoggedScope&) = delete;
    LoggedScope(LoggedScope&&) = delete;
    LoggedScope& operator=(LoggedScope&&) = delete;

    ~LoggedScope();

   private:
    explicit LoggedScope(ToolDisplay& parent, absl::string_view scope_name);

    void Start();
    void Done();

    ToolDisplay& parent_;
    // These should all be from literals.
    absl::string_view scope_name_;
  };

  // Get object that prints a start message and an exit message
  // automatically when it goes out of scope.
  [[maybe_unused]] LoggedScope StartS(absl::string_view scope_name);

 private:
  static std::string MakeLabel(absl::string_view tool_label);
  std::string label_;
  UserStream ostream_;
};

// Print art and info at cli startup.
void DumpPreamble(ToolDisplay& display);

}  // namespace litert::tools

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TOOLS_TOOL_DISPLAY_H_
