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

#include <ostream>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/tools/outstream.h"

namespace litert::tools {

std::string ToolDisplay::MakeLabel(absl::string_view tool_label) {
  return absl::StrFormat(
      "[LITERT_TOOLS%s] ",
      tool_label.empty() ? tool_label : absl::StrFormat(":%s", tool_label));
}

std::ostream& ToolDisplay::Display() { return ostream_.Get(); }

std::ostream& ToolDisplay::Labeled() {
  Display() << label_;
  return Display();
}

std::ostream& ToolDisplay::Indented() {
  Display() << "\t";
  return Display();
}

void ToolDisplay::Start(const absl::string_view scope_name) {
  static constexpr absl::string_view kStartFmt = "Starting %s...\n";
  Labeled() << absl::StreamFormat(kStartFmt, scope_name);
}

void ToolDisplay::Done(const absl::string_view scope_name) {
  static constexpr absl::string_view kDoneFmt = "%s Done!\n";
  Labeled() << "";
  Indented() << absl::StreamFormat(kDoneFmt, scope_name);
}

void ToolDisplay::Fail() {
  Labeled() << "";
  Indented() << "Failed\n";
}

ToolDisplay::LoggedScope ToolDisplay::StartS(absl::string_view scope_name) {
  return LoggedScope(*this, scope_name);
}

void ToolDisplay::LoggedScope::Start() { parent_.Start(scope_name_); }

void ToolDisplay::LoggedScope::Done() { parent_.Done(scope_name_); }

ToolDisplay::LoggedScope::~LoggedScope() { Done(); }

ToolDisplay::LoggedScope::LoggedScope(ToolDisplay& parent,
                                      absl::string_view scope_name)
    : parent_(parent), scope_name_(scope_name) {
  Start();
}

static constexpr absl::string_view kArt = R"(
    __    _ __       ____  __
   / /   (_/ /____  / __ \/ /_
  / /   / / __/ _ \/ /_/ / __/
 / /___/ / /_/  __/ _, _/ /_
/_____/_/\__/\___/_/ |_|\__/
)";

void DumpPreamble(ToolDisplay& display) { display.Display() << kArt << "\n"; }

}  // namespace litert::tools
