/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer.h"

#include "absl/strings/substitute.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace generator {
namespace cpp {

Renderer::Renderer(RendererContext context) : context_(context) {}

Renderer& Renderer::BlankLine() {
  context_.code.AddLineWithoutIndent("");
  return *this;
}

Renderer& Renderer::CodeLine(const string& text) {
  context_.code.AddLineWithoutIndent(text);
  return *this;
}

Renderer& Renderer::CodeLines(const string& text) {
  StringPiece trimmed_text(text);
  str_util::RemoveWhitespaceContext(&trimmed_text);
  for (const string& line : str_util::Split(trimmed_text, '\n')) {
    context_.code.AddLineWithoutIndent(line);
  }
  return *this;
}

Renderer& Renderer::Statement(const string& text) {
  if (str_util::EndsWith(text, ";")) {
    LOG(WARNING) << "Superfluous terminating ';' in '" << text << "'";
    context_.code.AddLineWithIndent(text);
  } else {
    context_.code.AddLineWithIndent(absl::StrCat(text, ";"));
  }
  return *this;
}

Renderer& Renderer::TFStatement(const string& text) {
  return Statement(absl::Substitute("TF_RETURN_IF_ERROR($0)", text));
}

Renderer& Renderer::CommentLine(const string& text) {
  context_.code.AddLineWithIndent(absl::StrCat("// ", text));
  return *this;
}

Renderer& Renderer::BlockOpen(const string& text) {
  context_.code.AddLineWithIndent(absl::StrCat(text, " {"));
  context_.code.IncreaseIndent();
  return *this;
}

Renderer& Renderer::BlockClose(const string& text) {
  context_.code.DecreaseIndent();
  context_.code.AddLineWithIndent(absl::StrCat("}", text));
  return *this;
}

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
