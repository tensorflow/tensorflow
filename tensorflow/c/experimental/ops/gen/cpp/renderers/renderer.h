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
#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_RENDERERS_RENDERER_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_RENDERERS_RENDERER_H_

#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer_context.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {
namespace cpp {

class Renderer {
 public:
  explicit Renderer(RendererContext context);

 protected:
  // Append a blank line.
  Renderer &BlankLine();

  // Append a line of source code, left-justified (not indented).
  // Use for preprocessors directives ("#include"), namespaces, etc.
  Renderer &CodeLine(const string &text);
  template <typename... Args>
  Renderer CodeLine(absl::string_view text, const Args &...args) {
    return CodeLine(absl::Substitute(text, args...));
  }

  // Append a multiline string of source code, left-justified (not indented).
  // Note: Trims leading/trailing whitespace including newlines, making this
  //       method convenient for multiline raw strings.
  // Newlines ('\n') are allowed/expected.
  Renderer &CodeLines(const string &text);
  template <typename... Args>
  Renderer CodeLines(absl::string_view text, const Args &...args) {
    return CodeLines(absl::Substitute(text, args...));
  }

  // Indent and append a C++ statement.
  // Note: do *not* include a trailing semicolon in the statement text.
  Renderer &Statement(const string &text);
  template <typename... Args>
  Renderer Statement(absl::string_view text, const Args &...args) {
    return Statement(absl::Substitute(text, args...));
  }

  // Indent and append a call to a TF method returning a Status to check.
  // Note: do *not* include a trailing semicolon in the statement text.
  Renderer &TFStatement(const string &text);
  template <typename... Args>
  Renderer TFStatement(absl::string_view text, const Args &...args) {
    return TFStatement(absl::Substitute(text, args...));
  }

  // Indent and append a C++ single-line style comment (using '//').
  Renderer &CommentLine(const string &text = "");
  template <typename... Args>
  Renderer CommentLine(absl::string_view text, const Args &...args) {
    return CommentLine(absl::Substitute(text, args...));
  }

  // Append a line of code which starts a new block: trailing with '{') and
  // indenting.
  Renderer &BlockOpen(const string &text);
  template <typename... Args>
  Renderer BlockOpen(absl::string_view text, const Args &...args) {
    return BlockOpen(absl::Substitute(text, args...));
  }

  // Append a line of code ending a block: unindenting and adding '}'.
  // Note: optional trailing text is often a comment, e.g. '// namespace xyz'.
  Renderer &BlockClose(const string &text = "");
  template <typename... Args>
  Renderer BlockClose(absl::string_view text, const Args &...args) {
    return BlockClose(absl::Substitute(text, args...));
  }

 protected:
  RendererContext context_;
};

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_CPP_RENDERERS_RENDERER_H_
