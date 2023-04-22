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
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/op_comment_renderer.h"

#include "tensorflow/c/experimental/ops/gen/cpp/views/op_view.h"

namespace tensorflow {
namespace generator {
namespace cpp {

OpCommentRenderer::OpCommentRenderer(RendererContext context, OpView op)
    : Renderer(context), op_(op) {}

void OpCommentRenderer::Render() {
  CommentLine("Op: $0()", op_.FunctionName());
  CommentLine("Summary: $0", op_.Summary());
  CommentLine("");
  CommentLine("Description:");
  for (const auto& line : op_.Description()) {
    CommentLine("  $0", line);
  }
}

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
