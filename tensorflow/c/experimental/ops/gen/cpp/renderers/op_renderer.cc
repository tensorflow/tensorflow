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
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/op_renderer.h"

#include <iterator>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/op_implementation_renderer.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer_context.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/op_argument_view.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/op_view.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {
namespace cpp {

string OpRenderer::Signature() const {
  std::vector<string> args_with_default_val;
  std::vector<string> args_without_default_val;
  for (OpArgumentView const& argument : op_.AllArguments()) {
    string text = argument.Declaration();
    if (context_.mode == RendererContext::kHeader) {
      absl::StrAppend(&text, argument.Initializer());
    }
    if (argument.HasDefaultValue()) {
      args_with_default_val.push_back(text);
    } else {
      args_without_default_val.push_back(text);
    }
  }
  std::vector<string> arguments;
  arguments.reserve(args_without_default_val.size() +
                    args_with_default_val.size());
  arguments.insert(arguments.end(),
                   std::make_move_iterator(args_without_default_val.begin()),
                   std::make_move_iterator(args_without_default_val.end()));
  arguments.insert(arguments.end(),
                   std::make_move_iterator(args_with_default_val.begin()),
                   std::make_move_iterator(args_with_default_val.end()));
  return absl::Substitute("$0 $1($2)", "Status", op_.FunctionName(),
                          absl::StrJoin(arguments, ", "));
}

OpRenderer::OpRenderer(RendererContext context, OpView op)
    : Renderer(context), op_(op), comment_(context, op) {}

void OpRenderer::Render() {
  comment_.Render();
  if (context_.mode == RendererContext::kHeader) {
    Statement(Signature());
  } else {
    BlockOpen(Signature());
    OpImplementationRenderer(context_, op_).Render();
    BlockClose();
  }
  BlankLine();
}

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
