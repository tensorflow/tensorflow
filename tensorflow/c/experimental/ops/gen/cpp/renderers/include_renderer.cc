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
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/include_renderer.h"

#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer_context.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {
namespace cpp {

IncludeRenderer::IncludeRenderer(RendererContext context) : Renderer(context) {}

void IncludeRenderer::SelfHeader() {
  Include(SelfHeaderPath());
  BlankLine();
}

string IncludeRenderer::SelfHeaderPath() const {
  return io::JoinPath(context_.path_config.tf_root_dir,
                      context_.path_config.tf_output_dir,
                      context_.cpp_config.unit + "_ops.h");
}

void IncludeRenderer::Include(const string &tf_file_path) {
  CodeLine("#include \"$0\"",
           io::JoinPath(context_.path_config.tf_prefix_dir, tf_file_path));
}

void IncludeRenderer::Headers() {
  Include("tensorflow/c/eager/abstract_context.h");
  Include("tensorflow/c/eager/abstract_tensor_handle.h");
  if (context_.mode == RendererContext::kSource) {
    Include("tensorflow/c/eager/tracing_utils.h");
    Include("tensorflow/core/framework/types.h");
    Include("tensorflow/core/platform/errors.h");
    BlankLine();
    Statement("using tensorflow::tracing::MaybeSetOpName");
  }
  BlankLine();
}

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
