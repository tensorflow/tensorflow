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
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/guard_renderer.h"

#include <algorithm>

#include "tensorflow/c/experimental/ops/gen/common/case_format.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer_context.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {
namespace cpp {

GuardRenderer::GuardRenderer(RendererContext context) : Renderer(context) {
  string self_path = io::JoinPath(context_.path_config.tf_root_dir,
                                  context_.path_config.tf_output_dir,
                                  context_.cpp_config.unit + "_ops.h");
  string with_underscores(self_path);
  std::replace(with_underscores.begin(), with_underscores.end(), '/', '_');
  std::replace(with_underscores.begin(), with_underscores.end(), '.', '_');
  guard_ = toUpperSnake(with_underscores) + "_";
}

void GuardRenderer::Open() {
  CodeLine("#ifndef $0", guard_);
  CodeLine("#define $0", guard_);
  BlankLine();
}

void GuardRenderer::Close() {
  BlankLine();
  CodeLine("#endif  // $0", guard_);
}

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
