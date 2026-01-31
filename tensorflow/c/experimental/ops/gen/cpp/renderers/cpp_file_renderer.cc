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
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/cpp_file_renderer.h"

#include <vector>

#include "tensorflow/c/experimental/ops/gen/cpp/renderers/op_renderer.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer.h"
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/renderer_context.h"
#include "tensorflow/c/experimental/ops/gen/cpp/views/op_view.h"

namespace tensorflow {
namespace generator {
namespace cpp {

static const char *copyright =
    R"(
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
)";

static const char *machine_generated =
    "// This file is MACHINE GENERATED! Do not edit.";

CppFileRenderer::CppFileRenderer(RendererContext context,
                                 const std::vector<OpView> &ops)
    : Renderer(context),
      guard_(context),
      name_space_(context),
      includes_(context),
      ops_(ops) {}

void CppFileRenderer::Render() {
  CodeLines(copyright);
  BlankLine();
  CodeLine(machine_generated);
  BlankLine();

  if (context_.mode == RendererContext::kHeader) {
    guard_.Open();
  } else {
    includes_.SelfHeader();
  }

  includes_.Headers();
  name_space_.Open();
  BlankLine();

  for (const OpView &op : ops_) {
    OpRenderer(context_, op).Render();
  }

  name_space_.Close();
  if (context_.mode == RendererContext::kHeader) {
    guard_.Close();
  }
}

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
