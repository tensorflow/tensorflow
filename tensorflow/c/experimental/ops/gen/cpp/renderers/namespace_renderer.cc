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
#include "tensorflow/c/experimental/ops/gen/cpp/renderers/namespace_renderer.h"

#include "absl/strings/str_split.h"

namespace tensorflow {
namespace generator {
namespace cpp {

NamespaceRenderer::NamespaceRenderer(RendererContext context)
    : Renderer(context) {}

void NamespaceRenderer::Open() {
  for (const string& ns : context_.cpp_config.namespaces) {
    CodeLine("namespace " + ns + " {");
  }
}

void NamespaceRenderer::Close() {
  for (auto it = context_.cpp_config.namespaces.rbegin();
       it != context_.cpp_config.namespaces.rend(); ++it) {
    CodeLine("}  // namespace " + *it);
  }
}

}  // namespace cpp
}  // namespace generator
}  // namespace tensorflow
