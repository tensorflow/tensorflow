/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Implementation of an DOT graph renderer that uses Javascript to render DOT to
// SVG in a browser.

#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace hlo_graph_dumper {
namespace {

class GraphHtmlRenderer : public GraphRendererInterface {
 public:
  string RenderGraph(const string& graph, GraphKind graph_kind,
                     const DebugOptions& debug_options) override {
    switch (graph_kind) {
      case DOT_GRAPH:
        return RenderDotAsHTMLFile(graph, debug_options);
      default:
        LOG(FATAL) << "Only DOT graphs can be rendered";
    }
  }
};

XLA_REGISTER_GRAPH_RENDERER(GraphHtmlRenderer);

}  // namespace
}  // namespace hlo_graph_dumper
}  // namespace xla
