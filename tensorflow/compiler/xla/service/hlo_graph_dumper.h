/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_GRAPH_DUMPER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_GRAPH_DUMPER_H_

#include <string>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {
namespace hlo_graph_dumper {

// Abstract interface for classes that render HLO graphs (e.g. DOT graph,
// tensorflow GraphDef).
class GraphRendererInterface {
 public:
  enum GraphKind {
    DOT_GRAPH,
    TF_GRAPHDEF,
  };

  virtual ~GraphRendererInterface() = default;

  // Renders a DOT graph, returning a description of the rendered output
  // (e.g., a URL)
  virtual string RenderGraph(const string& graph, GraphKind graph_kind,
                             const DebugOptions& debug_options) = 0;
};

// Dump the given HLO module if a dump is requested in its debug options. Based
// on the debug options, either a graph dump, a text dump or both may be
// generated. If a graph dump is generated, the description (e.g. an URL) is
// returned; otherwise an empty string is returned.
string MaybeDumpHloModule(const HloModule& module, const string& label,
                          const HloExecutionProfile* profile = nullptr);

// Dumps a graph of the computation and returns a description of the rendered
// graph (e.g., a URL) based on the renderer. The "best" renderer in the
// registry is used.
string DumpGraph(const HloComputation& computation, const string& label,
                 const DebugOptions& debug_options,
                 const HloExecutionProfile* hlo_execution_profile = nullptr,
                 bool show_backend_config = false);

// Like DumpGraph, but renders only nodes "near" the given node in the graph.
//
// The number of nodes dumped is controlled by the radius parameter, which
// (roughly) corresponds to the max distance a node may be from the primary node
// before it's omitted from the graph.
// 
// The optional boundary parameter specifies the set of boundary nodes which
// will be omitted when they are within the radius.
string DumpNeighborhoodAround(
    const HloInstruction& node, int radius,
    const std::set<const HloInstruction*>* boundary = nullptr,
    bool show_backend_config = false);

// Dumps nodes on any of the paths from `from` to `to`.  If there are more than
// max_nodes on all paths, restricts to the max_nodes nodes on the shortest
// paths.
string DumpAllPathsFromTo(const HloInstruction& from, const HloInstruction& to,
                          int64 max_nodes, bool show_backend_config = false);

// Dumps the HloModule::ToString() as a file into the provided directory path
// suffixed with the provided label.
//
// If do_prefix is true, a timestamp will be prepended onto the label to
// construct a filename in the directory path; otherwise, the label is used
// as the filename directly.
void DumpText(const HloModule& module, const string& label,
              const string& directory_path, bool do_prefix = true);

// Renders DOT graph as inline SVG and saves it in an HTML file in a temprary
// directory or directory specified via --xla_hlo_graph_path. Returns the file
// URI pointing to the file.
string RenderDotAsHTMLFile(const string& dot,
                           const DebugOptions& debug_options);

// Graph renderers may be added using a registration mechanism, e.g.:
// XLA_REGISTER_GRAPH_RENDERER(AGraphRendererClass, 100)
// The renderer with the highest numeric priority value is used.

#define XLA_REGISTER_GRAPH_RENDERER(factory, ...) \
  XLA_INTERNAL_REGISTER_GRAPH_RENDERER(factory, __COUNTER__, ##__VA_ARGS__)

// Internal implementation details below this point.

// Class that registers a graph renderer.
class Registrar {
 public:
  Registrar(std::shared_ptr<GraphRendererInterface> dumper);
};

#define XLA_INTERNAL_REGISTER_GRAPH_RENDERER(factory, ctr, ...) \
  static ::xla::hlo_graph_dumper::Registrar                     \
      XLA_INTERNAL_REGISTER_GRAPH_RENDERER_NAME(ctr)(           \
          std::make_shared<factory>(), ##__VA_ARGS__)

// __COUNTER__ must go through another macro to be properly expanded
#define XLA_INTERNAL_REGISTER_GRAPH_RENDERER_NAME(ctr) ___##ctr##__object_

}  // namespace hlo_graph_dumper
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_GRAPH_DUMPER_H_
