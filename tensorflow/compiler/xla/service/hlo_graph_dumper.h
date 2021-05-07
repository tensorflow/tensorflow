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

// This file contains routines for rendering HLO computations into a
// human-readable graphical format.
//
// Fundamentally all graphs are rendered using the DOT language, but they can be
// packaged four different ways:
//
//  - as a raw DOT file, which can be rendered using `graphviz`.
//
//  - as an HTML file with an embedded DOT file, which can be viewed in a
//    browser using a version of graphviz compiled to JavaScript
//
//  - as a URL hosted somewhere which somehow embeds the DOT file.
//
//  - as an HTML page showing the fusion progress.
//
// Two last options are not implemented by default, but you can add a plugin to
// implement it via RegisterGraphToURLRenderer.
//
// TODO(jlebar): Rename this file to hlo_graph_renderer.

namespace xla {

// Different formats that a graph can be packaged as.
enum class RenderedGraphFormat {
  kDot,
  kHtml,
  kUrl,
  kFusionVisualization,
};

struct HloRenderOptions {
  // Include the backend config string in the rendered graph.
  bool show_backend_config = false;

  // Include the fusion subcomputations in the rendered graph.
  bool show_fusion_subcomputations = true;
};

// Renders an HLO module as a human-readable visual graph.
//
// Note that this only works well for relatively small graphs (no more than a
// few hundred nodes).  Beyond that, the dot is usually unrenderable,
// unreadable, or both.  To view such graphs, use a tool such as
// interactive_graphviz, which calls RenderNeighborhoodAround to render subsets
// of a graph.
StatusOr<string> RenderGraph(
    const HloComputation& computation, absl::string_view label,
    const DebugOptions& debug_options, RenderedGraphFormat format,
    const HloExecutionProfile* hlo_execution_profile = nullptr,
    HloRenderOptions hlo_render_options = {});

// Like RenderGraph, but renders only nodes "near" the given node in the graph.
//
// The number of nodes dumped is controlled by the radius parameter, which
// (roughly) corresponds to the max distance a node may be from the primary node
// before it's omitted from the graph.
//
// The optional boundary specifies a set of boundary nodes, beyond which nodes
// will be omitted even if they are within the radius.
StatusOr<string> RenderNeighborhoodAround(
    const HloInstruction& node, int radius, RenderedGraphFormat format,
    HloRenderOptions hlo_render_options = {},
    const absl::flat_hash_set<const HloInstruction*>& boundary = {});

// Renders nodes on any of the paths from `from` to `to`.  If there are more
// than max_nodes on all paths, restricts to the max_nodes nodes on the shortest
// paths.
StatusOr<string> RenderAllPathsFromTo(const HloInstruction& from,
                                      const HloInstruction& to, int64 max_nodes,
                                      RenderedGraphFormat format,
                                      HloRenderOptions hlo_render_options = {});

// Registers the fusion state of the graph for future visualization using
// the kFusionVisulization render format.
Status RegisterFusionState(const HloComputation& computation,
                           absl::string_view label);

// Registers a function which implements RenderedGraphFormat::kUrl.
//
// The input to the function is dot, and the output should be a URL or an error.
//
// There can only be one active renderer, and the last call to this function
// wins.
void RegisterGraphToURLRenderer(
    std::function<StatusOr<string>(absl::string_view dot)> renderer);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_GRAPH_DUMPER_H_
