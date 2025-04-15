/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_HLO_PROTO_TO_GRAPH_VIEW_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_HLO_PROTO_TO_GRAPH_VIEW_H_

#include <string>
#include <string_view>
#include <vector>

#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_graph_dumper.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/convert/tool_options.h"

namespace tensorflow {
namespace profiler {

// All the parameters for graph viewer.
struct GraphViewerParams {
  // Whether to use GraphView or TxtView.
  std::string type;
  // Parameters for GraphView.
  std::string node_name;
  int graph_width;
  xla::HloRenderOptions render_options;
  xla::RenderedGraphFormat format;
  // Parameters for TxtView.
  bool verbose;
  bool show_metadata;
};

// Return mapping from style key word to op names separated by comma.
// following hlo_graph_dumper styling
absl::StatusOr<std::string> GetNodeStyles();

// Parse tool options to get the parameters for graph viewer.
absl::StatusOr<GraphViewerParams> ParseGraphViewerParams(
    const ToolOptions& options);

// Get graph render format.
xla::RenderedGraphFormat GetRenderFormat(const std::string& format_string);

// Convert `hlo_proto` to GraphView with the provided render options.
absl::StatusOr<std::string> ConvertHloProtoToGraph(
    const xla::HloProto& hlo_proto, const std::string& node_name,
    int graph_width, const xla::HloRenderOptions& render_options,
    const xla::RenderedGraphFormat& format);

// Convert `hlo_proto` to ModelExplorer Graph JSON data.
absl::StatusOr<std::string> ConvertHloProtoToMeGraph(
    const xla::HloProto& hlo_proto, const std::string& node_name,
    int graph_width);

// Render graph with the provided render options.
absl::StatusOr<std::string> RenderGraphView(
    const xla::HloComputation& computation, absl::string_view label,
    const xla::DebugOptions& debug_options, xla::RenderedGraphFormat format,
    xla::HloRenderOptions hlo_render_options = {});

// Render graph with centered node and depth
absl::StatusOr<std::string> RenderGraphNeighborhoodAround(
    const xla::HloInstruction& node, int radius,
    xla::RenderedGraphFormat format,
    xla::HloRenderOptions hlo_render_options = {},
    const absl::flat_hash_set<const xla::HloInstruction*>& boundary = {});

// Convert `hlo_proto` to StringView.
absl::StatusOr<std::string> ConvertHloProtoToStringView(
    const xla::HloProto& hlo_proto, bool verbose, bool metadata);

// Convert dot into certain format
absl::StatusOr<std::string> WrapDotInFormat(std::string dot,
                                            xla::RenderedGraphFormat format);

// Convert dot into visual graph in html
std::string WrapDotInHtml(std::string dot,
                          absl::string_view layout_engine = "dot");

// Registers a function which implements RenderedGraphFormat::kUrl.
// The input to the function is dot, and the output should be a URL or an error.
// There can only be one active renderer, and the last call to this function
// wins.
void RegisterGraphvizURLRenderer(
    std::function<absl::StatusOr<std::string>(absl::string_view dot)> renderer);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_HLO_PROTO_TO_GRAPH_VIEW_H_
