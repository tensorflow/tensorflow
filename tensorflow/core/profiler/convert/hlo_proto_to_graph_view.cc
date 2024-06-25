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

#include "tensorflow/core/profiler/convert/hlo_proto_to_graph_view.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_graph_dumper.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/convert/tool_options.h"
#include "tensorflow/core/profiler/utils/hlo_module_utils.h"
#include "tensorflow/core/profiler/utils/hlo_proto_to_module.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::StatusOr;
using ::tensorflow::errors::InvalidArgument;
using ::xla::HloComputation;
using ::xla::HloInstruction;
using ::xla::HloModule;
using ::xla::HloPrintOptions;
using ::xla::HloProto;
using ::xla::HloRenderOptions;
using ::xla::RenderedGraphFormat;

void CleanUpHloModuleForGraphviz(HloModule* hlo_module) {
  // Infeed config is escaped serialized proto, and graphviz server complains.
  for (HloComputation* computation : hlo_module->computations()) {
    for (HloInstruction* inst : computation->instructions()) {
      if (inst->opcode() == xla::HloOpcode::kInfeed) {
        inst->set_infeed_config("");
      } else if (inst->opcode() == xla::HloOpcode::kOutfeed) {
        inst->set_outfeed_config("");
      }
    }
  }
}

absl::StatusOr<std::string> Plot(std::unique_ptr<HloModule> module,
                                 const std::string& node_name, int graph_width,
                                 const HloRenderOptions& render_options,
                                 const RenderedGraphFormat& format) {
  if (node_name.empty()) {
    // This should not happen.
    return InvalidArgument("node_name should not be empty");
  }
  // Find the node with the given name.
  const HloInstruction* instr = FindInstruction(*module, node_name);
  const HloComputation* comp = FindComputation(*module, node_name);
  if (!instr && !comp) {
    return InvalidArgument(
        absl::StrCat("Couldn't find HloInstruction or HloComputation named ",
                     node_name, "."));
  }
  // Generate the graph and print the resulting string.
  absl::StatusOr<std::string> graph_handle;

  CleanUpHloModuleForGraphviz(module.get());
  if (comp) {
    graph_handle =
        RenderGraphView(*comp, "", comp->parent()->config().debug_options(),
                        format, render_options);
  } else {
    graph_handle = RenderGraphNeighborhoodAround(*instr, graph_width, format,
                                                 render_options);
  }
  if (graph_handle.ok()) {
    VLOG(1) << graph_handle.value();
  } else {
    LOG(ERROR) << "Unable to render graph: " << graph_handle.status();
  }

  return graph_handle;
}

// Default parameter constants for graph viewer.
static constexpr char kGraphTypeName[] = "graph";
static constexpr char kShortTxtTypeName[] = "short_txt";
static constexpr char kLongTxtTypeName[] = "long_txt";
static constexpr char kDefaultFormatString[] = "url";
static constexpr int kDefaultWidth = 3;
static constexpr int kDefaultShowMetadata = 0;
static constexpr int kDefaultMergeFusion = 0;

}  // namespace

absl::StatusOr<GraphViewerParams> ParseGraphViewerParams(
    const ToolOptions& options) {
  GraphViewerParams params;
  std::optional<std::string> type = GetParam<std::string>(options, "type");
  if (!type.has_value()) {
    return errors::InvalidArgument("Graph viewer must provide a type option.");
  }

  // For graph type.
  if (type == kGraphTypeName) {
    params.type = type.value();
    if (std::optional<std::string> node_name =
            GetParam<std::string>(options, "node_name")) {
      params.node_name = node_name.value();
    }

    params.graph_width =
        GetParamWithDefault<int>(options, "graph_width", kDefaultWidth);
    params.render_options.show_backend_config = GetParamWithDefault<int>(
        options, "show_metadata", kDefaultShowMetadata);
    params.render_options.show_fusion_subcomputations =
        !GetParamWithDefault<int>(options, "merge_fusion", kDefaultMergeFusion);
    params.format = GetRenderFormat(GetParamWithDefault<std::string>(
        options, "format", kDefaultFormatString));

    return params;
  }

  // For txt type.
  if (type == kShortTxtTypeName || type == kLongTxtTypeName) {
    params.type = type.value();
    params.verbose = (type == kLongTxtTypeName);
    params.show_metadata =
        GetParamWithDefault(options, "show_metadata", kDefaultShowMetadata);
    return params;
  }

  // Unknown type.
  return errors::InvalidArgument("Unknown graph viewer type option: ",
                                 type.value());
}

xla::RenderedGraphFormat GetRenderFormat(const std::string& format_string) {
  if (format_string == "html") {
    return xla::RenderedGraphFormat::kHtml;
  } else if (format_string == "dot") {
    return xla::RenderedGraphFormat::kDot;
  } else if (format_string == "url") {
    return xla::RenderedGraphFormat::kUrl;
  } else {
    LOG(ERROR) << "Invalid graph format argument: " << format_string
               << ", fallback to default url";
    return xla::RenderedGraphFormat::kUrl;
  }
}

absl::StatusOr<std::string> ConvertHloProtoToGraph(
    const HloProto& hlo_proto, const std::string& node_name, int graph_width,
    const HloRenderOptions& render_options, const RenderedGraphFormat& format) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      ConvertHloProtoToModule(hlo_proto));
  return Plot(std::move(hlo_module), node_name, graph_width, render_options,
              format);
}

absl::StatusOr<std::string> ConvertHloProtoToStringView(
    const HloProto& hlo_proto, bool verbose, bool metadata) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      ConvertHloProtoToModule(hlo_proto));
  HloPrintOptions options;
  if (!verbose) {
    options = HloPrintOptions::ShortParsable();
  }
  options.set_print_large_constants(verbose);
  options.set_print_metadata(metadata);
  return hlo_module->ToString(options);
}

std::function<absl::StatusOr<std::string>(absl::string_view)>* url_renderer =
    nullptr;

// Precondition: (url_renderer != nullptr || format != kUrl).
//
// (We specify this as a precondition rather than checking it in here and
// returning an error because we want to fail quickly when there's no URL
// renderer available, and this function runs only after we've done all the work
// of producing dot for the graph.)
absl::Status CheckPrecondition(xla::RenderedGraphFormat format) {
  if (format == xla::RenderedGraphFormat::kUrl && url_renderer == nullptr) {
    return absl::FailedPreconditionError(
        "Can't render as URL; no URL renderer was registered.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> RenderGraphView(
    const xla::HloComputation& computation, absl::string_view label,
    const xla::DebugOptions& debug_options, xla::RenderedGraphFormat format,
    xla::HloRenderOptions hlo_render_options) {
  auto precheck_status = CheckPrecondition(format);
  if (!precheck_status.ok()) {
    return precheck_status;
  }
  auto rendered_dot =
      xla::RenderGraph(computation, label, debug_options,
                       RenderedGraphFormat::kDot, hlo_render_options);
  if (!rendered_dot.ok()) {
    return rendered_dot.status();
  }
  return WrapDotInFormat(rendered_dot.value(), format);
}

absl::StatusOr<std::string> RenderGraphNeighborhoodAround(
    const xla::HloInstruction& node, int radius,
    xla::RenderedGraphFormat format, xla::HloRenderOptions hlo_render_options,
    const absl::flat_hash_set<const xla::HloInstruction*>& boundary) {
  auto precheck_status = CheckPrecondition(format);
  if (!precheck_status.ok()) {
    return precheck_status;
  }
  auto rendered_dot = xla::RenderNeighborhoodAround(
      node, radius, RenderedGraphFormat::kDot, hlo_render_options, boundary);
  if (!rendered_dot.ok()) {
    return rendered_dot.status();
  }
  return WrapDotInFormat(rendered_dot.value(), format);
}

absl::StatusOr<std::string> WrapDotInFormat(std::string dot,
                                            xla::RenderedGraphFormat format) {
  switch (format) {
    case xla::RenderedGraphFormat::kUrl:
      if (url_renderer == nullptr) {
        return absl::InternalError("url_renderer is null");
      }
      return (*url_renderer)(dot);
    case xla::RenderedGraphFormat::kHtml:
      return WrapDotInHtml(dot);
    case xla::RenderedGraphFormat::kDot:
      return std::string(dot);
  }
}

std::string WrapDotInHtml(std::string dot) {
  return absl::StrReplaceAll(R"html(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style type="text/css">
    body {
      height: 100vh;
      margin: 0;
    }
    #graph-container {height:95vh;width:100%;padding:10px;display:block;}
    #graph-container svg { height: 100% !important; width: 100% !important;}
    .node, .cluster {cursor:pointer;}
    .cluster:hover, .node:hover {outline: solid 3px black;}
  </style>
</head>
<body>
  <script src="https://www.gstatic.com/external_hosted/hpcc_js_wasm/index.min.js"
      integrity="sha384-LigJPbR3TOfU/Xbb+PjiN1dGJYPweLk7kiGnaMgmxnUmKWaCFKbb5tH6iLlyVhPZ"
      crossorigin="anonymous"></script>
  <script src="https://www.gstatic.com/external_hosted/svg_pan_zoom/svg-pan-zoom.js"></script>
  <div id="graph-container"></div>
  <script>
    const cssregex = new RegExp('stylesheet=<([^]*)\n>\n', 'gm');
    const hpccWasm = window["@hpcc-js/wasm"];
    const data = `$DOT`;
    const results = cssregex.exec(data);
    // graphviz has problem dealing with large stylesheets.
    // https://github.com/tensorflow/tensorflow/issues/17220#issuecomment-369228492
    // In order to avoid the problem, remove the stylesheet from the dot and
    // insert it directly info the rendered SVG.

    let dot_data = data;
    let css_data = '';
    if (results !== null) {
        css_data = results[1].replace(/\s*data:.*\s*,/,''); // Strip content-type field.
        // CSS inside DOT is URL-escaped, so we must unescape it
        // before we can insert it into SVG.
        css_data = unescape(css_data);
        dot_data = data.replace(cssregex, ''); // Remove the stylesheet
    }

    var render_start = performance.now()
    function add_controls(svg) {
        var htmlblob = new Blob([document.documentElement.innerHTML],
                                {type: 'text/html'});
        var savehtml = document.createElement('a');
        savehtml.setAttribute('href', URL.createObjectURL(htmlblob));
        savehtml.setAttribute('download', 'graph.html');
        savehtml.innerHTML = " [Save HTML+SVG] ";
        document.body.append(savehtml);
        var svgblob = new Blob([svg.outerHTML], {type: 'image/svg'});
        var savesvg = document.createElement('a');
        savesvg.setAttribute('href', URL.createObjectURL(svgblob));
        savesvg.setAttribute('download', 'graph.svg');
        savesvg.innerHTML = " [Save SVG] ";
        document.body.append(savesvg);
        var dotblob =  new Blob([data], {type: 'text/dot'});
        var savedot = document.createElement('a');
        savedot.setAttribute('href', URL.createObjectURL(dotblob));
        savedot.setAttribute('download', 'graph.dot');
        savedot.innerHTML = " [Save DOT] ";
        document.body.append(savedot);
        // Will get called after embed element was loaded
        var render_end = performance.now();
        var render_note = document.createElement('div')
        render_note.innerHTML = 'Rendering took '
                                + (render_end - render_start).toFixed(2) + "ms."
        document.body.append(render_note);
    }
    const render_callback = svg => {
      const container = document.getElementById('graph-container')
      container.innerHTML = `${svg}<style>${css_data}</style>`;
      const panZoom = svgPanZoom(container.children[0], {
        zoomEnabled: true,
        controlIconsEnabled: true,
        maxZoom: 200,
        minZoom: 0,
      });
      add_controls(svg);
    };
    hpccWasm.graphviz.layout(dot_data, "svg", "dot").then(render_callback);
  </script>
</body>
</html>
)html",
                             {
                                 {"$DOT", dot},
                             });
}

void RegisterGraphvizURLRenderer(
    std::function<absl::StatusOr<std::string>(absl::string_view)> renderer) {
  if (url_renderer != nullptr) {
    LOG(WARNING) << "Multiple calls to RegisterGraphToURLRenderer. Last call "
                    "wins, but because order of initialization in C++ is "
                    "nondeterministic, this may not be what you want.";
  }
  delete url_renderer;
  url_renderer =
      new std::function<absl::StatusOr<std::string>(absl::string_view)>(
          std::move(renderer));
}

}  // namespace profiler
}  // namespace tensorflow
