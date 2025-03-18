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
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/tsl/platform/statusor.h"
#ifdef PLATFORM_GOOGLE
#include "nlohmann/json.hpp"
#include "tensorflow/compiler/mlir/lite/experimental/google/tooling/hlo_adapter/direct_hlo_to_json_graph_convert.h"
#endif  // PLATFORM_GOOGLE
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/profiler/convert/tool_options.h"
#include "tensorflow/core/profiler/utils/hlo_module_utils.h"
#include "tensorflow/core/profiler/utils/hlo_proto_to_module.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tsl::StatusOr;
using ::tsl::errors::InvalidArgument;
using ::xla::HloComputation;
using ::xla::HloInstruction;
using ::xla::HloModule;
using ::xla::HloPrintOptions;
using ::xla::HloProto;
using ::xla::HloRenderOptions;
using ::xla::RenderedGraphFormat;

constexpr char kCenterNodeKey[] = "centerNode";

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

#ifdef PLATFORM_GOOGLE
// Add a custom group node on the graph level, for the center node chosen by the
// user set its attributes like `id`, `name` or `opcode` in `graph_json`.
void AddCenterNodeMetadata(nlohmann::json& graph_json, std::string id,
                           absl::string_view name, absl::string_view opcode) {
  nlohmann::json centerGroupNodeAttributes;
  centerGroupNodeAttributes["name"] = name;
  centerGroupNodeAttributes["id"] = id;
  if (!opcode.empty()) {
    centerGroupNodeAttributes["opcode"] = opcode;
  }
  // Follow ModelExplorer's Graph typing: GraphCollectionFromBuiltinAdapters
  graph_json[0]["subgraphs"][0]["groupNodeAttributes"][kCenterNodeKey] =
      centerGroupNodeAttributes;
}
#endif  // PLATFORM_GOOGLE

void AddGraphMetadata(std::string& graph_json_str,
                      const HloInstruction& instr) {
#ifdef PLATFORM_GOOGLE
  nlohmann::json graph_json = nlohmann::json::parse(graph_json_str);
  // 1. Fusion instruction is represented as a layer on client, use its
  // pinned node as the center node, id of the pinned node is the fusion name.
  // 2. Other instructions are represented as nodes on client, use iteself as
  // the center node, where node id is the instruction name.
  std::string id = absl::StrCat(instr.name());
  AddCenterNodeMetadata(graph_json, id, instr.name(),
                        HloOpcodeString(instr.opcode()));
  graph_json_str = graph_json.dump();
#endif  // PLATFORM_GOOGLE
}

void AddGraphMetadata(std::string& graph_json_str, const HloComputation& comp) {
#ifdef PLATFORM_GOOGLE
  nlohmann::json graph_json = nlohmann::json::parse(graph_json_str);
  // Computation is represented as a layer on client, use its pinned node as the
  // center node,id of the pinned node is the computation name.
  AddCenterNodeMetadata(graph_json, absl::StrCat(comp.name()), comp.name(), "");
  graph_json_str = graph_json.dump();
#endif  // PLATFORM_GOOGLE
}

// This function does the same thing as Plot() but uses the ModelExplorer
// instead of graphviz.
absl::StatusOr<std::string> PlotMe(std::unique_ptr<HloModule> module,
                                   const std::string& node_name,
                                   int graph_width) {
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
  std::string graph_json_str;
// b/360874576: Enable when the adapter is open sourced.
#ifdef PLATFORM_GOOGLE
  if (comp) {
    graph_handle = tooling::visualization_client::HloGraphAdapter(*comp);
  } else {
    graph_handle =
        tooling::visualization_client::HloGraphAdapter(*instr, graph_width);
  }
#endif  // PLATFORM_GOOGLE
  if (graph_handle.ok()) {
    VLOG(1) << graph_handle.value();
    graph_json_str = graph_handle.value();
    if (comp) {
      AddGraphMetadata(graph_json_str, *comp);
    } else {
      AddGraphMetadata(graph_json_str, *instr);
    }
    return graph_json_str;
  } else {
    LOG(ERROR) << "Unable to render graph: " << graph_handle.status();
  }

  return graph_handle;
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

absl::StatusOr<std::string> GetNodeStyles() {
  std::vector<xla::HloOpcode> async_op_codes = {xla::HloOpcode::kAsyncStart,
                                                xla::HloOpcode::kAsyncUpdate,
                                                xla::HloOpcode::kAsyncDone};
  std::vector<xla::HloOpcode> brown_op_codes = {
      xla::HloOpcode::kAllGather,
      xla::HloOpcode::kAllGatherStart,
      xla::HloOpcode::kAllGatherDone,
      xla::HloOpcode::kAllReduce,
      xla::HloOpcode::kReduceScatter,
      xla::HloOpcode::kAllReduceStart,
      xla::HloOpcode::kAllReduceDone,
      xla::HloOpcode::kAllToAll,
      xla::HloOpcode::kCollectiveBroadcast,
      xla::HloOpcode::kCollectivePermute,
      xla::HloOpcode::kCollectivePermuteStart,
      xla::HloOpcode::kCollectivePermuteDone,
      xla::HloOpcode::kInfeed,
      xla::HloOpcode::kOutfeed,
      xla::HloOpcode::kPartitionId,
      xla::HloOpcode::kRecv,
      xla::HloOpcode::kRecvDone,
      xla::HloOpcode::kSend,
      xla::HloOpcode::kSendDone,
      xla::HloOpcode::kReplicaId};
  std::vector<xla::HloOpcode> dark_blue_op_codes = {
      xla::HloOpcode::kConvolution, xla::HloOpcode::kDot, xla::HloOpcode::kFft,
      xla::HloOpcode::kTriangularSolve, xla::HloOpcode::kCholesky};
  std::vector<xla::HloOpcode> dark_green_op_codes = {
      xla::HloOpcode::kCall, xla::HloOpcode::kConditional,
      xla::HloOpcode::kCustomCall, xla::HloOpcode::kWhile};
  std::vector<xla::HloOpcode> gray_op_codes = {
      xla::HloOpcode::kDomain, xla::HloOpcode::kFusion, xla::HloOpcode::kMap,
      xla::HloOpcode::kGetDimensionSize, xla::HloOpcode::kSetDimensionSize};
  std::vector<xla::HloOpcode> green_op_codes = {
      xla::HloOpcode::kConcatenate, xla::HloOpcode::kDynamicSlice,
      xla::HloOpcode::kReshape,     xla::HloOpcode::kDynamicReshape,
      xla::HloOpcode::kReverse,     xla::HloOpcode::kTranspose,
      xla::HloOpcode::kCopy,        xla::HloOpcode::kCopyStart,
      xla::HloOpcode::kCopyDone};
  std::vector<xla::HloOpcode> orange_op_codes = {xla::HloOpcode::kParameter};
  std::vector<xla::HloOpcode> purple_op_codes = {
      xla::HloOpcode::kBatchNormGrad,     xla::HloOpcode::kBatchNormInference,
      xla::HloOpcode::kBatchNormTraining, xla::HloOpcode::kReduce,
      xla::HloOpcode::kReduceWindow,      xla::HloOpcode::kScatter,
      xla::HloOpcode::kSelectAndScatter,  xla::HloOpcode::kGather};
  std::vector<xla::HloOpcode> yellow_op_codes = {
      xla::HloOpcode::kBroadcast, xla::HloOpcode::kDynamicUpdateSlice};

  auto OpCodesToNames =
      [&](std::vector<xla::HloOpcode> op_codes) -> std::string {
    std::string op_names = "";
    for (const auto& op_code : op_codes) {
      if (!op_names.empty()) {
        op_names += ",";
      }
      op_names += std::string(xla::HloOpcodeString(op_code));
    }
    return op_names;
  };

  return absl::StrReplaceAll(
      R"json({
      "kBlue": "$asyncOpNames",
      "kBrown": "$brownOpNames",
      "kDarkBlue": "$darkBlueOpNames",
      "kDarkGreen": "$darkGreenOpNames",
      "kGray": "$grayOpNames",
      "kGreen": "$greenOpNames",
      "kOrange": "$orangeOpNames",
      "kPurple": "$purpleOpNames",
      "kYellow": "$yellowOpNames"
    })json",
      {
          {"$asyncOpNames", OpCodesToNames(async_op_codes)},
          {"$brownOpNames", OpCodesToNames(brown_op_codes)},
          {"$darkBlueOpNames", OpCodesToNames(dark_blue_op_codes)},
          {"$darkGreenOpNames", OpCodesToNames(dark_green_op_codes)},
          {"$grayOpNames", OpCodesToNames(gray_op_codes)},
          {"$greenOpNames", OpCodesToNames(green_op_codes)},
          {"$orangeOpNames", OpCodesToNames(orange_op_codes)},
          {"$purpleOpNames", OpCodesToNames(purple_op_codes)},
          {"$yellowOpNames", OpCodesToNames(yellow_op_codes)},
      });
}

absl::StatusOr<GraphViewerParams> ParseGraphViewerParams(
    const ToolOptions& options) {
  GraphViewerParams params;
  std::optional<std::string> type = GetParam<std::string>(options, "type");
  if (!type.has_value()) {
    return InvalidArgument("Graph viewer must provide a type option.");
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
  return InvalidArgument("Unknown graph viewer type option: ", type.value());
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

absl::StatusOr<std::string> ConvertHloProtoToMeGraph(
    const HloProto& hlo_proto, const std::string& node_name, int graph_width) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      ConvertHloProtoToModule(hlo_proto));
  return PlotMe(std::move(hlo_module), node_name, graph_width);
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
