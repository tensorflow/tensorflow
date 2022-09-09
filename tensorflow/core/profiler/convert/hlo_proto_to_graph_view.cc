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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/convert/tool_options.h"
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

const HloInstruction* FindInstruction(const HloModule& module,
                                      std::string node_name) {
  if (absl::StartsWith(node_name, "%")) {
    node_name.erase(node_name.begin());
  }
  for (const HloComputation* computation : module.computations()) {
    auto instrs = computation->instructions();
    auto it = absl::c_find_if(instrs, [&](const HloInstruction* instr) {
      // Try with and without "%" at the beginning of the node name.
      return absl::EqualsIgnoreCase(instr->name(), node_name) ||
             absl::EqualsIgnoreCase(instr->name(),
                                    absl::StrCat("%", node_name));
    });
    if (it != instrs.end()) {
      return *it;
    }
  }
  return nullptr;
}

const HloComputation* FindComputation(const HloModule& module,
                                      const std::string& comp_name) {
  for (const HloComputation* computation : module.computations()) {
    if (absl::EqualsIgnoreCase(computation->name(), comp_name)) {
      return computation;
    }
  }
  return nullptr;
}

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

StatusOr<std::string> Plot(std::unique_ptr<HloModule> module,
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
  StatusOr<std::string> graph_handle;

  CleanUpHloModuleForGraphviz(module.get());
  if (comp) {
    graph_handle =
        xla::RenderGraph(*comp, "", comp->parent()->config().debug_options(),
                         format, nullptr, render_options);
  } else {
    graph_handle = xla::RenderNeighborhoodAround(*instr, graph_width, format,
                                                 render_options);
  }
  if (graph_handle.ok()) {
    LOG(INFO) << graph_handle.value();
  } else {
    LOG(INFO) << "Unable to render graph: " << graph_handle.status();
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

StatusOr<GraphViewerParams> ParseGraphViewerParams(const ToolOptions& options) {
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

StatusOr<std::string> ConvertHloProtoToGraph(
    const HloProto& hlo_proto, const std::string& node_name, int graph_width,
    const HloRenderOptions& render_options, const RenderedGraphFormat& format) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      ConvertHloProtoToModule(hlo_proto));
  return Plot(std::move(hlo_module), node_name, graph_width, render_options,
              format);
}

StatusOr<std::string> ConvertHloProtoToStringView(const HloProto& hlo_proto,
                                                  bool verbose, bool metadata) {
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
}  // namespace profiler
}  // namespace tensorflow
