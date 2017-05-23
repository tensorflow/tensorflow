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

#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"

#include <unistd.h>
#include <string>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/legacy_flags/hlo_graph_dumper_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_tfgraph_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"

using ::tensorflow::Env;
using ::tensorflow::WriteStringToFile;
using ::tensorflow::io::JoinPath;
using ::tensorflow::strings::Appendf;
using ::tensorflow::strings::Printf;
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;
using ::tensorflow::str_util::Join;

namespace xla {
namespace hlo_graph_dumper {
namespace {

// Node color schemes, used by NodeColorAttributes.
enum ColorScheme {
  kBlue,
  kBrown,
  kDarkBlue,
  kDarkGreen,
  kDarkRed,
  kGray,
  kGreen,
  kOrange,
  kPurple,
  kRed,
  kWhite,
  kYellow,
};

// Given a ColorScheme, returns an attribute string for a node of that color.
// Sets the node's fill, stroke, and text colors.
//
// Colors are from https://material.io/color.
string NodeColorAttributes(ColorScheme color) {
  using std::make_tuple;

  const char *fill_color, *stroke_color, *font_color;
  std::tie(fill_color, stroke_color, font_color) =
      [color]() -> std::tuple<const char*, const char*, const char*> {
    switch (color) {
      case kBlue:
        return make_tuple("#bbdefb", "#8aacc8", "black");
      case kBrown:
        return make_tuple("#bcaaa4", "#8c7b75", "black");
      case kDarkBlue:
        return make_tuple("#1565c0", "#003c8f", "white");
      case kDarkGreen:
        return make_tuple("#2e7d32", "#005005", "white");
      case kDarkRed:
        return make_tuple("#b71c1c", "#7f0000", "white");
      case kGray:
        return make_tuple("#cfd8dc", "#9ea7aa", "black");
      case kGreen:
        return make_tuple("#c8e6c9", "#97b498", "black");
      case kOrange:
        return make_tuple("#ffe0b2", "#cbae82", "black");
      case kPurple:
        return make_tuple("#e1bee7", "#af8eb5", "black");
      case kRed:
        return make_tuple("#ffcdd2", "#cb9ca1", "black");
      case kWhite:
        return make_tuple("white", "black", "black");
      case kYellow:
        return make_tuple("#fff9c4", "#cbc693", "black");
    }
  }();

  return Printf(
      "style=filled, fontcolor=\"%s\", color=\"%s\", fillcolor=\"%s\"",
      font_color, stroke_color, fill_color);
}

// Replaces <> with &lt;&gt;, so that this string is safe(er) for use in a
// graphviz HTML-like string.
string HtmlLikeStringSanitize(tensorflow::StringPiece s) {
  return tensorflow::str_util::StringReplace(
      tensorflow::str_util::StringReplace(s, "<", "&lt;", /*replace_all=*/true),
      ">", "&gt;", /*replace_all=*/true);
}

// Returns the dot graph identifier for the given instruction.
string InstructionId(const HloInstruction* instruction) {
  return Printf("%lld", reinterpret_cast<uint64>(instruction));
}

// Returns the dot graph identifier for the given computation.
string ComputationId(const HloComputation* computation) {
  return Printf("%lld", reinterpret_cast<uint64>(computation));
}

// Returns the dot graph edges and nodes for the given instruction sequence.
// Edges which extend between computations are added to the vector
// intercomputation_edges. This is necessary because graphviz does not render
// the graph properly unless these inter-computation edges appear after all
// subgraph statements.
string InstructionSequenceGraph(
    const std::list<std::unique_ptr<HloInstruction>>& instructions,
    bool show_addresses, bool show_layouts,
    std::vector<string>* intercomputation_edges,
    const HloExecutionProfile* hlo_execution_profile) {
  string graph_body;

  // Create a single "record" node for the parameters. This node is a
  // partitioned rectangle with one partition per parameter node. The keeps
  // all the parameter instructions together.
  std::vector<HloInstruction*> param_instructions;
  for (auto& instruction : instructions) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      size_t param_number = instruction->parameter_number();

      if (param_instructions.size() < param_number + 1) {
        param_instructions.resize(param_number + 1, nullptr);
      }
      param_instructions[param_number] = instruction.get();
    }
  }
  string param_node_name;
  if (!param_instructions.empty()) {
    std::vector<string> param_ports;
    param_node_name =
        StrCat("parameters_", InstructionId(param_instructions[0]));
    for (auto& param : param_instructions) {
      string label = StrCat(param->parameter_name(), "\\n",
                            ShapeUtil::HumanString(param->shape()));
      if (show_addresses) {
        Appendf(&label, "\\n[%p]", param);
      }
      if (show_layouts) {
        StrAppend(&label, "\\nlayout=\\{",
                  Join(param->shape().layout().minor_to_major(), ","), "\\}");
      }
      param_ports.push_back(
          Printf("<%s> %s", InstructionId(param).c_str(), label.c_str()));
    }
    // (If we wanted the word "parameters" to be bold like the other op names,
    // we'd have to make this into an HTML-like table.  It is possible but
    // complicated; see http://www.graphviz.org/doc/info/shapes.html#html.)
    StrAppend(&graph_body, param_node_name, " [shape=record ",
              NodeColorAttributes(kOrange), "label=\"{parameters | {",
              Join(param_ports, "|"), "}}\"];\n");
  }

  for (auto& instruction : instructions) {
    ColorScheme color = kYellow;
    string shape = "box";
    string name =
        StrCat("<b>", HtmlLikeStringSanitize(instruction->ExtendedOpcodeStr()),
               "</b> ", HtmlLikeStringSanitize(instruction->name()));
    if (HloOpcode::kConvolution == instruction->opcode()) {
      StrAppend(
          &name, "<br/>",
          HtmlLikeStringSanitize(
              instruction->ConvolutionDimensionNumbersToString()),
          "<br/>",
          HtmlLikeStringSanitize(window_util::ToString(instruction->window())));
    }

    if (!instruction->metadata().op_name().empty()) {
      StrAppend(&name, "<br/>",
                HtmlLikeStringSanitize(instruction->metadata().op_name()));
    }
    if (!instruction->metadata().source_file().empty() &&
        instruction->metadata().source_line() != 0) {
      StrAppend(&name, "<br/>", instruction->metadata().source_file(), ":",
                instruction->metadata().source_line());
    }

    // Pick different colors or shapes for instructions which are particularly
    // expensive (eg, dot) and those which are unusual in some way or unique
    // (eg, parameter).
    switch (instruction->opcode()) {
      // "Normal" instructions. Mostly cheap and elementwise. No call to
      // embedded computations. In this case, use default color, shape and
      // label.
      case HloOpcode::kAbs:
      case HloOpcode::kAdd:
      case HloOpcode::kCeil:
      case HloOpcode::kClamp:
      case HloOpcode::kConvert:
      case HloOpcode::kDivide:
      case HloOpcode::kEq:
      case HloOpcode::kExp:
      case HloOpcode::kFloor:
      case HloOpcode::kGe:
      case HloOpcode::kGt:
      case HloOpcode::kIndex:
      case HloOpcode::kIsFinite:
      case HloOpcode::kLe:
      case HloOpcode::kLog:
      case HloOpcode::kLogicalAnd:
      case HloOpcode::kLogicalNot:
      case HloOpcode::kLogicalOr:
      case HloOpcode::kLt:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum:
      case HloOpcode::kMultiply:
      case HloOpcode::kNe:
      case HloOpcode::kNegate:
      case HloOpcode::kPower:
      case HloOpcode::kRemainder:
      case HloOpcode::kSelect:
      case HloOpcode::kSign:
      case HloOpcode::kSlice:
      case HloOpcode::kSort:
      case HloOpcode::kSubtract:
      case HloOpcode::kTanh:
        break;
      case HloOpcode::kRng:
        StrAppend(&name, "<br/>",
                  RandomDistribution_Name(instruction->random_distribution()));
        break;
      case HloOpcode::kBroadcast:
      case HloOpcode::kTranspose:
        StrAppend(&name, "<br/>", "dims={",
                  Join(instruction->dimensions(), ","), "}");
        break;
      case HloOpcode::kBitcast:
      case HloOpcode::kTuple:
      case HloOpcode::kTrace:
        color = kWhite;
        break;
      case HloOpcode::kGetTupleElement:
        color = kWhite;
        StrAppend(&name, "<br/>index=", instruction->tuple_index());
        break;
      case HloOpcode::kConcatenate:
      case HloOpcode::kCopy:
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice:
      case HloOpcode::kPad:
      case HloOpcode::kReshape:
      case HloOpcode::kReverse:
      case HloOpcode::kUpdate:
        color = kGreen;
        break;
      case HloOpcode::kConstant:
        color = kBlue;
        break;
      case HloOpcode::kConvolution:
      case HloOpcode::kDot:
        color = kDarkBlue;
        break;
      case HloOpcode::kParameter:
        // A single record node is created for all the parameter nodes with a
        // port for each parameter instruction. No need to emit anything in this
        // case.
        continue;
      case HloOpcode::kReduce:
        StrAppend(&name, " dims=", Join(instruction->dimensions(), ","));
        color = kPurple;
        break;
      case HloOpcode::kSelectAndScatter:
      case HloOpcode::kReduceWindow:
        color = kPurple;
        break;
      case HloOpcode::kWhile:
        shape = "ellipse";
        color = kDarkGreen;
        break;
      case HloOpcode::kMap:
      case HloOpcode::kFusion:
        color = kGray;
        break;
      case HloOpcode::kSend:
      case HloOpcode::kRecv:
      case HloOpcode::kInfeed:
      case HloOpcode::kOutfeed:
      case HloOpcode::kCrossReplicaSum:
        color = kBrown;
        break;
      case HloOpcode::kCall:
        color = kDarkGreen;
        break;
      case HloOpcode::kCustomCall:
        color = kDarkGreen;
        StrAppend(&name, "<br/>",
                  "custom_call_target=", instruction->custom_call_target());
        break;
    }

    // Create instruction node with appropriate label, shape, and color.
    // label is interpreted as an HTML-like string, so newlines must be
    // delimited with <br/>, rather than \n.
    string label =
        StrCat(name, "<br/>", ShapeUtil::HumanString(instruction->shape()));

    if (instruction->opcode() == HloOpcode::kConstant &&
        ShapeUtil::IsEffectiveScalar(instruction->shape())) {
      auto elem_idx = IndexUtil::LinearIndexToMultidimensionalIndex(
          instruction->shape(), /*linear_index=*/0);
      StrAppend(&label, " = {",
                LiteralUtil::GetAsString(instruction->literal(), elem_idx),
                "}");
    }

    if (show_addresses) {
      Appendf(&label, "<br/>[%p]", instruction.get());
    }
    if (show_layouts && LayoutUtil::HasLayout(instruction->shape())) {
      string layout_string;
      if (ShapeUtil::IsTuple(instruction->shape())) {
        // For tuples, emit the full shape because the layout of a tuple is not
        // represented in a single Layout field.
        layout_string = ShapeUtil::HumanStringWithLayout(instruction->shape());
      } else {
        layout_string =
            Join(instruction->shape().layout().minor_to_major(), ",");
      }
      StrAppend(&label, "<br/>layout={", layout_string, "}");
    }
    if (hlo_execution_profile != nullptr) {
      auto hlo_cycles_executed =
          hlo_execution_profile->GetProfileResult(*instruction);
      auto total_cycles_executed =
          hlo_execution_profile->total_cycles_executed(*instruction->parent());
      if (hlo_cycles_executed > 0 && total_cycles_executed > 0) {
        Appendf(&label, "<br/>%% of cycles executed=%.2f",
                (static_cast<double>(hlo_cycles_executed) /
                 static_cast<double>(total_cycles_executed)) *
                    100);
      }
    }

    Appendf(&graph_body, "%s [label=<%s>, shape=%s, %s];\n",
            InstructionId(instruction.get()).c_str(), label.c_str(),
            shape.c_str(), NodeColorAttributes(color).c_str());

    // Create edges from the instruction's operands to the instruction.
    int64 operand_number = 0;
    for (auto* operand : instruction->operands()) {
      string src;
      if (operand->opcode() == HloOpcode::kParameter) {
        // If operand is a parameter, then select the proper partition (port) in
        // the unified parameter node.
        src = param_node_name + ":" + InstructionId(operand);
      } else {
        src = InstructionId(operand);
      }
      Appendf(&graph_body, "%s -> %s", src.c_str(),
              InstructionId(instruction.get()).c_str());
      if (instruction->operand_count() > 1) {
        Appendf(&graph_body, " [headlabel=\"%lld\",labeldistance=2]",
                operand_number);
      }
      StrAppend(&graph_body, ";\n");
      ++operand_number;
    }

    // Fusion nodes are handled specially because they contain nested
    // expressions.
    if (instruction->opcode() == HloOpcode::kFusion) {
      string cluster_name =
          StrCat("cluster_", InstructionId(instruction.get()));
      StrAppend(&graph_body, "subgraph ", cluster_name, " {\n");
      StrAppend(&graph_body,
                "label=<<b>fused expression</b>>;\nstyle=\"rounded,filled\";\n"
                "color=lightgrey;\n");
      StrAppend(&graph_body, InstructionSequenceGraph(
                                 instruction->fused_instructions(),
                                 show_addresses, show_layouts,
                                 intercomputation_edges, hlo_execution_profile),
                "}\n");
      string fusion_edge =
          StrCat(InstructionId(instruction->fused_expression_root()), " -> ",
                 InstructionId(instruction.get()),
                 "  [ style = \"dotted\", arrowsize=0.0, ltail=", cluster_name,
                 " ];\n");
      intercomputation_edges->push_back(fusion_edge);
    } else {
      // Add a dotted edge between the instruction and any computations that the
      // instruction calls.
      for (const HloComputation* computation :
           instruction->called_computations()) {
        string cluster_name = StrCat("cluster_", ComputationId(computation));
        string call_edge = Printf(
            "%s -> %s [ style=dashed; ltail=%s ];\n",
            InstructionId(computation->root_instruction()).c_str(),
            InstructionId(instruction.get()).c_str(), cluster_name.c_str());
        intercomputation_edges->push_back(call_edge);
      }
    }
  }
  return graph_body;
}

// DOT graphs accept a stylesheet as a URL.  So naturally, an inline stylesheet
// is a data URI!
//
// We don't perform any escaping on this string, so be careful not to use double
// quotes inside.
static const char* dot_stylesheet = R"(
data:text/css,
@import url(https://fonts.googleapis.com/css?family=Roboto:400,700);
svg text {
  font-family: 'Roboto';
  font-size: 12px;
}
)";

string ComputationToDotGraph(const HloComputation& computation,
                             const string& label, bool show_addresses,
                             bool show_layouts,
                             const HloExecutionProfile* hlo_execution_profile) {
  string graph_label = StrCat(label, "<br/>", computation.name());
  if (hlo_execution_profile != nullptr) {
    auto cycles = hlo_execution_profile->total_cycles_executed(computation);
    Appendf(&graph_label, "<br/>total cycles = %lld (%s)", cycles,
            tensorflow::strings::HumanReadableNum(cycles).c_str());
  }
  string graph = Printf(
      R"(digraph G {
rankdir=TB;
compound=true;
label=<<b>%s</b>>;
labelloc=t;
stylesheet="%s"
)",
      graph_label.c_str(), dot_stylesheet);

  // Emit embedded computations as subgraph clusters.
  std::vector<string> intercomputation_edges;
  for (auto embedded : computation.MakeEmbeddedComputationsList()) {
    string graph_body = InstructionSequenceGraph(
        embedded->instructions(), show_addresses, show_layouts,
        &intercomputation_edges, hlo_execution_profile);
    Appendf(&graph,
            "subgraph cluster_%s "
            "{\nstyle=rounded;label=<<b>%s</b>>;labelloc=t;\n%s}\n",
            ComputationId(embedded).c_str(), embedded->name().c_str(),
            graph_body.c_str());
  }
  StrAppend(&graph,
            InstructionSequenceGraph(computation.instructions(), show_addresses,
                                     show_layouts, &intercomputation_edges,
                                     hlo_execution_profile));

  // Edges between computations (subgraph clusters) must be emitted last for the
  // graph to be rendered properly for some reason.
  StrAppend(&graph, Join(intercomputation_edges, "\n"), "}\n");

  return graph;
}

tensorflow::mutex& RendererMutex() {
  static tensorflow::mutex* mu = new tensorflow::mutex;
  return *mu;
}

std::map<int, GraphRendererInterface*>* GraphRenderers() {
  static auto* graph_renderers = new std::map<int, GraphRendererInterface*>();
  return graph_renderers;
}

GraphRendererInterface* GetGraphRenderer() {
  tensorflow::mutex_lock lock(RendererMutex());
  auto* graph_renderers = GraphRenderers();
  auto it = graph_renderers->rbegin();
  CHECK(it != graph_renderers->rend()) << "No registered graph dumpers";
  return it->second;
}

}  // namespace

Registrar::Registrar(GraphRendererInterface* dumper, int priority) {
  tensorflow::mutex_lock lock(RendererMutex());
  auto* graph_renderers = GraphRenderers();
  graph_renderers->emplace(priority, dumper);
}

namespace {

class FileGraphRenderer : public GraphRendererInterface {
 public:
  string RenderGraph(const string& graph, GraphKind graph_kind) override {
    static std::atomic<int> output_num(0);
    legacy_flags::HloGraphDumperFlags* flags =
        legacy_flags::GetHloGraphDumperFlags();
    string file_extension;
    switch (graph_kind) {
      case DOT_GRAPH:
        file_extension = ".dot";
        break;
      case TF_GRAPHDEF:
        file_extension = ".pbtxt";
        break;
    }
    string path =
        JoinPath(flags->xla_hlo_dump_graph_path,
                 StrCat("hlo_graph_", output_num++, ".XXXXXX", file_extension));
    auto status = Status::OK();
    int fd = mkstemps(&path[0], file_extension.length());
    if (fd < 0) {
      status =
          Status(tensorflow::error::Code::UNKNOWN,
                 StrCat("Failed to create temporary file to dump HLO graph: ",
                        strerror(errno)));
    } else {
      status = tensorflow::WriteStringToFile(tensorflow::Env::Default(), path,
                                             graph);
      close(fd);
    }
    if (!status.ok()) {
      LOG(WARNING) << "Saving HLO graph failed: " << status;
    }
    return path;
  }
};

XLA_REGISTER_GRAPH_RENDERER(FileGraphRenderer, 0);

}  // namespace

string DumpGraph(const HloComputation& computation, const string& label,
                 bool show_addresses, bool show_layouts,
                 const HloExecutionProfile* hlo_execution_profile) {
  string graph;
  string graph_url;
  legacy_flags::HloGraphDumperFlags* flags =
      legacy_flags::GetHloGraphDumperFlags();
  if (flags->xla_hlo_dump_as_graphdef) {
    HloTfGraphBuilder builder;
    TF_CHECK_OK(builder.AddComputation(computation));
    CHECK(tensorflow::protobuf::TextFormat::PrintToString(builder.GetGraphDef(),
                                                          &graph));
    // TODO(b/37198616): Use the default registered renderers when all
    // renderers support rendering GraphDefs. Always dump GraphDefs to files
    // for now.
    graph_url = FileGraphRenderer().RenderGraph(
        graph, GraphRendererInterface::TF_GRAPHDEF);
  } else {
    graph = ComputationToDotGraph(computation, label, show_addresses,
                                  show_layouts, hlo_execution_profile);
    graph_url = GetGraphRenderer()->RenderGraph(
        graph, GraphRendererInterface::DOT_GRAPH);
  }
  LOG(INFO) << "computation " << computation.name() << " [" << label
            << "]: " << graph_url;
  return graph_url;
}

void DumpText(const HloModule& module, const string& label,
              const string& directory_path, bool do_prefix) {
  Env* env = Env::Default();
  TF_CHECK_OK(env->RecursivelyCreateDir(directory_path));
  string prefix = StrCat(env->NowMicros());
  string filename =
      do_prefix ? StrCat(prefix, "-", label, ".txt") : StrCat(label, ".txt");
  string path = JoinPath(directory_path, filename);
  TF_CHECK_OK(WriteStringToFile(env, path, module.ToString()));
}

}  // namespace hlo_graph_dumper
}  // namespace xla
