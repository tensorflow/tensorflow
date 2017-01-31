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

#include <string>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/legacy_flags/hlo_graph_dumper_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
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

// Returns the dot graph identifier for the given instruction.
string InstructionId(const HloInstruction* instruction) {
  return Printf("%lld", reinterpret_cast<uint64>(instruction));
}

// Returns the dot graph identifier for the given computation.
string ComputationId(const HloComputation* computation) {
  return Printf("%lld", reinterpret_cast<uint64>(computation));
}

// Returns a compact string that represents the convolution dimension numbers.
string ConvolutionDimensionNumbersToString(
    const ConvolutionDimensionNumbers& dim_numbers) {
  return Printf("B@%lld,Z@%lld,KIZ@%lld,KOZ@%lld",
                dim_numbers.batch_dimension(), dim_numbers.feature_dimension(),
                dim_numbers.kernel_input_feature_dimension(),
                dim_numbers.kernel_output_feature_dimension());
}

// Returns a compact string that represents the non-trivial fields in the window
// description. If there are no non-trivial fields, the empty string is
// returned.
string WindowToString(const Window& window) {
  bool display_padding = false;
  bool display_window_dilation = false;
  bool display_base_dilation = false;
  bool display_stride = false;
  for (const WindowDimension& dimension : window.dimensions()) {
    display_padding |=
        dimension.padding_low() != 0 || dimension.padding_high() != 0;
    display_window_dilation |= dimension.window_dilation() != 1;
    display_base_dilation |= dimension.base_dilation() != 1;
    display_stride |= dimension.stride() != 1;
  }
  std::vector<string> pieces = {};
  if (display_padding) {
    pieces.push_back("\\n");
    pieces.push_back("padding=[");
    for (const WindowDimension& dimension : window.dimensions()) {
      pieces.push_back(StrCat("(", dimension.padding_low(), ",",
                              dimension.padding_high(), ")"));
      pieces.push_back(", ");
    }
    pieces.pop_back();
    pieces.push_back("]");
  }
  // Make a convenient lambda that adds a simple int64 field in each
  // WindowDimension.
  auto add_field = [&pieces, &window](
      const string& label,
      tensorflow::protobuf_int64 (WindowDimension::*member)() const) {
    pieces.push_back("\\n");
    pieces.push_back(label + "=[");
    for (const WindowDimension& dimension : window.dimensions()) {
      pieces.push_back(StrCat(((&dimension)->*member)()));
      pieces.push_back(", ");
    }
    pieces.pop_back();
    pieces.push_back("]");
  };
  if (display_window_dilation) {
    add_field("window_dilation", &WindowDimension::window_dilation);
  }
  if (display_base_dilation) {
    add_field("base_dilation", &WindowDimension::base_dilation);
  }
  if (display_stride) {
    add_field("stride", &WindowDimension::stride);
  }
  return Join(pieces, "");
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
      int64 param_number = instruction->parameter_number();
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
    StrAppend(&graph_body, param_node_name,
              " [shape=record,style=filled,fillcolor=\"lightblue1\",",
              "label=\"{parameters | {", Join(param_ports, "|"), "}}\"];\n");
  }

  for (auto& instruction : instructions) {
    string color = "peachpuff";
    string shape = "ellipse";
    string name = HloOpcodeString(instruction->opcode());
    if (HloOpcode::kFusion == instruction->opcode()) {
      name += ": " + FusionKindString(instruction->fusion_kind());
    }
    if (HloOpcode::kConvolution == instruction->opcode()) {
      name += ":\\n" + ConvolutionDimensionNumbersToString(
                           instruction->convolution_dimension_numbers()) +
              WindowToString(instruction->window());
    }
    name += "\\n" + instruction->name();
    std::vector<HloComputation*> called_computations;

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
      case HloOpcode::kConcatenate:
      case HloOpcode::kConvert:
      case HloOpcode::kDivide:
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice:
      case HloOpcode::kEq:
      case HloOpcode::kExp:
      case HloOpcode::kFloor:
      case HloOpcode::kGe:
      case HloOpcode::kGt:
      case HloOpcode::kIndex:
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
      case HloOpcode::kPad:
      case HloOpcode::kPower:
      case HloOpcode::kRemainder:
      case HloOpcode::kReshape:
      case HloOpcode::kReverse:
      case HloOpcode::kSelect:
      case HloOpcode::kSign:
      case HloOpcode::kSlice:
      case HloOpcode::kSort:
      case HloOpcode::kSubtract:
      case HloOpcode::kTanh:
      case HloOpcode::kTuple:
      case HloOpcode::kUpdate:
        break;

      case HloOpcode::kBroadcast:
      case HloOpcode::kTranspose:
        StrAppend(&name, "\\n", "dims={", Join(instruction->dimensions(), ","),
                  "}");
        break;
      case HloOpcode::kGetTupleElement:
        StrAppend(&name, "\\nindex=", instruction->tuple_index());
        break;
      case HloOpcode::kRng:
        StrAppend(&name, "\\n",
                  RandomDistribution_Name(instruction->random_distribution()));
        break;
      case HloOpcode::kConstant:
        shape = "boxed";
        color = "palegreen";
        if (ShapeUtil::IsScalar(instruction->shape())) {
          StrAppend(&name, "\\n", "value=", LiteralUtil::GetAsString(
                                                instruction->literal(), {}));
        }
        break;
      case HloOpcode::kBitcast:
      case HloOpcode::kCopy:
        color = "white";
        break;
      case HloOpcode::kCall:
        color = "tomato";
        break;
      case HloOpcode::kCustomCall:
        color = "tomato4";
        StrAppend(&name, "\\n",
                  "custom_call_target=", instruction->custom_call_target());
        break;
      case HloOpcode::kDot:
        color = "slateblue";
        break;
      case HloOpcode::kSend:
        color = "purple";
        break;
      case HloOpcode::kRecv:
        color = "orange";
        break;
      case HloOpcode::kMap:
        color = "palevioletred";
        break;
      case HloOpcode::kParameter:
        // A single record node is created for all the parameter nodes with a
        // port for each parameter instruction. No need to emit anything in this
        // case.
        continue;
      case HloOpcode::kReduce:
        StrAppend(&name, " dims=", Join(instruction->dimensions(), ","));
        color = "lightsalmon";
        break;
      case HloOpcode::kSelectAndScatter:
      case HloOpcode::kReduceWindow:
        color = "lightsalmon";
        break;
      case HloOpcode::kTrace:
        color = "white";
        break;
      case HloOpcode::kWhile:
        color = "forestgreen";
        break;
      case HloOpcode::kFusion:
        color = "gray";
        break;
      case HloOpcode::kConvolution:
        color = "red";
        break;
      case HloOpcode::kCrossReplicaSum:
        color = "turquoise";
        break;
      case HloOpcode::kInfeed:
      case HloOpcode::kOutfeed:
        color = "blue";
        break;
    }

    // Create instruction node with appropriate label, shape, and color.
    string label =
        StrCat(name, "\\n", ShapeUtil::HumanString(instruction->shape()));
    if (show_addresses) {
      Appendf(&label, "\\n[%p]", instruction.get());
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
      StrAppend(&label, "\\nlayout={", layout_string, "}");
    }
    if (hlo_execution_profile != nullptr) {
      auto hlo_cycles_executed =
          hlo_execution_profile->GetProfileResult(*instruction);
      auto total_cycles_executed =
          hlo_execution_profile->total_cycles_executed();
      if (hlo_cycles_executed > 0 && total_cycles_executed > 0) {
        Appendf(&label, "\\n%% of cycles executed=%.2f",
                (static_cast<double>(hlo_cycles_executed) /
                 static_cast<double>(total_cycles_executed)) *
                    100);
      }
    }
    Appendf(&graph_body,
            "%s [label=\"%s\", shape=%s, style=filled, fillcolor=%s];\n",
            InstructionId(instruction.get()).c_str(), label.c_str(),
            shape.c_str(), color.c_str());

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
                "label=\"fused expression\";\nstyle=filled;\n"
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
      for (auto* computation : instruction->MakeCalledComputationsSet()) {
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

string ComputationToDotGraph(const HloComputation& computation,
                             const string& label, bool show_addresses,
                             bool show_layouts,
                             const HloExecutionProfile* hlo_execution_profile) {
  string graph_label = StrCat(label, "\\n", computation.name());
  if (hlo_execution_profile != nullptr) {
    auto cycles = hlo_execution_profile->total_cycles_executed();
    Appendf(&graph_label, "\\ntotal cycles = %lld (%s)", cycles,
            tensorflow::strings::HumanReadableNum(cycles).c_str());
  }
  string graph =
      Printf("digraph G {\nrankdir=TB;\ncompound=true;\nlabel=\"%s\"\n",
             graph_label.c_str());

  // Emit embedded computations as subgraph clusters.
  std::vector<string> intercomputation_edges;
  for (auto embedded : computation.MakeEmbeddedComputationsList()) {
    string graph_body = InstructionSequenceGraph(
        embedded->instructions(), show_addresses, show_layouts,
        &intercomputation_edges, hlo_execution_profile);
    Appendf(&graph, "subgraph cluster_%s {\nlabel=\"%s\";\n%s}\n",
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
  string RenderGraph(const string& graph) override {
    static std::atomic<int> output_num(0);
    legacy_flags::HloGraphDumperFlags* flags =
        legacy_flags::GetHloGraphDumperFlags();
    string path = StrCat(flags->xla_hlo_dump_graph_path, "hlo_graph_",
                         output_num++, ".dot");
    tensorflow::Status status =
        tensorflow::WriteStringToFile(tensorflow::Env::Default(), path, graph);
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
  string graph = ComputationToDotGraph(computation, label, show_addresses,
                                       show_layouts, hlo_execution_profile);

  string graph_url = GetGraphRenderer()->RenderGraph(graph);
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
