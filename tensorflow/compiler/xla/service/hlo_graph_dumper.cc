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
#include <algorithm>
#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_tfgraph_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/regexp.h"

using ::tensorflow::Env;
using ::tensorflow::gtl::nullopt;
using ::tensorflow::gtl::optional;
using ::tensorflow::io::JoinPath;
using ::tensorflow::strings::Appendf;
using ::tensorflow::strings::Printf;
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;
using ::tensorflow::str_util::Join;
using ::tensorflow::WriteStringToFile;

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

  // Causes the node's border to be a dashed line, and its content to be gray
  // text on a white background, suggesting that this is an "unimportant" node.
  kDashedBorder,
};

// Used to indicate how we should treat a given HLOInstruction in the graph.
// should we treat it like normal, hide it, and so on?
enum NodeFilterResult {
  kNormalNode,
  kHideNode,
  // Make the node easy to find in the final graph.
  kHighlightNode,
  // "Gray out" the node to indicate that some of its operands have been
  // omitted.
  kSomeOperandsOmitted,
  // Style the node the same as kSomeOperandsOmitted, but also don't connect it
  // to its operands, even if they're present in the graph.
  kOmitNodeOperands,
};

// NodeFilter is essentially a map from HloInstruction*s to NodeFilterResult.
// It lets callers tell the graph-drawing routines which nodes they want to be
// shown, hidden, or highlighted.
class NodeFilter {
 public:
  NodeFilter() : filter_([](const HloInstruction*) { return kNormalNode; }) {}

  explicit NodeFilter(
      std::function<NodeFilterResult(const HloInstruction* instr)> filter)
      : filter_(std::move(filter)) {}

  bool Show(const HloInstruction* instr) const {
    return filter_(instr) != kHideNode;
  }
  bool Highlight(const HloInstruction* instr) const {
    return filter_(instr) == kHighlightNode;
  }
  bool OmitOperands(const HloInstruction* instr) const {
    return filter_(instr) == kOmitNodeOperands;
  }
  bool SomeOrAllOperandsOmitted(const HloInstruction* instr) const {
    auto result = filter_(instr);
    return result == kOmitNodeOperands || result == kSomeOperandsOmitted;
  }

 private:
  std::function<NodeFilterResult(const HloInstruction* instr)> filter_;
};

// Given a ColorScheme, returns an attribute string for a node of that color.
// Sets the node's style and fill/stroke/text colors.
//
// Colors are from https://material.io/color.
string NodeColorAttributes(ColorScheme color) {
  using std::make_tuple;

  const char *style, *fill_color, *stroke_color, *font_color;
  std::tie(style, fill_color, stroke_color, font_color) = [color] {
    switch (color) {
      case kBlue:
        return make_tuple("filled", "#bbdefb", "#8aacc8", "black");
      case kBrown:
        return make_tuple("filled", "#bcaaa4", "#8c7b75", "black");
      case kDarkBlue:
        return make_tuple("filled", "#1565c0", "#003c8f", "white");
      case kDarkGreen:
        return make_tuple("filled", "#2e7d32", "#005005", "white");
      case kDarkRed:
        return make_tuple("filled", "#b71c1c", "#7f0000", "white");
      case kGray:
        return make_tuple("filled", "#cfd8dc", "#9ea7aa", "black");
      case kGreen:
        return make_tuple("filled", "#c8e6c9", "#97b498", "black");
      case kOrange:
        return make_tuple("filled", "#ffe0b2", "#cbae82", "black");
      case kPurple:
        return make_tuple("filled", "#e1bee7", "#af8eb5", "black");
      case kRed:
        return make_tuple("filled", "#ffcdd2", "#cb9ca1", "black");
      case kWhite:
        return make_tuple("filled", "white", "black", "black");
      case kYellow:
        return make_tuple("filled", "#fff9c4", "#cbc693", "black");
      case kDashedBorder:
        return make_tuple("dashed", "white", "#757575", "#757575");
    }
  }();

  return Printf(
      R"(style=%s, fontcolor="%s", color="%s", fillcolor="%s")", style,
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

// Tries to generates a human-readable one-word description of the given
// computation.
//
// Currently we support:
//
//   "return param0 + param1;"      --> "add"
//   "return param0 * param1;"      --> "multiply"
//   "return min(param0, param1);"  --> "min"
//   "return max(param0, param1);"  --> "max"
//
// where param0 and param1 are effective scalars.  Since all of the ops above
// are commutative, we also support them with param0 and param1 swapped.
//
// This is useful primarily for reduce and map nodes.  These take a
// subcomputation which is almost always one of the four above, and pattern
// matching it to a short string lets us tell the user what the subcomputation
// is without drawing it as a graph.
optional<string> MatchTrivialComputation(const HloComputation* computation) {
  if (computation->instruction_count() != 3) {
    return nullopt;
  }

  HloInstruction* root = computation->root_instruction();
  if (root->operand_count() != 2) {
    return nullopt;
  }

  // Check that both of the operands to the root are parameters.
  const HloInstruction* operand0 = root->operand(0);
  const HloInstruction* operand1 = root->operand(1);
  if (operand0->opcode() != HloOpcode::kParameter ||
      operand1->opcode() != HloOpcode::kParameter) {
    return nullopt;
  }
  // Check that the two operands of root are param0 and param1.  All of the
  // opcodes we recognize are commutative, so we're OK with either order.
  auto n0 = operand0->parameter_number();
  auto n1 = operand1->parameter_number();
  if (!(n0 == 0 && n1 == 1) && !(n1 == 0 && n0 == 1)) {
    return nullopt;
  }

  // Check that the root and params are all effective scalars.
  if (!ShapeUtil::IsEffectiveScalar(root->shape()) ||
      !ShapeUtil::IsEffectiveScalar(operand0->shape()) ||
      !ShapeUtil::IsEffectiveScalar(operand1->shape())) {
    return nullopt;
  }

  // If we recognize the root's opcode, we've successfully pattern-matched!
  switch (root->opcode()) {
    case HloOpcode::kAdd:
      return "add";
    case HloOpcode::kMultiply:
      return "multiply";
    case HloOpcode::kMinimum:
      return "min";
    case HloOpcode::kMaximum:
      return "max";
    default:
      return nullopt;
  }
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
    const HloExecutionProfile* hlo_execution_profile,
    const NodeFilter& filter) {
  string graph_body;

  for (auto& instruction : instructions) {
    if (!filter.Show(instruction.get())) {
      continue;
    }

    // We don't display constants as separate nodes; they're merged into their
    // users.
    if (instruction->opcode() == HloOpcode::kConstant) {
      continue;
    }

    ColorScheme color = kYellow;
    string shape = "box";

    // Build the first line or two of the node, containing its name and opcode
    // (if the opcode isn't redundant with the name).
    string name;
    if (instruction->opcode() == HloOpcode::kParameter) {
      // If we have a parameter, put the param number in the name.
      name = StrCat("<b>Parameter ", instruction->parameter_number(),
                    "</b><br/>", HtmlLikeStringSanitize(instruction->name()));
    } else if (tensorflow::StringPiece(instruction->name())
                   .starts_with(
                       StrCat("%", instruction->ExtendedOpcodeStr()))) {
      // The HLO instruction name contains usually the opcode, e.g. "%add.42" is
      // an add instruction.  In this case we render just the name.
      name = StrCat("<b>", HtmlLikeStringSanitize(instruction->name()), "</b>");
    } else if (instruction->opcode() == HloOpcode::kFusion &&
               tensorflow::StringPiece(instruction->name())
                   .starts_with(
                       StrCat("%", HloOpcodeString(instruction->opcode())))) {
      // Fusion nodes are usually named e.g. "%fusion.5".  We render these as
      // e.g. "%fusion.5<br/>input fusion".
      name = StrCat("<b>", HtmlLikeStringSanitize(instruction->name()),
                    "</b><br/>",
                    HtmlLikeStringSanitize(instruction->ToCategory()));
    } else {
      // If the name does not contain the opcode, render both.
      name = StrCat("<b>",
                    HtmlLikeStringSanitize(instruction->ExtendedOpcodeStr()),
                    "</b><br/>", HtmlLikeStringSanitize(instruction->name()));
    }

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
      case HloOpcode::kCos:
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
      case HloOpcode::kConvolution:
      case HloOpcode::kDot:
        color = kDarkBlue;
        break;
      case HloOpcode::kParameter:
        color = kOrange;
        break;
      case HloOpcode::kBatchNormTraining:
        StrAppend(&name, " feature_index=", instruction->feature_index());
        color = kPurple;
        break;
      case HloOpcode::kBatchNormGrad:
        StrAppend(&name, " feature_index=", instruction->feature_index());
        color = kPurple;
        break;
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
      case HloOpcode::kReducePrecision:
        // Make ReducePrecision ops a bit more visible, since typically they
        // will be inserted as modifications to an existing graph.
        color = kRed;
        break;
      case HloOpcode::kConstant:
        LOG(FATAL) << "Constants don't get their own nodes in the graph.";
    }

    // Create instruction node with appropriate label, shape, and color.
    // label is interpreted as an HTML-like string, so newlines must be
    // delimited with <br/>, rather than \n.
    string label =
        StrCat(name, "<br/>", ShapeUtil::HumanString(instruction->shape()));

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

    // If this node's operands are omitted, style it accordingly.
    if (filter.SomeOrAllOperandsOmitted(instruction.get())) {
      color = kDashedBorder;
    }

    // If this node is highlighted, override its formatting.
    if (filter.Highlight(instruction.get())) {
      shape = "diamond";
      color = kDarkRed;
    }

    // Create edges from the instruction's operands to the instruction.
    if (!filter.OmitOperands(instruction.get())) {
      int64 operand_number = 0;
      for (auto* operand : instruction->operands()) {
        if (!filter.Show(operand) ||
            operand->opcode() == HloOpcode::kConstant) {
          ++operand_number;
          continue;
        }
        Appendf(&graph_body, "%s -> %s", InstructionId(operand).c_str(),
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
        StrAppend(&graph_body, "label=<fused expression for <b>",
                  HtmlLikeStringSanitize(instruction->name()),
                  "</b>>;\nstyle=\"rounded,filled\";\n"
                  "color=lightgrey;\n");
        StrAppend(&graph_body,
                  InstructionSequenceGraph(instruction->fused_instructions(),
                                           show_addresses, show_layouts,
                                           intercomputation_edges,
                                           hlo_execution_profile, NodeFilter()),
                  "}\n");
        string fusion_edge = StrCat(
            InstructionId(instruction->fused_expression_root()), " -> ",
            InstructionId(instruction.get()),
            "  [ style = \"dotted\", arrowsize=0.0, ltail=", cluster_name,
            " ];\n");
        intercomputation_edges->push_back(fusion_edge);
      } else {
        // If instruction has just one computation and it's trivial (e.g.
        // "return param0 + param1"), put the trivial computation type (e.g.
        // "add") into instruction's label.  Otherwise, add a dotted edge
        // between the instruction and its subcomputations.
        const auto& subcomputations = instruction->called_computations();

        bool trivial_subcomputation = false;
        if (subcomputations.size() == 1) {
          optional<string> computation_type =
              MatchTrivialComputation(subcomputations.front());
          if (computation_type) {
            trivial_subcomputation = true;
            StrAppend(&label, "<br/>Subcomputation: <b>", *computation_type,
                      "</b>");
          }
        }

        if (!trivial_subcomputation) {
          for (const HloComputation* computation :
               instruction->called_computations()) {
            string cluster_name =
                StrCat("cluster_", ComputationId(computation));
            string call_edge = Printf(
                "%s -> %s [ style=dashed; ltail=%s ];\n",
                InstructionId(computation->root_instruction()).c_str(),
                InstructionId(instruction.get()).c_str(), cluster_name.c_str());
            intercomputation_edges->push_back(call_edge);
          }
        }
      }
    }

    // Inline constant operands into the node.
    for (int64 i = 0; i < instruction->operand_count(); ++i) {
      const HloInstruction* operand = instruction->operand(i);
      if (operand->opcode() != HloOpcode::kConstant) {
        continue;
      }

      StrAppend(&label, "<br/><b>operand ", i, "</b> = ");
      if (ShapeUtil::IsEffectiveScalar(operand->shape())) {
        auto elem_idx = IndexUtil::LinearIndexToMultidimensionalIndex(
            operand->shape(), /*linear_index=*/0);
        StrAppend(&label, ShapeUtil::HumanString(operand->shape()), "{",
                  operand->literal().GetAsString(elem_idx), "}");
      } else {
        if (tensorflow::StringPiece(operand->name()).starts_with("%constant")) {
          StrAppend(&label, operand->name());
        } else {
          StrAppend(&label, "constant ", operand->name());
        }
      }
    }

    Appendf(&graph_body, "%s [label=<%s>, shape=%s, %s];\n",
            InstructionId(instruction.get()).c_str(), label.c_str(),
            shape.c_str(), NodeColorAttributes(color).c_str());
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
                             const HloExecutionProfile* hlo_execution_profile,
                             const NodeFilter& filter) {
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

  // Dump the subcomputations of each instruction that's shown and doesn't have
  // its operands omitted.  If an instruction has just one subcomputation and
  // it's trivial, omit it: We'll display that subcomputation inlined into the
  // instruction's node when we draw it.
  std::unordered_set<const HloComputation*> computations_to_dump;
  for (const auto& instr : computation.instructions()) {
    if (!filter.Show(instr.get()) || filter.OmitOperands(instr.get())) {
      continue;
    }
    if (instr->opcode() == HloOpcode::kFusion) {
      computations_to_dump.insert(instr->fused_instructions_computation());
    }

    const auto& subcomputations = instr->called_computations();
    if (subcomputations.size() != 1 ||
        !MatchTrivialComputation(subcomputations.front())) {
      for (const HloComputation* computation : instr->called_computations()) {
        computations_to_dump.insert(computation);
      }
    }
  }

  // Emit embedded computations as subgraph clusters.
  std::vector<string> intercomputation_edges;
  for (const HloComputation* embedded :
       computation.MakeEmbeddedComputationsList()) {
    if (!computations_to_dump.count(embedded)) {
      continue;
    }
    // Don't pass our filter down into the subcomputation -- always render the
    // whole thing.
    string graph_body = InstructionSequenceGraph(
        embedded->instructions(), show_addresses, show_layouts,
        &intercomputation_edges, hlo_execution_profile, NodeFilter());
    Appendf(&graph,
            "subgraph cluster_%s "
            "{\nstyle=rounded;label=<<b>%s</b>>;labelloc=t;\n%s}\n",
            ComputationId(embedded).c_str(), embedded->name().c_str(),
            graph_body.c_str());
  }
  StrAppend(&graph,
            InstructionSequenceGraph(computation.instructions(), show_addresses,
                                     show_layouts, &intercomputation_edges,
                                     hlo_execution_profile, filter));

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
  string RenderGraph(const string& graph, GraphKind graph_kind,
                     const DebugOptions& debug_options) override {
    static std::atomic<int> output_num(0);
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
        JoinPath(debug_options.xla_hlo_graph_path(),
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

// Gets a NodeFilter that includes roughly all instructions whose distance from
// root is <= radius.
//
// It's confusing to draw a node and include only some of its operands.  So if
// some but not all of a node's operands are <= radius units away from the root,
// we include the other operands (unless there are a lot of them, as often in a
// tuple node).  These additional operands may have as inputs other nodes
// already present in the graph, but we don't draw those edges unless *all* of
// the inputs are present.  (Otherwise we'd have the same problem we were trying
// to solve in the first place!)
NodeFilter MakeNodeFilter(const HloInstruction* root, int64 radius) {
  // First, find the neighborhood of nodes with distance from root <= radius.
  // These nodes are our initial set of "normal" nodes.
  std::unordered_map<const HloInstruction*, NodeFilterResult> nodes;
  std::deque<std::pair<const HloInstruction*, /*depth*/ int64>> worklist;
  worklist.push_back({root, 0});
  while (!worklist.empty()) {
    const HloInstruction* instr;
    int64 depth;
    std::tie(instr, depth) = worklist.front();
    worklist.pop_front();

    nodes[instr] = kNormalNode;
    if (depth == radius) {
      continue;
    }

    // Traverse into instr's operands.
    //
    // Don't traverse into tuples' operands unless the tuple is the root.
    // Usually a tuple is the bottommost node in the graph, and so its operands
    // are not interesting to the graph at hand.
    if (instr == root || instr->opcode() != HloOpcode::kTuple) {
      for (const HloInstruction* operand : instr->operands()) {
        if (!nodes.count(operand)) {
          worklist.push_back({operand, depth + 1});
        }
      }
    }

    // If you're looking at node X, it's probably not interesting that node Y
    // also happens to use the same constant, so we don't traverse into
    // constants' users.
    if (instr->opcode() != HloOpcode::kConstant) {
      for (const HloInstruction* user : instr->users()) {
        if (!nodes.count(user)) {
          worklist.push_back({user, depth + 1});
        }
      }
    }
  }

  auto is_displayed = [&](const HloInstruction* instr) {
    return nodes.count(instr) > 0;
  };

  // If a node has some but not all of its operands omitted, add the operands to
  // the map with type kOmitNodeOperands.  Unless the node has a lot of
  // operands, in which case just mark the node as "some operands omitted".
  std::vector<const HloInstruction*> extra_operands;
  for (auto& kv : nodes) {
    const HloInstruction* instr = kv.first;
    NodeFilterResult& filter_result = kv.second;
    const auto& operands = instr->operands();

    // Mark nodes with many operands and some omitted as "some operands omitted"
    // and carry on -- don't add their omitted operands to extra_operands.
    if (operands.size() > 4) {
      if (std::any_of(operands.begin(), operands.end(), is_displayed) &&
          !std::all_of(operands.begin(), operands.end(), is_displayed)) {
        filter_result = kSomeOperandsOmitted;
      }
      continue;
    }

    if (std::any_of(operands.begin(), operands.end(), is_displayed)) {
      for (const HloInstruction* operand : operands) {
        if (!is_displayed(operand)) {
          extra_operands.push_back(operand);
        }
      }
    }
  }
  for (const HloInstruction* instr : extra_operands) {
    nodes[instr] = kOmitNodeOperands;
  }

  // Some of the nodes in extra_operands may now have all of their inputs
  // present in nodes.  We can promote these to normal nodes.
  for (const HloInstruction* instr : extra_operands) {
    const auto& operands = instr->operands();
    if (std::all_of(operands.begin(), operands.end(), is_displayed)) {
      nodes[instr] = kNormalNode;
    }
  }

  // If none of a node's operands appear in nodes, mark it as type
  // kOmitNodeOperands so it gets styled appropriately.
  for (auto& kv : nodes) {
    const auto& operands = kv.first->operands();
    if (!operands.empty() &&
        std::none_of(operands.begin(), operands.end(), is_displayed)) {
      kv.second = kOmitNodeOperands;
    }
  }

  // Highlight the root node.
  nodes[root] = kHighlightNode;

  return NodeFilter([=](const HloInstruction* instr) {
    auto it = nodes.find(instr);
    if (it != nodes.end()) {
      return it->second;
    }
    return kHideNode;
  });
}

XLA_REGISTER_GRAPH_RENDERER(FileGraphRenderer, 0);

}  // namespace

string DumpGraph(const HloComputation& computation, const string& label,
                 const DebugOptions& debug_options,
                 const HloExecutionProfile* hlo_execution_profile) {
  string graph;
  string graph_url;
  if (debug_options.xla_hlo_dump_as_graphdef()) {
    HloTfGraphBuilder builder;
    TF_CHECK_OK(builder.AddComputation(computation));
    CHECK(tensorflow::protobuf::TextFormat::PrintToString(builder.GetGraphDef(),
                                                          &graph));
    // TODO(b/37198616): Use the default registered renderers when all
    // renderers support rendering GraphDefs. Always dump GraphDefs to files
    // for now.
    graph_url = FileGraphRenderer().RenderGraph(
        graph, GraphRendererInterface::TF_GRAPHDEF, debug_options);
  } else {
    graph = ComputationToDotGraph(computation, label,
                                  debug_options.xla_hlo_graph_addresses(),
                                  debug_options.xla_hlo_graph_layout(),
                                  hlo_execution_profile, NodeFilter());
    graph_url = GetGraphRenderer()->RenderGraph(
        graph, GraphRendererInterface::DOT_GRAPH, debug_options);
  }
  LOG(INFO) << "computation " << computation.name() << " [" << label
            << "]: " << graph_url;
  return graph_url;
}

string DumpNeighborhoodAround(const HloInstruction& node, int radius) {
  auto debug_options = node.GetModule()->config().debug_options();
  string label =
      StrCat("Neighborhood of ", radius, " nodes around ", node.name());
  NodeFilter filter = MakeNodeFilter(&node, radius);
  string graph = ComputationToDotGraph(
      *node.parent(), label,
      /*show_addresses=*/debug_options.xla_hlo_graph_addresses(),
      /*show_layouts=*/debug_options.xla_hlo_graph_layout(),
      /*hlo_execution_profile=*/nullptr, filter);
  return GetGraphRenderer()->RenderGraph(
      graph, GraphRendererInterface::DOT_GRAPH, debug_options);
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
  LOG(INFO) << "dumping module '" << module.name() << "' to " << path;
}

string MaybeDumpHloModule(const HloModule& module, const string& label,
                          const HloExecutionProfile* profile) {
  VLOG(2) << "MaybeDumpHloModule called on module " << module.name();
  string graph_url;
  const DebugOptions& debug_options = module.config().debug_options();
  if (!debug_options.xla_generate_hlo_graph().empty() &&
      RE2::PartialMatch(module.name(),
                        debug_options.xla_generate_hlo_graph())) {
    graph_url =
        DumpGraph(*module.entry_computation(), label, debug_options, profile);
  }
  if (!debug_options.xla_log_hlo_text().empty() &&
      RE2::PartialMatch(module.name(), debug_options.xla_log_hlo_text())) {
    LOG(INFO) << "HLO for module " << module.name();
    LOG(INFO) << "Label: " << label;
    XLA_LOG_LINES(2, module.ToString());
  }
  if (!debug_options.xla_generate_hlo_text_to().empty()) {
    DumpText(module, label, debug_options.xla_generate_hlo_text_to());
  }
  return graph_url;
}

}  // namespace hlo_graph_dumper
}  // namespace xla
