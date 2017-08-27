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
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;
using ::tensorflow::str_util::Join;
using ::tensorflow::str_util::StringReplace;
using ::tensorflow::WriteStringToFile;

namespace xla {
namespace hlo_graph_dumper {
namespace {

// Helpers for Printf and Appendf.
template <typename T>
struct PrintfConvert {
  const T& operator()(const T& t) const { return t; }
};
template <>
struct PrintfConvert<string> {
  const char* operator()(const string& s) const { return s.c_str(); }
};

// Like tensorflow::strings::Printf/Appendf, but you don't need to call c_str()
// on strings.
template <typename... Ts>
string Printf(const char* fmt, const Ts&... ts) {
  return tensorflow::strings::Printf(fmt, PrintfConvert<Ts>()(ts)...);
}
template <typename... Ts>
void Appendf(string* s, const char* fmt, const Ts&... ts) {
  tensorflow::strings::Appendf(s, fmt, PrintfConvert<Ts>()(ts)...);
}

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
  // Same style as kSomeOperandsOmitted, but used to indicate that some of the
  // node's *users* have been omitted.
  kSomeUsersOmitted,
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
  bool Deemphasized(const HloInstruction* instr) const {
    auto result = filter_(instr);
    return result == kOmitNodeOperands || result == kSomeOperandsOmitted ||
           result == kSomeUsersOmitted;
  }

  bool ShowFusionSubcomputation(const HloInstruction* instr) const {
    CHECK_EQ(instr->opcode(), HloOpcode::kFusion);
    return Show(instr) && !SomeOrAllOperandsOmitted(instr);
  }

 private:
  std::function<NodeFilterResult(const HloInstruction* instr)> filter_;
};

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
        // "filled,dashed" looks the same as "dashed", since we have a white
        // background.  But we use "filled,dashed" so that when you hover over
        // any part of the node (not just the text inside the node), our css
        // :hover rule is triggered.
        return make_tuple("filled,dashed", "white", "#757575", "#757575");
    }
  }();

  return Printf(
      R"(style="%s", fontcolor="%s", color="%s", fillcolor="%s")", style,
      font_color, stroke_color, fill_color);
}

// Replaces <> with &lt;&gt;, so that this string is safe(er) for use in a
// graphviz HTML-like string.
string HtmlLikeStringSanitize(tensorflow::StringPiece s) {
  return StringReplace(StringReplace(s, "<", "&lt;", /*replace_all=*/true), ">",
                       "&gt;", /*replace_all=*/true);
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
//   "return param0 <= param1;"     --> "less-or-equal"
//   "return param0 >= param1;"     --> "greater-or-equal"
//   "return param0 >  param1;"     --> "greater-than"
//   "return param0 <  param1;"     --> "less-than"
//   "return param0 == param1;"     --> "equal-to"
//   "return param0 != param1;"     --> "not-equal-to"
//
// where param0 and param1 are effective scalars.  For the ops that are
// commutative, we also support them with param0 and param1 swapped.
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

  // If the params are reversed, check that the operation being performed is
  // commutative.
  if (n0 == 1) {
    switch (root->opcode()) {
      case HloOpcode::kLe:
      case HloOpcode::kGe:
      case HloOpcode::kGt:
      case HloOpcode::kLt:
        return nullopt;
      default:
        break;
    }
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
    case HloOpcode::kLe:
      return "less-or-equal";
    case HloOpcode::kGe:
      return "greater-or-equal";
    case HloOpcode::kGt:
      return "greater-than";
    case HloOpcode::kLt:
      return "less-than";
    case HloOpcode::kEq:
      return "equal-to";
    case HloOpcode::kNe:
      return "not-equal-to";
    default:
      return nullopt;
  }
}

// Encapsulates logic for dumping an HLO module to DOT (i.e. graphviz syntax).
class HloDotDumper {
 public:
  HloDotDumper(const HloComputation* computation, tensorflow::StringPiece label,
               bool show_addresses, const HloExecutionProfile* profile,
               NodeFilter filter)
      : computation_(computation),
        label_(label.ToString()),
        show_addresses_(show_addresses),
        profile_(profile),
        filter_(std::move(filter)) {}

  string Dump();

 private:
  // Returns the dot graph identifier for the given instruction.
  string InstructionId(const HloInstruction* instruction) {
    return StrCat(reinterpret_cast<uint64>(instruction));
  }

  // Returns the dot graph identifier for the given computation.
  string SubcomputationId(const HloComputation* computation) {
    return StrCat("cluster_", reinterpret_cast<uint64>(computation));
  }

  // Generates graph header/footer.  These should be called *after* dumping all
  // of the instructions and subcomputations for the graph, as they both use
  // data generated while dumping the graph.
  string Header();
  string Footer();

  // Maps HloComputations we should dump to their parent instruction in the
  // outer computation.
  std::unordered_map<const HloComputation*, const HloInstruction*>
  SubcomputationsToDump();

  string DumpSubcomputation(const HloComputation* subcomp,
                            const HloInstruction* parent_instr);
  string DumpComputation(const HloComputation* comp);
  string DumpInstruction(const HloInstruction* instr);
  ColorScheme GetInstructionColor(const HloInstruction* instr);
  string GetInstructionNodeShape(const HloInstruction* instr);
  string GetInstructionNodeLabel(const HloInstruction* instr);
  string GetInstructionNodeExtraInfo(const HloInstruction* instr);
  string GetInstructionNodeInlinedConstants(const HloInstruction* instr);
  void AddInstructionIncomingEdges(const HloInstruction* instr);

  // If instr has just one computation and it's trivial (e.g. "return param0 +
  // param1"), returns a string you can put into the node's body that names the
  // subcomputation, e.g. "Subcomputation: <b>add</b>".
  string GetInstructionTrivialComputationStr(const HloInstruction* instr);

  const HloComputation* computation_;  // never null
  const string label_;                 // overall name for the graph
  const bool show_addresses_;
  const HloExecutionProfile* profile_;  // may be null
  const NodeFilter filter_;

  // Each HloInstruction dumped gets a monotically-increasing node ID.  This
  // must start at 1, because that's where graphviz's accounting starts.
  int64 next_node_id_ = 1;
  std::unordered_map<const HloInstruction*, int64> node_ids_;

  // Each (from, to) edge gets a monotonically-increasing ID.  This is a
  // multimap because it's possible for the same edge to appear multiple times
  // in the graph (e.g. x^2 may be represented as mul(x, x)).
  int64 next_edge_id_ = 1;
  std::unordered_multimap<
      std::pair<const HloInstruction*, const HloInstruction*>, int64,
      tensorflow::hash<std::pair<const HloInstruction*, const HloInstruction*>>>
      edge_ids_;

  // Each HloComputation that's emitted gets a monotonically-increasing ID.
  int64 next_cluster_id_ = 1;
  std::unordered_map<const HloComputation*, int64> cluster_ids_;

  // Edges to print from Footer().  Edges come at the end because graphviz is
  // unhappy if an edge from a subcomputation to a node in the outer computation
  // appears before both the inner computation and the destination node are
  // defined.
  std::vector<string> edges_;
};

string HloDotDumper::Dump() {
  string body;
  for (const auto& kv : SubcomputationsToDump()) {
    const HloComputation* subcomp = kv.first;
    const HloInstruction* parent = kv.second;
    StrAppend(&body, DumpSubcomputation(subcomp, parent));
  }
  StrAppend(&body, DumpComputation(computation_));

  // By contract, Header() and Footer() have to be called after we've dumped all
  // our instructions, because they use state generated during that process.
  string g = Header();
  StrAppend(&g, body);
  StrAppend(&g, Footer());
  return g;
}

string HloDotDumper::Header() {
  const char* fmt = R"(digraph G {
rankdir = TB;
compound = true;
label = <<b>%s</b>>;
labelloc = t;
// Disable the tooltip.  Interestingly, "" doesn't work!
tooltip = " ";
// DOT graphs accept a stylesheet as a URI.  So naturally, an inline
// stylesheet is a data URI!
stylesheet="
  data:text/css,
  @import url(https://fonts.googleapis.com/css?family=Roboto:400,700);
  svg text {
    font-family: 'Roboto';
    font-size: 12px;
  }

  %s
"

)";

  string graph_label = StrCat(label_, "<br/>", computation_->name());
  if (profile_ != nullptr) {
    auto cycles = profile_->total_cycles_executed(*computation_);
    Appendf(&graph_label, "<br/>total cycles = %lld (%s)", cycles,
            tensorflow::strings::HumanReadableNum(cycles));
  }

  // Create CSS rules that say, when you hover over the given node or cluster,
  // turn the given edge the given color.
  //
  // We rely on a few properties of how graphviz generates SVGs:
  //
  //  - Nodes are named "nodeN", where N corresponds to the 1-based index of
  //    the node in our DOT (i.e. the first node in the DOT is "node1", etc.).
  //    Edges are similarly named "edgeN", and clusters are named "clustN".
  //  - Nodes come before their in- and out-edges in the SVG.  We need this
  //    because the "X ~ Y" CSS selector finds a sibling of X that *comes
  //    after X in the DOM* and matches Y.
  std::vector<string> edge_css_rules;
  const char* kBlue = "#1976d2";
  const char* kRed = "#d32f2f";
  for (const auto& kv : edge_ids_) {
    const HloInstruction* from_node = kv.first.first;
    const HloInstruction* to_node = kv.first.second;
    int64 edge_id = kv.second;

    auto add_hover_css_rule = [&](string elem_type, int64 elem_id,
                                  const char* color) {
      // One could imagine other ways of writing this CSS rule that involve less
      // duplication, but this way seems to be relatively performant.
      edge_css_rules.push_back(Printf(
          "  #%s%d:hover ~ #edge%lld text { fill: %s; }\n"
          "  #%s%d:hover ~ #edge%lld path { stroke: %s; stroke-width: .2em; }\n"
          "  #%s%d:hover ~ #edge%lld polygon { "
          "fill: %s; stroke: %s; stroke-width: .2em; }\n",
          elem_type, elem_id, edge_id, color,  //
          elem_type, elem_id, edge_id, color,  //
          elem_type, elem_id, edge_id, color, color));
    };

    int64 from_node_id = node_ids_.at(from_node);
    int64 to_node_id = node_ids_.at(to_node);
    add_hover_css_rule("node", from_node_id, kBlue);
    add_hover_css_rule("node", to_node_id, kRed);

    // If this edge crosses a fusion cluster boundary, highlight it when the
    // cluster is hovered over.
    if (from_node->IsFused() &&
        from_node->fusion_instruction()->fused_expression_root() == from_node) {
      int64 cluster_id = cluster_ids_.at(from_node->parent());
      add_hover_css_rule("clust", cluster_id, kBlue);
    }
    if (to_node->IsFused() && to_node->opcode() == HloOpcode::kParameter) {
      int64 cluster_id = cluster_ids_.at(to_node->parent());
      add_hover_css_rule("clust", cluster_id, kRed);
    }
  }

  return Printf(fmt, graph_label, Join(edge_css_rules, "\n"));
}

string HloDotDumper::Footer() { return StrCat(Join(edges_, "\n"), "\n}"); }

std::unordered_map<const HloComputation*, const HloInstruction*>
HloDotDumper::SubcomputationsToDump() {
  // Dump the subcomputations of each instruction that's shown and doesn't have
  // its operands omitted.  If an instruction has just one subcomputation and
  // it's trivial, omit it: We'll display that subcomputation inlined into the
  // instruction's node when we draw it.
  std::unordered_map<const HloComputation*, const HloInstruction*> to_dump;
  for (const auto& instr : computation_->instructions()) {
    if (!filter_.Show(instr.get()) ||
        filter_.SomeOrAllOperandsOmitted(instr.get())) {
      continue;
    }
    if (instr->opcode() == HloOpcode::kFusion) {
      to_dump[instr->fused_instructions_computation()] = instr.get();
    }

    for (const HloComputation* comp : instr->called_computations()) {
      if (!MatchTrivialComputation(comp)) {
        to_dump[comp] = instr.get();
      }
    }
  }
  return to_dump;
}

string HloDotDumper::DumpSubcomputation(const HloComputation* subcomp,
                                        const HloInstruction* parent_instr) {
  const char* computation_fmt = R"(subgraph %s {
%s
label = <%s>;
labelloc = t;
tooltip = " ";
%s
}  // %s

)";

  cluster_ids_[subcomp] = next_cluster_id_++;

  string id = SubcomputationId(subcomp);

  string subcomp_label, style;
  if (parent_instr->opcode() == HloOpcode::kFusion) {
    subcomp_label = Printf("Fused expression for <b>%s</b><br/>%s",
                           HtmlLikeStringSanitize(parent_instr->name()),
                           HtmlLikeStringSanitize(parent_instr->ToCategory()));
    string extra_info = GetInstructionNodeExtraInfo(parent_instr);
    if (!extra_info.empty()) {
      StrAppend(&subcomp_label, "<br/>", extra_info);
    }

    // Subcomputation's fill/stroke color is light/dark red/gray, depending on
    // whether or not the subcomputation's fusion node is highlighted.
    bool highlight = filter_.Highlight(parent_instr);
    const char* fillcolor = highlight ? "#ffcdd2" : "#f5f5f5";
    const char* strokecolor = highlight ? "#b71c1c" : "#c2c2c2";
    style =
        Printf(R"(style="rounded,filled,bold"; fillcolor="%s"; color="%s;")",
               fillcolor, strokecolor);
  } else {
    subcomp_label = Printf("Subcomputation for <b>%s</b><br/>%s",
                           HtmlLikeStringSanitize(parent_instr->name()),
                           HtmlLikeStringSanitize(subcomp->name()));
    style = "style=rounded; color=black;";
  }

  string comp_body = DumpComputation(subcomp);
  string computation =
      Printf(computation_fmt, id, style, subcomp_label, comp_body, id);

  // Add an edge from the subcomputation to its parent node.  If subcomp
  // belongs to a fusion node, it's drawn in place of the fusion instruction, so
  // there's no need to link those.
  if (parent_instr->opcode() != HloOpcode::kFusion) {
    edge_ids_.insert(
        {{subcomp->root_instruction(), parent_instr}, next_edge_id_++});
    const char* edge_fmt =
        R"(%s -> %s [ltail="%s", style="dashed" tooltip="%s -> %s"];)";
    edges_.push_back(
        Printf(edge_fmt, InstructionId(subcomp->root_instruction()),
               InstructionId(parent_instr), SubcomputationId(subcomp),
               subcomp->name(), parent_instr->name()));
  }

  return computation;
}

string HloDotDumper::DumpComputation(const HloComputation* comp) {
  string g;
  for (const auto& instr : comp->instructions()) {
    if (!filter_.Show(instr.get())) {
      continue;
    }
    StrAppend(&g, DumpInstruction(instr.get()));
  }
  return g;
}

string HloDotDumper::DumpInstruction(const HloInstruction* instr) {
  // We don't display constants as separate nodes; they're merged into their
  // users.
  if (instr->opcode() == HloOpcode::kConstant) {
    return "";
  }
  // Omit the fusion node if its subcomputation is drawn, since the
  // subcomputation will be drawn inline.
  if (instr->opcode() == HloOpcode::kFusion &&
      filter_.ShowFusionSubcomputation(instr)) {
    return "";
  }

  node_ids_[instr] = next_node_id_++;

  ColorScheme color = GetInstructionColor(instr);
  string node_shape = GetInstructionNodeShape(instr);
  string node_label = GetInstructionNodeLabel(instr);
  string extra_info = GetInstructionNodeExtraInfo(instr);
  string inlined_constants = GetInstructionNodeInlinedConstants(instr);
  string trivial_subcomputation = GetInstructionTrivialComputationStr(instr);
  AddInstructionIncomingEdges(instr);

  // Override the node's styling if it should be (de-)emphasized.
  if (filter_.Deemphasized(instr)) {
    color = kDashedBorder;
  }
  if (filter_.Highlight(instr)) {
    node_shape = "diamond";
    color = kDarkRed;
  }

  // Build the text that will be displayed inside the node.
  string node_body = node_label;
  for (const string& s :
       {trivial_subcomputation, extra_info, inlined_constants}) {
    if (!s.empty()) {
      StrAppend(&node_body, "<br/>", s);
    }
  }

  return Printf(R"(%s [label=<%s>, shape=%s, tooltip=" ", %s];)"
                "\n",
                InstructionId(instr), node_body, node_shape,
                NodeColorAttributes(color));
}

string HloDotDumper::GetInstructionNodeInlinedConstants(
    const HloInstruction* instr) {
  auto stringify_constant = [](const HloInstruction* constant) {
    if (ShapeUtil::IsEffectiveScalar(constant->shape())) {
      auto elem_idx = IndexUtil::LinearIndexToMultidimensionalIndex(
          constant->shape(), /*linear_index=*/0);
      return Printf("%s (%s)", constant->literal().GetAsString(elem_idx),
                    ShapeUtil::HumanString(constant->shape()));
    }
    if (tensorflow::StringPiece(constant->name()).starts_with("%constant")) {
      return constant->name();
    }
    return StrCat("constant ", constant->name());
  };

  // Special case: If instr is a parameter to a fusion node, check whether the
  // corresponding operand to the fusion node is a constant.
  if (instr->opcode() == HloOpcode::kParameter && instr->IsFused()) {
    const HloInstruction* fusion = instr->fusion_instruction();
    const HloInstruction* operand = fusion->operand(instr->parameter_number());
    if (operand->opcode() != HloOpcode::kConstant) {
      return "";
    }
    return StrCat("<b>constant</b> ", stringify_constant(operand));
  }

  std::vector<string> lines;
  for (int64 i = 0; i < instr->operand_count(); ++i) {
    const HloInstruction* operand = instr->operand(i);
    if (operand->opcode() != HloOpcode::kConstant) {
      continue;
    }
    lines.push_back(
        Printf("<b>operand %lld</b> = %s", i, stringify_constant(operand)));
  }
  return Join(lines, "<br/>");
}

ColorScheme HloDotDumper::GetInstructionColor(const HloInstruction* instr) {
  // Pick different colors or shapes for instructions which are particularly
  // expensive (eg, dot) and those which are unusual in some way or unique
  // (eg, parameter).
  switch (instr->opcode()) {
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
    case HloOpcode::kSin:
    case HloOpcode::kSlice:
    case HloOpcode::kSort:
    case HloOpcode::kSubtract:
    case HloOpcode::kTanh:
    case HloOpcode::kRng:
    case HloOpcode::kBroadcast:
    case HloOpcode::kTranspose:
      return kYellow;
    case HloOpcode::kBitcast:
    case HloOpcode::kTuple:
    case HloOpcode::kTrace:
    case HloOpcode::kGetTupleElement:
      return kWhite;
    case HloOpcode::kConcatenate:
    case HloOpcode::kCopy:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kPad:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kUpdate:
      return kGreen;
    case HloOpcode::kConvolution:
    case HloOpcode::kDot:
      return kDarkBlue;
    case HloOpcode::kReducePrecision:
      return kRed;
    case HloOpcode::kParameter:
      return kOrange;
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kReduce:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kReduceWindow:
      return kPurple;
    case HloOpcode::kMap:
    case HloOpcode::kFusion:
      return kGray;
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kCrossReplicaSum:
      return kBrown;
    case HloOpcode::kCustomCall:
    case HloOpcode::kWhile:
    case HloOpcode::kCall:
      return kDarkGreen;
    case HloOpcode::kConstant:
      LOG(FATAL) << "Constants don't get their own nodes in the graph.";
  }
}

string HloDotDumper::GetInstructionNodeShape(const HloInstruction* instr) {
  // Give while loops a different shape so they're easier to pick out.
  switch (instr->opcode()) {
    case HloOpcode::kWhile:
      return "ellipse";
    default:
      return "rect";
  }
}

string HloDotDumper::GetInstructionNodeLabel(const HloInstruction* instr) {
  // If we have a parameter, put the param number in the name.
  if (instr->opcode() == HloOpcode::kParameter) {
    return Printf("<b>Parameter %lld</b>", instr->parameter_number());
  }

  // The HLO instruction name contains usually the opcode, e.g. "%add.42" is
  // an add instruction.  In this case we render just the name.
  if (tensorflow::StringPiece(instr->name())
          .starts_with(StrCat("%", HloOpcodeString(instr->opcode())))) {
    return Printf("<b>%s</b>", HtmlLikeStringSanitize(instr->name()));
  }

  // If the name does not contain the opcode, render both.
  return Printf("<b>%s</b><br/>%s",
                HtmlLikeStringSanitize(instr->ExtendedOpcodeStr()),
                HtmlLikeStringSanitize(instr->name()));
}

string HloDotDumper::GetInstructionNodeExtraInfo(const HloInstruction* instr) {
  string opcode_specific_info = [&]() -> string {
    switch (instr->opcode()) {
      case HloOpcode::kRng:
        return RandomDistribution_Name(instr->random_distribution());
      case HloOpcode::kConvolution:
        return StrCat(
            HtmlLikeStringSanitize(
                instr->ConvolutionDimensionNumbersToString()),
            "<br/>",
            HtmlLikeStringSanitize(window_util::ToString(instr->window())));
      case HloOpcode::kBroadcast:
      case HloOpcode::kTranspose:
      case HloOpcode::kReduce:
        return Printf("dims={%s}", Join(instr->dimensions(), ","));
      case HloOpcode::kGetTupleElement:
        return Printf("index=%lld", instr->tuple_index());
      case HloOpcode::kBatchNormTraining:
      case HloOpcode::kBatchNormGrad:
        return Printf("feature_index=%lld", instr->feature_index());
      case HloOpcode::kCustomCall:
        return Printf("custom_call_target=%s", instr->custom_call_target());
      default:
        return "";
    }
  }();

  std::vector<string> lines;
  if (!opcode_specific_info.empty()) {
    lines.push_back(opcode_specific_info);
  }

  // Show the shape and layout of the instruction, unless it's an inlined fusion
  // node -- there the shape and layout is present in the output node.
  if (instr->opcode() != HloOpcode::kFusion ||
      !filter_.ShowFusionSubcomputation(instr)) {
    string instr_shape = ShapeUtil::HumanString(instr->shape());

    // Show layout of non-tuple shapes with more than one dimension.
    if (LayoutUtil::HasLayout(instr->shape()) &&
        instr->shape().dimensions_size() > 1 &&
        !ShapeUtil::IsTuple(instr->shape())) {
      StrAppend(&instr_shape, "{",
                Join(instr->shape().layout().minor_to_major(), ","), "}");
    }

    // Some instructions have giant tuples as their shapes, so truncate the
    // HLO's shape to kMaxShapeLen characters.
    constexpr int kMaxShapeLen = 64;
    if (instr_shape.length() > kMaxShapeLen) {
      instr_shape = StrCat(
          tensorflow::StringPiece(instr_shape).substr(0, kMaxShapeLen - 3),
          "...");
    }
    lines.push_back(instr_shape);
  }

  if (show_addresses_) {
    lines.push_back(Printf("[%p]", instr));
  }
  if (profile_ != nullptr) {
    double hlo_cycles_executed = profile_->GetProfileResult(*instr);
    double total_cycles_executed =
        profile_->total_cycles_executed(*instr->parent());
    if (hlo_cycles_executed > 0 && total_cycles_executed > 0) {
      lines.push_back(
          Printf("%% of cycles executed=%.2f",
                 100 * hlo_cycles_executed / total_cycles_executed));
    }
  }
  return Join(lines, "<br/>");
}

void HloDotDumper::AddInstructionIncomingEdges(const HloInstruction* instr) {
  auto add_edge = [&](const HloInstruction* from, const HloInstruction* to,
                      int64 operand_num, bool control_edge = false) {
    // Fusion nodes' subcomputations are displayed inline, so if 'from' is a
    // fusion node and the node's subcomputation is shown, we draw our edge
    // starting at the fusion node's root instead of at the fusion node itself.
    if (from->opcode() == HloOpcode::kFusion &&
        filter_.ShowFusionSubcomputation(from)) {
      from = from->fused_expression_root();
    }
    if (!filter_.Show(from) || from->opcode() == HloOpcode::kConstant) {
      return;
    }
    edge_ids_.insert({{from, to}, next_edge_id_++});

    string edge_label;
    if (instr->operand_count() > 1 && !control_edge) {
      edge_label = Printf(R"( headlabel="%lld", labeldistance=2)", operand_num);
    } else if (control_edge) {
      edge_label = "style=\"dotted\" color=\"gray\" label=\"ctrl\"";
    }
    const char* kEdgeFmt = R"(%s -> %s [tooltip="%s -> %s" %s];)";
    edges_.push_back(Printf(kEdgeFmt, InstructionId(from), InstructionId(to),
                            from->name(), to->name(), edge_label));
  };

  // Add edges from instr's operands to instr.  Parameters within fusion
  // expressions are handled specially -- we draw an edge from the corresponding
  // operand on the fusion node itself to the parameter.
  if (instr->opcode() == HloOpcode::kParameter && instr->IsFused()) {
    const HloInstruction* fusion = instr->fusion_instruction();
    add_edge(fusion->operand(instr->parameter_number()), instr,
             /*operand_num=*/0);
  } else {
    for (int64 i = 0; i < instr->operand_count(); ++i) {
      add_edge(instr->operand(i), instr, i);
    }
    for (const HloInstruction* pred : instr->control_predecessors()) {
      add_edge(pred, instr, /*operand_num=*/0, /*control_edge=*/true);
    }
  }
}

string HloDotDumper::GetInstructionTrivialComputationStr(
    const HloInstruction* instr) {
  // called_computations() on a fusion node "inherits" any called computations
  // of the fused root, which isn't what we want.  Just ignore fusion nodes
  // here; they're handled separately.
  if (instr->opcode() == HloOpcode::kFusion) {
    return "";
  }

  std::vector<string> lines;
  for (int64 i = 0; i < instr->called_computations().size(); ++i) {
    optional<string> computation_type =
        MatchTrivialComputation(instr->called_computations()[i]);
    if (!computation_type) {
      continue;
    }
    if (instr->called_computations().size() == 1) {
      lines.push_back(Printf("Subcomputation: <b>%s</b>",
                             HtmlLikeStringSanitize(*computation_type)));
    } else {
      lines.push_back(Printf("Subcomputation %lld: <b>%s</b>", i,
                             HtmlLikeStringSanitize(*computation_type)));
    }
  }
  return Join(lines, "<br/>");
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

    // Traverse into instr's users, unless:
    //
    //  - there are a ton of them, in which case they're probably not
    //    interesting (and anyway, rendering them all would make the graph
    //    unreadable), or
    //  - instr is a constant, in which case its users are probably not
    //    interesting.
    if (instr->opcode() == HloOpcode::kConstant) {
      continue;
    }
    constexpr int kMaxUsersToRender = 16;
    if (instr->user_count() > kMaxUsersToRender) {
      // If we're going to skip this node's users, style it as such.
      nodes[instr] = kSomeUsersOmitted;
      continue;
    }
    for (const HloInstruction* user : instr->users()) {
      if (!nodes.count(user)) {
        worklist.push_back({user, depth + 1});
      }
    }
  }

  auto is_displayed = [&](const HloInstruction* instr) {
    // Constants are displayed inline with their users; they're never omitted.
    return nodes.count(instr) > 0 || instr->opcode() == HloOpcode::kConstant;
  };

  // Make a second pass over 'nodes' to fix up the NodeFilterResults now that we
  // know which nodes will be included in the graph.
  for (auto& kv : nodes) {
    const HloInstruction* instr = kv.first;
    NodeFilterResult& filter_result = kv.second;
    const auto& operands = instr->operands();

    if (std::any_of(operands.begin(), operands.end(), is_displayed) &&
        !std::all_of(operands.begin(), operands.end(), is_displayed)) {
      // Mark nodes with some operands omitted appropriately.
      filter_result = kSomeOperandsOmitted;
    } else if (!operands.empty() &&
               std::none_of(operands.begin(), operands.end(), is_displayed)) {
      // Mark nodes with *all* operands omitted appropriately.
      filter_result = kOmitNodeOperands;
    }

    // Promote nodes with type kSomeUsersOmitted to kNormalNode if all of their
    // users made it into the graph.
    if (filter_result == kSomeUsersOmitted &&
        std::all_of(instr->users().begin(), instr->users().end(),
                    is_displayed)) {
      filter_result = kNormalNode;
    }
  }

  // Highlight the root node.
  nodes[root] = kHighlightNode;

  return NodeFilter([=](const HloInstruction* instr) {
    auto it = nodes.find(instr);
    if (it != nodes.end()) {
      return it->second;
    }
    // Show all nodes in subcomputations.
    if (instr->parent() != root->parent()) {
      return kNormalNode;
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
    graph =
        HloDotDumper(&computation, label,
                     /*show_addresses=*/debug_options.xla_hlo_graph_addresses(),
                     hlo_execution_profile, NodeFilter())
            .Dump();
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
  string graph =
      HloDotDumper(node.parent(), label,
                   /*show_addresses=*/debug_options.xla_hlo_graph_addresses(),
                   /*profile=*/nullptr, filter)
          .Dump();
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
