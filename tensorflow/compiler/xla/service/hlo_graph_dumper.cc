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
#include <queue>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/regexp.h"

namespace xla {
namespace {

using absl::nullopt;
using absl::optional;
using absl::StrAppend;
using absl::StrCat;
using absl::StrFormat;
using absl::StrJoin;

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

 private:
  std::function<NodeFilterResult(const HloInstruction* instr)> filter_;
};

// We arbitrarily set this as the boundary between "large" and "small"
// instructions.
bool IsSmall(const HloInstruction* instr) {
  if (ShapeUtil::HasPrimitiveType(instr->shape(), OPAQUE_TYPE) ||
      ShapeUtil::HasPrimitiveType(instr->shape(), TOKEN)) {
    return true;
  }
  return ShapeUtil::ElementsInRecursive(instr->shape()) < 4096;
}

// Node color schemes, used by NodeColorAttributes.
enum ColorScheme {
  kBlue,
  kBrown,
  kDarkBlue,
  kDarkGreen,
  kDarkOrange,
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

// Graphviz attributes/colors that make up a color scheme.
struct NodeColors {
  const char* style;
  const char* fill_color;
  const char* stroke_color;
  const char* font_color;
};

NodeColors NodeColorsForScheme(ColorScheme color) {
  switch (color) {
    case kBlue:
      return NodeColors{"filled", "#bbdefb", "#8aacc8", "black"};
    case kBrown:
      return NodeColors{"filled", "#bcaaa4", "#8c7b75", "black"};
    case kDarkBlue:
      return NodeColors{"filled", "#1565c0", "#003c8f", "white"};
    case kDarkGreen:
      return NodeColors{"filled", "#2e7d32", "#005005", "white"};
    case kDarkOrange:
      // This is more of a "medium" orange, made to look close to kOrange;
      // there's probably room for a darker weight if desired.
      return NodeColors{"filled", "#ffb74d", "#c88719", "black"};
    case kDarkRed:
      return NodeColors{"filled", "#b71c1c", "#7f0000", "white"};
    case kGray:
      return NodeColors{"filled", "#cfd8dc", "#9ea7aa", "black"};
    case kGreen:
      return NodeColors{"filled", "#c8e6c9", "#97b498", "black"};
    case kOrange:
      return NodeColors{"filled", "#ffe0b2", "#cbae82", "black"};
    case kPurple:
      return NodeColors{"filled", "#e1bee7", "#af8eb5", "black"};
    case kRed:
      return NodeColors{"filled", "#ffcdd2", "#cb9ca1", "black"};
    case kWhite:
      return NodeColors{"filled", "white", "black", "black"};
    case kYellow:
      return NodeColors{"filled", "#fff9c4", "#cbc693", "black"};
    case kDashedBorder:
      // "filled,dashed" looks the same as "dashed", since we have a white
      // background.  But we use "filled,dashed" so that when you hover over
      // any part of the node (not just the text inside the node), our css
      // :hover rule is triggered.
      return NodeColors{"filled,dashed", "white", "#757575", "#757575"};
  }
}

// Given a ColorScheme, returns an attribute string for a node of that color.
// Sets the node's style and fill/stroke/text colors.
//
// Colors are from https://material.io/color.
string NodeColorAttributes(ColorScheme color) {
  NodeColors node_colors = NodeColorsForScheme(color);

  return StrFormat(R"(style="%s", fontcolor="%s", color="%s", fillcolor="%s")",
                   node_colors.style, node_colors.font_color,
                   node_colors.stroke_color, node_colors.fill_color);
}

// Replaces <> with &lt;&gt;, so that this string is safe(er) for use in a
// graphviz HTML-like string.
string HtmlLikeStringSanitize(absl::string_view s) {
  return absl::StrReplaceAll(s, {{"<", "&lt;"}, {">", "&gt;"}});
}

bool IsFusedBroadcastOfConstantEffectiveScalar(const HloInstruction* instr) {
  namespace m = match;
  return instr->parent()->IsFusionComputation() &&
         Match(instr, m::Broadcast(m::ConstantEffectiveScalar()));
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
// subcomputation which is almost always one of the above, and pattern matching
// it to a short string lets us tell the user what the subcomputation is without
// drawing it as a graph.
optional<string> MatchTrivialComputation(const HloComputation* computation) {
  namespace m = match;

  if (computation->instruction_count() != 3) {
    return nullopt;
  }
  HloInstruction* root = computation->root_instruction();
  const HloInstruction *param0, *param1;
  if (!Match(root, m::Op()
                       .WithNumOperands(2)
                       .WithShape(m::Shape().IsEffectiveScalar())
                       .WithBinaryOperandsAnyOrder(
                           m::Parameter(&param0, 0)
                               .WithShape(m::Shape().IsEffectiveScalar()),
                           m::Parameter(&param1, 1)
                               .WithShape(m::Shape().IsEffectiveScalar())))) {
    return nullopt;
  }

  // If the params are reversed (i.e. operand0 is param1 and operand1 is
  // param0), check that the operation being performed is commutative.
  if (root->operand(0) == param1) {
    CHECK_EQ(root->operand(1), param0);
    if (root->opcode() == HloOpcode()) {
      switch (root->comparison_direction()) {
        case ComparisonDirection::kLe:
        case ComparisonDirection::kGe:
        case ComparisonDirection::kGt:
        case ComparisonDirection::kLt:
          return nullopt;
        default:
          break;
      }
    }
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
    case HloOpcode::kCompare: {
      switch (root->comparison_direction()) {
        case ComparisonDirection::kLe:
          return "less-or-equal";
        case ComparisonDirection::kGe:
          return "greater-or-equal";
        case ComparisonDirection::kGt:
          return "greater-than";
        case ComparisonDirection::kLt:
          return "less-than";
        case ComparisonDirection::kEq:
          return "equal-to";
        case ComparisonDirection::kNe:
          return "not-equal-to";
      }
    }
    default:
      return nullopt;
  }
}

// Encapsulates logic for dumping an HLO module to DOT (i.e. graphviz syntax).
class HloDotDumper {
 public:
  HloDotDumper(const HloComputation* computation, absl::string_view label,
               const DebugOptions& debug_options, bool show_backend_config,
               const HloExecutionProfile* profile, NodeFilter filter)
      : computation_(computation),
        label_(label),
        debug_options_(debug_options),
        show_backend_config_(show_backend_config),
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

  bool ShouldShowSubcomputation(const HloComputation* subcomp);
  bool ShouldShowFusionSubcomputation(const HloInstruction* instr);

  // We omit some nodes from the graph, instead drawing them inlined into the
  // nodes that use them.
  bool ShouldMergeIntoUsers(const HloInstruction* instr) const;

  string DumpSubcomputation(const HloComputation* subcomp,
                            const HloInstruction* parent_instr);
  string DumpComputation(const HloComputation* comp);
  string DumpRootTag();
  string DumpInstruction(const HloInstruction* instr);
  ColorScheme GetInstructionColor(const HloInstruction* instr);
  string GetInstructionNodeShape(const HloInstruction* instr);
  string GetInstructionNodeLabel(const HloInstruction* instr);
  string GetInstructionNodeMetadata(const HloInstruction* instr);
  string GetInstructionNodeBackendConfig(const HloInstruction* instr);
  string GetInstructionNodeExtraInfo(const HloInstruction* instr);
  string GetInstructionNodeInlinedOperands(const HloInstruction* instr);
  void AddInstructionIncomingEdges(const HloInstruction* instr);

  // For most instructions, GetNodeForEdge(instr) returns instr.
  //
  // The exception is fusion nodes.  For these, we walk up the chain of nested
  // fusion nodes starting at instr until we reach a node that either (a) isn't
  // a fusion node, or (b) is a fusion node for which
  // ShouldShowFusionSubcomputation is false.
  //
  // We do this because fusion nodes are expanded inline -- if
  // ShouldShowFusionSubcomputation is true, the fusion node won't be present in
  // the graph.
  //
  // In general when you want to draw an edge from A to B, you should actually
  // draw an edge from GetNodeForEdge(A) to GetNodeForEdge(B).
  const HloInstruction* GetNodeForEdge(const HloInstruction* instr);

  // If instr has just one computation and it's trivial (e.g. "return param0 +
  // param1"), returns a string you can put into the node's body that names the
  // subcomputation, e.g. "Subcomputation: <b>add</b>".
  string GetInstructionTrivialComputationStr(const HloInstruction* instr);

  const HloComputation* computation_;  // never null
  const string label_;                 // overall name for the graph
  const DebugOptions& debug_options_;
  const bool show_backend_config_;
  const HloExecutionProfile* profile_;  // may be null
  const NodeFilter filter_;

  // Each HloInstruction dumped gets a monotonically-increasing node ID.  This
  // must start at 1, because that's where graphviz's accounting starts.
  int64 next_node_id_ = 1;
  absl::flat_hash_map<const HloInstruction*, int64> node_ids_;

  // The "root" tag doesn't have an associated HloInstruction pointer, so we
  // need to store it outside the map.
  int64 root_node_id_;

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
  absl::flat_hash_map<const HloComputation*, int64> cluster_ids_;

  // Edges to print from Footer().  Edges come at the end because graphviz is
  // unhappy if an edge from a subcomputation to a node in the outer computation
  // appears before both the inner computation and the destination node are
  // defined.
  std::vector<string> edges_;

  // When coloring by sharding information, we track the sharding string
  // representation to color association, by round-robin the color schemes.
  absl::flat_hash_map<HloSharding, ColorScheme, HloSharding::Hasher>
      sharding_colors_;
  int64 next_shard_color_ = 0;
};

string HloDotDumper::Dump() {
  string body;
  StrAppend(&body, DumpComputation(computation_));
  StrAppend(&body, DumpRootTag());

  // By contract, Header() and Footer() have to be called after we've dumped all
  // our instructions, because they use state generated during that process.
  string g = Header();
  StrAppend(&g, body);
  StrAppend(&g, Footer());
  return g;
}

string HloDotDumper::Header() {
  constexpr char fmt[] = R"(digraph G {
rankdir = TB;
compound = true;
label = <<b>%s</b>>;
labelloc = t;
// Disable the tooltip.  Interestingly, "" doesn't work!
tooltip = " ";
// DOT graphs accept a stylesheet as a URI.  So naturally, an inline
// stylesheet is a data URI!
stylesheet=<
  data:text/css,
  @import url(https://fonts.googleapis.com/css?family=Roboto:400,700);
  svg text {
    font-family: 'Roboto';
    font-size: 12px;
  }

  %s
>

)";

  VLOG(3) << "Generating Header";

  string graph_label =
      StrCat(label_, "<br/>Computation ", computation_->name());
  if (computation_->IsFusionComputation()) {
    StrAppend(&graph_label, " (in fusion instruction ",
              computation_->FusionInstruction()->name(), ")");
  }
  if (profile_ != nullptr) {
    auto cycles = profile_->total_cycles_executed(*computation_);
    absl::StrAppendFormat(&graph_label, "<br/>total cycles = %d (%s)", cycles,
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
      // One could imagine other ways of writing this CSS rule that involve
      // less duplication, but this way seems to be relatively performant.
      edge_css_rules.push_back(
          StrFormat("  #%s%d:hover ~ #edge%d text { fill: %s; }\n"
                    "  #%s%d:hover ~ #edge%d path { "
                    "stroke: %s; stroke-width: .2em; }\n"
                    "  #%s%d:hover ~ #edge%d polygon { "
                    "fill: %s; stroke: %s; stroke-width: .2em; }\n",
                    elem_type, elem_id, edge_id, color,  //
                    elem_type, elem_id, edge_id, color,  //
                    elem_type, elem_id, edge_id, color, color));
    };

    // The "to_node" value may be a NULL, indicating that this points to the
    // "root" tag rather than a normal node.
    int64 from_node_id =
        tensorflow::gtl::FindWithDefault(node_ids_, from_node, -1);
    if (from_node_id == -1) {
      LOG(FATAL) << from_node->name() << " was added to edges but not to nodes";
    }
    int64 to_node_id =
        to_node ? tensorflow::gtl::FindWithDefault(node_ids_, to_node, -1)
                : root_node_id_;
    if (to_node != nullptr && to_node_id == -1) {
      LOG(FATAL) << to_node->name() << " was added to edges but not to nodes";
    }

    add_hover_css_rule("node", from_node_id, kBlue);
    add_hover_css_rule("node", to_node_id, kRed);

    if (to_node) {
      VLOG(3) << "Adding css for edge " << edge_id << " from node "
              << from_node->name() << " to node " << to_node->name();
    } else {
      VLOG(3) << "Adding css for edge " << edge_id << " from node "
              << from_node->name() << " to root tag";
    }

    // If this edge crosses a fusion cluster boundary, highlight it when the
    // cluster is hovered over.
    if (to_node) {
      if (from_node->IsFused() &&
          from_node->parent()->root_instruction() == from_node) {
        int64 cluster_id = cluster_ids_.at(from_node->parent());
        add_hover_css_rule("clust", cluster_id, kBlue);
      }
      if (to_node->IsFused() && to_node->opcode() == HloOpcode::kParameter) {
        int64 cluster_id = cluster_ids_.at(to_node->parent());
        add_hover_css_rule("clust", cluster_id, kRed);
      }
    }
  }

  // Browsers require that we URI-encode the contents of our data URI.  (It
  // seems this was a relatively recent change?) In practice, this means that we
  // need to escape '#'.
  return StrFormat(
      fmt, graph_label,
      absl::StrReplaceAll(StrJoin(edge_css_rules, "\n"), {{"#", "%23"}}));
}

string HloDotDumper::Footer() { return StrCat(StrJoin(edges_, "\n"), "\n}"); }

bool HloDotDumper::ShouldShowFusionSubcomputation(const HloInstruction* instr) {
  CHECK_EQ(instr->opcode(), HloOpcode::kFusion);
  return ShouldShowSubcomputation(instr->fused_instructions_computation());
}

bool HloDotDumper::ShouldShowSubcomputation(const HloComputation* subcomp) {
  if (subcomp->IsFusionComputation()) {
    const HloInstruction* fusion = subcomp->FusionInstruction();
    if (!filter_.Show(fusion) || filter_.SomeOrAllOperandsOmitted(fusion)) {
      return false;
    }
  }

  // Don't show trivial subcomputations on non-fusion nodes -- these are inlined
  // into the graph.
  if (!subcomp->IsFusionComputation() && MatchTrivialComputation(subcomp)) {
    return false;
  }

  // Show the subcomputation if we're showing any of its members.
  return absl::c_any_of(
      subcomp->instructions(),
      [&](const HloInstruction* instr) { return filter_.Show(instr); });
}

string HloDotDumper::DumpSubcomputation(const HloComputation* subcomp,
                                        const HloInstruction* parent_instr) {
  VLOG(2) << "Dumping subcomputation " << subcomp->name();
  // Add an edge from the subcomputation to its parent node.  If subcomp
  // belongs to a fusion node, it's drawn in place of the fusion instruction,
  // so there's no need to link those.
  if (parent_instr->opcode() != HloOpcode::kFusion) {
    const HloInstruction* from = GetNodeForEdge(subcomp->root_instruction());
    VLOG(2) << "Edge: from " << from->name() << " to " << parent_instr->name()
            << " as " << next_edge_id_;
    edge_ids_.insert({{from, parent_instr}, next_edge_id_++});
    constexpr char edge_fmt[] =
        R"(%s -> %s [ltail="%s", style="dashed" tooltip="%s -> %s"];)";
    edges_.push_back(StrFormat(
        edge_fmt, InstructionId(from), InstructionId(parent_instr),
        SubcomputationId(subcomp), subcomp->name(), parent_instr->name()));
  }

  // Have we already dumped this subcomputation?  If so, generating the edge
  // linking it and parent_instr is all we want to do in this function.
  if (cluster_ids_.find(subcomp) != cluster_ids_.end()) {
    return "";
  }

  cluster_ids_[subcomp] = next_cluster_id_++;

  string id = SubcomputationId(subcomp);

  string subcomp_label, style;
  if (parent_instr->opcode() == HloOpcode::kFusion) {
    subcomp_label =
        StrFormat("Fused expression for <b>%s</b><br/>%s",
                  HtmlLikeStringSanitize(parent_instr->name()),
                  HtmlLikeStringSanitize(parent_instr->ToCategory()));
    string extra_info = GetInstructionNodeExtraInfo(parent_instr);
    if (!extra_info.empty()) {
      StrAppend(&subcomp_label, "<br/>", extra_info);
    }
    string node_backend_config = GetInstructionNodeBackendConfig(parent_instr);
    if (!node_backend_config.empty()) {
      StrAppend(&subcomp_label, "<br/>", node_backend_config);
    }

    bool highlight = filter_.Highlight(parent_instr);
    const char* fillcolor;
    const char* strokecolor;
    if (debug_options_.xla_hlo_graph_sharding_color() && !highlight) {
      // Use the sharding color, if the node isn't highlighted.
      NodeColors node_colors =
          NodeColorsForScheme(GetInstructionColor(parent_instr));
      fillcolor = node_colors.fill_color;
      strokecolor = node_colors.stroke_color;
    } else {
      // Subcomputation's fill/stroke color is light/dark red/gray, depending on
      // whether or not the subcomputation's fusion node is highlighted.
      fillcolor = highlight ? "#ffcdd2" : "#f5f5f5";
      strokecolor = highlight ? "#b71c1c" : "#c2c2c2";
    }
    style =
        StrFormat(R"(style="rounded,filled,bold"; fillcolor="%s"; color="%s;")",
                  fillcolor, strokecolor);
  } else {
    subcomp_label = StrFormat("Subcomputation for <b>%s</b><br/>%s",
                              HtmlLikeStringSanitize(parent_instr->name()),
                              HtmlLikeStringSanitize(subcomp->name()));
    style = "style=rounded; color=black;";
  }

  string comp_body = DumpComputation(subcomp);

  constexpr char computation_fmt[] = R"(subgraph %s {
%s
label = <%s>;
labelloc = t;
tooltip = " ";
%s
}  // %s

)";
  return StrFormat(computation_fmt, id, style, subcomp_label, comp_body, id);
}

string HloDotDumper::DumpComputation(const HloComputation* comp) {
  string g;
  for (const auto* instr : comp->instructions()) {
    if (!filter_.Show(instr)) {
      continue;
    }

    // Dump subcomputations within instr.
    for (const HloComputation* subcomp : instr->called_computations()) {
      if (ShouldShowSubcomputation(subcomp)) {
        StrAppend(&g, DumpSubcomputation(subcomp, instr));
      }
    }

    StrAppend(&g, DumpInstruction(instr));
  }
  return g;
}

string HloDotDumper::DumpRootTag() {
  const HloInstruction* from = GetNodeForEdge(computation_->root_instruction());

  // We didn't display constants or broadcasts of effective scalars within
  // fusions as separate nodes; so if the root is a constant/broadcast of
  // scalar, we don't add root tag or edge for it.
  if (!filter_.Show(from) || from->opcode() == HloOpcode::kConstant ||
      IsFusedBroadcastOfConstantEffectiveScalar(from)) {
    return "";
  }

  auto from_id = InstructionId(from);

  // The ID of the root computation is otherwise unused, so it makes a good ID
  // to use for the root-tag node.  However, the edge_ids_ map requires a
  // HloInstruction* pointer for the 'to' value, so we use a NULL value there
  // (rather than a pointer type-cast) to make it obvious if it is erroneously
  // dereferenced.
  HloInstruction* to = nullptr;
  auto to_id = SubcomputationId(computation_);

  string node_body = "ROOT";
  string node_shape = "circle";
  ColorScheme color = kBrown;

  VLOG(2) << "Adding root tag as node " << next_node_id_;
  root_node_id_ = next_node_id_++;

  VLOG(2) << "Adding edge from " << from->name() << " to root tag as "
          << next_edge_id_;
  edge_ids_.insert({{from, to}, next_edge_id_++});
  edges_.push_back(StrFormat(R"(%s -> %s [tooltip=" "];)", from_id, to_id));

  return StrFormat(R"(%s [label=<%s>, shape=%s, tooltip=" ", %s];)"
                   "\n",
                   to_id, node_body, node_shape, NodeColorAttributes(color));
}

static const HloConstantInstruction* TryGetFusionParameterConstant(
    const HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kParameter || !instr->IsFused()) {
    return nullptr;
  }
  const HloInstruction* fusion = instr->parent()->FusionInstruction();
  const HloInstruction* operand = fusion->operand(instr->parameter_number());
  return DynCast<HloConstantInstruction>(operand);
}

bool HloDotDumper::ShouldMergeIntoUsers(const HloInstruction* instr) const {
  // If a node:
  //
  //  - is a parameter of a fusion node which is bound to a constant,
  //
  // or
  //
  //  - is a tuple-shaped parameter, and
  //  - is not a parameter to a fusion node, and
  //  - has at least kMinUsersToOmit users shown, and
  //  - all of the shown users are get-tuple-elements,
  //
  // then we omit it from the graph, merging it with its users.
  //
  // This helps us handle the common case where a while loop body has one big
  // tuple-shaped parameter.
  if (TryGetFusionParameterConstant(instr) != nullptr) {
    return true;
  }
  const int kMinUsersToOmit = 3;
  return instr->opcode() == HloOpcode::kParameter && instr->shape().IsTuple() &&
         !instr->IsFused() &&
         absl::c_count_if(instr->users(),
                          [&](const HloInstruction* user) {
                            return filter_.Show(user);
                          }) > kMinUsersToOmit &&
         absl::c_all_of(instr->users(), [&](const HloInstruction* user) {
           return !filter_.Show(user) ||
                  user->opcode() == HloOpcode::kGetTupleElement;
         });
}

string HloDotDumper::DumpInstruction(const HloInstruction* instr) {
  // We don't display constants or broadcasts of effective scalar constants
  // within fusions as separate nodes; they're merged into their users.
  if (instr->opcode() == HloOpcode::kConstant ||
      IsFusedBroadcastOfConstantEffectiveScalar(instr)) {
    return "";
  }
  // Skip this node if it's merged into its users.
  if (ShouldMergeIntoUsers(instr)) {
    return "";
  }
  // Omit the fusion node if its subcomputation is drawn, since the
  // subcomputation will be drawn inline.
  if (instr->opcode() == HloOpcode::kFusion &&
      ShouldShowFusionSubcomputation(instr)) {
    return "";
  }

  VLOG(2) << "Adding node " << instr->name() << " as " << next_node_id_;
  node_ids_[instr] = next_node_id_++;

  ColorScheme color = GetInstructionColor(instr);
  string node_shape = GetInstructionNodeShape(instr);
  string node_label = GetInstructionNodeLabel(instr);
  string node_metadata = GetInstructionNodeMetadata(instr);
  string node_backend_config = GetInstructionNodeBackendConfig(instr);
  string extra_info = GetInstructionNodeExtraInfo(instr);
  string inlined_constants = GetInstructionNodeInlinedOperands(instr);
  string trivial_subcomputation = GetInstructionTrivialComputationStr(instr);
  AddInstructionIncomingEdges(instr);

  if (!debug_options_.xla_hlo_graph_sharding_color()) {
    // Override the node's styling if it should be (de-)emphasized.
    if (filter_.Deemphasized(instr)) {
      color = kDashedBorder;
    }
    if (filter_.Highlight(instr)) {
      node_shape = "diamond";
      color = kDarkRed;
    }
  }
  // Build the text that will be displayed inside the node.
  string node_body = node_label;
  for (const string& s : {trivial_subcomputation, node_backend_config,
                          extra_info, inlined_constants}) {
    if (!s.empty()) {
      StrAppend(&node_body, "<br/>", s);
    }
  }

  return StrFormat(R"(%s [label=<%s>, shape=%s, tooltip="%s", %s];)"
                   "\n",
                   InstructionId(instr), node_body, node_shape, node_metadata,
                   NodeColorAttributes(color));
}

string HloDotDumper::GetInstructionNodeInlinedOperands(
    const HloInstruction* instr) {
  // The constant's shape is a parameter because, in the case of a broadcasted
  // scalar constant, we want to show the broadcasted shape, not the constant's
  // scalar shape.
  auto stringify_constant = [](const HloConstantInstruction* constant,
                               const Shape& shape) {
    // If the shape has a dimension of size zero, print it as e.g.
    // "{} (f32[42, 0, 10])".  The alternative, calling Literal::ToString(),
    // enumerates all of its empty dimensions (e.g.  "{ { {}, {} }, ..."), which
    // is just noise.
    if (ShapeUtil::IsZeroElementArray(shape)) {
      return StrFormat("{} (%s)", ShapeUtil::HumanString(constant->shape()));
    }

    // Print the literal value of constants with <= K elements.  Note that we
    // use `constant->shape()` rather than `shape`, because if `constant` is a
    // scalar that's broadcasted into `shape`, we want to print the constant.
    optional<int64> elem_count;
    if (shape.IsArray()) {
      elem_count = ShapeUtil::ElementsIn(constant->shape());
    }
    // Allow HloDotDumper to print HloInstruction reconstructed from HloProto
    // collected from profiling tools. Those constants may not have a valid
    // literal.
    if (elem_count.has_value() && *elem_count <= 8 && constant->HasLiteral()) {
      return StrFormat("%s %s", shape.ToString(),
                       constant->literal().ToStringWithoutShape());
    }

    // Otherwise, print e.g. "%constant.42 (s32[100])".
    string constant_name;
    if (absl::StartsWith(constant->name(), "constant")) {
      constant_name = constant->name();
    } else {
      constant_name = StrCat("constant ", constant->name());
    }
    return StrFormat("%s %s", constant_name, ShapeUtil::HumanString(shape));
  };

  std::vector<string> lines;
  for (int64 i = 0; i < instr->operand_count(); ++i) {
    const HloInstruction* operand = instr->operand(i);
    optional<string> operand_str;
    if (const auto* constant_operand =
            DynCast<HloConstantInstruction>(operand)) {
      operand_str =
          stringify_constant(constant_operand, constant_operand->shape());
    } else if (IsFusedBroadcastOfConstantEffectiveScalar(operand)) {
      operand_str = stringify_constant(
          Cast<HloConstantInstruction>(operand->operand(0)), operand->shape());
    } else if (ShouldMergeIntoUsers(operand)) {
      // Special case: If the operand is a parameter to a fusion node and it
      // always has a constant value, display it like a regular constant.
      //
      // For other parameters, use the parameter number rather than the proper
      // name, because that's generally how people think of the node.
      if (operand->opcode() == HloOpcode::kParameter) {
        if (const HloConstantInstruction* constant =
                TryGetFusionParameterConstant(operand)) {
          operand_str = stringify_constant(constant, constant->shape());
        } else {
          operand_str = StrFormat("Parameter %d", operand->parameter_number());
        }
      } else {
        operand_str = operand->name();
      }
    }

    if (operand_str) {
      if (instr->operand_count() > 1) {
        lines.push_back(StrFormat("<b>operand %d</b> = %s", i, *operand_str));
      } else {
        lines.push_back(StrFormat("<b>operand</b> = %s", *operand_str));
      }
    }
  }
  return StrJoin(lines, "<br/>");
}

ColorScheme HloDotDumper::GetInstructionColor(const HloInstruction* instr) {
  if (debug_options_.xla_hlo_graph_sharding_color()) {
    if (!instr->has_sharding()) {
      return kDashedBorder;
    }
    auto it = sharding_colors_.find(instr->sharding());
    if (it != sharding_colors_.end()) {
      return it->second;
    }
    ColorScheme color = static_cast<ColorScheme>(
        kBlue + (next_shard_color_++ % (kDashedBorder - kBlue)));
    sharding_colors_.emplace(instr->sharding(), color);
    return color;
  }

  // Choose different weights of orange for small vs large parameters.  This
  // distinction is often important, especially in fusion nodes.
  auto parameter_color = IsSmall(instr) ? kOrange : kDarkOrange;

  // Special case: If this instruction has a parameter merged into it, paint it
  // the same color as a parameter.  Unless the merged-in parameter is a
  // parameter to a fusion node that is bound to a constant -- these aren't
  // "real" parameters from the user's perspective.
  if (absl::c_any_of(instr->operands(), [&](const HloInstruction* operand) {
        return operand->opcode() == HloOpcode::kParameter &&
               ShouldMergeIntoUsers(operand) &&
               TryGetFusionParameterConstant(operand) == nullptr;
      })) {
    return parameter_color;
  }

  // Pick different colors or shapes for instructions which are particularly
  // expensive (eg, dot) and those which are unusual in some way or unique
  // (eg, parameter).
  switch (instr->opcode()) {
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kAtan2:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kDivide:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIota:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kPower:
    case HloOpcode::kReal:
    case HloOpcode::kRemainder:
    case HloOpcode::kRng:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSlice:
    case HloOpcode::kSort:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kTanh:
      // De-emphasize scalar-shaped elementwise ops -- they're generally
      // uninteresting.
      if (ShapeUtil::IsEffectiveScalar(instr->shape())) {
        return kWhite;
      }
      return kYellow;
    case HloOpcode::kBitcast:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kTrace:
    case HloOpcode::kAfterAll:
    case HloOpcode::kAddDependency:
    case HloOpcode::kTuple:
      return kWhite;
    case HloOpcode::kBroadcast:
      // De-emphasize nodes which broadcast a scalar within a fusion node --
      // these are essentially free.
      if (instr->IsFused() &&
          ShapeUtil::IsEffectiveScalar(instr->operand(0)->shape())) {
        return kWhite;
      }
      return kGreen;
    case HloOpcode::kConcatenate:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kGather:
    case HloOpcode::kPad:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kTupleSelect:
    case HloOpcode::kTranspose:
      // De-emphasize scalar-shaped data movement ops and all data movement ops
      // inside fusion nodes, both of which are essentially free.
      if (ShapeUtil::IsEffectiveScalar(instr->shape()) || instr->IsFused()) {
        return kWhite;
      }
      return kGreen;
    case HloOpcode::kDynamicUpdateSlice:
      // Unlike the data-movement ops above, dynamic-update-slice is not ~free
      // inside of fusion nodes, so we de-emphasize it only if it's
      // scalar-shaped.
      if (ShapeUtil::IsEffectiveScalar(instr->shape())) {
        return kWhite;
      }
      return kGreen;
    case HloOpcode::kScatter:
      // Do not de-emphasize Scatter, since it involves significant work.
    case HloOpcode::kCopy:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCopyDone:
      // Emphasize copy nodes, which are either physical transposes (and thus
      // significant), or copies of read-only buffers (and thus dead weight).
      return kGreen;
    case HloOpcode::kConvolution:
    case HloOpcode::kDot:
    case HloOpcode::kFft:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kCholesky:
      return kDarkBlue;
    case HloOpcode::kReducePrecision:
      return kRed;
    case HloOpcode::kParameter:
      return parameter_color;
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSelectAndScatter:
      return kPurple;
    case HloOpcode::kDomain:
    case HloOpcode::kFusion:
    case HloOpcode::kMap:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
      return kGray;
    case HloOpcode::kAllGather:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kPartitionId:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kReplicaId:
      return kBrown;
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kCustomCall:
    case HloOpcode::kWhile:
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
    return StrFormat("<b>Parameter %d</b>", instr->parameter_number());
  }

  // The HLO instruction name contains usually the opcode, e.g. "%add.42" is
  // an add instruction.  In this case we render just the name.
  if (absl::StartsWith(instr->name(), HloOpcodeString(instr->opcode()))) {
    return StrFormat("<b>%s</b>", HtmlLikeStringSanitize(instr->name()));
  }
  string extended_opcode =
      StrCat(HloOpcodeString(instr->opcode()),
             instr->opcode() != HloOpcode::kFusion
                 ? ""
                 : StrCat(":", xla::ToString(instr->fusion_kind())));
  // If the name does not contain the opcode, render both.
  return StrFormat("<b>%s</b><br/>%s", HtmlLikeStringSanitize(extended_opcode),
                   HtmlLikeStringSanitize(instr->name()));
}

string HloDotDumper::GetInstructionNodeMetadata(const HloInstruction* instr) {
  std::vector<string> lines;
  if (!instr->metadata().op_name().empty()) {
    lines.push_back(HtmlLikeStringSanitize(instr->metadata().op_name()));
  }
  if (!instr->metadata().op_type().empty()) {
    lines.push_back(StrFormat(
        "op_type: %s", HtmlLikeStringSanitize(instr->metadata().op_type())));
  }
  if (!instr->metadata().source_file().empty() &&
      instr->metadata().source_line() != 0) {
    lines.push_back(StrFormat("op_type: %s:%d", instr->metadata().source_file(),
                              instr->metadata().source_line()));
  }

  return StrJoin(lines, "\n");
}

string HloDotDumper::GetInstructionNodeBackendConfig(
    const HloInstruction* instr) {
  if (!show_backend_config_ || instr->raw_backend_config_string().empty()) {
    return "";
  }

  return StrCat("backend_config=\"", instr->raw_backend_config_string(), "\"");
}

string HloDotDumper::GetInstructionNodeExtraInfo(const HloInstruction* instr) {
  std::vector<string> lines;

  // Get the instruction's extra attributes excluding the names of its
  // subcomputations, since those are drawn explicitly in the graph.
  for (const auto& line : instr->ExtraAttributesToString(
           HloPrintOptions().set_print_subcomputation_mode(
               HloPrintOptions::PrintSubcomputationMode::kOff))) {
    lines.push_back(HtmlLikeStringSanitize(line));
  }

  // Show the shape and layout of the instruction, unless it's an inlined fusion
  // node -- there the shape and layout is present in the output node.
  if (instr->opcode() != HloOpcode::kFusion ||
      !ShouldShowFusionSubcomputation(instr)) {
    // Show layout of instructions with more than one dimension.  Don't show
    // layout on tuples or tensors with just one dimension (which only have one
    // possible layout) to avoid visual noise.
    bool shape_is_multidim = false;
    ShapeUtil::ForEachSubshape(instr->shape(),
                               [&](const Shape& s, const ShapeIndex&) {
                                 shape_is_multidim |= s.dimensions_size() > 1;
                               });
    string instr_shape;
    if (instr->opcode() != HloOpcode::kTuple && shape_is_multidim) {
      instr_shape = ShapeUtil::HumanStringWithLayout(instr->shape());
    } else {
      instr_shape = ShapeUtil::HumanString(instr->shape());
    }

    // Some instructions have giant tuples as their shapes, so truncate the
    // HLO's shape to kMaxShapeLen characters.
    constexpr int kMaxShapeLen = 64;
    if (instr_shape.length() > kMaxShapeLen) {
      instr_shape = StrCat(
          absl::string_view(instr_shape).substr(0, kMaxShapeLen - 3), "...");
    }
    lines.push_back(instr_shape);
  }
  if (debug_options_.xla_hlo_graph_addresses()) {
    lines.push_back(StrFormat("[%p]", instr));
  }
  if (profile_ != nullptr) {
    double hlo_cycles_executed = profile_->GetCyclesTakenBy(*instr);
    double total_cycles_executed =
        profile_->total_cycles_executed(*instr->parent());
    if (hlo_cycles_executed > 0 && total_cycles_executed > 0) {
      lines.push_back(
          StrFormat("%% of cycles executed=%.2f",
                    100 * hlo_cycles_executed / total_cycles_executed));
    }
  }
  return StrJoin(lines, "<br/>");
}

void HloDotDumper::AddInstructionIncomingEdges(const HloInstruction* instr) {
  auto add_edge = [&](const HloInstruction* from, const HloInstruction* to,
                      int64 operand_num, bool control_edge = false) {
    from = GetNodeForEdge(from);

    if (!filter_.Show(from) || from->opcode() == HloOpcode::kConstant ||
        IsFusedBroadcastOfConstantEffectiveScalar(from) ||
        ShouldMergeIntoUsers(from)) {
      return;
    }
    VLOG(2) << "Adding edge from " << from->name() << " to " << to->name()
            << " as " << next_edge_id_;
    edge_ids_.insert({{from, to}, next_edge_id_++});

    string edge_label;
    if (instr->operand_count() > 1 && !control_edge) {
      edge_label =
          StrFormat(R"( headlabel="%d", labeldistance=2)", operand_num);
    } else if (control_edge) {
      edge_label = "style=\"dotted\" color=\"gray\" label=\"ctrl\"";
    }

    // We print "small" arrays using a hollow arrowhead and "large" arrays using
    // a filled arrowhead.
    constexpr char kEdgeFmt[] =
        R"(%s -> %s [arrowhead=%s tooltip="%s -> %s" %s];)";
    edges_.push_back(StrFormat(kEdgeFmt, InstructionId(from), InstructionId(to),
                               (IsSmall(from) ? "empty" : "normal"),
                               from->name(), to->name(), edge_label));
  };

  // Add edges from instr's operands to instr.  Parameters within fusion
  // expressions are handled specially -- we draw an edge from the corresponding
  // operand on the fusion node itself to the parameter.
  if (instr->opcode() == HloOpcode::kParameter && instr->IsFused()) {
    // Only add the edge if this is not the outermost computation; otherwise it
    // will lead from a node we're not drawing.
    if (instr->parent() != computation_) {
      const HloInstruction* fusion = instr->parent()->FusionInstruction();
      add_edge(fusion->operand(instr->parameter_number()), instr,
               /*operand_num=*/0);
    }
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
      lines.push_back(StrFormat("Subcomputation: <b>%s</b>",
                                HtmlLikeStringSanitize(*computation_type)));
    } else {
      lines.push_back(StrFormat("Subcomputation %d: <b>%s</b>", i,
                                HtmlLikeStringSanitize(*computation_type)));
    }
  }
  return StrJoin(lines, "<br/>");
}

const HloInstruction* HloDotDumper::GetNodeForEdge(
    const HloInstruction* instr) {
  while (instr->opcode() == HloOpcode::kFusion &&
         ShouldShowFusionSubcomputation(instr)) {
    instr = instr->fused_expression_root();
  }
  return instr;
}

// Gets a NodeFilter that includes roughly all instructions whose distance from
// root is <= radius.
NodeFilter MakeNodeRadiusAroundFilter(
    const HloInstruction* root, int64 radius,
    const absl::flat_hash_set<const HloInstruction*>& boundary) {
  // First, find the neighborhood of nodes with distance from root <= radius.
  // These nodes are our initial set of "normal" nodes.
  absl::flat_hash_map<const HloInstruction*, NodeFilterResult> nodes;
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
    if (boundary.contains(instr)) {
      continue;
    }

    // Traverse into instr's operands.
    //
    // Don't traverse into tuples' operands unless the tuple is the root.
    // Usually a tuple is the bottommost node in the graph, and so its operands
    // are not interesting to the graph at hand.
    if (instr == root || instr->opcode() != HloOpcode::kTuple) {
      for (const HloInstruction* operand : instr->operands()) {
        if (!nodes.contains(operand)) {
          worklist.push_back({operand, depth + 1});
        }
      }
    }

    // Traverse into instr's nested computations.
    for (const HloComputation* computation : instr->called_computations()) {
      worklist.push_back({computation->root_instruction(), depth + 1});
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
      if (!nodes.contains(user)) {
        worklist.push_back({user, depth + 1});
      }
    }
  }

  auto is_displayed = [&](const HloInstruction* instr) {
    // Constants are displayed inline with their users; they're never omitted.
    // Nodes in subcomputations are always shown.
    return nodes.contains(instr) || instr->opcode() == HloOpcode::kConstant ||
           instr->parent() != root->parent();
  };

  // Make a second pass over 'nodes' to fix up the NodeFilterResults now that we
  // know which nodes will be included in the graph.
  for (auto& kv : nodes) {
    const HloInstruction* instr = kv.first;
    NodeFilterResult& filter_result = kv.second;
    const auto& operands = instr->operands();

    if (absl::c_any_of(operands, is_displayed) &&
        !absl::c_all_of(operands, is_displayed)) {
      // Mark nodes with some operands omitted appropriately.
      filter_result = kSomeOperandsOmitted;
    } else if (!operands.empty() && absl::c_none_of(operands, is_displayed)) {
      // Mark nodes with *all* operands omitted appropriately.
      filter_result = kOmitNodeOperands;
    }

    // Promote nodes with type kSomeUsersOmitted to kNormalNode if all of their
    // users made it into the graph.
    if (filter_result == kSomeUsersOmitted &&
        absl::c_all_of(instr->users(), is_displayed)) {
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

// Gets a node filter that includes nodes on all paths from `from` to `to`.  If
// the all-paths set contains more than max_nodes elements, includes the nodes
// on the shortest paths and sets hit_limit to true.
NodeFilter MakeNodeFromToFilter(const HloInstruction* from,
                                const HloInstruction* to, int64 max_nodes,
                                bool* hit_limit) {
  *hit_limit = false;

  // Elements in the queue are paths through the graph.
  std::deque<std::vector<const HloInstruction*>> queue;
  queue.push_front({from});

  // Compute the set of nodes we want to show using a slightly-modified
  // Djikstra's algorithm.  The only real difference is, rather than stopping
  // when we find a (shortest) path, we continue until we've found max_nodes
  // nodes on some path.
  std::unordered_set<const HloInstruction*> visited;
  std::unordered_set<const HloInstruction*> to_display = {from, to};
  while (!queue.empty() && to_display.size() < max_nodes) {
    std::vector<const HloInstruction*> path = std::move(queue.front());
    queue.pop_front();
    if (!visited.insert(path.back()).second) {
      continue;
    }

    for (const auto* user : path.back()->users()) {
      if (user == to) {
        auto it = path.begin();
        for (; it != path.end() && to_display.size() < max_nodes; ++it) {
          to_display.insert(*it);
        }
        if (it != path.end()) {
          *hit_limit = true;
        }
      } else if (!visited.count(user)) {
        auto new_path = path;
        new_path.push_back(user);
        queue.push_back(std::move(new_path));
      }
    }
  }

  return NodeFilter([=](const HloInstruction* instr) {
    if (instr == from || instr == to) {
      return kHighlightNode;
    }
    return to_display.count(instr) ? kNormalNode : kHideNode;
  });
}

string WrapDotInHtml(absl::string_view dot) {
  static const char html_prefix[] = R"html(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style type="text/css">
    body {
      height: 100vh;
      margin: 0;
    }
  </style>
</head>
<body>
  <!-- Integrity hash is generated by https://www.srihash.org/ -->
  <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.1/viz.js"
     integrity="sha384-aD1MJYb0WKIUT+CtwJp5LTuV3U4pLAS6B/nUxL7ECimC2pN9N8vjlMr/yQCAkzxE"
     crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.1/full.render.js"
     integrity="sha384-bAixY275aIpCj6Te19y0MILZ4V+VEC8CVFujFEH+Lf7W+4XYYeYLwW5IBI6yQmMT"
     crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.0/dist/svg-pan-zoom.min.js"
     integrity="sha384-3008WpYB2pOBvE7lwkrKf+qTmbTPGGPYxA9C1YVhvbPukns4ZFj7E98QPLkNW9dS"
     crossorigin="anonymous"></script>
  <div id="container" style="height:95vh; border:1px solid black; "></div>
  <script>
    var data = `
)html";

  static const char html_suffix[] = R"html(
`;
    var cssregex = new RegExp('stylesheet=<([^]*)\n>\n', 'gm');
    var results = cssregex.exec(data)
    // graphviz has problem dealing with large stylesheets.
    // https://github.com/tensorflow/tensorflow/issues/17220#issuecomment-369228492
    // In order to avoid the problem, remove the stylesheet from the dot and
    // insert it directly info the rendered SVG.
    var dot_data = data;
    var css_data = ''
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
        var panzoom = svgPanZoom(svg, {
            zoomEnabled: true,
            controlIconsEnabled: true,
        });
        document.getElementsByTagName("BODY")[0].onresize = function() {
            panzoom.resize();
            panzoom.fit();
            panzoom.center();
        };
        var render_end = performance.now();
        var render_note = document.createElement('div')
        render_note.innerHTML = 'Rendering took '
                                + (render_end - render_start).toFixed(2) + "ms."
        document.body.append(render_note);
    }
    var svg = document.getElementById('graph')
    if (svg == null) {
        // Need to render SVG first.
        var viz = new Viz();
        viz.renderSVGElement(dot_data)
            .then(function(svg){
                var container = document.getElementById('container')
                var style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
                var node = document.createTextNode(css_data);
                style.appendChild(node);
                svg.setAttribute('width', '100%');
                svg.setAttribute('height', '100%');
                svg.setAttribute('id', 'graph');
                svg.appendChild(style);
                container.appendChild(svg);
                add_controls(svg);
            })
    } else {
        // HTML already has rendered SVG embedded, so we just need to add
        // controls.
        add_controls(svg);
    }
  </script>
</body>
</html>
)html";

  return absl::StrCat(html_prefix, dot, html_suffix);
}

tensorflow::mutex url_renderer_mu(tensorflow::LINKER_INITIALIZED);
std::function<StatusOr<string>(absl::string_view)>* url_renderer
    TF_GUARDED_BY(url_renderer_mu) = nullptr;

// Precondition: url_renderer != nullptr.
//
// (We specify this as a precondition rather than checking it in here and
// returning an error because we want to fail quickly when there's no URL
// renderer available, and this function runs only after we've done all the work
// of producing dot for the graph.)
StatusOr<string> WrapDotInFormat(absl::string_view dot,
                                 RenderedGraphFormat format)
    TF_EXCLUSIVE_LOCKS_REQUIRED(url_renderer_mu) {
  switch (format) {
    case RenderedGraphFormat::kUrl:
      CHECK(url_renderer != nullptr)
          << "Should have checked url_renderer != null before calling.";
      return (*url_renderer)(dot);
    case RenderedGraphFormat::kHtml:
      return WrapDotInHtml(dot);
    case RenderedGraphFormat::kDot:
      return string(dot);
  }
}

}  // namespace

void RegisterGraphToURLRenderer(
    std::function<StatusOr<string>(absl::string_view)> renderer) {
  tensorflow::mutex_lock lock(url_renderer_mu);
  if (url_renderer != nullptr) {
    LOG(WARNING) << "Multiple calls to RegisterGraphToURLRenderer.  Last call "
                    "wins, but because order of initialization in C++ is "
                    "nondeterministic, this may not be what you want.";
  }
  delete url_renderer;
  url_renderer = new std::function<StatusOr<string>(absl::string_view)>(
      std::move(renderer));
}

StatusOr<string> RenderGraph(const HloComputation& computation,
                             absl::string_view label,
                             const DebugOptions& debug_options,
                             RenderedGraphFormat format,
                             const HloExecutionProfile* hlo_execution_profile,
                             bool show_backend_config) {
  tensorflow::mutex_lock lock(url_renderer_mu);
  if (format == RenderedGraphFormat::kUrl && url_renderer == nullptr) {
    return Unavailable("Can't render as URL; no URL renderer was registered.");
  }

  string rendered_dot =
      HloDotDumper(&computation, label, debug_options, show_backend_config,
                   hlo_execution_profile, NodeFilter())
          .Dump();
  return WrapDotInFormat(rendered_dot, format);
}

StatusOr<string> RenderNeighborhoodAround(
    const HloInstruction& node, int radius, RenderedGraphFormat format,
    bool show_backend_config,
    const absl::flat_hash_set<const HloInstruction*>& boundary) {
  tensorflow::mutex_lock lock(url_renderer_mu);
  if (format == RenderedGraphFormat::kUrl && url_renderer == nullptr) {
    return FailedPrecondition(
        "Can't render as URL; no URL renderer was registered.");
  }

  string label =
      StrCat("Neighborhood of ", radius, " nodes around ", node.name());
  string rendered_dot =
      HloDotDumper(node.parent(), label,
                   node.GetModule()->config().debug_options(),
                   show_backend_config, /*profile=*/nullptr,
                   MakeNodeRadiusAroundFilter(&node, radius, boundary))
          .Dump();
  return WrapDotInFormat(rendered_dot, format);
}

StatusOr<string> RenderAllPathsFromTo(const HloInstruction& from,
                                      const HloInstruction& to, int64 max_nodes,
                                      RenderedGraphFormat format,
                                      bool show_backend_config) {
  tensorflow::mutex_lock lock(url_renderer_mu);
  if (format == RenderedGraphFormat::kUrl && url_renderer == nullptr) {
    return FailedPrecondition(
        "Can't render as URL; no URL renderer was registered.");
  }

  CHECK_EQ(from.parent(), to.parent()) << "Nodes must be in same computation!";
  auto debug_options = from.GetModule()->config().debug_options();

  bool hit_limit = false;
  NodeFilter filter = MakeNodeFromToFilter(&from, &to, max_nodes, &hit_limit);
  string label;
  if (!hit_limit) {
    label = StrCat("All paths from ", from.name(), " to ", to.name());
  } else {
    label = StrCat(max_nodes, " nodes on the shortest paths from ", from.name(),
                   " to ", to.name(),
                   "<br/><br/>***SHOWING ONLY A SUBSET OF ALL PATHS BETWEEN "
                   "NODES***<br/><br/>");
  }
  string rendered_dot =
      HloDotDumper(from.parent(), label, debug_options, show_backend_config,
                   /*profile=*/nullptr, filter)
          .Dump();
  return WrapDotInFormat(rendered_dot, format);
}

}  // namespace xla
