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
#include "tensorflow/lite/toco/dump_graphviz.h"

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "re2/re2.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_graphviz_dump_options.h"
#include "tensorflow/lite/toco/toco_port.h"
#include "tensorflow/lite/toco/toco_types.h"
#include "tensorflow/lite/toco/tooling_util.h"

using toco::port::AppendF;
using toco::port::StringF;

namespace toco {
namespace {

// 'nslimit' is a graphviz (dot) parameter that limits the iterations during
// the layout phase. Omitting it allows infinite iterations, causing some
// complex graphs to never finish. A value of 125 produces good graphs
// while allowing complex graphs to finish.
constexpr char kGraphFmt[] = R"CODE(digraph Computegraph { tooltip = "/"
    nslimit=125 margin=36 ranksep = 2 labelloc="t" label=%s
)CODE";
// Note: tooltip's are only supported on SVGs in Chrome.
constexpr char kSubgraphFmt[] =
    R"CODE(    subgraph "cluster_%s" { style=rounded bgcolor="%s" penwidth=0.0 label=%s
)CODE";
constexpr char kArrayNodeFmt[] =
    R"CODE(        "%s" [label=%s tooltip="%s" shape=%s style=filled fillcolor="%s" fontcolor="%sDD"];
)CODE";
constexpr char kOpNodeFmt[] =
    R"CODE(        %s [label=%s tooltip=" " shape=box margin=0 style=filled fillcolor="%s" fontcolor="%sDD"];
)CODE";
constexpr char kInputEdgeFmt[] =
    R"CODE(        "%s"%s -> %s:i%d:n [penwidth=%f weight=%f];
)CODE";
constexpr char kOutputEdgeFmt[] =
    R"CODE(        %s:o%d:s -> "%s"%s [penwidth=%f weight=%f];
)CODE";
constexpr char kRNNBackEdgeFmt[] =
    R"CODE(        "%s":s -> "%s":n [color="#0F9D58" constraint=false];
)CODE";
constexpr char kUnicodeMult[] = "\u00D7";
constexpr char kUnicodeEllipsis[] = " \u2026 ";

class Color {
 public:
  Color() {}
  Color(uint8 r, uint8 g, uint8 b) : r_(r), g_(g), b_(b) {}
  explicit Color(uint32 word)
      : r_((word & 0x00FF0000) >> 16),
        g_((word & 0x0000FF00) >> 8),
        b_((word & 0x000000FF) >> 0) {}

  // Returns the string serialization of this color in graphviz format,
  // for use as 'fillcolor' in boxes.
  string AsHexString() const { return StringF("#%.2X%.2X%.2X", r_, g_, b_); }
  // The color to use for this node; will be used as 'fillcolor'
  // for its box. See Color::AsHexString. A suitable, different
  // color will be chosen for the 'fontcolor' for the inside text
  // label, see Color::TextColorString.
  // Returns the serialization in graphviz format of a suitable color to use
  // 'fontcolor' in the same boxes. It should black or white, whichever offers
  // the better contrast from AsHexString().
  string TextColorString() const {
    // https://en.wikipedia.org/wiki/Relative_luminance
    const float luminance = 0.2126f * r_ + 0.7152f * g_ + 0.0722f * b_;
    const uint8 l = luminance > 128.f ? 0 : 255;
    return StringF("#%.2X%.2X%.2X", l, l, l);
  }

 private:
  uint8 r_ = 0, g_ = 0, b_ = 0;
};

Color HashStringToColor(string s) {
  // Return a unique color for a name.
  //
  // This function removes Tensorflow anti-collision suffixes (eg "_2"), hashes
  // the string to a uint_32, then twiddles some bits to get a light and subtle
  // color. This seems to be a good heuristic for keeping enough of the name to
  // hash to a unique color while still revealing structure through naming
  // similarities.
  //
  // The regular expression "_\d+" matches any underscore followed by numbers,
  // which we strip out. Examples:
  //
  //     "Conv"      -> "Conv"
  //     "Conv_2"    -> "Conv"
  //     "Conv_72"   -> "Conv"
  //     "Pad_1_bias -> "Pad_bias"
  //     "Conv_abc"  -> "Conv_abc"

  RE2::GlobalReplace(&s, R"CODE(_\d+)CODE", "");
  uint32 color_word = std::hash<std::string>{}(s);
  color_word |= 0x00E0E0E0;
  return Color(color_word);
}

void GetArrayColorAndShape(const Model& model, const string& array_name,
                           Color* color, string* shape) {
  // All colors in this file are from:
  // https://material.io/guidelines/style/color.html
  // Arrays involved in RNN back-edges have a different color
  for (const auto& rnn_state : model.flags.rnn_states()) {
    // RNN state, fed by a back-edge. Bold color.
    if (array_name == rnn_state.state_array()) {
      *color = Color(0x0F, 0x9D, 0x58);
      *shape = "invhouse";
      return;
    }
    // RNN back-edge source, feeding a RNN state.
    // Light tone of the same color as RNN states.
    if (array_name == rnn_state.back_edge_source_array()) {
      *color = Color(0xB7, 0xE1, 0xCD);
      *shape = "house";
      return;
    }
  }
  // Constant parameter arrays have their own bold color
  if (model.GetArray(array_name).buffer) {
    *color = Color(0x42, 0x85, 0xF4);
    *shape = "cylinder";
    return;
  }
  // Remaining arrays are activations.
  // We use gray colors for them because they are the majority
  // of arrays so we want to highlight other arrays instead of them.
  // First, we use a bolder gray for input/output arrays:
  if (IsInputArray(model, array_name)) {
    *color = Color(0x9E, 0x9E, 0x9E);
    *shape = "invhouse";
    return;
  }
  if (IsOutputArray(model, array_name)) {
    *color = Color(0x9E, 0x9E, 0x9E);
    *shape = "house";
    return;
  }
  // Remaining arrays are intermediate activation arrays.
  // Lighter tone of the same grey as for input/output arrays:
  // We want these to be very discrete.
  *color = Color(0xF5, 0xF5, 0xF5);
  *shape = "box";
}

string GetArrayCompassPt(const Model& model, const string& array_name) {
  // The "compass point" is the point on the node where edge connections are
  // made. For most arrays we don't care, but input's and outputs look better
  // connected at the tip of the "house" and "invhouse" shapes used. So we
  // append ":n" and ":s" respectively for those.
  for (const auto& rnn_state : model.flags.rnn_states()) {
    // RNN state is essentially an input
    if (array_name == rnn_state.state_array()) {
      return ":s";
    }
    // RNN back-edge source is essentially an output
    if (array_name == rnn_state.back_edge_source_array()) {
      return ":n";
    }
  }
  if (IsInputArray(model, array_name)) {
    return ":s";
  }
  if (IsOutputArray(model, array_name)) {
    return ":n";
  }
  return "";
}

void AppendArrayVal(string* string, Array const& array, int index) {
  if (array.buffer->type == ArrayDataType::kFloat) {
    const auto& data = array.GetBuffer<ArrayDataType::kFloat>().data;
    if (index >= data.size()) {
      return;
    }
    AppendF(string, "%.3f", data[index]);
  } else if (array.buffer->type == ArrayDataType::kUint8) {
    const auto& data = array.GetBuffer<ArrayDataType::kUint8>().data;
    if (index >= data.size()) {
      return;
    }
    AppendF(string, "%d", data[index]);
  } else if (array.buffer->type == ArrayDataType::kInt16) {
    const auto& data = array.GetBuffer<ArrayDataType::kInt16>().data;
    if (index >= data.size()) {
      return;
    }
    AppendF(string, "%d", data[index]);
  } else if (array.buffer->type == ArrayDataType::kInt32) {
    const auto& data = array.GetBuffer<ArrayDataType::kInt32>().data;
    if (index >= data.size()) {
      return;
    }
    AppendF(string, "%d", data[index]);
  } else if (array.buffer->type == ArrayDataType::kInt64) {
    const auto& data = array.GetBuffer<ArrayDataType::kInt64>().data;
    if (index >= data.size()) {
      return;
    }
    AppendF(string, "%d", data[index]);
  } else if (array.buffer->type == ArrayDataType::kBool) {
    const auto& data = array.GetBuffer<ArrayDataType::kBool>().data;
    if (index >= data.size()) {
      return;
    }
    AppendF(string, "%d", data[index]);
  }
}

typedef std::map<string, string> Attributes;

string AttributesToHtml(Attributes attributes) {
  string html;
  for (const auto& attr : attributes) {
    html += R"CODE(<TR><TD CELLPADDING="1" ALIGN="RIGHT">)CODE";
    html += attr.first;
    html += R"CODE(:</TD><TD CELLPADDING="1" ALIGN="LEFT">)CODE";
    html += attr.second;
    html += "</TD></TR>";
  }
  return html;
}

string GetArrayLabel(const Model& model, const string& array_id) {
  string html;

  // Use HTML-like labels (http://www.graphviz.org/doc/info/shapes.html#html)
  html += "<";

  // Begin Table
  html += R"CODE(<FONT POINT-SIZE="10" FACE="Courier">)CODE";
  html += R"CODE(<TABLE BORDER="0" CELLSPACING="2" CELLPADDING="0">)CODE";

  auto& array = model.GetArray(array_id);
  if (array.buffer) {
    // "cylinder" shapes require some extra head room.
    html += R"CODE(<TR><TD COLSPAN="2"> </TD></TR>)CODE";
  }

  // "Primary" name of array (last non-slash delimited group of characters).
  html += R"CODE(<TR><TD COLSPAN="2" ALIGN="CENTER">)CODE";
  html += R"CODE(<FONT POINT-SIZE="16" FACE="Helvetica"><I>)CODE";
  AppendF(&html, R"CODE(%s)CODE",
          std::vector<string>(absl::StrSplit(array_id, '/')).back());
  html += R"CODE(</I></FONT>)CODE";
  html += "</TD></TR>";

  // Array data type and dimensions
  html += R"CODE(<TR><TD COLSPAN="2" ALIGN="CENTER">)CODE";
  html += R"CODE(<FONT POINT-SIZE="14" FACE="Courier"><B>)CODE";
  // Type
  html += ArrayDataTypeName(array.data_type);
  // Shape
  if (array.has_shape()) {
    auto& array_shape = array.shape();
    html += "[";
    for (int dim = 0; dim < array_shape.dimensions_count(); dim++) {
      AppendF(&html, "%d", array_shape.dims(dim));
      if (dim + 1 < array_shape.dimensions_count()) {
        html += kUnicodeMult;
      }
    }
    html += "]";
  }

  // Small buffer sample
  int buffer_size = 0;
  if (array.buffer) {
    buffer_size = RequiredBufferSizeForShape(array.shape());
  }
  if ((buffer_size > 0) && (buffer_size <= 4)) {
    html += " = ";
    if (array.shape().dimensions_count() > 0) {
      html += "{";
    }
    for (int i = 0; i < buffer_size; i++) {
      AppendArrayVal(&html, array, i);
      if (i + 1 < buffer_size) {
        html += ", ";
      }
    }
    if (array.shape().dimensions_count() > 0) {
      html += "}";
    }
  }
  html += R"CODE(</B></FONT>)CODE";
  html += "</TD></TR>";

  // Large buffer samples get their own line
  if (buffer_size > 4) {
    html += R"CODE(<TR><TD COLSPAN="2" ALIGN="CENTER"> = {)CODE";
    AppendArrayVal(&html, array, 0);
    html += ", ";
    AppendArrayVal(&html, array, 1);
    html += kUnicodeEllipsis;
    AppendArrayVal(&html, array, buffer_size - 2);
    html += ", ";
    AppendArrayVal(&html, array, buffer_size - 1);
    html += "}</TD></TR>";
  }

  // Other array properties
  Attributes attrs;
  if (array.minmax) {
    attrs["minmax"] =
        StringF("[%.7g, %.7g]", array.minmax->min, array.minmax->max);
  }
  if (array.quantization_params) {
    attrs["quant"] = StringF("%7g\u00B7(x-%d)",  // Unicode "cdot"
                             array.quantization_params->scale,
                             array.quantization_params->zero_point);
  }
  if (array.alloc) {
    attrs["alloc"] = StringF("[%d, %d)", array.alloc->start, array.alloc->end);
  }
  html += AttributesToHtml(attrs);

  // output array_id in ultra-small font so it can be searched and copied.
  html += R"CODE(<TR><TD COLSPAN="2" ALIGN="CENTER">)CODE";
  html += R"CODE(<FONT POINT-SIZE="3" FACE="">)CODE";
  AppendF(&html, R"CODE("%s")CODE", array_id);
  html += R"CODE(</FONT>)CODE";
  html += "</TD></TR>";

  // End Table and HTML-like label
  html += R"CODE(</TABLE></FONT>)CODE";
  html += ">";
  return html;
}

Attributes GetOpAttributes(const Model& model, const Operator& op) {
  Attributes attrs;
  switch (op.fused_activation_function) {
    case FusedActivationFunctionType::kRelu:
      attrs["func"] = "ReLU";
      break;
    case FusedActivationFunctionType::kRelu6:
      attrs["func"] = "ReLU6";
      break;
    case FusedActivationFunctionType::kRelu1:
      attrs["func"] = "ReLU1";
      break;
    default:
      break;
  }
  // Output state of member vars on derived operators.
  switch (op.type) {
    case OperatorType::kConv: {
      const auto& conv_op = static_cast<const ConvOperator&>(op);
      string stride;
      AppendF(&stride, "%d", conv_op.stride_width);
      stride += kUnicodeMult;
      AppendF(&stride, "%d", conv_op.stride_height);
      attrs["stride"] = stride;
      attrs["padding"] =
          (conv_op.padding.type == PaddingType::kSame) ? "same" : "valid";
      break;
    }
    case OperatorType::kDepthwiseConv: {
      const auto& depthconv_op = static_cast<const ConvOperator&>(op);
      string stride;
      AppendF(&stride, "%d", depthconv_op.stride_width);
      stride += kUnicodeMult;
      AppendF(&stride, "%d", depthconv_op.stride_height);
      attrs["stride"] = stride;
      attrs["padding"] =
          (depthconv_op.padding.type == PaddingType::kSame) ? "same" : "valid";
      break;
    }
    case OperatorType::kFakeQuant: {
      const auto& fakequant_op = static_cast<const FakeQuantOperator&>(op);
      attrs["bits"] = StringF("%d", fakequant_op.num_bits);
      if (fakequant_op.minmax) {
        attrs["range"] = StringF("[%g,%g]", fakequant_op.minmax->min,
                                 fakequant_op.minmax->max);
      } else {
        attrs["range"] = "[?,?]";
      }
      break;
    }
    default:
      break;
  }
  int64 math_ops_count;
  if (EstimateArithmeticOpsCount(model, op, &math_ops_count) &&
      (math_ops_count != 0)) {
    attrs["math"] = FormattedNumber(math_ops_count) + "ops";
  }

  return attrs;
}

Color GetOpColor(const Operator& op) {
  if ((op.type == OperatorType::kDepthwiseConv) ||
      (op.type == OperatorType::kConv) ||
      (op.type == OperatorType::kFullyConnected) ||
      (op.type == OperatorType::kFakeQuant)) {
    // Give some ops a bolder red
    return Color(0xC5, 0x39, 0x29);
  } else {
    return Color(0xDB, 0x44, 0x37);
  }
}

string GetOpLabel(const Model& model, const Operator& op) {
  // Use HTML-like labels (http://www.graphviz.org/doc/info/shapes.html#html)
  string html;
  html += "<";

  // Begin Table
  html += R"CODE(<FONT POINT-SIZE="10" FACE="Courier">)CODE";
  html +=
      R"CODE(<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">)CODE";

  // Input Ports
  if (!op.inputs.empty()) {
    html += R"CODE(<TR><TD COLSPAN="2" ALIGN="CENTER">)CODE";
    // Distribute evenly using a sub-table
    html += R"CODE(<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">)CODE";
    html += R"CODE(<TR>)CODE";
    for (int i = 0; i < op.inputs.size(); i++) {
      html += R"CODE(<TD PORT=")CODE";
      AppendF(&html, "i%d", i);
      html += R"CODE(">)CODE";
      if (op.inputs.size() > 1) {
        // Only number inputs when op has two or more inputs
        AppendF(&html, "%d", i);
      }
      html += "</TD>";
    }
    html += "</TR>";
    html += R"CODE(</TABLE></TD></TR>)CODE";
  }

  // Name
  html += R"CODE(<TR><TD COLSPAN="2" CELLPADDING="3" ALIGN="CENTER">)CODE";
  html += R"CODE(<FONT POINT-SIZE="16" FACE="Helvetica"><B>)CODE";
  if (op.type == OperatorType::kUnsupported) {
    html += static_cast<const TensorFlowUnsupportedOperator&>(op).tensorflow_op;
  } else {
    html += string(absl::StripPrefix(OperatorTypeName(op.type), "TensorFlow"));
  }
  html += R"CODE(</B></FONT>)CODE";
  html += "</TD></TR>";

  // Attributes
  Attributes attrs = GetOpAttributes(model, op);
  html += AttributesToHtml(attrs);

  // Output Ports
  if (!op.outputs.empty()) {
    html += R"CODE(<TR><TD COLSPAN="2" ALIGN="CENTER">)CODE";
    // Distribute evenly using a sub-table
    html += R"CODE(<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">)CODE";
    html += R"CODE(<TR>)CODE";
    for (int i = 0; i < op.outputs.size(); i++) {
      html += R"CODE(<TD PORT=")CODE";
      AppendF(&html, "o%d", i);
      html += R"CODE(">)CODE";
      if (op.outputs.size() > 1) {
        // Only number outputs when op has two or more outputs
        AppendF(&html, "%d", i);
      }
      html += "</TD>";
    }
    html += "</TR>";
    html += R"CODE(</TABLE></TD></TR>)CODE";
  }

  // End Table and HTML-like label
  html += R"CODE(</TABLE></FONT>)CODE";
  html += ">";

  return html;
}

float GetLog2BufferSize(const Model& model, const string& array_id) {
  auto& array = model.GetArray(array_id);
  if (array.has_shape()) {
    int buffer_size = 0;
    if (IsNonEmpty(array.shape())) {
      buffer_size = RequiredBufferSizeForShape(array.shape());
      return std::log2(static_cast<float>(buffer_size));
    }
  }
  return 0.0f;
}

string GetOpId(int op_index) { return StringF("op%05d", op_index); }

void DumpOperator(const Model& model, string* output_file, int op_index) {
  // Dump node for operator.
  const Operator& op = *model.operators[op_index];
  Color color = GetOpColor(op);
  string label = GetOpLabel(model, op);
  string op_id = GetOpId(op_index);
  AppendF(output_file, kOpNodeFmt, op_id, label, color.AsHexString(),
          color.TextColorString());
}

void DumpOperatorEdges(const Model& model, string* output_file, int op_index) {
  // Inputs
  const Operator& op = *model.operators[op_index];
  string op_id = GetOpId(op_index);
  for (int i = 0; i < op.inputs.size(); i++) {
    const auto& input = op.inputs[i];
    if (!model.HasArray(input)) {
      // Connected arrays should _always_ exist. Except, perhaps, during
      // development.
      continue;
    }
    float log2_buffer_size = GetLog2BufferSize(model, input);
    // Draw lines that transport more data thicker (Otherwise, where would the
    // data fit? right?).
    float line_width = std::max(0.5f, log2_buffer_size / 3.0f);
    // Keep edges that transport more data shorter than those with less.
    float weight = std::max(1.0f, log2_buffer_size);
    if (!IsInputArray(model, input) &&
        GetOpWithOutput(model, input) == nullptr) {
      // Give the main line of data flow a straighter path by penalizing edges
      // to standalone buffers. Weights are generally very large buffers that
      // would otherwise skew the layout.
      weight = 1.0f;
    }
    string compass_pt = GetArrayCompassPt(model, input);
    AppendF(output_file, kInputEdgeFmt, input, compass_pt, op_id, i, line_width,
            weight);
  }
  // Outputs
  for (int i = 0; i < op.outputs.size(); i++) {
    const auto& output = op.outputs[i];
    if (!model.HasArray(output)) {
      continue;
    }
    float log2_buffer_size = GetLog2BufferSize(model, output);
    // See comments above regarding weight and line_width calculations.
    float line_width = std::max(0.5f, log2_buffer_size / 3.0f);
    float weight = std::max(1.0f, log2_buffer_size);
    if (!IsArrayConsumed(model, output)) {
      weight = 1.0f;
    }
    string compass_pt = GetArrayCompassPt(model, output);
    AppendF(output_file, kOutputEdgeFmt, op_id, i, output, compass_pt,
            line_width, weight);
  }
}

struct Node {
  Node() : math_ops(0) {}
  // Name used as a key in the model's array map
  string array_id;

  // Estimated number of math ops incurred by this node (the sum of the op
  // with this array as 1st output, plus all children nodes).
  int64 math_ops;

  // A map of child nodes keyed by name.
  std::map<const string, std::unique_ptr<Node>> children;
};

string GetSubgraphLabel(Node const& node, const string& subgraph) {
  // Use HTML-like labels (http://www.graphviz.org/doc/info/shapes.html#html)
  string html;
  html += "<";

  // Begin Table
  html += R"CODE(<FONT POINT-SIZE="12" FACE="Courier">)CODE";
  html +=
      R"CODE(<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">)CODE";

  // Name
  html += R"CODE(<TR><TD COLSPAN="2" CELLPADDING="3" ALIGN="CENTER">)CODE";
  html += R"CODE(<FONT POINT-SIZE="18" FACE="Helvetica"><I>)CODE";
  html += subgraph;
  html += R"CODE(</I></FONT>)CODE";
  html += "</TD></TR>";

  // Attributes
  Attributes attrs;
  if (node.math_ops > 0) {
    attrs["math"] = FormattedNumber(node.math_ops) + "ops";
  }
  html += AttributesToHtml(attrs);

  // End Table and HTML-like label
  html += R"CODE(</TABLE></FONT>)CODE";
  html += ">";

  return html;
}

void DumpSubgraphHeader(string* output_file, Node const& node,
                        const string& node_name) {
  Color color = HashStringToColor(node_name);
  string label = GetSubgraphLabel(node, node_name);
  AppendF(output_file, kSubgraphFmt, node_name, color.AsHexString(), label);
}

void DumpArray(const Model& model, string* output_file,
               const string& array_id) {
  Color color;
  string shape;
  GetArrayColorAndShape(model, array_id, &color, &shape);
  string label = GetArrayLabel(model, array_id);
  AppendF(output_file, kArrayNodeFmt, array_id, label, array_id, shape,
          color.AsHexString(), color.TextColorString());

  // Ops are placed in the same subgraph as their first output.
  for (int op_index = 0; op_index < model.operators.size(); op_index++) {
    const Operator& op = *model.operators[op_index];
    if (!op.outputs.empty() && (op.outputs[0] == array_id)) {
      DumpOperator(model, output_file, op_index);
    }
  }
}

void DumpNode(const Model& model, string* output_file, const string& node_name,
              Node const& node) {
  bool not_root = !node_name.empty();
  if (not_root) {
    DumpSubgraphHeader(output_file, node, node_name);
  }

  for (const auto& child : node.children) {
    if (!child.second->array_id.empty()) {
      // Dump array if this node posesses one.
      DumpArray(model, output_file, child.second->array_id);
    }
    // Note that it is always possible to have children. Unlike a filesystem,
    // the existence of array "foo/bar" does _not_ prevent other arrays, such as
    // and "foo/bar/baz", from being nested beneath it.
    DumpNode(model, output_file, child.first, *child.second);
  }

  if (not_root) {
    // End subgraph
    AppendF(output_file, "    }\n");
  }
}

int64 GetArithmeticOpsCount(const Model& model, const string& array_id) {
  for (const auto& op : model.operators) {
    if (!op->outputs.empty() && op->outputs[0] == array_id) {
      int64 count;
      if (EstimateArithmeticOpsCount(model, *op, &count)) {
        return count;
      } else {
        return 0;
      }
    }
  }
  return 0;
}

void InsertNode(const Model& model, const string& array_id, Node* node,
                std::vector<string> prefixes, int64* math_ops) {
  if (prefixes.empty()) {
    // Base case: store array in this node.
    node->array_id = array_id;
    *math_ops = GetArithmeticOpsCount(model, array_id);
  } else {
    // Insert into the sub-tree for that prefix.
    string prefix = prefixes.back();
    prefixes.pop_back();
    if (node->children.count(prefix) == 0) {
      // Create a new node if this prefix is unseen.
      node->children[prefix] = absl::make_unique<Node>();
    }
    InsertNode(model, array_id, node->children[prefix].get(), prefixes,
               math_ops);
  }
  // Sum estimated math ops into all nodes.
  node->math_ops += *math_ops;
}

void BuildArrayTree(const Model& model, Node* tree) {
  // Delimit array names by path "/", then place into a tree based on this path.
  for (const auto& array_id : model.GetArrayMap()) {
    std::vector<string> prefixes = absl::StrSplit(array_id.first, '/');
    std::reverse(prefixes.begin(), prefixes.end());
    int64 math_ops;  // Temporary storage for math ops used during recursion.
    InsertNode(model, array_id.first, tree, prefixes, &math_ops);
  }
}

string GetGraphLabel(const Model& model, const string& graph_name) {
  // Use HTML-like labels (http://www.graphviz.org/doc/info/shapes.html#html)
  string html;
  html += "<";

  // Begin Table
  html += R"CODE(<FONT POINT-SIZE="36" FACE="Courier">)CODE";
  html +=
      R"CODE(<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">)CODE";

  // Name
  html += R"CODE(<TR><TD COLSPAN="2" CELLPADDING="3" ALIGN="CENTER">)CODE";
  html += R"CODE(<FONT POINT-SIZE="64" FACE="Helvetica"><B><I>)CODE";
  html += graph_name;
  html += R"CODE(</I></B></FONT>)CODE";
  html += "</TD></TR>";

  // Attributes
  Attributes attrs;
  attrs["arrays"] = StringF("%d", model.GetArrayMap().size());
  if (!model.optional_arrays.empty()) {
    attrs["optional arrays"] = StringF("%d", model.optional_arrays.size());
  }
  attrs["operators"] = StringF("%d", model.operators.size());
  int64 ops_count;
  if (EstimateArithmeticOpsCount(model, &ops_count) && (ops_count > 0)) {
    attrs["math"] = FormattedNumber(ops_count) + "ops";
  }
  if (model.transient_data_size > 0) {
    attrs["transient data size"] =
        StringF("%d KiB", model.transient_data_size / 1024);
  }
  if (model.transient_data_alignment > 0) {
    attrs["transient data alignment"] =
        StringF("%d bytes", model.transient_data_alignment);
  }
  html += AttributesToHtml(attrs);

  // End Table and HTML-like label
  html += R"CODE(</TABLE></FONT>)CODE";
  html += ">";

  return html;
}
}  // namespace

void DumpGraphviz(const Model& model, string* output_file,
                  const string& graph_name) {
  // Start graphviz format
  AppendF(output_file, kGraphFmt, GetGraphLabel(model, graph_name));

  // Organize arrays into a tree for subgraphing
  Node tree;
  BuildArrayTree(model, &tree);
  DumpNode(model, output_file, "", tree);

  // Dump edges outside all subgraphs (otherwise the referred-to nodes are
  // implicitly included in that subgraph).
  for (int op_index = 0; op_index < model.operators.size(); op_index++) {
    DumpOperatorEdges(model, output_file, op_index);
  }

  // Dump RNN Backedges
  for (const auto& rnn_state : model.flags.rnn_states()) {
    AppendF(output_file, kRNNBackEdgeFmt, rnn_state.back_edge_source_array(),
            rnn_state.state_array());
  }
  // End graphviz format
  AppendF(output_file, "}\n");
}
}  // namespace toco
