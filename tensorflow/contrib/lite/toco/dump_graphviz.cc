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
#include "tensorflow/contrib/lite/toco/dump_graphviz.h"

#include <cmath>
#include <memory>
#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/strings/strip.h"
#include "tensorflow/contrib/lite/toco/model_flags.pb.h"
#include "tensorflow/contrib/lite/toco/toco_graphviz_dump_options.h"
#include "tensorflow/contrib/lite/toco/toco_port.h"
#include "tensorflow/contrib/lite/toco/toco_types.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

using toco::port::AppendF;
using toco::port::StringF;

namespace toco {
namespace {

class Color {
 public:
  Color() {}
  Color(uint8 r, uint8 g, uint8 b) : r_(r), g_(g), b_(b) {}
  // Returns the string serialization of this color in graphviz format,
  // for use as 'fillcolor' in boxes.
  string FillColorString() const { return StringF("%.2X%.2X%.2X", r_, g_, b_); }
  // Returns the serialization in graphviz format of a suitable color to use
  // 'fontcolor' in the same boxes. It should black or white, whichever offers
  // the better contrast from FillColorString().
  string TextColorString() const {
    // https://en.wikipedia.org/wiki/Relative_luminance
    const float luminance = 0.2126f * r_ + 0.7152f * g_ + 0.0722f * b_;
    const uint8 l = luminance > 128.f ? 0 : 255;
    return StringF("%.2X%.2X%.2X", l, l, l);
  }

 private:
  uint8 r_ = 0, g_ = 0, b_ = 0;
};

struct NodeProperties {
  // The text to display inside the box for this node.
  string label;
  // The color to use for this node; will be used as 'fillcolor'
  // for its box. See Color::FillColorString. A suitable, different
  // color will be chosen for the 'fontcolor' for the inside text
  // label, see Color::TextColorString.
  Color color;
  float log2_buffer_size;
};

// All colors in this file are from:
// https://material.io/guidelines/style/color.html

Color GetColorForArray(const Model& model, const string& array_name) {
  // Arrays involved in RNN back-edges have a different color
  for (const auto& rnn_state : model.flags.rnn_states()) {
    // RNN state, fed by a back-edge. Bold color.
    if (array_name == rnn_state.state_array()) {
      return Color(0x0F, 0x9D, 0x58);
    }
    // RNN back-edge source, feeding a RNN state.
    // Light tone of the same color as RNN states.
    if (array_name == rnn_state.back_edge_source_array()) {
      return Color(0xB7, 0xE1, 0xCD);
    }
  }
  // Constant parameter arrays have their own bold color
  if (model.GetArray(array_name).buffer) {
    return Color(0x42, 0x85, 0xF4);
  }
  // Remaining arrays are activations.
  // We use gray colors for them because they are the majority
  // of arrays so we want to highlight other arrays instead of them.
  // First, we use a bolder gray for input/output arrays:
  if (IsInputArray(model, array_name)) {
    return Color(0x9E, 0x9E, 0x9E);
  }
  if (IsOutputArray(model, array_name)) {
    return Color(0x9E, 0x9E, 0x9E);
  }
  // Remaining arrays are intermediate activation arrays.
  // Lighter tone of the same grey as for input/output arrays:
  // We want these to be very discrete.
  return Color(0xF5, 0xF5, 0xF5);
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

NodeProperties GetPropertiesForArray(const Model& model,
                                     const string& array_name) {
  NodeProperties node_properties;
  node_properties.color = GetColorForArray(model, array_name);
  node_properties.label = absl::StrReplaceAll(array_name, {{"/", "/\\n"}});
  node_properties.log2_buffer_size = 0.0f;

  // Append array shape to the label.
  auto& array = model.GetArray(array_name);
  AppendF(&node_properties.label, "\\nType: %s",
          ArrayDataTypeName(array.data_type));

  if (array.has_shape()) {
    auto& array_shape = array.shape();
    node_properties.label += "\\n[";
    for (int id = 0; id < array_shape.dimensions_count(); id++) {
      if (id == 0) {
        AppendF(&node_properties.label, "%d", array_shape.dims(id));
      } else {
        // 0x00D7 is the unicode multiplication symbol
        AppendF(&node_properties.label, "\u00D7%d", array_shape.dims(id));
      }
    }
    node_properties.label += "]";

    int buffer_size = 0;
    if (IsValid(array.shape())) {
      buffer_size = RequiredBufferSizeForShape(array.shape());
      node_properties.log2_buffer_size =
          std::log2(static_cast<float>(buffer_size));
    }

    if (array.buffer) {
      const auto& array = model.GetArray(array_name);
      if (buffer_size <= 4) {
        AppendF(&node_properties.label, " = ");
        if (array.shape().dimensions_count() > 0) {
          AppendF(&node_properties.label, "{");
        }
        for (int i = 0; i < buffer_size; i++) {
          AppendArrayVal(&node_properties.label, array, i);
          if (i + 1 < buffer_size) {
            AppendF(&node_properties.label, ", ");
          }
        }
      } else {
        AppendF(&node_properties.label, "\\n = ");
        if (array.shape().dimensions_count() > 0) {
          AppendF(&node_properties.label, "{");
        }
        AppendArrayVal(&node_properties.label, array, 0);
        AppendF(&node_properties.label, ", ");
        AppendArrayVal(&node_properties.label, array, 1);
        // 0x2026 is the unicode ellipsis symbol
        AppendF(&node_properties.label, " \u2026 ");
        AppendArrayVal(&node_properties.label, array, buffer_size - 2);
        AppendF(&node_properties.label, ", ");
        AppendArrayVal(&node_properties.label, array, buffer_size - 1);
      }
      if (array.shape().dimensions_count() > 0) {
        AppendF(&node_properties.label, "}");
      }
    }
  }

  if (array.minmax) {
    AppendF(&node_properties.label, "\\nMinMax: [%.7g, %.7g]",
            array.minmax->min, array.minmax->max);
  }

  if (array.quantization_params) {
    AppendF(&node_properties.label, "\\nQuantization: %7g * (x - %d)",
            array.quantization_params->scale,
            array.quantization_params->zero_point);
  }

  if (array.alloc) {
    AppendF(&node_properties.label, "\\nTransient Alloc: [%d, %d)",
            array.alloc->start, array.alloc->end);
  }

  return node_properties;
}

NodeProperties GetPropertiesForOperator(const Operator& op) {
  NodeProperties node_properties;
  if (op.type == OperatorType::kTensorFlowUnsupported) {
    node_properties.label =
        static_cast<const TensorFlowUnsupportedOperator&>(op).tensorflow_op;
  } else {
    node_properties.label =
        string(absl::StripPrefix(OperatorTypeName(op.type), "TensorFlow"));
  }
  switch (op.fused_activation_function) {
    case FusedActivationFunctionType::kRelu:
      AppendF(&node_properties.label, "\\nReLU");
      break;
    case FusedActivationFunctionType::kRelu6:
      AppendF(&node_properties.label, "\\nReLU6");
      break;
    case FusedActivationFunctionType::kRelu1:
      AppendF(&node_properties.label, "\\nReLU1");
      break;
    default:
      break;
  }
  // Additional information for some of the operators.
  switch (op.type) {
    case OperatorType::kConv: {
      const auto& conv_op = static_cast<const ConvOperator&>(op);
      node_properties.color = Color(0xC5, 0x39, 0x29);  // Bolder color
      AppendF(&node_properties.label, "\\n%dx%d/%s", conv_op.stride_width,
              conv_op.stride_height,
              conv_op.padding.type == PaddingType::kSame ? "S" : "V");
      break;
    }
    case OperatorType::kDepthwiseConv: {
      const auto& conv_op = static_cast<const DepthwiseConvOperator&>(op);
      node_properties.color = Color(0xC5, 0x39, 0x29);  // Bolder color
      AppendF(&node_properties.label, "\\n%dx%d/%s", conv_op.stride_width,
              conv_op.stride_height,
              conv_op.padding.type == PaddingType::kSame ? "S" : "V");
      break;
    }
    case OperatorType::kFullyConnected: {
      node_properties.color = Color(0xC5, 0x39, 0x29);  // Bolder color
      break;
    }
    case OperatorType::kFakeQuant: {
      const auto& fakequant_op = static_cast<const FakeQuantOperator&>(op);
      node_properties.color = Color(0xC5, 0x39, 0x29);  // Bolder color
      if (fakequant_op.minmax) {
        AppendF(&node_properties.label, "\\n%dbit [%g,%g]",
                fakequant_op.num_bits, fakequant_op.minmax->min,
                fakequant_op.minmax->max);
      } else {
        AppendF(&node_properties.label, "\\n%dbit [?,?]",
                fakequant_op.num_bits);
      }
      break;
    }
    default:
      node_properties.color = Color(0xDB, 0x44, 0x37);
      break;
  }

  return node_properties;
}

}  // namespace

void DumpGraphviz(const Model& model, string* output_file_contents) {
  AppendF(output_file_contents, "digraph Computegraph {\n");
  // 'nslimit' is a graphviz (dot) paramater that limits the iterations during
  // the layout phase. Omitting it allows infinite iterations, causing some
  // complex graphs to never finish. A value of 125 produces good graphs
  // while allowing complex graphs to finish.
  AppendF(output_file_contents, "\t nslimit=125;\n");

  constexpr char kNodeFormat[] =
      "\t \"%s\" [label=\"%s\", shape=%s, style=filled, fillcolor=\"#%s\", "
      "fontcolor = \"#%sDD\"];\n";

  constexpr char kEdgeFormat[] =
      "\t \"%s\" -> \"%s\" [penwidth=%f, weight=%f];\n";

  constexpr char kRNNBackEdgeFormat[] =
      "\t \"%s\" -> \"%s\" [color=\"#0F9D58\"];\n";

  for (const auto& array_kv : model.GetArrayMap()) {
    // Add node for array.
    const string& array_name = array_kv.first;
    const auto& array_properties = GetPropertiesForArray(model, array_name);
    AppendF(output_file_contents, kNodeFormat, array_name,
            array_properties.label, "octagon",
            array_properties.color.FillColorString().c_str(),
            array_properties.color.TextColorString().c_str());
  }
  for (int op_index = 0; op_index < model.operators.size(); op_index++) {
    const Operator& op = *model.operators[op_index];
    // Add node for operator.
    auto op_properties = GetPropertiesForOperator(op);
    string operator_id = StringF("op%05d", op_index);
    AppendF(output_file_contents, kNodeFormat, operator_id, op_properties.label,
            "box", op_properties.color.FillColorString().c_str(),
            op_properties.color.TextColorString().c_str());
    // Add edges for all inputs of the operator.
    for (const auto& input : op.inputs) {
      if (!model.HasArray(input)) {
        // Arrays should _always_ exist. Except, perhaps, during development.
        continue;
      }
      auto array_properties = GetPropertiesForArray(model, input);
      // Draw lines that transport more data thicker (Otherwise, where would the
      // data fit? right?).
      float line_width =
          std::max(0.5f, array_properties.log2_buffer_size / 3.0f);
      // Keep edges that transport more data shorter than those with less.
      float weight = std::max(1.0f, array_properties.log2_buffer_size);
      if (!IsInputArray(model, input) &&
          GetOpWithOutput(model, input) == nullptr) {
        // Give the main line of data flow a straighter path by penalizing edges
        // to standalone buffers. Weights are generally very large buffers that
        // otherwise skew the layout without this.
        weight = 1.0f;
      }
      AppendF(output_file_contents, kEdgeFormat, input, operator_id, line_width,
              weight);
    }
    // Add edges for all outputs of the operator.
    for (const auto& output : op.outputs) {
      if (!model.HasArray(output)) {
        // Arrays should _always_ exist. Except, perhaps, during development.
        continue;
      }
      auto array_properties = GetPropertiesForArray(model, output);
      // See comments above regarding weight and line_width calculations.
      float line_width =
          std::max(0.5f, array_properties.log2_buffer_size / 3.0f);
      float weight = std::max(1.0f, array_properties.log2_buffer_size);
      if (!IsArrayConsumed(model, output)) {
        weight = 1.0f;
      }
      AppendF(output_file_contents, kEdgeFormat, operator_id, output,
              line_width, weight);
    }
  }

  for (const auto& rnn_state : model.flags.rnn_states()) {
    AppendF(output_file_contents, kRNNBackEdgeFormat,
            rnn_state.back_edge_source_array(), rnn_state.state_array());
  }

  AppendF(output_file_contents, "}\n");
}
}  // namespace toco
