/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Holds the information we need to translate from a float version of this op
// into the quantized equivalent.
struct QuantizedOpInfo {
  // The name of the float op.
  string float_name;
  // Which attributes to copy directly over.
  std::vector<string> attrs_to_copy;
  // Extra data type attributes we need to set.
  std::vector<std::pair<string, DataType>> dtypes_to_set;
  // What depth of inputs the op can read in.
  DataType input_bit_depth;
  // The depth of the op's quantized outputs.
  DataType output_bit_depth;
  // Which inputs (e.g. shapes) aren't involved in the quantization process.
  std::set<int32> unquantized_inputs;
  // How the outputs are arranged, either
  // [input0, input1, min0, max0, min1, max1] for contiguous, or
  // [input0, input1, min0, min1, max0, max1] for separate.
  // The separate order is needed because it's the only way to specify unknown
  // numbers of inputs for ops like Concat.
  enum { CONTIGUOUS_MIN_MAX, SEPARATE_MIN_MAX } min_max_order;
};

// Every op that has a quantized equivalent should be listed here, so that the
// conversion process can transform them.
const std::vector<QuantizedOpInfo>& GetQuantizedOpList() {
  static const std::vector<QuantizedOpInfo> op_list = {
      {"Add",
       {},
       {{"T1", DT_QUINT8}, {"T2", DT_QUINT8}, {"Toutput", DT_QINT32}},
       DT_QUINT8,
       DT_QINT32,
       {},
       QuantizedOpInfo::CONTIGUOUS_MIN_MAX},
      {"AvgPool",
       {"ksize", "strides", "padding"},
       {{"T", DT_QUINT8}},
       DT_QUINT8,
       DT_QUINT8,
       {},
       QuantizedOpInfo::CONTIGUOUS_MIN_MAX},
      {"BiasAdd",
       {},
       {{"T1", DT_QUINT8}, {"T2", DT_QUINT8}, {"out_type", DT_QINT32}},
       DT_QUINT8,
       DT_QINT32,
       {},
       QuantizedOpInfo::CONTIGUOUS_MIN_MAX},
      {"Concat",
       {"N"},
       {{"T", DT_QUINT8}},
       DT_QUINT8,
       DT_QUINT8,
       {0},
       QuantizedOpInfo::SEPARATE_MIN_MAX},
      {"Conv2D",
       {"strides", "padding"},
       {{"Tinput", DT_QUINT8}, {"Tfilter", DT_QUINT8}, {"out_type", DT_QINT32}},
       DT_QUINT8,
       DT_QINT32,
       {},
       QuantizedOpInfo::CONTIGUOUS_MIN_MAX},
      {"MatMul",
       {"transpose_a", "transpose_b"},
       {{"T1", DT_QUINT8}, {"T2", DT_QUINT8}, {"Toutput", DT_QINT32}},
       DT_QUINT8,
       DT_QINT32,
       {},
       QuantizedOpInfo::CONTIGUOUS_MIN_MAX},
      {"MaxPool",
       {"ksize", "strides", "padding"},
       {{"T", DT_QUINT8}},
       DT_QUINT8,
       DT_QUINT8,
       {},
       QuantizedOpInfo::CONTIGUOUS_MIN_MAX},
      {"Mul",
       {},
       {{"T1", DT_QUINT8}, {"T2", DT_QUINT8}, {"Toutput", DT_QINT32}},
       DT_QUINT8,
       DT_QINT32,
       {},
       QuantizedOpInfo::CONTIGUOUS_MIN_MAX},
      {"Relu",
       {},
       {{"Tinput", DT_QUINT8}},
       DT_QUINT8,
       DT_QUINT8,
       {},
       QuantizedOpInfo::CONTIGUOUS_MIN_MAX},
      {"ResizeBilinear",
       {"align_corners"},
       {{"T", DT_QUINT8}},
       DT_QUINT8,
       DT_QUINT8,
       {1},
       QuantizedOpInfo::CONTIGUOUS_MIN_MAX},
      {"Relu6",
       {},
       {{"Tinput", DT_QUINT8}},
       DT_QUINT8,
       DT_QUINT8,
       {},
       QuantizedOpInfo::CONTIGUOUS_MIN_MAX},
      {"Reshape",
       {},
       {{"T", DT_QUINT8}},
       DT_QUINT8,
       DT_QUINT8,
       {1},
       QuantizedOpInfo::CONTIGUOUS_MIN_MAX},
  };
  return op_list;
}

namespace {
// Replaces invalid characters in input names to get a unique node name.
string UniqueNodeNameFromInput(const string& input_name) {
  string prefix;
  string node_name;
  string suffix;
  NodeNamePartsFromInput(input_name, &prefix, &node_name, &suffix);
  string result;
  if (prefix == "^") {
    result += "__hat__";
  }
  result += node_name;
  if (!suffix.empty()) {
    result += "__port__" + suffix.substr(1, suffix.size() - 1);
  }
  return result;
}

// Pulls two float values from the named parameters, with a lot of checking.
Status ExtractRangeFromParams(const TransformFuncContext& context,
                              const string& min_name, const string& max_name,
                              float* min_value, float* max_value,
                              bool* has_range) {
  // See if we've been given quantized inputs with a known range.
  const bool has_min = (context.params.count(min_name) != 0);
  const bool has_max = (context.params.count(max_name) != 0);
  *has_range = (has_min || has_max);
  if (!*has_range) {
    return Status::OK();
  }
  if (!has_min || !has_max) {
    return errors::InvalidArgument("You must pass both ", min_name, " and ",
                                   max_name, " into quantize_nodes");
  }
  TF_RETURN_IF_ERROR(context.GetOneFloatParameter(min_name, 0.0f, min_value));
  TF_RETURN_IF_ERROR(context.GetOneFloatParameter(max_name, 0.0f, max_value));
  return Status::OK();
}

}  // namespace

// Analyzes all the nodes in the graph to figure out which ones are duplicates
// apart from their names. This commonly includes identical Const nodes, but can
// also be simple operations that are repeated on multiple outputs of a
// particular node. The complexity is managed using a hash function that avoids
// the need for any O(n^2) algorithms when identifying duplicates.
Status MergeDuplicateNodes(const GraphDef& input_graph_def,
                           const TransformFuncContext& context,
                           GraphDef* output_graph_def) {
  // Make sure we can look up inputs and outputs quickly.
  std::set<string> input_names(context.input_names.begin(),
                               context.input_names.end());
  std::set<string> output_names(context.output_names.begin(),
                                context.output_names.end());
  GraphDef current_graph_def = input_graph_def;
  // Keep running the merging until no more duplicates are found.
  bool any_duplicates_found;
  do {
    any_duplicates_found = false;
    // First arrange all of the nodes by a hash of their contents.
    std::map<uint64, std::vector<const NodeDef*>> hashed_nodes;
    for (const NodeDef& node : current_graph_def.node()) {
      NodeDef nameless_node = node;
      // The name matters if it's being used as an input or output node,
      // otherwise ignore it when looking for duplicates.
      if (!input_names.count(node.name()) && !output_names.count(node.name())) {
        nameless_node.set_name("");
      }
      const uint64 hash = HashNodeDef(nameless_node);
      hashed_nodes[hash].push_back(&node);
    }
    // If we have multiple nodes with the same hash, then we know they're
    // duplicates and can be removed, unless they're stateful.
    std::map<string, string> inputs_to_rename;
    GraphDef merged_graph_def;
    for (const std::pair<const uint64, std::vector<const NodeDef*>>&
             hashed_node_info : hashed_nodes) {
      const std::vector<const NodeDef*>& hash_node_list =
          hashed_node_info.second;
      for (int i = 0; i < hash_node_list.size(); ++i) {
        const NodeDef* current_node = hash_node_list[i];
        const OpDef* op_def = nullptr;
        TF_RETURN_IF_ERROR(
            OpRegistry::Global()->LookUpOpDef(current_node->op(), &op_def));
        const bool is_duplicate = ((!op_def->is_stateful()) && (i > 0));
        if (is_duplicate) {
          const string original_name = hash_node_list[0]->name();
          inputs_to_rename[current_node->name() + ":*"] = original_name;
          any_duplicates_found = true;
        } else {
          NodeDef* new_node = merged_graph_def.mutable_node()->Add();
          *new_node = *current_node;
        }
      }
    }
    // Update the graph so that any nodes that referred to removed inputs now
    // pull from the remaining duplicate.
    TF_RETURN_IF_ERROR(RenameNodeInputs(merged_graph_def, inputs_to_rename,
                                        std::unordered_set<string>(),
                                        &current_graph_def));
  } while (any_duplicates_found);

  *output_graph_def = current_graph_def;

  return Status::OK();
}

// Looks for the patterns that indicate there are two eight-bit ops feeding into
// each other, separated by a conversion up to float and back again. These occur
// during the initial conversion of ops to their quantized forms. Because we're
// only looking at an individual op in that phase and don't know if its inputs
// and outputs are eight-bit-capable, we start by converting the actual op into
// quantized form, but add float conversions before and after. This pass gets
// rid of those conversions if it turns out we do have adjacent ops capable of
// eight-bit processing.
Status RemoveRedundantQuantizations(const GraphDef& input_graph_def,
                                    const TransformFuncContext& context,
                                    GraphDef* output_graph_def) {
  std::set<string> graph_outputs;
  for (const string& output_name : context.output_names) {
    graph_outputs.insert(NodeNameFromInput(output_name));
  }
  std::map<string, string> inputs_to_rename;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"QuantizeV2",
        {
          {"Dequantize"},
          {"Min"},
          {"Max"},
        }
      },  // clang-format on
      [&inputs_to_rename, &graph_outputs](const NodeMatch& match,
                                          const std::set<string>& input_nodes,
                                          const std::set<string>& output_nodes,
                                          std::vector<NodeDef>* new_nodes) {
        const NodeDef& quantize_node = match.node;
        const NodeDef& dequantize_node = match.inputs[0].node;
        inputs_to_rename[quantize_node.name() + ":0"] =
            dequantize_node.input(0);
        inputs_to_rename[quantize_node.name() + ":1"] =
            dequantize_node.input(1);
        inputs_to_rename[quantize_node.name() + ":2"] =
            dequantize_node.input(2);

        // Are other sub-graphs using the float intermediate result? If so,
        // preserve it, but the input renaming still rewires the eight-bit ops
        // so they don't go through float.
        if (output_nodes.count(dequantize_node.name()) ||
            graph_outputs.count(dequantize_node.name())) {
          CopyOriginalMatch(match, new_nodes);
        }

        return Status::OK();
      },
      {true}, &replaced_graph_def));

  return RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                          std::unordered_set<string>(), output_graph_def);
}

// If the user has passed in the input_min and input_max args, then we need to
// convert any input placeholders from float to eight bit, so quantized inputs
// can be fed directly into the graph.
Status QuantizePlaceholders(const GraphDef& input_graph_def,
                            const TransformFuncContext& context,
                            GraphDef* output_graph_def) {
  float input_min;
  float input_max;
  bool has_input_range;
  TF_RETURN_IF_ERROR(ExtractRangeFromParams(context, "input_min", "input_max",
                                            &input_min, &input_max,
                                            &has_input_range));
  if (!has_input_range) {
    *output_graph_def = input_graph_def;
    return Status::OK();
  }
  std::map<string, string> inputs_to_rename_first_pass;
  std::map<string, string> inputs_to_rename_second_pass;
  GraphDef placeholder_graph_def;
  placeholder_graph_def.Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    if (node.op() != "Placeholder") {
      *(placeholder_graph_def.mutable_node()->Add()) = node;
    } else {
      string namespace_prefix = node.name() + "_eightbit";

      NodeDef quantized_placeholder;
      quantized_placeholder = node;
      SetNodeAttr("dtype", DT_QUINT8, &quantized_placeholder);
      *(placeholder_graph_def.mutable_node()->Add()) = quantized_placeholder;

      NodeDef min_node;
      min_node.set_op("Const");
      min_node.set_name(namespace_prefix + "/min");
      SetNodeAttr("dtype", DT_FLOAT, &min_node);
      Tensor min_tensor(DT_FLOAT, {});
      min_tensor.flat<float>()(0) = input_min;
      SetNodeTensorAttr<float>("value", min_tensor, &min_node);
      *(placeholder_graph_def.mutable_node()->Add()) = min_node;

      NodeDef max_node;
      max_node.set_op("Const");
      max_node.set_name(namespace_prefix + "/max");
      SetNodeAttr("dtype", DT_FLOAT, &max_node);
      Tensor max_tensor(DT_FLOAT, {});
      max_tensor.flat<float>()(0) = input_max;
      SetNodeTensorAttr<float>("value", max_tensor, &max_node);
      *(placeholder_graph_def.mutable_node()->Add()) = max_node;

      const string rename_suffix = "__RENAMED_PLACEHOLDER__";
      NodeDef dequantize_node;
      dequantize_node.set_op("Dequantize");
      dequantize_node.set_name(namespace_prefix + "/dequantize");
      SetNodeAttr("T", DT_QUINT8, &dequantize_node);
      SetNodeAttr("mode", "MIN_FIRST", &dequantize_node);
      AddNodeInput(node.name() + rename_suffix, &dequantize_node);
      AddNodeInput(min_node.name(), &dequantize_node);
      AddNodeInput(max_node.name(), &dequantize_node);
      *(placeholder_graph_def.mutable_node()->Add()) = dequantize_node;

      // First make sure that any internal references to the old placeholder
      // now point to the dequantize result.
      inputs_to_rename_first_pass[node.name()] = dequantize_node.name();
      // Then fix up the dequantize op so that it really points to the
      // placeholder.
      inputs_to_rename_second_pass[node.name() + rename_suffix] = node.name();
    }
  }

  GraphDef first_pass_graph_def;
  TF_RETURN_IF_ERROR(
      RenameNodeInputs(placeholder_graph_def, inputs_to_rename_first_pass,
                       std::unordered_set<string>(), &first_pass_graph_def));
  TF_RETURN_IF_ERROR(
      RenameNodeInputs(first_pass_graph_def, inputs_to_rename_second_pass,
                       std::unordered_set<string>(), output_graph_def));

  return Status::OK();
}

// During training, FakeQuantWithMinMaxVars ops capture a good min/max range for
// an activation layer. To use these during inference, this pass converts those
// ops into Requantizes with the trained min/maxes as constant inputs.
Status ConvertFakeQuantsToRequantize(const GraphDef& input_graph_def,
                                     const TransformFuncContext& context,
                                     GraphDef* output_graph_def) {
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"FakeQuantWithMinMaxVars",
        {
          {"*"},
          {"Const"},
          {"Const"},
        }
      },  // clang-format on
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        const NodeDef& fake_quant_node = match.node;
        const NodeDef& original_op_node = match.inputs[0].node;
        const NodeDef& fake_quant_min_node = match.inputs[1].node;
        const NodeDef& fake_quant_max_node = match.inputs[2].node;

        string namespace_prefix = fake_quant_node.name() + "_eightbit";

        new_nodes->push_back(original_op_node);
        new_nodes->push_back(fake_quant_min_node);
        new_nodes->push_back(fake_quant_max_node);

        NodeDef quantize_node;
        quantize_node.set_op("QuantizeV2");
        quantize_node.set_name(namespace_prefix + "/quantize");
        SetNodeAttr("T", DT_QINT32, &quantize_node);
        SetNodeAttr("mode", "MIN_FIRST", &quantize_node);
        AddNodeInput(fake_quant_node.input(0), &quantize_node);
        AddNodeInput(fake_quant_min_node.name(), &quantize_node);
        AddNodeInput(fake_quant_max_node.name(), &quantize_node);
        new_nodes->push_back(quantize_node);

        NodeDef requantize_node;
        requantize_node.set_op("Requantize");
        requantize_node.set_name(namespace_prefix + "/requantize");
        SetNodeAttr("Tinput", DT_QINT32, &requantize_node);
        SetNodeAttr("out_type", DT_QUINT8, &requantize_node);
        AddNodeInput(quantize_node.name() + ":0", &requantize_node);
        AddNodeInput(quantize_node.name() + ":1", &requantize_node);
        AddNodeInput(quantize_node.name() + ":2", &requantize_node);
        AddNodeInput(fake_quant_min_node.name(), &requantize_node);
        AddNodeInput(fake_quant_max_node.name(), &requantize_node);
        new_nodes->push_back(requantize_node);

        // Convert the 8-bit result back into float for the final output.
        NodeDef dequantize_node;
        dequantize_node.set_op("Dequantize");
        dequantize_node.set_name(fake_quant_node.name());
        SetNodeAttr("T", DT_QUINT8, &dequantize_node);
        SetNodeAttr("mode", "MIN_FIRST", &dequantize_node);
        AddNodeInput(requantize_node.name() + ":0", &dequantize_node);
        AddNodeInput(requantize_node.name() + ":1", &dequantize_node);
        AddNodeInput(requantize_node.name() + ":2", &dequantize_node);
        new_nodes->push_back(dequantize_node);

        return Status::OK();
      },
      {}, output_graph_def));

  return Status::OK();
}

// We always generate Requantize ops driven by dynamic RequantizationRange
// calculations when we produce quantized ops like Conv2D or BiasAdd with
// 32-bit results. If there were FakeQuant ops already for those activation
// layers, then there will be a later Requantize op with constant min/max
// inputs, which is preferable for fast inference. This pass looks for those
// later Requantize ops, and replaces the dynamic version with them.
Status MergeAdjacentRequantizes(const GraphDef& input_graph_def,
                                const TransformFuncContext& context,
                                GraphDef* output_graph_def) {
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"Requantize",
        {
          {"QuantizeV2",
            {
              {"Dequantize",
                {
                  {"Requantize",
                    {
                      {"*"},
                      {"*"},
                      {"*"},
                      {"RequantizationRange"},
                      {"RequantizationRange"},
                    }
                  },
                  {"Requantize"},
                  {"Requantize"},
                }
              },
              {"Const"},
              {"Const"},
            },
          },
          {"QuantizeV2"},
          {"QuantizeV2"},
          {"Const"},
          {"Const"},
        }
      },  // clang-format on
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        const NodeDef& fake_requantize_node = match.node;
        const NodeDef& original_op_node =
            match.inputs[0].inputs[0].inputs[0].inputs[0].node;
        const NodeDef& fake_requantize_min_node = match.inputs[3].node;
        const NodeDef& fake_requantize_max_node = match.inputs[4].node;

        new_nodes->push_back(original_op_node);
        new_nodes->push_back(fake_requantize_min_node);
        new_nodes->push_back(fake_requantize_max_node);

        NodeDef requantize_node;
        requantize_node = fake_requantize_node;
        requantize_node.mutable_input()->Clear();
        AddNodeInput(original_op_node.name() + ":0", &requantize_node);
        AddNodeInput(original_op_node.name() + ":1", &requantize_node);
        AddNodeInput(original_op_node.name() + ":2", &requantize_node);
        AddNodeInput(fake_requantize_min_node.name(), &requantize_node);
        AddNodeInput(fake_requantize_max_node.name(), &requantize_node);
        new_nodes->push_back(requantize_node);

        return Status::OK();
      },
      {}, output_graph_def));

  return Status::OK();
}

// Sometimes FakeQuantWithMinMaxVars ops are added at the end of a chain of
// linear ops like Relu, MaxPool, etc, several steps from the Conv2D or BiasAdd
// op that we want to apply the trained constant conversions to. This pass tries
// to move FakeQuant ops up the input chain, so they're as close as possible to
// the 32-bit conversion, and so can be easily merged into the automatic dynamic
// Requantizes.
Status HoistFakeQuants(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def) {
  GraphDef current_graph_def = input_graph_def;
  const int max_depth = 3;
  for (int depth = max_depth; depth > 0; --depth) {
    OpTypePattern pattern = {"*"};
    for (int i = 0; i < depth; ++i) {
      pattern = {"*", {pattern}};
    }
    pattern = {"FakeQuantWithMinMaxVars", {pattern, {"Const"}, {"Const"}}};
    GraphDef hoisted_graph_def;
    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def, pattern,
        [depth](const NodeMatch& match, const std::set<string>& input_nodes,
                const std::set<string>& output_nodes,
                std::vector<NodeDef>* new_nodes) {
          const NodeDef& fake_quant_node = match.node;
          const NodeDef& fake_quant_min_node = match.inputs[1].node;
          const NodeDef& fake_quant_max_node = match.inputs[2].node;
          std::vector<NodeDef> linear_nodes;
          NodeMatch current_match = match;
          for (int i = 0; i <= depth; ++i) {
            linear_nodes.push_back(current_match.inputs[0].node);
            current_match = current_match.inputs[0];
          }
          NodeDef new_fake_quant_node;
          new_fake_quant_node = fake_quant_node;
          new_fake_quant_node.set_name(fake_quant_node.name() + "_hoisted");
          new_fake_quant_node.set_input(
              0, linear_nodes[linear_nodes.size() - 2].input(0));
          new_nodes->push_back(new_fake_quant_node);

          new_nodes->push_back(fake_quant_min_node);
          new_nodes->push_back(fake_quant_max_node);

          linear_nodes[linear_nodes.size() - 2].set_input(
              0, new_fake_quant_node.name());
          linear_nodes.front().set_name(fake_quant_node.name());
          for (const NodeDef& linear_node : linear_nodes) {
            new_nodes->push_back(linear_node);
          }

          return Status::OK();
        },
        {}, &hoisted_graph_def));
    current_graph_def = hoisted_graph_def;
  }
  *output_graph_def = current_graph_def;

  return Status::OK();
}

// Converts any float ops that have eight-bit equivalents into their quantized
// forms, so that as much calculation as possible is done in the lower-precision
// format.
Status QuantizeNodes(const GraphDef& input_graph_def,
                     const TransformFuncContext& context,
                     GraphDef* output_graph_def) {
  // Loop through all of the quantizable op types, and replace any occurrences
  // with equivalent sub-graphs with quantized ops at their core. For example
  // this one-input operation:
  //
  //            Input(float)
  //                |
  //                v
  //            Operation
  //                |
  //                v
  //             (float)
  //
  // Will be turned into it's quantized equivalent:
  //
  //      Input(float)          ReshapeDims
  //         +------v v-------------+
  //         |    Reshape
  //         |      |
  //         |      |          ReductionDims
  //         |      +-----+         |
  //         |      | +---c---------+
  //         |      v v   v v-------+
  //         |      Min   Max
  //         |  +----+      |
  //         v  v  v--------+
  //        Quantize
  //            |
  //            v
  //     QuantizedOperation
  //        |   |   |
  //        v   v   v
  //        Dequantize
  //            |
  //            v
  //         (float)
  //
  // This keeps the inputs and outputs visible to the rest of the graph in
  // float
  // and converts them down to quantized buffers internally for the
  // computation.
  // The result will end up with a lot of redundant dequantize/quantize pairs
  // between adjacent quantized ops, but a later pass removes these where it
  // can.

  std::set<string> ops_to_ignore;
  if (context.params.count("ignore_op") > 0) {
    for (const string& name : context.params.at("ignore_op")) {
      ops_to_ignore.insert(name);
    }
  }

  const std::vector<QuantizedOpInfo>& op_list = GetQuantizedOpList();
  string op_pattern;
  bool is_first = true;
  std::map<string, QuantizedOpInfo> op_map;
  for (const QuantizedOpInfo& op_info : op_list) {
    if (ops_to_ignore.count(op_info.float_name) == 0) {
      strings::StrAppend(&op_pattern, (is_first ? "" : "|"),
                         op_info.float_name);
      op_map.insert({op_info.float_name, op_info});
      is_first = false;
    }
  }

  // If input_min and input max have been passed in, then we convert all float
  // Placeholder nodes into quantized versions, with the supplied values as
  // their range.
  GraphDef placeholder_graph_def;
  TF_RETURN_IF_ERROR(
      QuantizePlaceholders(input_graph_def, context, &placeholder_graph_def));
  TF_RETURN_IF_ERROR(IsGraphValid(placeholder_graph_def));

  // If there are any FakeQuantWithMinMaxVars at the end of a chain of linear
  // operations like Relu or MaxPool, move them up so that they're as close as
  // possible to ops with 32-bit outputs like BiasAdd or Conv2D.
  GraphDef hoisted_graph_def;
  TF_RETURN_IF_ERROR(
      HoistFakeQuants(placeholder_graph_def, context, &hoisted_graph_def));
  TF_RETURN_IF_ERROR(IsGraphValid(hoisted_graph_def));

  // Convert any FakeQuantWithMinMaxVars, which hold the trained ranges of
  // activation layers, into Requantize ops with those ranges instead. This
  // makes it easier to replace the dynamic range calculations that are used
  // by default.
  GraphDef converted_graph_def;
  TF_RETURN_IF_ERROR(ConvertFakeQuantsToRequantize(hoisted_graph_def, context,
                                                   &converted_graph_def));
  TF_RETURN_IF_ERROR(IsGraphValid(converted_graph_def));

  // If fallback_min and fallback_max are set, then we'll use hardwired ranges
  // for all the 32-bit to 8-bit requantizations.
  float fallback_min;
  float fallback_max;
  bool has_fallback_range;
  TF_RETURN_IF_ERROR(ExtractRangeFromParams(
      context, "fallback_min", "fallback_max", &fallback_min, &fallback_max,
      &has_fallback_range));

  // Replace all occurrences of the current float op with its quantized
  // equivalent.
  GraphDef quantized_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      converted_graph_def, {op_pattern},
      [&op_map, fallback_min, fallback_max, has_fallback_range](
          const NodeMatch& match, const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {
        const NodeDef& float_node = match.node;
        const QuantizedOpInfo& op_info = op_map[float_node.op()];

        DataTypeVector input_types;
        DataTypeVector output_types;
        TF_RETURN_IF_ERROR(
            GetInOutTypes(float_node, &input_types, &output_types));
        bool are_all_float = true;
        for (int i = 0; i < float_node.input_size(); ++i) {
          // Skip any known non-float inputs.
          if (op_info.unquantized_inputs.count(i)) {
            continue;
          }
          if (i >= input_types.size()) {
            LOG(ERROR) << "input_types has incorrect size "
                       << input_types.size() << " <= " << i
                       << ". Assuming everything else is floats.";
          }
          if (i < input_types.size() && input_types[i] != DT_FLOAT) {
            are_all_float = false;
          }
        }
        for (const DataType& output_type : output_types) {
          if (output_type != DT_FLOAT) {
            are_all_float = false;
          }
        }
        // This isn't a float op, so don't quantize it.
        if (!are_all_float) {
          CopyOriginalMatch(match, new_nodes);
          return Status::OK();
        }

        string namespace_prefix = float_node.name() + "_eightbit";

        // Quantize all of the inputs.
        std::vector<string> quantized_input_names;
        for (int i = 0; i < float_node.input_size(); ++i) {
          // Skip any non-float inputs.
          if (op_info.unquantized_inputs.count(i)) {
            continue;
          }

          const string& input_name = float_node.input(i);
          string unique_input_name =
              namespace_prefix + "/" + UniqueNodeNameFromInput(input_name);

          // Add some common constants we need for reshaping inputs.
          NodeDef reshape_dims;
          reshape_dims.set_op("Const");
          reshape_dims.set_name(unique_input_name + "/reshape_dims");
          AddNodeInput("^" + NodeNameFromInput(input_name), &reshape_dims);
          SetNodeAttr("dtype", DT_INT32, &reshape_dims);
          Tensor reshape_dims_tensor(DT_INT32, {1});
          reshape_dims_tensor.flat<int32>()(0) = -1;
          SetNodeTensorAttr<int32>("value", reshape_dims_tensor, &reshape_dims);
          new_nodes->push_back(reshape_dims);

          NodeDef reduction_dims;
          reduction_dims.set_op("Const");
          reduction_dims.set_name(unique_input_name + "/reduction_dims");
          AddNodeInput("^" + NodeNameFromInput(input_name), &reduction_dims);
          SetNodeAttr("dtype", DT_INT32, &reduction_dims);
          Tensor reduction_dims_tensor(DT_INT32, {1});
          reduction_dims_tensor.flat<int32>()(0) = 0;
          SetNodeTensorAttr<int32>("value", reduction_dims_tensor,
                                   &reduction_dims);
          new_nodes->push_back(reduction_dims);

          NodeDef reshape_node;
          reshape_node.set_op("Reshape");
          reshape_node.set_name(unique_input_name + "/reshape");
          SetNodeAttr("T", DT_FLOAT, &reshape_node);
          AddNodeInput(input_name, &reshape_node);
          AddNodeInput(reshape_dims.name(), &reshape_node);
          new_nodes->push_back(reshape_node);

          NodeDef min_node;
          min_node.set_op("Min");
          min_node.set_name(unique_input_name + "/min");
          SetNodeAttr("T", DT_FLOAT, &min_node);
          SetNodeAttr("keep_dims", false, &min_node);
          AddNodeInput(reshape_node.name(), &min_node);
          AddNodeInput(reduction_dims.name(), &min_node);
          new_nodes->push_back(min_node);

          NodeDef max_node;
          max_node.set_op("Max");
          max_node.set_name(unique_input_name + "/max");
          SetNodeAttr("T", DT_FLOAT, &max_node);
          SetNodeAttr("keep_dims", false, &max_node);
          AddNodeInput(reshape_node.name(), &max_node);
          AddNodeInput(reduction_dims.name(), &max_node);
          new_nodes->push_back(max_node);

          NodeDef quantize_node;
          quantize_node.set_op("QuantizeV2");
          quantize_node.set_name(unique_input_name + "/quantize");
          SetNodeAttr("T", DT_QUINT8, &quantize_node);
          SetNodeAttr("mode", "MIN_FIRST", &quantize_node);
          AddNodeInput(input_name, &quantize_node);
          AddNodeInput(min_node.name(), &quantize_node);
          AddNodeInput(max_node.name(), &quantize_node);
          new_nodes->push_back(quantize_node);
          quantized_input_names.push_back(quantize_node.name());
        }

        // Set up the quantized version of the current op.
        NodeDef quantized_main_node;
        quantized_main_node.set_op("Quantized" + float_node.op());
        quantized_main_node.set_name(float_node.name() + "/eightbit");
        for (const string& attr_to_copy : op_info.attrs_to_copy) {
          CopyNodeAttr(float_node, attr_to_copy, attr_to_copy,
                       &quantized_main_node);
        }
        for (const std::pair<string, DataType>& dtype_to_set :
             op_info.dtypes_to_set) {
          SetNodeAttr(dtype_to_set.first, dtype_to_set.second,
                      &quantized_main_node);
        }
        int quantized_input_index = 0;
        for (int i = 0; i < float_node.input_size(); ++i) {
          if (op_info.unquantized_inputs.count(i)) {
            AddNodeInput(float_node.input(i), &quantized_main_node);
          } else {
            const string& quantized_input_name =
                quantized_input_names[quantized_input_index];
            AddNodeInput(quantized_input_name + ":0", &quantized_main_node);
            ++quantized_input_index;
          }
        }
        if (op_info.min_max_order == QuantizedOpInfo::CONTIGUOUS_MIN_MAX) {
          for (const string& quantized_input_name : quantized_input_names) {
            AddNodeInput(quantized_input_name + ":1", &quantized_main_node);
            AddNodeInput(quantized_input_name + ":2", &quantized_main_node);
          }
        } else {
          for (const string& quantized_input_name : quantized_input_names) {
            AddNodeInput(quantized_input_name + ":1", &quantized_main_node);
          }
          for (const string& quantized_input_name : quantized_input_names) {
            AddNodeInput(quantized_input_name + ":2", &quantized_main_node);
          }
        }
        new_nodes->push_back(quantized_main_node);

        string eight_bit_node_name;
        if (op_info.output_bit_depth == DT_QINT32) {
          // Shrink the range of the output down from 32 bits to 8.
          string requantize_min_input;
          string requantize_max_input;
          if (has_fallback_range) {
            // Use constant values for the min/max range if they were given.
            NodeDef fallback_min_node;
            fallback_min_node.set_op("Const");
            fallback_min_node.set_name(quantized_main_node.name() +
                                       "/fallback_min");
            SetNodeAttr("dtype", DT_FLOAT, &fallback_min_node);
            Tensor fallback_min_tensor(DT_FLOAT, {});
            fallback_min_tensor.flat<float>()(0) = fallback_min;
            SetNodeTensorAttr<float>("value", fallback_min_tensor,
                                     &fallback_min_node);
            new_nodes->push_back(fallback_min_node);

            NodeDef fallback_max_node;
            fallback_max_node.set_op("Const");
            fallback_max_node.set_name(quantized_main_node.name() +
                                       "/fallback_max");
            SetNodeAttr("dtype", DT_FLOAT, &fallback_max_node);
            Tensor fallback_max_tensor(DT_FLOAT, {});
            fallback_max_tensor.flat<float>()(0) = fallback_max;
            SetNodeTensorAttr<float>("value", fallback_max_tensor,
                                     &fallback_max_node);
            new_nodes->push_back(fallback_max_node);

            requantize_min_input = fallback_min_node.name();
            requantize_max_input = fallback_max_node.name();
          } else {
            // Otherwise dynamically measure the range each time.
            NodeDef requant_range_node;
            requant_range_node.set_op("RequantizationRange");
            requant_range_node.set_name(quantized_main_node.name() +
                                        "/requant_range");
            SetNodeAttr("Tinput", DT_QINT32, &requant_range_node);
            AddNodeInput(quantized_main_node.name() + ":0",
                         &requant_range_node);
            AddNodeInput(quantized_main_node.name() + ":1",
                         &requant_range_node);
            AddNodeInput(quantized_main_node.name() + ":2",
                         &requant_range_node);
            new_nodes->push_back(requant_range_node);

            requantize_min_input = requant_range_node.name() + ":0";
            requantize_max_input = requant_range_node.name() + ":1";
          }
          NodeDef requantize_node;
          requantize_node.set_op("Requantize");
          requantize_node.set_name(quantized_main_node.name() + "/requantize");
          SetNodeAttr("Tinput", DT_QINT32, &requantize_node);
          SetNodeAttr("out_type", DT_QUINT8, &requantize_node);
          AddNodeInput(quantized_main_node.name() + ":0", &requantize_node);
          AddNodeInput(quantized_main_node.name() + ":1", &requantize_node);
          AddNodeInput(quantized_main_node.name() + ":2", &requantize_node);
          AddNodeInput(requantize_min_input, &requantize_node);
          AddNodeInput(requantize_max_input, &requantize_node);
          new_nodes->push_back(requantize_node);
          eight_bit_node_name = requantize_node.name();
        } else {
          eight_bit_node_name = quantized_main_node.name();
        }

        // Convert the 8-bit result back into float for the final output.
        NodeDef dequantize_node;
        dequantize_node.set_op("Dequantize");
        dequantize_node.set_name(float_node.name());
        SetNodeAttr("T", DT_QUINT8, &dequantize_node);
        SetNodeAttr("mode", "MIN_FIRST", &dequantize_node);
        AddNodeInput(eight_bit_node_name + ":0", &dequantize_node);
        AddNodeInput(eight_bit_node_name + ":1", &dequantize_node);
        AddNodeInput(eight_bit_node_name + ":2", &dequantize_node);
        new_nodes->push_back(dequantize_node);

        return Status::OK();
      },
      {}, &quantized_graph_def));
  TF_RETURN_IF_ERROR(IsGraphValid(quantized_graph_def));

  // If we've ended up with two Requantize ops in a row (for example if there
  // was a Conv2D feeding into a FakeQuantWithMinMaxVars) merge them together,
  // using the trained range from the second op.
  GraphDef merged_graph_def;
  TF_RETURN_IF_ERROR(MergeAdjacentRequantizes(quantized_graph_def, context,
                                              &merged_graph_def));
  TF_RETURN_IF_ERROR(IsGraphValid(merged_graph_def));

  // There can be duplicate quantize nodes if multiple ops pull from a single
  // input, which makes it harder to remove redundant ones, so strip them out.
  GraphDef deduped_graph_def;
  TF_RETURN_IF_ERROR(
      MergeDuplicateNodes(merged_graph_def, context, &deduped_graph_def));
  TF_RETURN_IF_ERROR(IsGraphValid(deduped_graph_def));

  // Look for Dequantizes that immediately go into Quantizes, and remove them
  // since the two together cancel each other out. This allows us to keep the
  // data flow in eight bit where two adjacent ops are in eight bit, but still
  // keep interoperability with float ops.
  TF_RETURN_IF_ERROR(RemoveRedundantQuantizations(deduped_graph_def, context,
                                                  output_graph_def));
  TF_RETURN_IF_ERROR(IsGraphValid(*output_graph_def));

  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("quantize_nodes", QuantizeNodes);

REGISTER_GRAPH_TRANSFORM("merge_duplicate_nodes", MergeDuplicateNodes);

}  // namespace graph_transforms
}  // namespace tensorflow
