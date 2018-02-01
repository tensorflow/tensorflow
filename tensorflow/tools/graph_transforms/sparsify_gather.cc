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

#include <cmath>
#include <memory>
#include <unordered_map>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
using str_util::Join;
using str_util::Split;
using str_util::StringReplace;
using strings::StrCat;

namespace graph_transforms {

// Sparsify Tensor of shape [N, 1]. Return the indices and values vectors for
// non-zero tensor content.
Status SparsifyWeights(const Tensor& tensor, Tensor* indices_tensor,
                       Tensor* values_tensor) {
  if (tensor.dims() != 2 || tensor.dim_size(1) != 1) {
    return tensorflow::errors::FailedPrecondition(
        "Transform only applicable to subgraph with 'Const' with "
        "tensor of shape [N, 1]. But instead get shape ",
        tensor.shape().DebugString(), ".");
  }

  auto flat = tensor.flat<float>();
  std::vector<int64> indices;
  std::vector<float> values;

  for (int64 i = 0; i < flat.size(); i++) {
    float val = flat(i);
    if (std::abs(val) >= 1.0e-5) {
      indices.push_back(i);
      values.push_back(val);
    }
  }

  // During model initialization, InitializeTableOp makes use of
  // KeyValueTensorIterator, which does not accept empty keys or values.
  // Consequently, adding a dummy pair of indices and values as a walkaround.
  if (indices.empty() || values.empty()) {
    indices.push_back(0);
    values.push_back(0);
  }
  *indices_tensor = Tensor(DataTypeToEnum<int64>::value,
                           {static_cast<int64>(indices.size())});
  std::copy_n(indices.begin(), indices.size(),
              indices_tensor->flat<int64>().data());

  *values_tensor =
      Tensor(DataTypeToEnum<float>::value, {static_cast<int64>(values.size())});
  std::copy_n(values.begin(), values.size(),
              values_tensor->flat<float>().data());

  return Status::OK();
}

void CreateConstNode(const Tensor& tensor, const string& name,
                     NodeDef* node_def) {
  node_def->set_op("Const");
  node_def->set_name(name);
  SetNodeTensorAttr<float>("value", tensor, node_def);
}

Status ObtainTensorSlice(const GraphDef& input_graph_def,
                         const string& tensor_name,
                         string* shape_slice_string) {
  string restore_node_name;
  for (const auto& node : input_graph_def.node()) {
    std::vector<string> node_name_parts = Split(node.name(), "/");
    if (node_name_parts.size() == 2 &&
        StringPiece(node_name_parts[0]).starts_with("save") &&
        StringPiece(node_name_parts[1]).starts_with("Assign") &&
        node.input(0) == tensor_name) {
      restore_node_name = node.input(1);
      break;
    }
  }
  string shape_and_slices_node;
  for (const auto& node : input_graph_def.node()) {
    if ((node.name() == restore_node_name) && (node.op() == "RestoreV2")) {
      shape_and_slices_node = node.input(2);
      break;
    }
  }
  for (const auto& node : input_graph_def.node()) {
    if (node.name() == shape_and_slices_node) {
      Tensor shape_and_slices_tensor;
      TF_RETURN_IF_ERROR(GetNodeAttr(node, "value", &shape_and_slices_tensor));
      const auto& shape_and_slices_value =
          shape_and_slices_tensor.flat<string>();
      *shape_slice_string = shape_and_slices_value(0);
      return Status::OK();
    }
  }
  return errors::Internal("Unable to find slice for variable: ", tensor_name);
}

string GetMonolithicTensorKey(const string& tensor_slice_name) {
  std::vector<string> names = Split(tensor_slice_name, "/");
  CHECK_GE(names.size(), 2);
  CHECK(StringPiece(names[names.size() - 1]).starts_with("part_"));

  // Remove the "part_x" suffix
  names.pop_back();
  return Join(names, "/");
}

Status ReadTensorFromCheckpoint(
    const string& tensor_name, const std::unique_ptr<BundleReader>& ckpt_reader,
    const string& shape_and_slice, Tensor* tensor) {
  if (ckpt_reader) {
    TensorShape parsed_full_shape;
    TensorSlice parsed_slice;
    TensorShape parsed_slice_shape;

    bool get_slice = false;
    if (!shape_and_slice.empty()) {
      TF_RETURN_IF_ERROR(
          checkpoint::ParseShapeAndSlice(shape_and_slice, &parsed_full_shape,
                                         &parsed_slice, &parsed_slice_shape));
      get_slice = (parsed_full_shape != parsed_slice_shape);
    }
    if (get_slice) {
      TF_RETURN_IF_ERROR(ckpt_reader->LookupSlice(
          GetMonolithicTensorKey(tensor_name), parsed_slice, tensor));
    } else {
      TF_RETURN_IF_ERROR(
          ckpt_reader->Lookup(GetMonolithicTensorKey(tensor_name), tensor));
    }
    return Status::OK();
  }
  return errors::Internal("Checkpoint reader was not initialized. ");
}

Status InitializeCheckpointReader(const TransformFuncContext& context,
                                  std::unique_ptr<BundleReader>* ckpt_reader) {
  if (context.params.count("input_checkpoint")) {
    const string input_checkpoint = context.params.at("input_checkpoint")[0];
    ckpt_reader->reset(new BundleReader(Env::Default(), input_checkpoint));
    TF_RETURN_IF_ERROR((*ckpt_reader)->status());
  }
  return Status::OK();
}

Status ObtainVariableInfo(
    const GraphDef& input_graph_def,
    std::unique_ptr<std::unordered_map<string, string> >* shapes_and_slices) {
  shapes_and_slices->reset(new std::unordered_map<string, string>());
  for (const auto& node : input_graph_def.node()) {
    if ((node.op() == "Variable") || (node.op() == "VariableV2")) {
      string s;
      TF_RETURN_IF_ERROR(ObtainTensorSlice(input_graph_def, node.name(), &s));
      (**shapes_and_slices)[node.name()] = s;
    }
  }
  return Status::OK();
}

Status RemoveInputAtIndex(NodeDef* n, int index) {
  for (int i = index; i < n->input_size() - 1; i++) {
    n->mutable_input()->SwapElements(i, i + 1);
  }
  n->mutable_input()->RemoveLast();
  return Status::OK();
}

Status SparsifyGatherInternal(
    const GraphDef& input_graph_def,
    const std::unique_ptr<std::unordered_map<string, string> >&
        shapes_and_slices,
    const TransformFuncContext& context, const OpTypePattern& pattern,
    const std::unique_ptr<BundleReader>& ckpt_reader,
    GraphDef* output_graph_def) {
  string group_init_node = "group_deps";
  if (context.params.count("group_init_node")) {
    group_init_node = context.params.at("group_init_node")[0];
  }
  GraphDef current_graph_def = input_graph_def;
  bool any_match_found = false;

  // Populate references.
  std::unordered_map<string, int> refs;
  for (const auto& node : current_graph_def.node()) {
    for (const auto& input : node.input()) {
      auto parsed_input = StringReplace(input, "^", "", true);
      refs[parsed_input] += 1;
    }
  }

  // The subgraphs may have overlapping components, therefore GraphMatcher
  // doesn't return all subgraphs in one round -- this has to be multi-round
  // update.
  do {
    any_match_found = false;
    GraphDef replaced_graph_def = current_graph_def;
    std::vector<string> init_table_node_names;
    std::vector<string> removed_node_names;

    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def, pattern,
        [&ckpt_reader, &any_match_found, &init_table_node_names,
         &shapes_and_slices, &removed_node_names,
         &refs](const NodeMatch& match, const std::set<string>& input_nodes,
                const std::set<string>& output_nodes,
                std::vector<NodeDef>* new_nodes) {
          any_match_found = true;

          // The captured subgraph should be of the following pattern:
          // Const --> Identity --> Gather --> ...
          //                          ^
          //                          |
          //                        (ids)
          //
          // After transform, it becomes:
          //                   --> NoOp(group_deps)
          //                   |
          // Const --> InitializeTable --> HashTable
          //                   ^              |
          //                   |              |
          // Const -------------              |
          //                                  v
          //               (ids) ---> LookupTableFind <--- Const(default)
          //                                  |
          //                                  v
          //                                 ...

          // clang-format off
          // For each subgraph, do the following
          // 1. Sparsify the `Const`, creating two `Const`, for hashtable
          // key/val.
          // 2. Create a `InitializeTable` op connecting to the above 2 `Const`.
          // 3. Create a `HashTable` op connecting to `InitializeTable` op.
          // 4. Replace the `Gather` with a `LookupTableFind` op.
          // 5. Connect the `LookupTableFind` with
          //    a. `HashTable`
          //    b. `Gather`'s ids input
          //    c. a `default_val` arg, valued at 0
          // clang-format on
          const NodeDef& gather_node = match.node;

          // GatherV2 adds an "axis" parameter. sparsify_gather only supports
          // axis 0 gathers.
          if (gather_node.op() == "GatherV2") {
            // Per the OpTypePattern, the 3rd input to Gather must be a Const.
            const NodeDef& axis_node = match.inputs[2].node;

            Tensor axis_t;
            TF_RETURN_IF_ERROR(GetNodeAttr(axis_node, "value", &axis_t));
            int64 axis = 0;
            if (axis_t.dtype() == DT_INT32) {
              axis = axis_t.scalar<int32>()();
            } else if (axis_t.dtype() == DT_INT64) {
              axis = axis_t.scalar<int64>()();
            } else {
              return tensorflow::errors::FailedPrecondition(
                  "Gather axis was not int32 or int64.");
            }

            if (axis != 0) {
              return tensorflow::errors::FailedPrecondition(
                  "Transform only applicable to subgraph with GatherV2 over "
                  "axis 0. Found axis ",
                  axis, ".");
            }
          }

          const NodeDef& weights_node = match.inputs[0].inputs[0].node;

          DataType data_type;
          TF_RETURN_IF_ERROR(GetNodeAttr(weights_node, "dtype", &data_type));
          if (data_type != DT_FLOAT) {
            return tensorflow::errors::FailedPrecondition(
                "Transform only applicable to subgraph with 'Const',"
                "'Variable', or 'VariableV2' of dtype "
                "'DT_FLOAT'. Found '" +
                    weights_node.op() + "' with name '",
                weights_node.name(), "' and dtype '", data_type, "'.");
          }

          Tensor weight;
          if (weights_node.op() == "Const") {
            weight = GetNodeTensorAttr(weights_node, "value");
          } else {
            TF_RETURN_IF_ERROR(ReadTensorFromCheckpoint(
                weights_node.name(), ckpt_reader,
                (*shapes_and_slices)[weights_node.name()], &weight));
          }
          // Add both both weight and identity node names.
          removed_node_names.push_back(weights_node.name());
          removed_node_names.push_back(match.inputs[0].node.name());
          for (auto input_node : match.inputs[0].node.input()) {
            auto parsed_input = StringReplace(input_node, "^", "", true);
            refs[parsed_input]--;
          }
          Tensor indices_tensor;
          Tensor values_tensor;
          TF_RETURN_IF_ERROR(
              SparsifyWeights(weight, &indices_tensor, &values_tensor));

          // indices and values of sparsified `Const`
          DataType key_dtype = DT_INT64;
          NodeDef indices_node;
          CreateConstNode(indices_tensor,
                          StrCat(weights_node.name(), "/indices"),
                          &indices_node);
          SetNodeAttr("dtype", key_dtype, &indices_node);

          NodeDef values_node;
          CreateConstNode(values_tensor, StrCat(weights_node.name(), "/values"),
                          &values_node);
          SetNodeAttr("dtype", data_type, &values_node);

          // HashTable node
          NodeDef hashtable_node;
          hashtable_node.set_op("HashTable");
          hashtable_node.set_name(StrCat(weights_node.name(), "/HashTable"));
          SetNodeAttr("key_dtype", key_dtype, &hashtable_node);
          SetNodeAttr("value_dtype", data_type, &hashtable_node);

          // InitializeTable node
          NodeDef init_table_node;
          init_table_node.set_op("InitializeTable");
          init_table_node.set_name(
              StrCat(weights_node.name(), "/InitializeTable"));
          SetNodeAttr("Tkey", key_dtype, &init_table_node);
          SetNodeAttr("Tval", data_type, &init_table_node);
          init_table_node_names.push_back(init_table_node.name());

          // LookupTableFind node
          NodeDef lookup_node;
          lookup_node.set_op("LookupTableFind");
          lookup_node.set_name(StrCat(gather_node.name(), "/LookupTableFind"));
          SetNodeAttr("Tin", key_dtype, &lookup_node);
          SetNodeAttr("Tout", data_type, &lookup_node);

          // Default return value of hashtable lookup
          Tensor zero_tensor(data_type, TensorShape({}));
          zero_tensor.flat<float>()(0) = 0.0;
          NodeDef default_value_node;
          CreateConstNode(zero_tensor, StrCat(gather_node.name(), "/Const"),
                          &default_value_node);
          SetNodeAttr("dtype", data_type, &default_value_node);

          // ExpandDims argument
          Tensor dim_idx(DT_INT32, TensorShape({}));
          dim_idx.flat<int32>()(0) = -1;
          NodeDef dim_idx_node;
          dim_idx_node.set_op("Const");
          dim_idx_node.set_name(
              StrCat(gather_node.name(), "/ExpandDims/Const"));
          SetNodeAttr("value", dim_idx, &dim_idx_node);
          SetNodeAttr("dtype", DT_INT32, &dim_idx_node);

          // ExpandDims node
          NodeDef expand_dims_node;
          expand_dims_node.set_op("ExpandDims");
          // Reuse gather_node's name so not to change dependent's inputs
          expand_dims_node.set_name(gather_node.name());
          SetNodeAttr("T", data_type, &expand_dims_node);

          // Connect nodes
          AddNodeInput(hashtable_node.name(), &init_table_node);
          refs[hashtable_node.name()]++;
          AddNodeInput(indices_node.name(), &init_table_node);
          refs[indices_node.name()]++;
          AddNodeInput(values_node.name(), &init_table_node);
          refs[values_node.name()]++;

          AddNodeInput(hashtable_node.name(), &lookup_node);
          refs[hashtable_node.name()]++;
          AddNodeInput(gather_node.input(1), &lookup_node);
          refs[gather_node.input(1)]++;
          AddNodeInput(default_value_node.name(), &lookup_node);
          refs[default_value_node.name()]++;

          AddNodeInput(lookup_node.name(), &expand_dims_node);
          refs[lookup_node.name()]++;
          AddNodeInput(dim_idx_node.name(), &expand_dims_node);
          refs[dim_idx_node.name()]++;

          // Copy 'ids' input of original 'Gather'
          new_nodes->push_back(match.inputs[1].node);
          new_nodes->push_back(indices_node);
          new_nodes->push_back(values_node);
          new_nodes->push_back(hashtable_node);
          new_nodes->push_back(init_table_node);
          new_nodes->push_back(lookup_node);
          new_nodes->push_back(default_value_node);
          new_nodes->push_back(dim_idx_node);
          new_nodes->push_back(expand_dims_node);

          return Status::OK();
        },
        {true}, &replaced_graph_def));

    NodeDef* init_op = nullptr;
    for (int i = 0; i < replaced_graph_def.node_size(); i++) {
      if (replaced_graph_def.node(i).name() == group_init_node &&
          replaced_graph_def.node(i).op() == "NoOp") {
        init_op = replaced_graph_def.mutable_node(i);
        break;
      }
    }
    if (!init_op) {
      // Init node
      init_op = replaced_graph_def.mutable_node()->Add();
      init_op->set_op("NoOp");
      init_op->set_name(group_init_node);
    }
    for (const string& name : init_table_node_names) {
      // Add control dependence from init_table_node to group_deps_node
      AddNodeInput(StrCat("^", name), init_op);
      refs[name]++;
    }

    // Erase inputs and outputs as they are not considered for deletion.
    for (const auto& output : context.output_names) {
      refs.erase(output);
    }

    for (const auto& input : context.input_names) {
      refs.erase(input);
    }

    // Add nodes with a reference count of 0 for deletion.
    for (auto entry : refs) {
      if (entry.second == 0) {
        removed_node_names.push_back(entry.first);
      }
    }

    while (!removed_node_names.empty()) {
      auto name = removed_node_names.back();
      removed_node_names.pop_back();

      int i = 0;
      while (i < replaced_graph_def.node_size()) {
        // Revisit this to see if we can safely remove RestoreV2 nodes.
        if ((replaced_graph_def.node(i).name() == name) &&
            (replaced_graph_def.node(i).op() != "RestoreV2")) {
          for (const auto& input : replaced_graph_def.node(i).input()) {
            auto parsed_input = StringReplace(input, "^", "", true);
            refs[parsed_input] -= 1;
            if (refs[parsed_input] == 0) {
              removed_node_names.push_back(parsed_input);
            }
          }
          replaced_graph_def.mutable_node()->SwapElements(
              i, replaced_graph_def.node_size() - 1);
          replaced_graph_def.mutable_node()->RemoveLast();
          continue;
        }
        int j = 0;
        bool deleted_inputs = false;
        while (j < replaced_graph_def.node(i).input_size()) {
          if (replaced_graph_def.node(i).input(j) == name ||
              replaced_graph_def.node(i).input(j) == ("^" + name)) {
            TF_RETURN_IF_ERROR(
                RemoveInputAtIndex(replaced_graph_def.mutable_node(i), j));
            deleted_inputs = true;
            continue;
          }
          j++;
        }
        if (deleted_inputs) {
          if (replaced_graph_def.node(i).op() == "ConcatV2") {
            if (replaced_graph_def.node(i).input_size() > 2) {
              SetNodeAttr("N", replaced_graph_def.node(i).input_size() - 1,
                          replaced_graph_def.mutable_node(i));
            } else if (replaced_graph_def.node(i).input_size() == 2) {
              if (refs[replaced_graph_def.node(i).input(1)] != 1) {
                return errors::Internal(
                    "Expect axis tensor of ConcatV2 node to only be referenced "
                    "once.");
              }
              refs[replaced_graph_def.node(i).input(1)] -= 1;
              removed_node_names.push_back(replaced_graph_def.node(i).input(1));
              replaced_graph_def.mutable_node(i)->mutable_input()->RemoveLast();
              replaced_graph_def.mutable_node(i)->mutable_attr()->erase("N");
              replaced_graph_def.mutable_node(i)->set_op("Identity");
            } else {
              return errors::Internal(
                  "ConcatV2 should have at least two elements");
            }
          }
          if ((replaced_graph_def.node(i).op() == "Assign" ||
               replaced_graph_def.node(i).op() == "Reshape" ||
               replaced_graph_def.node(i).op() == "Equal" ||
               replaced_graph_def.node(i).op() == "Mean" ||
               replaced_graph_def.node(i).op() == "ScalarSummary") &&
              replaced_graph_def.node(i).input_size() == 1) {
            removed_node_names.push_back(replaced_graph_def.node(i).name());
          }
          if (!replaced_graph_def.node(i).input_size()) {
            removed_node_names.push_back(replaced_graph_def.node(i).name());
          }
        }
        i++;
      }
    }
    current_graph_def = replaced_graph_def;
  } while (any_match_found);
  *output_graph_def = current_graph_def;
  return Status::OK();
}

Status SparsifyGather(const GraphDef& input_graph_def,
                      const TransformFuncContext& context,
                      GraphDef* output_graph_def) {
  // clang-format off
  const OpTypePattern gather_pattern =
    {"Gather",
     {
       {"Identity",
        {
          {"Const|Variable|VariableV2"}
        }
       },
       {"*"},
     }
    };
  const OpTypePattern gather_v2_pattern =
    {"GatherV2",
      {
        {"Identity",
          {
            {"Const|Variable|VariableV2"}
          }
        },
        {"*"},
        // GatherV2's axis must be constant.
        {"Const"},
      }
    };
  // clang-format on

  GraphDef cleaned_input_graph_def;
  RemoveAttributes(input_graph_def, {"_output_shapes"},
                   &cleaned_input_graph_def);

  GraphDef temp_output;

  std::unique_ptr<BundleReader> ckpt_reader;
  TF_RETURN_IF_ERROR(InitializeCheckpointReader(context, &ckpt_reader));

  std::unique_ptr<std::unordered_map<string, string> > shapes_and_slices;
  TF_RETURN_IF_ERROR(
      ObtainVariableInfo(cleaned_input_graph_def, &shapes_and_slices));

  TF_RETURN_IF_ERROR(SparsifyGatherInternal(
      cleaned_input_graph_def, shapes_and_slices, context, gather_pattern,
      ckpt_reader, &temp_output));

  TF_RETURN_IF_ERROR(SparsifyGatherInternal(temp_output, shapes_and_slices,
                                            context, gather_v2_pattern,
                                            ckpt_reader, output_graph_def));

  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("sparsify_gather", SparsifyGather);

}  // namespace graph_transforms
}  // namespace tensorflow
