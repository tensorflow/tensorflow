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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
using strings::StrCat;
namespace graph_transforms {
namespace {

// Sparsify Tensor of shape [N, 1]. Return the indices and values vectors for
// non-zero tensor content.
Status SparsifyWeights(const Tensor& tensor, Tensor* indices_tensor,
                       Tensor* values_tensor) {
  if (tensor.dims() != 2 || tensor.dim_size(1) != 1) {
    return tensorflow::errors::FailedPrecondition(
        "Transform only applicable to subgraph with 'Const' with "
        "tensor of shpae [N, 1]. But instead get shape ",
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
}  // namespace

Status SparsifyGather(const GraphDef& input_graph_def,
                      const TransformFuncContext& context,
                      GraphDef* output_graph_def) {
  GraphDef current_graph_def = input_graph_def;
  bool any_match_found = false;
  // The subgraphs may have overlapping components, therefore GraphMatcher
  // doesn't return all subgraphs in one round -- this has to be multi-round
  // update.
  do {
    any_match_found = false;
    GraphDef replaced_graph_def = current_graph_def;
    std::vector<string> init_table_node_names;

    TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
        current_graph_def,  // clang-format off
      {"Gather",
        {
          {"Identity",
            {
              {"Const"}
            }
          },
          {"*"},
        }
      },  // clang-format on
        [&any_match_found, &init_table_node_names](
            const NodeMatch& match, const std::set<string>& input_nodes,
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
          const NodeDef& const_node = match.inputs[0].inputs[0].node;

          DataType data_type;
          TF_RETURN_IF_ERROR(GetNodeAttr(const_node, "dtype", &data_type));
          if (data_type != DT_FLOAT) {
            return tensorflow::errors::FailedPrecondition(
                "Transform only applicable to subgraph with 'Const' of dtype "
                "'DT_FLOAT'. Found 'Const' with name '",
                const_node.name(), "' and dtype '", data_type, "'.");
          }
          Tensor weight = GetNodeTensorAttr(const_node, "value");
          Tensor indices_tensor;
          Tensor values_tensor;
          TF_RETURN_IF_ERROR(
              SparsifyWeights(weight, &indices_tensor, &values_tensor));

          // indices and values of sparsified `Const`
          DataType key_dtype = DT_INT64;
          NodeDef indices_node;
          CreateConstNode(indices_tensor, StrCat(const_node.name(), "/indices"),
                          &indices_node);
          SetNodeAttr("dtype", key_dtype, &indices_node);

          NodeDef values_node;
          CreateConstNode(values_tensor, StrCat(const_node.name(), "/values"),
                          &values_node);
          SetNodeAttr("dtype", data_type, &values_node);

          // HashTable node
          NodeDef hashtable_node;
          hashtable_node.set_op("HashTable");
          hashtable_node.set_name(StrCat(const_node.name(), "/HashTable"));
          SetNodeAttr("key_dtype", key_dtype, &hashtable_node);
          SetNodeAttr("value_dtype", data_type, &hashtable_node);

          // InitializeTable node
          NodeDef init_table_node;
          init_table_node.set_op("InitializeTable");
          init_table_node.set_name(
              StrCat(const_node.name(), "/InitializeTable"));
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
          AddNodeInput(indices_node.name(), &init_table_node);
          AddNodeInput(values_node.name(), &init_table_node);

          AddNodeInput(hashtable_node.name(), &lookup_node);
          AddNodeInput(gather_node.input(1), &lookup_node);
          AddNodeInput(default_value_node.name(), &lookup_node);

          AddNodeInput(lookup_node.name(), &expand_dims_node);
          AddNodeInput(dim_idx_node.name(), &expand_dims_node);

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
      if (replaced_graph_def.node(i).name() == "group_deps" &&
          replaced_graph_def.node(i).op() == "NoOp") {
        if (init_op != nullptr) {
          return tensorflow::errors::FailedPrecondition(
              "Multiple nodes with name: 'group_deps' and type: 'NoOp'.");
        }
        init_op = replaced_graph_def.mutable_node(i);
      }
    }
    if (!init_op) {
      return tensorflow::errors::FailedPrecondition(
          "No node found with name: 'group_deps' and type: 'NoOp'");
    }
    for (const string& name : init_table_node_names) {
      // Add control dependence from init_table_node to group_deps_node
      AddNodeInput(StrCat("^", name), init_op);
    }
    current_graph_def = replaced_graph_def;
  } while (any_match_found);
  *output_graph_def = current_graph_def;
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("sparsify_gather", SparsifyGather);

}  // namespace graph_transforms
}  // namespace tensorflow
