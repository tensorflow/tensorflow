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

#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

using tensorflow::strings::StrCat;

namespace tensorflow {
namespace grappler {
namespace {

template <typename T>
bool AreInversePermutations(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int i = 0; i < a.size(); ++i) {
    if (a[b[i]] != i) {
      return false;
    }
  }
  return true;
}

// Extract values from a Const op to `values`. Returns true if succeeds.
template <typename T>
bool ValuesFromConstNode(const NodeDef& node, std::vector<T>* values) {
  if (node.op() != "Const") {
    return false;
  }

  if (node.attr().at("dtype").type() != DataTypeToEnum<T>::value) {
    return false;
  }

  // TensorProto represents the content of the tensor in either <type>_val or
  // tensor_content.
  const TensorProto& tensor = node.attr().at("value").tensor();
  typename checkpoint::SaveTypeTraits<T>::RepeatedField* tensor_values =
      checkpoint::MutableTensorProtoData<T>(const_cast<TensorProto*>(&tensor));

  if (!tensor_values->empty() && tensor.has_tensor_shape()) {
    // When tensor_shape is set, theoretically the representation of the data
    // could be compressed. So, before copying values to the returned vector,
    // make sure no compression happens.
    const TensorShapeProto& shape = tensor.tensor_shape();
    if (shape.dim_size() == 1 && shape.dim(0).size() == tensor_values->size()) {
      values->insert(values->end(), tensor_values->begin(),
                     tensor_values->end());
      return true;
    }
  }

  const auto tensor_content_size = tensor.tensor_content().size();
  if (tensor_content_size > 0) {
    CHECK_EQ(0, tensor_content_size % sizeof(T))
        << "tensor_content_size (" << tensor_content_size
        << ") is not a multiple of " << sizeof(T);
    values->resize(tensor_content_size / sizeof(T));
    port::CopyToArray(tensor.tensor_content(),
                      reinterpret_cast<char*>(values->data()));
    return true;
  }

  return false;
}

template <typename T>
bool IsInnerMatrixTranspose(const std::vector<T>& perm) {
  const T n = perm.size();
  if (n < 2) {
    return false;
  }
  for (T i = 0; i < n - 2; ++i) {
    if (perm[i] != i) {
      return false;
    }
  }
  return perm[n - 1] == n - 2 && perm[n - 2] == n - 1;
}

bool IsInnerMatrixTransposeNode(const NodeDef& transpose_node,
                                const NodeMap* node_map) {
  if (transpose_node.op() != "Transpose" &&
      transpose_node.op() != "ConjugateTranspose") {
    return false;
  }
  const NodeDef* perm_node = node_map->GetNode(transpose_node.input(1));
  std::vector<int> perm32;
  if (ValuesFromConstNode(*perm_node, &perm32)) {
    return IsInnerMatrixTranspose(perm32);
  }
  std::vector<int64> perm64;
  if (ValuesFromConstNode(*perm_node, &perm64)) {
    return IsInnerMatrixTranspose(perm64);
  }
  return false;
}

bool MaybeAddControlInput(const string& new_input, NodeDef* node,
                          GraphDef* graph, NodeMap* node_map) {
  bool already_exists = false;
  for (const string& input : node->input()) {
    if (input == new_input || AsControlDependency(input) == new_input) {
      already_exists = true;
      break;
    }
  }
  if (!already_exists) {
    const string ctrl_dep =
        ConstantFolding::AddControlDependency(new_input, graph, node_map);
    node->add_input(ctrl_dep);
    node_map->AddOutput(NodeName(new_input), node->name());
  }
  return !already_exists;
}

int CopyControlInputs(const NodeDef& from, NodeDef* to, GraphDef* graph,
                      NodeMap* node_map) {
  int num_copied = 0;
  for (const string& input : from.input()) {
    if (IsControlInput(input) &&
        MaybeAddControlInput(input, to, graph, node_map)) {
      ++num_copied;
    }
  }
  return num_copied;
}

void SetDataTypeToAttr(DataType dtype, const string& attr_name, NodeDef* node) {
  (*node->mutable_attr())[attr_name].set_type(dtype);
}

void FlipBooleanAttr(const string& attr_name, NodeDef* node) {
  const bool old_value =
      !node->attr().count(attr_name) ? false : node->attr().at(attr_name).b();
  (*node->mutable_attr())[attr_name].set_b(!old_value);
}

string SourceDataTypeAttrName(const NodeDef& node) {
  if (node.op() == "Bitcast") {
    return "T";
  } else if (node.op() == "Cast") {
    return "SrcT";
  } else {
    LOG(FATAL) << "SourceDataTypeAttrName not implemented for op " << node.op();
  }
}

string DestinationDataTypeAttrName(const NodeDef& node) {
  if (node.op() == "Bitcast") {
    return "type";
  } else if (node.op() == "Cast") {
    return "DstT";
  } else {
    LOG(FATAL) << "DestinationDataTypeAttrName not implemented for op "
               << node.op();
  }
}

DataType GetSourceDataType(const NodeDef& node) {
  return GetDataTypeFromAttr(node, SourceDataTypeAttrName(node));
}

DataType GetDestinationDataType(const NodeDef& node) {
  return GetDataTypeFromAttr(node, DestinationDataTypeAttrName(node));
}

void SetSourceDataType(DataType dtype, NodeDef* node) {
  SetDataTypeToAttr(dtype, SourceDataTypeAttrName(*node), node);
}

bool IsNumberType(DataType dtype) { return kNumberTypes.Contains(dtype); }

const char kOutputShapesAttr[] = "_output_shapes";

PartialTensorShape GetInputShape(const string& input, const NodeMap& node_map) {
  int output_pos;
  string node_name = ParseNodeName(input, &output_pos);
  const NodeDef* input_node = node_map.GetNode(node_name);
  auto attr = input_node->attr();
  if (attr.find(kOutputShapesAttr) == attr.end()) {
    return PartialTensorShape();  // unknown shape
  } else {
    return attr.at(kOutputShapesAttr).list().shape(output_pos);
  }
}

bool ShapesEqual(const string& input_x, const string& input_y,
                 const NodeMap& node_map) {
  PartialTensorShape x_shape = GetInputShape(input_x, node_map);
  PartialTensorShape y_shape = GetInputShape(input_y, node_map);
  if (x_shape.unknown_rank() || y_shape.unknown_rank() ||
      x_shape.dims() != y_shape.dims()) {
    return false;
  }
  for (int i = 0; i < x_shape.dims(); ++i) {
    if (x_shape.dim_size(i) == -1 || y_shape.dim_size(i) == -1 ||
        x_shape.dim_size(i) != y_shape.dim_size(i)) {
      return false;
    }
  }
  return true;
}

// Returns whether `reshape` is an identity op. The tensor that `reshape`
// reshapes is the `output_pos`-th output of node `input`.
bool ReshapeIsIdentity(const NodeDef& reshape, const NodeDef& input,
                       const int output_pos) {
  if (!reshape.attr().count(kOutputShapesAttr) ||
      !input.attr().count(kOutputShapesAttr)) {
    return false;
  }

  PartialTensorShape src_shape(
      input.attr().at(kOutputShapesAttr).list().shape(output_pos));
  PartialTensorShape dst_shape(
      reshape.attr().at(kOutputShapesAttr).list().shape(0));
  if (src_shape.unknown_rank() || dst_shape.unknown_rank()) {
    return false;
  }

  if (!dst_shape.IsCompatibleWith(src_shape)) {
    return false;
  }

  // Returns false when src_shape or dst_shape has >=2 dimensions with unknown
  // sizes.
  auto num_unknown_dim_sizes = [](const PartialTensorShape& partial_shape) {
    auto dim_sizes = partial_shape.dim_sizes();
    return std::count(dim_sizes.begin(), dim_sizes.end(), -1);
  };
  int src_num_unknown_dim_sizes = num_unknown_dim_sizes(src_shape);
  int dst_num_unknown_dim_sizes = num_unknown_dim_sizes(dst_shape);
  if (src_num_unknown_dim_sizes > 1 || dst_num_unknown_dim_sizes > 1) {
    return false;
  }

  // Now, src_shape and dst_shape have at most one dimension with unknown
  // sizes, and are compatible. Therefore, the reshape is a no-op when
  //
  // 1. at least one of them is fully-defined, or
  // 2. both are partially defined and the -1 appears on the same dimension,
  //    i.e., IsIdenticalTo returns true.
  if (src_num_unknown_dim_sizes == 1 && dst_num_unknown_dim_sizes == 1) {
    return dst_shape.IsIdenticalTo(src_shape);
  }

  return true;
}

NodeDef* GetTailOfValuePreservingChain(
    const NodeDef& node, const NodeMap& node_map,
    const std::unordered_set<string>& nodes_to_preserve) {
  auto is_value_preserving_non_branching = [&](const NodeDef& node) {
    return IsValuePreserving(node) &&
           NumNonControlOutputs(node, node_map) == 1 &&
           nodes_to_preserve.count(node.name()) == 0;
  };
  return GetTailOfChain(node, node_map, /*follow_control_input=*/false,
                        is_value_preserving_non_branching);
}

// Context passed to each arithmetic optimizer stage. Optimizer stage is
// responsible for updating the node map for all added or deleted nodes, to keep
// it consistent with optimized graph.
struct ArithmeticOptimizerContext {
  ArithmeticOptimizerContext(
      const std::unordered_set<string>* nodes_to_preserve,
      GraphDef* optimized_graph, NodeMap* node_map,
      SetVector<NodeDef*>* nodes_to_simplify)
      : nodes_to_preserve(nodes_to_preserve),
        optimized_graph(optimized_graph),
        node_map(node_map),
        nodes_to_simplify(nodes_to_simplify) {}

  const std::unordered_set<string>* nodes_to_preserve;
  GraphDef* optimized_graph;
  NodeMap* node_map;
  SetVector<NodeDef*>* nodes_to_simplify;
};

// Base class for single arithmetic optimization: e.g. Bitcast optimization,
// AddOps optimization, etc...
class ArithmeticOptimizerStage {
 public:
  explicit ArithmeticOptimizerStage(ArithmeticOptimizerContext ctx)
      : ctx_(ctx) {}
  virtual ~ArithmeticOptimizerStage() = default;

  // Check if we should try to simplify node. Returning true doesn't
  // guarantee that node will be simplified.
  //
  // Should implement just a basic sanity check, without any expensive graph
  // traversals.
  virtual bool IsSupported(const NodeDef* node) const = 0;

  // Try to simplify the given node. If successfully simplified a given node,
  // return a name of a new simplified version using output parameter.
  //
  // Consumers of an old node's outputs will be automatically re-wired to
  // consume outputs of a new simplified node.
  //
  // Return error status only if some precondition is failed, or got an
  // incorrect graph. In every other case return Status:OK(), even if didn't
  // simplify anything.
  //
  // A simplified node will be always considered for further optimization and
  // will be automatically added to the optimization queue. If a simplified node
  // has the same name as original node it has to be explicitly added to the
  // optimization queue for second pass.
  virtual Status TrySimplify(const NodeDef* node,
                             string* simplified_node_name) = 0;

 protected:
  // Simplification graph rewrite can create additional nodes that are inputs
  // to final simplified node, they can be also added to the arithmetic
  // optimizer queue for further optimization.
  void AddToOptimizationQueue(NodeDef* node) {
    ctx_.nodes_to_simplify->PushBack(node);
  }

  // Get a node by input name from a node map. Return a error if node was not
  // found.
  Status GetInputNode(const string& input, NodeDef** node) const {
    string node_name = NodeName(input);
    NodeDef* node_by_name = ctx_.node_map->GetNode(node_name);
    if (node_by_name == nullptr) {
      return errors::FailedPrecondition("Node ", node_name,
                                        " doesn't exists in a node map");
    }
    *node = node_by_name;
    return Status::OK();
  }

  // Get input shape from a node map. If node doesn't exists return unknown
  // shape.
  PartialTensorShape GetInputShape(const string& input) const {
    int position;
    string node_name = ParseNodeName(input, &position);
    NodeDef* node;
    Status node_status = GetInputNode(node_name, &node);
    if (!node_status.ok()) {
      return PartialTensorShape();  // unknown shape
    }
    auto attr = node->attr();
    if (attr.find(kOutputShapesAttr) == attr.end()) {
      return PartialTensorShape();  // unknown shape
    } else {
      return attr.at(kOutputShapesAttr).list().shape(position);
    }
  }

  ArithmeticOptimizerContext ctx_;
};

// Rewrite a tree of Add/AddN with a single AddN operation, consuming all the
// original inputs of absorbed nodes.
//
// All nodes in a Add/AddN subgraph must have fully specified and identical
// shape. All nodes must have the same device placement.
//
// Example:
//                AddN_1
//             /    |    \
//          Add_1   z   Add_2       -> AddN(z, y, z, w, q, e)
//          /  \        /  \
//         x    y      w    Add_3
//                          / \
//                         q   e
class AddOpsRewriteStage : public ArithmeticOptimizerStage {
 public:
  explicit AddOpsRewriteStage(ArithmeticOptimizerContext ctx)
      : ArithmeticOptimizerStage(ctx), rewritten_nodes_() {}

  ~AddOpsRewriteStage() override = default;

  // Check if a node can become a root of AddOpsGroup
  bool IsSupported(const NodeDef* node) const override {
    // check basic preconditions
    if (!IsRewritable(node)) {
      return false;
    }
    // and must have fully defined shape
    // TODO(ezhulenev): support partially defined shapes, when we can prove that
    // unknown dimensions in the rewritten subgraph are the same.
    PartialTensorShape shape = GetInputShape(node->name());
    if (!shape.IsFullyDefined()) {
      return false;
    }
    // and must have inputs of fully defined shape identical to the output
    // TODO(ezhulenev): relax this condition to support equal unknown dimensions
    return HasAllInputsOfIdenticalShape(*node, shape);
  }

  Status TrySimplify(const NodeDef* node,
                     string* simplified_node_name) override {
    CHECK(IsSupported(node))
        << "Node " << node->name()
        << " is not supported by add ops group optimizer step";
    AddOpsGroup group;
    TF_RETURN_IF_ERROR(CreateAddOpsGroup(node, &group));

    if (!group.absorbed_nodes.empty()) {
      *simplified_node_name = RewriteAddOpsGroup(group);
    }

    return Status::OK();
  }

 private:
  // Holds together an add ops subgraph that we want to rewrite together.
  //
  // For the graph above the AddOpsGroup will be:
  //   root_node: AddN_1
  //   absorbed_nodes: [Add_1, Add_2]
  //   input_nodes: [x, y, z, w, q, e]
  struct AddOpsGroup {
    const NodeDef* root_node;
    PartialTensorShape root_shape;
    // Add/AddN operations below the root level that were absorbed by this group
    std::vector<NodeDef*> absorbed_nodes;
    // Inputs of absorbed nodes that will be forwarded to rewritten AddN node
    std::vector<string> inputs;
  };

  // Check if all inputs are fully defined and identical to expected shape
  bool HasAllInputsOfIdenticalShape(const NodeDef& node,
                                    const PartialTensorShape& shape) const {
    const AddOpsRewriteStage* self = this;
    return std::all_of(node.input().begin(), node.input().end(),
                       [self, &shape](const string& input) {
                         auto input_shape = self->GetInputShape(input);
                         return input_shape.IsFullyDefined() &&
                                input_shape.IsIdenticalTo(shape);
                       });
  }

  // TODO(ezhulenev): use GraphRewriter?
  bool IsDrivenByControlDependency(const NodeDef& node) const {
    return std::any_of(node.input().begin(), node.input().end(),
                       IsControlInput);
  }

  // TODO(ezhulenev): use GraphRewriter?
  bool DrivesControlDependency(const NodeDef& node) const {
    int position;
    for (const NodeDef* output : ctx_.node_map->GetOutputs(node.name())) {
      for (int i = 0; i < output->input_size(); ++i) {
        auto input = output->input(i);
        string name = ParseNodeName(input, &position);
        if (name == node.name() && /*control input*/ position < 0) {
          return true;
        }
      }
    }
    return false;
  }

  // Check if a node can be absorbed by current AddOpsGroup
  bool IsAbsorbableByAddOpsGroup(const string& name, const AddOpsGroup& group) {
    NodeDef* node;
    Status node_status = GetInputNode(name, &node);
    if (!node_status.ok()) {
      return false;
    }

    PartialTensorShape shape = GetInputShape(name);
    CHECK(shape.IsIdenticalTo(group.root_shape))
        << "Cannot absorb a node of incompatible shape";

    // check basic preconditions
    if (!IsRewritable(node)) {
      return false;
    }
    // with a single output consumer (presumably if we reach this node from
    // previously absorbed or a root node, it means that this node is not used
    // as an input to any other op, outside of the group)
    if (ctx_.node_map->GetOutputs(node->name()).size() != 1) {
      return false;
    }
    // must be on the same device as a root node
    if (node->device() != group.root_node->device()) {
      return false;
    }
    // All input shapes must be fully defined and equal to the node shape
    return HasAllInputsOfIdenticalShape(*node, shape);
  }

  // Node requirements both for a root node and an absorbed node
  bool IsRewritable(const NodeDef* node) const {
    // only Add or AddN can be a root node
    // TODO(ezhulenev): check if AccumulateNV2 can be supported too
    if (!IsAdd(*node) && !IsAddN(*node)) {
      return false;
    }
    // it must not be in a preserve set
    if (ctx_.nodes_to_preserve->find(node->name()) !=
        ctx_.nodes_to_preserve->end()) {
      return false;
    }
    // it must not be a node created or absorbed by previous iteration
    if (rewritten_nodes_.find(node->name()) != rewritten_nodes_.end()) {
      return false;
    }
    // should not drive or be driven by control dependency
    // TODO(ezhulenev): relax this condition for root node
    return !(IsDrivenByControlDependency(*node) ||
             DrivesControlDependency(*node));
  }

  // Create an AddOpsGroup with a root in a given node
  Status CreateAddOpsGroup(const NodeDef* root_node, AddOpsGroup* group) {
    group->root_node = root_node;
    group->root_shape = GetInputShape(root_node->name());

    group->absorbed_nodes.reserve(root_node->input_size());
    for (int i = 0; i < root_node->input_size(); ++i) {
      TF_RETURN_IF_ERROR(AbsorbInputByAddOpsGroup(root_node->input(i), group));
    }

    return Status::OK();
  }

  Status AbsorbInputByAddOpsGroup(const string& input, AddOpsGroup* group) {
    NodeDef* node;
    TF_RETURN_IF_ERROR(GetInputNode(input, &node));

    if (IsAbsorbableByAddOpsGroup(input, *group)) {
      group->absorbed_nodes.push_back(node);
      for (int i = 0; i < node->input_size(); ++i) {
        TF_RETURN_IF_ERROR(AbsorbInputByAddOpsGroup(node->input(i), group));
      }
    } else {
      // If node can't be absorbed, add it to AddOpsGroup input
      group->inputs.push_back(input);
    }
    return Status::OK();
  }

  const std::pair<string, string> ParseNodeScopeAndName(const string& name) {
    auto pos = name.find_last_of("/");
    if (pos == string::npos) {
      return {"", name};
    } else {
      return {name.substr(0, pos), name.substr(pos + 1)};
    }
  }

  // New node for AddOpsGroup is added to the same scope as a root_node. All
  // absorbed nodes are stripped of their scope, and only names are used in a
  // new node name.
  //
  // Example: AddOpsGroup(root="a/b/c/Add_2", absorbed=["d/Add_1", "e/Add"])
  //          node_name="a/b/c/AddOpsGroup_Add_2_Add_1_Add
  string AddOpsGroupName(const AddOpsGroup& group) {
    CHECK_NOTNULL(group.root_node);
    string node_name;

    auto root_node = ParseNodeScopeAndName(group.root_node->name());
    auto root_scope = root_node.first;
    auto root_name = root_node.second;
    if (!root_scope.empty()) {
      strings::StrAppend(&node_name, root_scope, "/");
    }

    strings::StrAppend(&node_name, kArithmeticOptimizer, "/", "AddOpsGroup_",
                       root_name);
    for (const NodeDef* absorbed : group.absorbed_nodes) {
      auto absorbed_node = ParseNodeScopeAndName(absorbed->name());
      strings::StrAppend(&node_name, "_", absorbed_node.second);
    }
    return node_name;
  }

  // Create a new node for a AddOpsGroup and return it's name.
  string RewriteAddOpsGroup(const AddOpsGroup& group) {
    CHECK_GT(group.absorbed_nodes.size(), 0)
        << "AddOpsGroup must have non empty absorbed nodes";

    // name for a new node constructed from AddOpsGroup
    string node_name = AddOpsGroupName(group);

    // copy attributes from a root node
    DataType dtype = group.root_node->attr().at("T").type();

    // add new node
    NodeDef* added_node = ctx_.optimized_graph->add_node();
    added_node->set_name(node_name);
    added_node->set_op("AddN");
    added_node->set_device(group.root_node->device());
    (*added_node->mutable_attr())["T"].set_type(dtype);
    (*added_node->mutable_attr())["N"].set_i(group.inputs.size());

    ctx_.node_map->AddNode(node_name, added_node);
    for (string input : group.inputs) {
      ctx_.node_map->AddOutput(input, node_name);
      added_node->add_input(std::move(input));
    }

    VLOG(1) << "Absorbed " << group.absorbed_nodes.size()
            << " Add/AddN nodes from the graph";

    // keep track of nodes that were created or absorbed as a part of rewrite
    rewritten_nodes_.insert(node_name);
    for (const NodeDef* absorbed : group.absorbed_nodes) {
      rewritten_nodes_.insert(absorbed->name());
    }

    return node_name;
  }

  // keep nodes that were added or absorbed as a part of AddOpsGroup rewrite
  std::unordered_set<string> rewritten_nodes_;
};

}  // namespace

class UniqueNodes {
 public:
  NodeDef* FindOrAddRepresentative(NodeDef* node) {
    std::size_t sig = ComputeSignature(*node);
    std::vector<NodeDef*>& candidates = rep_[sig];
    for (auto& candidate : candidates) {
      if (SameNode(*candidate, *node)) {
        return candidate;
      }
    }
    candidates.push_back(node);
    return node;
  }

 private:
  std::size_t ComputeSignature(const NodeDef& node) const;
  bool SameNode(const NodeDef& node1, const NodeDef& node2) const;

  std::unordered_map<std::size_t, std::vector<NodeDef*>> rep_;
};

std::size_t UniqueNodes::ComputeSignature(const NodeDef& node) const {
  std::size_t h = std::hash<string>{}(node.op());
  h ^= std::hash<string>{}(node.device());
  for (const auto& input : node.input()) {
    int pos;
    string node_name = ParseNodeName(input, &pos);
    h ^= std::hash<string>{}(node_name);
    h ^= static_cast<std::size_t>(pos);
  }
  for (const auto& attr : node.attr()) {
    h ^= std::hash<string>{}(attr.first);
    string tmp;
    attr.second.AppendToString(&tmp);
    h ^= std::hash<string>{}(tmp);
  }
  return h;
}

bool UniqueNodes::SameNode(const NodeDef& node1, const NodeDef& node2) const {
  if (node1.op() != node2.op()) {
    return false;
  }
  if (node1.device() != node2.device()) {
    return false;
  }
  if (node1.input_size() != node2.input_size()) {
    return false;
  }
  if (node1.attr_size() != node2.attr_size()) {
    return false;
  }

  // Compare inputs.
  if (IsCommutative(node1)) {
    std::vector<string> inputs1(node1.input().begin(), node1.input().end());
    std::vector<string> inputs2(node2.input().begin(), node2.input().end());
    std::sort(inputs1.begin(), inputs1.end());
    std::sort(inputs2.begin(), inputs2.end());
    return inputs1 == inputs2;
  } else {
    std::vector<string> regular_inputs1;
    std::vector<string> regular_inputs2;
    std::vector<string> ctrl_inputs1;
    std::vector<string> ctrl_inputs2;
    for (int index = 0; index < node1.input_size(); ++index) {
      if (IsControlInput(node1.input(index))) {
        ctrl_inputs1.push_back(node1.input(index));
        ctrl_inputs2.push_back(node2.input(index));
      } else {
        regular_inputs1.push_back(node1.input(index));
        regular_inputs2.push_back(node2.input(index));
      }
    }
    if (regular_inputs1 != regular_inputs2) {
      return false;
    }
    std::sort(ctrl_inputs1.begin(), ctrl_inputs1.end());
    std::sort(ctrl_inputs2.begin(), ctrl_inputs2.end());
    if (ctrl_inputs1 != ctrl_inputs2) {
      return false;
    }
  }

  // Compare attributes.
  for (const auto& attr1 : node1.attr()) {
    auto it = node2.attr().find(attr1.first);
    if (it == node2.attr().end()) {
      return false;
    }
    const auto& attr2 = *it;
    string val1;
    attr1.second.AppendToString(&val1);
    string val2;
    attr2.second.AppendToString(&val2);
    if (val1 != val2) {
      return false;
    }
  }

  return true;
}

NodeDef* ArithmeticOptimizer::AddNode(const NodeDef& node, StringPiece suffix,
                                      bool copy_node) {
  return AddNode(OptimizedNodeName(node, suffix), copy_node ? &node : nullptr);
}

NodeDef* ArithmeticOptimizer::AddNode(const string& name,
                                      const NodeDef* node_to_copy) {
  NodeDef* new_node = optimized_graph_->add_node();
  node_map_->AddNode(NodeName(name), new_node);
  if (node_to_copy != nullptr) {
    *new_node = *node_to_copy;
  }
  new_node->set_name(name);
  return new_node;
}

string ArithmeticOptimizer::OptimizedNodeName(const NodeDef& node,
                                              StringPiece suffix) const {
  return AddPrefixToNodeName(strings::StrCat(node.name(), "_", suffix),
                             kArithmeticOptimizer);
}

bool ArithmeticOptimizer::OptimizedNodeExists(const NodeDef& node,
                                              StringPiece suffix) const {
  return node_map_->NodeExists(OptimizedNodeName(node, suffix));
}

bool ArithmeticOptimizer::CanDedup(const NodeDef& node) const {
  if (nodes_to_preserve_.find(node.name()) != nodes_to_preserve_.end()) {
    return false;
  }
  if (IsEnter(node) || IsExit(node)) {
    return false;
  }
  if (node.device().find("SPU") != string::npos) {
    return false;
  }
  // Workaround for Assert mistakenly being labeled as stateful.
  if (IsAssert(node)) {
    return true;
  }
  return IsFreeOfSideEffect(node);
}

void ArithmeticOptimizer::DedupComputations() {
  bool stop = true;
  std::set<int> duplicates;
  do {
    stop = true;
    UniqueNodes nodes;
    for (int i = 0; i < optimized_graph_->node_size(); ++i) {
      if (duplicates.find(i) != duplicates.end()) {
        continue;
      }
      NodeDef* node = optimized_graph_->mutable_node(i);
      if (!CanDedup(*node)) {
        continue;
      }
      NodeDef* rep = nodes.FindOrAddRepresentative(node);
      if (rep == node) {
        continue;
      }
      const std::set<NodeDef*>& fanouts = node_map_->GetOutputs(node->name());
      for (NodeDef* fanout : fanouts) {
        for (string& name : *fanout->mutable_input()) {
          int position;
          const string nodename = ParseNodeName(name, &position);
          if (nodename == node->name()) {
            // Update name in-place.
            if (position > 0) {
              name = StrCat(rep->name(), ":", position);
            } else if (position == 0) {
              name = rep->name();
            } else {
              name = StrCat("^", rep->name());
            }
            node_map_->AddOutput(rep->name(), fanout->name());
          }
        }
      }
      duplicates.insert(i);
      stop = false;
    }
  } while (!stop);

  // Delete duplicates
  if (fetch_nodes_known_ && !duplicates.empty()) {
    int last = optimized_graph_->node_size() - 1;
    for (auto it = duplicates.rbegin(); it != duplicates.rend(); ++it) {
      int index = *it;
      optimized_graph_->mutable_node()->SwapElements(index, last);
      last--;
    }
    optimized_graph_->mutable_node()->DeleteSubrange(last + 1,
                                                     duplicates.size());
    // Rebuild the NodeMap which was invalidated by the node  swapping above.
    node_map_.reset(new NodeMap(optimized_graph_));
  }
}

void ArithmeticOptimizer::AddFrameControlDeps(
    const NodeDef* old_node, const std::vector<NodeDef*>& new_nodes,
    const string& source_for_ctrl_dep,
    const std::vector<NodeDef*>& sinks_for_control_dep) {
  const auto frame_it = frame_map_.find(old_node);
  if (frame_it != frame_map_.end()) {
    for (auto node : new_nodes) {
      frame_map_.emplace(node, frame_it->second);
    }
    if (!source_for_ctrl_dep.empty() && !sinks_for_control_dep.empty()) {
      const string ctrl_dep = ConstantFolding::AddControlDependency(
          source_for_ctrl_dep, optimized_graph_, node_map_.get());
      for (auto node : sinks_for_control_dep) {
        MaybeAddControlInput(ctrl_dep, node, optimized_graph_, node_map_.get());
      }
    }
  }
}

// TODO(ezhulenev): extract each individual simplify rewrite into separate
// ArithmeticOptimizerStage
string ArithmeticOptimizer::TrySimplifyAndReplaceUses(
    const NodeDef* node, SetVector<NodeDef*>* nodes_to_simplify) {
  // Remove involutions applied twice.
  if (IsInvolution(*node)) {
    // An involution is an element-wise function f(x) that is its own inverse,
    // i.e. f(f(x)) = x. If we can find a chain of ops
    //   f->op1->op2->...opn->f
    // where op1 through opn preserve the values of their inputs, we can remove
    // the two instances of the involution from the graph, since they cancel
    // each other.
    NodeDef* tail =
        GetTailOfValuePreservingChain(*node, *node_map_, nodes_to_preserve_);
    NodeDef* involution = node_map_->GetNode(tail->input(0));
    if (involution->op() == node->op()) {
      // Skip both *node and *involution since they cancel each other.
      if (tail == node) {
        // The two nodes to eliminate are adjacent.
        return involution->input(0);
      } else {
        tail->set_input(0, involution->input(0));
        node_map_->UpdateInput(tail->name(), involution->name(),
                               involution->input(0));
        return node->input(0);
      }
    }
  }

  // Remove inverse transposes.
  if (node->op() == "Transpose" || node->op() == "ConjugateTranspose") {
    NodeDef* input = node_map_->GetNode(node->input(0));
    if (input->op() == node->op()) {
      const NodeDef* node_perm = node_map_->GetNode(node->input(1));
      const NodeDef* input_perm = node_map_->GetNode(input->input(1));
      // Try 32-bit indices.
      std::vector<int> node_perm_values;
      std::vector<int> input_perm_values;
      if (ValuesFromConstNode(*node_perm, &node_perm_values) &&
          ValuesFromConstNode(*input_perm, &input_perm_values) &&
          AreInversePermutations(node_perm_values, input_perm_values)) {
        return input->input(0);
      }
      // Try 64-bit indices.
      std::vector<int64> node_perm_values64;
      std::vector<int64> input_perm_values64;
      if (ValuesFromConstNode(*node_perm, &node_perm_values64) &&
          ValuesFromConstNode(*input_perm, &input_perm_values64) &&
          AreInversePermutations(node_perm_values64, input_perm_values64)) {
        return input->input(0);
      }
    }
  }

  if (node->op() == "Reshape") {
    //   Reshape
    //      ^
    //      |
    //   Reshape
    //      ^
    //      |
    //    input
    //
    // becomes
    //
    //   Reshape <-+
    //             |
    //   Reshape   |
    //      ^      |
    //      |      |
    //    input ---+
    NodeDef* reshape = node_map_->GetNode(node->name());
    int output_pos = 0;
    string input_node_name = ParseNodeName(node->input(0), &output_pos);
    const NodeDef* input = node_map_->GetNode(input_node_name);
    if (input->op() == "Reshape") {
      reshape->set_input(0, input->input(0));
      node_map_->UpdateInput(reshape->name(), input->name(), input->input(0));
      nodes_to_simplify->PushBack(reshape);
      return reshape->name();
    }

    // If the reshape is a no-op, forward its input to its consumers. This is
    // considered aggressive, because users may state that the placeholder
    // outputs tensors of shape [M, N] while feeding it with tensors of shape
    // [M*N] (or worse). The reshape nodes are then necessary to update the
    // tensor metadata to the required shape.
    if (ReshapeIsIdentity(*reshape, *input, output_pos)) {
      return reshape->input(0);
    }
  }

  if (node->op() == "Transpose") {
    // Reorder Cast and Transpose if beneficial.
    //
    // A common pattern after the layout optimizer is casting an uint8 NHWC
    // image to float before transposing it to NCHW. It is beneficial to reorder
    // the cast and the transpose to make the transpose process smaller amount
    // of data. This optimization converts
    //   Transpose(Cast(image, dst_type), perm)
    // to
    //   Cast(Transpose(image, perm), dst_type)
    // when sizeof(image.type) < sizeof(dst_type).
    //
    // TODO(jingyue): This optimization can be generalized to a cast followed by
    // a chain of ops that merely reorder elements (e.g. Reshape and
    // DepthToSpace).
    const NodeDef* transpose = node;
    string dontcare;
    string device;
    // This optimization can be dangerous on devices other than CPU and GPU. The
    // transpose might not be implemented for image.type, or might be slower
    // with image.type than with dst_type.
    if (DeviceNameUtils::SplitDeviceName(transpose->device(), &dontcare,
                                         &device) &&
        (StringPiece(device).contains(DEVICE_CPU) ||
         StringPiece(device).contains(DEVICE_GPU))) {
      const NodeDef* cast = node_map_->GetNode(transpose->input(0));
      if (cast->op() == "Cast") {
        const NodeDef* input = node_map_->GetNode(cast->input(0));
        const DataType src_type = GetSourceDataType(*cast);
        const DataType dst_type = GetDestinationDataType(*cast);
        if (IsNumberType(src_type) && IsNumberType(dst_type) &&
            DataTypeSize(src_type) < DataTypeSize(dst_type) &&
            !OptimizedNodeExists(*cast, DataTypeString(dst_type)) &&
            !OptimizedNodeExists(*transpose, DataTypeString(src_type))) {
          NodeDef* new_transpose = AddNode(*transpose, DataTypeString(src_type),
                                           /*copy_node=*/true);
          (*new_transpose->mutable_attr())["T"].set_type(src_type);
          new_transpose->set_input(0, cast->input(0));
          node_map_->AddOutput(input->name(), new_transpose->name());
          node_map_->AddOutput(NodeName(new_transpose->input(1)),
                               new_transpose->name());

          NodeDef* new_cast =
              AddNode(*cast, DataTypeString(dst_type), /*copy_node=*/true);
          new_cast->set_input(0, new_transpose->name());
          node_map_->AddOutput(new_transpose->name(), new_cast->name());

          nodes_to_simplify->PushBack(new_transpose);
          //  Add frame dependencies that the original node might have had.
          AddFrameControlDeps(node, {new_transpose, new_cast},
                              new_transpose->input(0), {new_transpose});

          return new_cast->name();
        }
      }
    }
  }

  if (node->op() == "Bitcast") {
    NodeDef* bitcast = node_map_->GetNode(node->name());
    // Bypass bitcasts whose source type and destination type are equal.
    if (GetSourceDataType(*bitcast) == GetDestinationDataType(*bitcast)) {
      return bitcast->input(0);
    }

    const NodeDef* operand = node_map_->GetNode(bitcast->input(0));
    if (operand->op() == bitcast->op()) {
      // Bitcast(Bitcast(x, type1), type2) => Bitcast(x, type2)
      bitcast->set_input(0, operand->input(0));
      SetSourceDataType(GetSourceDataType(*operand), bitcast);
      node_map_->UpdateInput(bitcast->name(), bitcast->input(0),
                             operand->input(0));
      nodes_to_simplify->PushBack(bitcast);
      return bitcast->name();
    }
  }

  if (node->op() == "Cast") {
    // Bypass casts whose source type and destination type are equal.
    if (GetSourceDataType(*node) == GetDestinationDataType(*node)) {
      return node->input(0);
    }
  }

  // Fold a multiply of a scalar into the following convolution. This folding
  // can jump across nodes that merely reorders data (such as reshape and
  // transpose). For example, we can optimize
  //
  //
  //         Conv2D
  //        /      \
  //    Transpose  weights
  //       |
  //      Mul
  //     /   \
  //   inputs 255.0
  //
  // to
  //
  //         Conv2D
  //        /      \
  //    Transpose   Mul
  //       |       /   \
  //       |   weights  255.0
  //       |
  //     inputs
  //
  // when `weights` are constant. `Mul` in the optimized graph can be
  // constant-folded.
  //
  // TODO(jingyue): Fold scalar multiplies to Conv?DBackpropFilter and
  // Conv?DBackpropInput.
  if (node->op() == "Conv2D" || node->op() == "Conv3D") {
    NodeDef* conv = const_cast<NodeDef*>(node);
    const NodeDef* weights = node_map_->GetNode(NodeName(conv->input(1)));
    // Fold the multiply to conv only when the weights are constant, so the
    // multiply can be constant-folded. TODO(jingyue): When the weights aren't
    // constant, this should also help performance a bit and memory usage a lot,
    // since the weights tend to be smaller than the activations.
    if (weights->op() == "Const" &&
        !OptimizedNodeExists(*weights, StrCat("scaled_", conv->name()))) {
      const NodeDef* source = node_map_->GetNode(
          GetTailOfValuePreservingChain(*node, *node_map_, nodes_to_preserve_)
              ->input(0));
      if (source->op() == "Mul" &&
          node_map_->GetOutputs(source->name()).size() == 1) {
        const NodeDef* mul = source;
        // `scale` is the scalar multiplier, and `other` is the other operand.
        // TODO(jingyue): handle the case where `scale` is 0-th operand.
        const NodeDef* scale = node_map_->GetNode(mul->input(1));
        const NodeDef* other = node_map_->GetNode(mul->input(0));
        if (scale->op() == "Const" && scale->attr().at("dtype").type() ==
                                          weights->attr().at("dtype").type()) {
          const TensorProto& scale_tensor = scale->attr().at("value").tensor();
          // Test whether `scale` is a scalar.
          if (scale_tensor.has_tensor_shape() &&
              scale_tensor.tensor_shape().dim_size() == 0) {
            // Create new node `scaled_weights`.
            NodeDef* scaled_weights = AddNode(
                *weights, StrCat("scaled_", conv->name()), /*copy_node=*/false);
            scaled_weights->set_op("Mul");
            scaled_weights->set_device(weights->device());
            (*scaled_weights->mutable_attr())["T"] =
                weights->attr().at("dtype");
            nodes_to_simplify->PushBack(scaled_weights);

            // Link in its inputs.
            scaled_weights->add_input(conv->input(1));
            node_map_->AddOutput(weights->name(), scaled_weights->name());
            scaled_weights->add_input(mul->input(1));
            node_map_->AddOutput(scale->name(), scaled_weights->name());
            AddFrameControlDeps(node, {scaled_weights}, "", {});

            // Update `conv`'s weights to `scaled_weights`.
            conv->set_input(1, scaled_weights->name());
            node_map_->UpdateInput(conv->name(), weights->name(),
                                   scaled_weights->name());
            nodes_to_simplify->PushBack(conv);

            // Update `mul`'s consumer to bypass `mul` because it's folded to
            // the weights.
            CHECK_EQ(node_map_->GetOutputs(mul->name()).size(), 1);
            NodeDef* consumer_of_mul =
                *node_map_->GetOutputs(mul->name()).begin();
            consumer_of_mul->set_input(0, mul->input(0));
            node_map_->UpdateInput(consumer_of_mul->name(), mul->name(),
                                   other->name());
            nodes_to_simplify->PushBack(consumer_of_mul);
            return conv->name();
          }
        }
      }
    }
  }

  if (node->op() == "Mul" && node->input(0) == node->input(1) &&
      !OptimizedNodeExists(*node, "square")) {
    NodeDef* new_square_node = AddNode(*node, "square", /*copy_node=*/true);
    new_square_node->set_op("Square");
    for (int i = 1; i < new_square_node->input_size(); ++i) {
      new_square_node->set_input(i - 1, new_square_node->input(i));
    }
    new_square_node->mutable_input()->RemoveLast();
    return new_square_node->name();
  }

  if (IsAggregate(*node) && NumNonControlInputs(*node) > 0) {
    // Discard aggregate nodes with a single input.
    if (node->input_size() == 1) {
      return node->input(0);
    }

    // Try to rewrite aggregations of N >= 2 identical terms (possibly due
    // to deduping or other rewrites) so we can get rid of the sum entirely.
    // The expression (using AddN as an example of an aggregate op):
    //   AddN(x, x, x, ... ,x)
    //        <-- N terms -->
    // can be rewritten to
    //   Mul(Const(N), x))
    //
    bool all_equal = true;
    int num_inputs = 1;
    for (int i = 1; i < node->input_size(); ++i) {
      if (IsControlInput(node->input(i))) {
        break;
      }
      ++num_inputs;
      if (node->input(i) != node->input(0)) {
        all_equal = false;
        break;
      }
    }
    if (all_equal && !OptimizedNodeExists(*node, "const") &&
        !OptimizedNodeExists(*node, "mul")) {
      // 1. Create constant node with value N.
      const auto type = GetDataTypeFromAttr(*node, "T");
      Tensor t(type, TensorShape({}));
      Status status = SetTensorValue(type, num_inputs, &t);
      if (!status.ok()) {
        LOG(WARNING) << "Failed to create const node: "
                     << status.error_message();
        return "";
      }
      TensorValue value(&t);
      NodeDef* new_const_node = AddNode(*node, "const", /*copy_node=*/false);
      status = ConstantFolding::CreateNodeDef(new_const_node->name(), value,
                                              new_const_node);
      if (!status.ok()) {
        LOG(WARNING) << "Failed to create const node: "
                     << status.error_message();
        return "";
      }
      new_const_node->set_device(node->device());
      nodes_to_simplify->PushBack(new_const_node);

      // 2. Replace the aggregate node with Mul(Const(N), x).
      NodeDef* new_mul_node = AddNode(*node, "mul", /*copy_node=*/false);
      new_mul_node->set_op("Mul");
      new_mul_node->set_device(node->device());
      SetDataTypeToAttr(type, "T", new_mul_node);
      new_mul_node->add_input(new_const_node->name());
      node_map_->AddOutput(new_const_node->name(), new_mul_node->name());
      new_mul_node->add_input(node->input(0));
      node_map_->AddOutput(node->input(0), new_mul_node->name());

      CopyControlInputs(*node, new_mul_node, optimized_graph_, node_map_.get());
      AddFrameControlDeps(node, {new_const_node, new_mul_node}, node->input(0),
                          {new_const_node});
      return new_mul_node->name();
    }
  }

  // Use the commutativity and (left- and right-) distributive property of
  // multiplication over addition to hoist common factors out of aggregate nodes
  // where all the inputs are Mul nodes. This pattern occurs frequently in
  // regularization terms for the gradients during training.
  // For example, we can rewrite an expression of the form:
  //   AddN(Mul(x, y1), Mul(y2, x), Mul(x, y3), ... Mul(x, yn))
  // to the following:
  //   Mul(x, AddN(y1, y2, y3, ... yn))
  if (IsAggregate(*node) && NumNonControlInputs(*node) > 1 &&
      !OptimizedNodeExists(*node, "hoist_add") &&
      !OptimizedNodeExists(*node, "hoist_mul")) {
    // Determine the set of common factors if the input nodes are all Mul nodes.
    std::set<string> common_factors;
    for (int i = 0; i < node->input_size(); ++i) {
      if (i > 0 && common_factors.empty()) {
        break;
      }
      if (IsControlInput(node->input(i))) {
        break;
      }
      const NodeDef* input = node_map_->GetNode(node->input(i));
      if (input->op() == "Mul") {
        std::set<string> factors_i{input->input(0), input->input(1)};
        if (i == 0) {
          std::swap(common_factors, factors_i);
        } else {
          std::set<string> intersection;
          std::set_intersection(
              factors_i.begin(), factors_i.end(), common_factors.begin(),
              common_factors.end(),
              std::inserter(intersection, intersection.begin()));
          std::swap(common_factors, intersection);
        }
      } else {
        common_factors.clear();
      }
    }
    if (common_factors.size() == 1) {
      const string& common_factor = *common_factors.begin();

      // Gather up the non-shared factors (the y's in the example).
      // Unless the aggregation is Add, we have to make sure that all the y's
      // have the same shape since the other aggregation ops do not support
      // broadcasting.
      std::vector<string> unique_factors;
      unique_factors.reserve(node->input_size());
      bool shapes_match = true;
      for (int i = 0; i < node->input_size() && shapes_match; ++i) {
        const string& input = node->input(i);
        if (IsControlInput(input)) {
          break;
        }
        const NodeDef* mul_node = node_map_->GetNode(input);
        const int unique_factor_index =
            mul_node->input(0) == common_factor ? 1 : 0;
        unique_factors.push_back(mul_node->input(unique_factor_index));
        if (i > 0 && !IsAdd(*node)) {
          shapes_match = ShapesEqual(unique_factors.front(),
                                     unique_factors.back(), *node_map_);
        }
      }

      if (shapes_match) {
        // 1. Use a copy of the first Mul node for the outer multiplication.
        NodeDef* new_mul_node = AddNode(OptimizedNodeName(*node, "hoist_mul"),
                                        node_map_->GetNode(node->input(0)));
        NodeDef* new_add_node = AddNode(*node, "hoist_add", /*copy_node=*/true);
        new_mul_node->set_device(node->device());
        new_mul_node->set_input(0, common_factor);
        node_map_->AddOutput(common_factor, new_mul_node->name());
        new_mul_node->set_input(1, new_add_node->name());
        node_map_->AddOutput(new_add_node->name(), new_mul_node->name());

        // 2. Hoist non-shared factors up into the new AddN node.
        nodes_to_simplify->PushBack(new_add_node);
        for (int i = 0; i < node->input_size(); ++i) {
          const string& input = node->input(i);
          if (IsControlInput(input)) {
            break;
          }
          new_add_node->set_input(i, unique_factors[i]);
        }

        // 3. Add frame dependencies that the original node might have had.
        AddFrameControlDeps(node, {new_add_node, new_mul_node}, common_factor,
                            {new_add_node});

        return new_mul_node->name();
      }
    }
  }

  // Fold Transpose into matrix multiplication.
  if ((node->op() == "MatMul" || node->op() == "SparseMatMul" ||
       node->op() == "BatchMatMul") &&
      !OptimizedNodeExists(*node, "fused")) {
    const NodeDef* a = node_map_->GetNode(node->input(0));
    const NodeDef* b = node_map_->GetNode(node->input(1));
    bool is_complex = false;
    if (node->op() != "SparseMatMul") {
      const DataType type = GetDataTypeFromAttr(*node, "T");
      is_complex = (type == DT_COMPLEX64) || (type == DT_COMPLEX128);
    }
    const std::set<string> foldable_transpose_ops =
        !is_complex ? std::set<string>{"ConjugateTranspose", "Transpose"}
                    : (node->op() == "BatchMatMul"
                           ? std::set<string>{"ConjugateTranspose"}
                           : std::set<string>{"Transpose"});
    const bool a_is_foldable = foldable_transpose_ops.count(a->op()) > 0 &&
                               IsInnerMatrixTransposeNode(*a, node_map_.get());
    const bool b_is_foldable = foldable_transpose_ops.count(b->op()) > 0 &&
                               IsInnerMatrixTransposeNode(*b, node_map_.get());
    if (a_is_foldable || b_is_foldable) {
      NodeDef* new_op = AddNode(*node, "fused", /*copy_node=*/true);
      if (a_is_foldable) {
        const string attr_a =
            node->op() == "BatchMatMul" ? "adj_x" : "transpose_a";
        FlipBooleanAttr(attr_a, new_op);
        new_op->set_input(0, a->input(0));
        node_map_->UpdateInput(new_op->name(), a->name(), a->input(0));
        AddFrameControlDeps(node, {new_op}, a->input(0), {new_op});
      }
      if (b_is_foldable) {
        const string attr_b =
            node->op() == "BatchMatMul" ? "adj_y" : "transpose_b";
        FlipBooleanAttr(attr_b, new_op);
        new_op->set_input(1, b->input(0));
        node_map_->UpdateInput(new_op->name(), b->name(), b->input(0));
        if (!a_is_foldable) {
          AddFrameControlDeps(node, {new_op}, b->input(0), {new_op});
        }
      }
    }
  }

  // Fold Conj into Transpose or ConjugateTranspose.
  if ((node->op() == "Conj" || node->op() == "Transpose" ||
       node->op() == "ConjugateTranspose") &&
      !OptimizedNodeExists(*node, "fused")) {
    const NodeDef* input = node_map_->GetNode(node->input(0));
    const NodeDef* transpose_op = node->op() == "Conj" ? input : node;
    const NodeDef* conj_op = node->op() == "Conj" ? node : input;

    if ((transpose_op->op() == "Transpose" ||
         transpose_op->op() == "ConjugateTranspose") &&
        conj_op->op() == "Conj") {
      NodeDef* new_op =
          AddNode(OptimizedNodeName(*node, "fused"), transpose_op);
      // Flip the type of transpose op to absorb the conjugation.
      new_op->set_op(transpose_op->op() == "Transpose" ? "ConjugateTranspose"
                                                       : "Transpose");
      new_op->set_input(0, input->input(0));
      node_map_->UpdateInput(new_op->name(), node->name(), input->input(0));
      AddFrameControlDeps(node, {new_op}, "", {});
      return new_op->name();
    }
  }

  return "";
}

Status ArithmeticOptimizer::SimplifyArithmeticOps() {
  SetVector<NodeDef*> nodes_to_simplify;
  nodes_to_simplify.Reserve(optimized_graph_->node_size());
  for (int i = 0; i < optimized_graph_->node_size(); ++i) {
    nodes_to_simplify.PushBack(optimized_graph_->mutable_node(i));
  }

  ArithmeticOptimizerContext ctx(&nodes_to_preserve_, optimized_graph_,
                                 node_map_.get(), &nodes_to_simplify);

  std::vector<std::unique_ptr<ArithmeticOptimizerStage>> stages;

  // Add/AddN tree rewrites
  if (options_.enable_add_to_addn_combining) {
    stages.push_back(
        std::unique_ptr<ArithmeticOptimizerStage>(new AddOpsRewriteStage(ctx)));
  }

  VLOG(1) << "Simplify arithmetic ops using " << stages.size()
          << " arithmetic optimization stages";

  while (!nodes_to_simplify.Empty()) {
    const NodeDef* node = nodes_to_simplify.PopBack();

    // TODO(ezhulenev): move all rewrites into separate stages
    string simplified_tensor =
        TrySimplifyAndReplaceUses(node, &nodes_to_simplify);

    // if it was not simplified try to run it through all configured stages
    if (simplified_tensor.empty()) {
      for (auto& stage : stages) {
        if (stage->IsSupported(node)) {
          TF_RETURN_IF_ERROR(stage->TrySimplify(node, &simplified_tensor));
          if (!simplified_tensor.empty()) {
            break;
          }
        }
      }
    }

    // if it's still empty go to the next Node
    if (simplified_tensor.empty()) {
      continue;
    }

    // re-wire consumers of an old node to the new one
    if (NodeName(simplified_tensor) != node->name()) {
      // Always consider simplified_tensor for further optimizations.
      NodeDef* simplified_node = node_map_->GetNode(simplified_tensor);
      if (simplified_node != nullptr) {
        nodes_to_simplify.PushBack(simplified_node);
      }
      // When `node` is simplified to another node rather than in-place, the
      // consumers of `node` are already redirected to `simplified_tensor`.
      // Re-push the consumers into `nodes_to_simplify` for further
      // optimizations.
      const std::set<NodeDef*> outputs = node_map_->GetOutputs(node->name());
      std::vector<NodeDef*> consumers(outputs.begin(), outputs.end());
      std::sort(consumers.begin(), consumers.end(),
                [](const NodeDef* n1, const NodeDef* n2) {
                  return n1->name() < n2->name();
                });
      for (NodeDef* consumer : consumers) {
        // Update `consumer`'s use of `node` to `input`'s operand.
        for (int i = 0; i < consumer->input_size(); ++i) {
          int operand_pos;
          string operand_node_name =
              ParseNodeName(consumer->input(i), &operand_pos);
          if (operand_node_name == node->name()) {
            *consumer->mutable_input(i) =
                (operand_pos < 0
                     ? AsControlDependency(NodeName(simplified_tensor))
                     : simplified_tensor);
          }
        }
        node_map_->UpdateInput(consumer->name(), node->name(),
                               simplified_tensor);
        nodes_to_simplify.PushBack(consumer);
      }
    }
  }
  return Status::OK();
}

Status ArithmeticOptimizer::Optimize(Cluster* /*cluster*/,
                                     const GrapplerItem& item,
                                     GraphDef* optimized_graph) {
  optimized_graph_ = optimized_graph;
  *optimized_graph_ = item.graph;

  // Set up helper data structures.
  nodes_to_preserve_ = item.NodesToPreserve();
  fetch_nodes_known_ = !item.fetch.empty();
  node_map_.reset(new NodeMap(optimized_graph_));
  int num_frames;
  TF_RETURN_IF_ERROR(IdentifyFramesWithNodeMap(*optimized_graph_, *node_map_,
                                               &frame_map_, &num_frames));
  // Shapes are only needed in aggressive mode.
  graph_properties_.reset(new GraphProperties(item));
  TF_RETURN_IF_ERROR(graph_properties_->InferStatically(false));
  // TODO(ezhulenev): Use GraphProperties to lookup tensor shapes directly
  TF_RETURN_IF_ERROR(graph_properties_->AnnotateOutputShapes(optimized_graph_));

  // Perform the optimizations.
  DedupComputations();
  TF_RETURN_IF_ERROR(SimplifyArithmeticOps());

  // Clear output shapes.
  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    optimized_graph_->mutable_node(i)->mutable_attr()->erase(kOutputShapesAttr);
  }

  return Status::OK();
}

void ArithmeticOptimizer::Feedback(Cluster* /*cluster*/,
                                   const GrapplerItem& /*item*/,
                                   const GraphDef& /*optimized_graph*/,
                                   double /*result*/) {
  // Nothing to do for ArithmeticOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
