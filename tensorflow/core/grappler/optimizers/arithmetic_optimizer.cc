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
#include <deque>
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
#include "tensorflow/core/grappler/optimizers/graph_optimizer_stage.h"
#include "tensorflow/core/grappler/optimizers/symbolic_shapes.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

using tensorflow::strings::StrCat;

namespace tensorflow {
namespace grappler {
namespace {

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

// Returns whether `reshape` is an identity op. The tensor that `reshape`
// reshapes is the `output_pos`-th output of node `input`.
bool ReshapeIsIdentity(const NodeDef& reshape, const NodeDef& input,
                       const int output_pos,
                       const GraphProperties& graph_properties) {
  const std::vector<OpInfo::TensorProperties>& reshape_props =
      graph_properties.GetOutputProperties(reshape.name());
  const std::vector<OpInfo::TensorProperties>& input_props =
      graph_properties.GetOutputProperties(input.name());
  if (reshape_props.empty() || input_props.empty() ||
      input_props.size() <= output_pos) {
    return false;
  }

  const PartialTensorShape& src_shape = input_props[output_pos].shape();
  const PartialTensorShape& dst_shape = reshape_props[0].shape();

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
    return std::count_if(dim_sizes.begin(), dim_sizes.end(),
                         [](int dim) { return dim < 0; });
  };
  int src_num_unknown_dim_sizes = num_unknown_dim_sizes(src_shape);
  int dst_num_unknown_dim_sizes = num_unknown_dim_sizes(dst_shape);
  if (src_num_unknown_dim_sizes > 1 || dst_num_unknown_dim_sizes > 1) {
    return false;
  }

  // If dst_num_unknown_dim_sizes != src_num_unknown_dim_sizes we would weaken
  // shape inference in subsequent passes if we removed this reshape.
  if (src_num_unknown_dim_sizes != dst_num_unknown_dim_sizes) {
    return false;
  }

  // Remove the reshape if both are fully defined or partially defined and the
  // unknown or symbolic shape appears on the same dimension, i.e., if
  // IsIdenticalTo returns true.
  return dst_shape.IsIdenticalTo(src_shape);
}

NodeDef* GetTailOfValuePreservingChain(
    const NodeDef& node, const NodeMap& node_map,
    const std::unordered_set<string>& nodes_to_preserve) {
  auto is_value_preserving_non_branching = [&](const NodeDef& node) {
    return nodes_to_preserve.find(node.name()) == nodes_to_preserve.end() &&
           IsValuePreserving(node) && NumNonControlOutputs(node, node_map) == 1;
  };
  return GetTailOfChain(node, node_map, /*follow_control_input=*/false,
                        is_value_preserving_non_branching);
}

// Graph optimizer context extension specific to ArithmeticOptimizer
struct ArithmeticOptimizerContext {
  explicit ArithmeticOptimizerContext(SetVector<NodeDef*>* nodes_to_simplify)
      : nodes_to_simplify(nodes_to_simplify) {}
  SetVector<NodeDef*>* nodes_to_simplify;
};

// Base class for single arithmetic optimization: e.g. Bitcast optimization,
// AddOps optimization, etc...
class ArithmeticOptimizerStage : public GraphOptimizerStage<string> {
 public:
  explicit ArithmeticOptimizerStage(const string& name,
                                    const GraphOptimizerContext& ctx,
                                    const ArithmeticOptimizerContext ctx_ext)
      : GraphOptimizerStage("ArithmeticOptimizer", name, ctx),
        ctx_ext_(ctx_ext) {}
  virtual ~ArithmeticOptimizerStage() = default;

 protected:
  // Simplification graph rewrite can create additional nodes that are inputs
  // to final simplified node, they can be also added to the arithmetic
  // optimizer queue for further optimization.
  void AddToOptimizationQueue(NodeDef* node) {
    ctx_ext_.nodes_to_simplify->PushBack(node);
  }

  // TODO(ezhulenev): remove this method from ArithmeticOptimizer when all
  // optimizations will be migrated to stages
  void ForwardControlDependencies(
      NodeDef* target_node, const std::vector<const NodeDef*>& src_nodes) {
    for (const auto& src : src_nodes) {
      for (int i = src->input_size() - 1; i >= 0; --i) {
        if (IsControlInput(src->input(i))) {
          *target_node->add_input() = src->input(i);
          ctx_.node_map->AddOutput(NodeName(src->input(i)),
                                   target_node->name());
        } else {
          break;
        }
      }
    }
  }

 private:
  // Extended context required for ArithmeticOptimizer.
  const ArithmeticOptimizerContext ctx_ext_;
};

// Subtype of ArithmeticOptimizerStage that does optimization by rewriting a
// group of nodes from the optimized graph.
//
// * AddOpsRewrite:
//   Rewrite a group of Add/AddN with compact Add/AddN tree
//
// * MinimizeBroadcasts:
//   Rewrite a group of binary associative ops, reordering
//   inputs, to minimize the cost of broadcast
class ArithmeticNodesGroupOptimizerStage : public ArithmeticOptimizerStage {
 public:
  explicit ArithmeticNodesGroupOptimizerStage(
      const string& name, const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext ctx_ext)
      : ArithmeticOptimizerStage(name, ctx, ctx_ext), optimized_nodes_{} {}
  ~ArithmeticNodesGroupOptimizerStage() override = default;

  // Input name with a statically inferred shape from GraphProperties
  struct InputAndShape {
    InputAndShape(const string& input, const TensorShapeProto& shape)
        : input(input), shape(shape) {}
    string input;
    TensorShapeProto shape;
  };

  // Subgraph (subtree) of nodes, that we want to optimize in "one shot" (e.g.
  // all the Add nodes that we plan to rewrite with a single AddN). Subgraph is
  // obtained by graph traversal, starting from a root node.
  struct OptimizedNodesGroup {
    NodeDef* root_node;
    TensorShapeProto root_shape;
    // Optimized nodes that will be updated or removed by rewrite
    std::vector<NodeDef*> optimized_nodes;
    // Inputs to optimized nodes
    std::vector<InputAndShape> inputs;
  };

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    TF_RETURN_IF_ERROR(EnsureNodeIsSupported(node));

    OptimizedNodesGroup group;
    TF_RETURN_IF_ERROR(CreateOptimizedNodesGroup(node, &group));

    if (!group.optimized_nodes.empty()) {
      *simplified_node_name = RewriteOptimizedNodesGroup(group);
    }

    return Status::OK();
  }

 protected:
  // Modify the optimized graph after nodes group was successfully identified
  virtual string RewriteOptimizedNodesGroup(
      const OptimizedNodesGroup& group) = 0;

  // Check if input can become a part of current optimized nodes group.
  virtual bool IsAbsorbableByOptimizedNodesGroup(
      const OptimizedNodesGroup& group, const string& input) const = 0;

  Status AbsorbInputByOptimizedNodesGroup(const string& input,
                                          OptimizedNodesGroup* group) const {
    NodeDef* node;
    TF_RETURN_IF_ERROR(GetInputNode(input, &node));

    if (IsAbsorbableByOptimizedNodesGroup(*group, input)) {
      for (int i = 0; i < node->input_size(); ++i) {
        const string& input_i = node->input(i);
        if (!IsControlInput(input)) {
          TF_RETURN_IF_ERROR(AbsorbInputByOptimizedNodesGroup(input_i, group));
        }
      }
      group->optimized_nodes.push_back(node);
    } else {
      // If node can't be absorbed, add it to OptimizedNodesGroup input
      OpInfo::TensorProperties properties;
      TF_RETURN_IF_ERROR(GetTensorProperties(input, &properties));
      group->inputs.emplace_back(input, properties.shape());
    }
    return Status::OK();
  }

  Status CreateOptimizedNodesGroup(NodeDef* root_node,
                                   OptimizedNodesGroup* group) const {
    OpInfo::TensorProperties root_node_output_properties;
    TF_RETURN_IF_ERROR(
        GetTensorProperties(root_node->name(), &root_node_output_properties));

    group->root_node = root_node;
    group->root_shape = root_node_output_properties.shape();

    group->optimized_nodes.reserve(root_node->input_size());
    for (int i = 0; i < root_node->input_size(); ++i) {
      const string& input_i = root_node->input(i);
      if (!IsControlInput(input_i)) {
        TF_RETURN_IF_ERROR(AbsorbInputByOptimizedNodesGroup(input_i, group));
      }
    }

    return Status::OK();
  }

  // Check if all inputs can be broadcasted to the same shape
  // TODO(ezhulenev): move to GraphOptimizerStage?
  bool HasAllInputsBroadcastableToShape(
      const NodeDef& node, const OpInfo::TensorProperties& properties) const {
    auto is_broadcastable = [this, &properties](const string& input) {
      OpInfo::TensorProperties input_props;
      Status has_input_properties = GetTensorProperties(input, &input_props);
      return has_input_properties.ok() &&
             ShapesBroadcastable(properties, input_props);
    };
    return std::all_of(node.input().begin(), node.input().end(),
                       is_broadcastable);
  }

  // TODO(ezhulenev): move to GraphOptimizerStage?
  bool IsDrivenByControlDependency(const NodeDef& node) const {
    return std::any_of(node.input().begin(), node.input().end(),
                       IsControlInput);
  }

  // TODO(ezhulenev): move to GraphOptimizerStage?
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

  string ShapeSignature(const TensorShapeProto& shape) const {
    string signature = strings::StrCat("rank:", shape.dim_size(), ":dim");
    for (int i = 0; i < shape.dim_size(); ++i)
      strings::StrAppend(&signature, ":", shape.dim(i).size());
    return signature;
  }

  void AddToOptimizedNodes(const NodeDef* node) {
    optimized_nodes_.insert(node->name());
  }

  bool IsOnTheSameDevice(const OptimizedNodesGroup& group,
                         const NodeDef& node) const {
    return group.root_node->device() == node.device();
  }

  bool IsInPreserveSet(const NodeDef& node) const {
    return ctx_.nodes_to_preserve->find(node.name()) !=
           ctx_.nodes_to_preserve->end();
  }

  bool IsAlreadyOptimized(const NodeDef& node) const {
    return optimized_nodes_.find(node.name()) != optimized_nodes_.end();
  }

 private:
  // set of nodes already processed by this optimizer stage
  std::unordered_set<string> optimized_nodes_;
};

// Rewrite a tree of Add/AddN with a single AddN operation, consuming all the
// original inputs of absorbed nodes.
//
// 1) All nodes must have the same device placement.
//
// 2) If All nodes in a Add/AddN subgraph have symbolically equal shape, tree is
//    optimized to a single AddN node.
//
//                AddN_1
//             /    |    \
//          Add_1   z   Add_2       -> AddN(x, y, z, w, q, e)
//          /  \        /  \
//         x    y      w    Add_3
//                          / \
//                         q   e
//
// 3) If some nodes have different shape (it needs to be broadcastable to the
//    shape of a "root), tree is optimized to AddNs for symbolically equal
//    shapes, and a tree of Add ops, that minimize broadcasts.
//
//                AddN_1                                 Add
//             /    |    \                              /  \
//          Add_1   z   Add_2       ->               Add    w
//          /  \        /  \                        /   \
//         x    y      w    Add_3      AddN(x, y, q, e)  z
//                          / \
//                         q   e
class AddOpsRewriteStage : public ArithmeticNodesGroupOptimizerStage {
 public:
  explicit AddOpsRewriteStage(const GraphOptimizerContext& ctx,
                              const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticNodesGroupOptimizerStage("AddOpsRewrite", ctx, ctx_ext) {}
  ~AddOpsRewriteStage() override = default;

  // Check if a node can become a root of AddOpsGroup
  bool IsSupported(const NodeDef* node) const override {
    if (!CanOptimize(node)) return false;

    // shape must be symbolically defined and all inputs compatible with it
    OpInfo::TensorProperties properties;
    Status has_properties = GetTensorProperties(node->name(), &properties);
    return has_properties.ok() && ShapeIsSymbolicallyDefined(properties) &&
           HasAllInputsBroadcastableToShape(*node, properties);
  }

 protected:
  // Check if a node can be absorbed by current OptimizedNodesGroup
  bool IsAbsorbableByOptimizedNodesGroup(const OptimizedNodesGroup& group,
                                         const string& input) const override {
    NodeDef* node;
    Status node_status = GetInputNode(input, &node);
    if (!node_status.ok() || !CanOptimize(node)) return false;

    if (!IsOnTheSameDevice(group, *node)) {
      return false;
    }
    // with a single output data consumer (presumably if we reach this node from
    // previously absorbed or a root node, it means that this node is not used
    // as an input to any other op, outside of the group)
    if (NumNonControlDataOutputs(*node, *ctx_.node_map) != 1) {
      return false;
    }
    // All input shapes must be broadcastable to the node shape
    OpInfo::TensorProperties properties;
    Status has_properties = GetTensorProperties(input, &properties);
    return has_properties.ok() &&
           HasAllInputsBroadcastableToShape(*node, properties);
  }

  // Node requirements both for a root node and an absorbed node
  bool CanOptimize(const NodeDef* node) const {
    // TODO(ezhulenev): check if AccumulateNV2 can be supported too
    if (!IsAdd(*node) && !IsAddN(*node)) {
      return false;
    }
    if (IsInPreserveSet(*node) || IsAlreadyOptimized(*node)) {
      return false;
    }
    // it must not be created by this stage at any of previous optimization runs
    if (str_util::StrContains(node->name(), stage_name_)) {
      return false;
    }
    // TODO(ezhulenev): relax this condition for root node
    return !(IsDrivenByControlDependency(*node) ||
             DrivesControlDependency(*node));
  }

  // Rewrite a group of add ops into a single AddN if all input shapes are
  // symbolically equal. If not, create AddN for equal shapes first, and then
  // build an Add tree, minimizing the cost of broadcasts.
  string RewriteOptimizedNodesGroup(const OptimizedNodesGroup& group) override {
    // all new nodes will be placed under the scope of a root node
    auto root_scope_and_name = ParseNodeScopeAndName(group.root_node->name());

    // Find what shapes are present in the inputs of absorbed nodes
    std::unordered_map<string, std::vector<InputAndShape>> shape_sig_to_inputs;
    for (const auto& input : group.inputs) {
      shape_sig_to_inputs[ShapeSignature(input.shape)].push_back(input);
    }

    // Collect all the shapes from representative elements
    std::vector<TensorShapeProto> shapes;
    shapes.reserve(shape_sig_to_inputs.size());
    for (const auto& el : shape_sig_to_inputs)
      shapes.push_back(el.second[0].shape);

    // If all inputs have the same shape, rewrite whole group with a single AddN
    if (shapes.size() == 1) {
      string node_name = OptimizedNodeName(root_scope_and_name);
      AddInputsOfSymbolicallyEqualShape(*group.root_node, node_name,
                                        group.inputs);
      return node_name;
    }

    // For inputs of different shapes:
    // 1. Rewrite inputs of the same shape using AddN (leaf nodes)
    // 2. Build a tree of Add nodes, minimizing cost of broadcast
    std::sort(shapes.begin(), shapes.end(),
              [](const TensorShapeProto& left, const TensorShapeProto& right) {
                return CompareSymbolicallyShapedTensorSizes(left, right);
              });

    // optimized name for leaf AddN nodes
    auto leaf_node_name = [&root_scope_and_name, this](int i) {
      return OptimizedNodeName(root_scope_and_name,
                               strings::StrCat("Leaf_", i));
    };
    // optimized name for internal nodes of a tree built up from AddN leaves
    auto internal_node_name = [&root_scope_and_name, this](int i) {
      return OptimizedNodeName(root_scope_and_name,
                               strings::StrCat("Internal_", i));
    };

    // Add/AddN nodes that must be added to the tree
    std::deque<InputAndShape> add_ops;

    // Prepare leaf AddN nodes for inputs of equal shape
    for (int i = 0; i < shapes.size(); ++i) {
      const auto node_name = leaf_node_name(i);
      const auto& inputs = shape_sig_to_inputs[ShapeSignature(shapes[i])];
      add_ops.push_back(AddInputsOfSymbolicallyEqualShape(*group.root_node,
                                                          node_name, inputs));
    }

    // Build up a tree of Add ops
    int internal_nodes = 0;
    do {
      const InputAndShape lhs = add_ops.front();
      add_ops.pop_front();
      const InputAndShape rhs = add_ops.front();
      add_ops.pop_front();
      string name = add_ops.empty() ? OptimizedNodeName(root_scope_and_name)
                                    : internal_node_name(internal_nodes++);
      InputAndShape add = AddAggregatedInputs(*group.root_node, name, lhs, rhs);
      add_ops.push_front(add);
    } while (add_ops.size() > 1);

    InputAndShape optimized_root_node = add_ops.front();
    return optimized_root_node.input;
  }

  // Add 'AddN' node to aggregate inputs of symbolically equal shape
  InputAndShape AddInputsOfSymbolicallyEqualShape(
      const NodeDef& root_node, const string& node_name,
      const std::vector<InputAndShape>& inputs) {
    CHECK(!inputs.empty()) << "Inputs must be non-empty";

    // Do not create redundant AddN nodes
    if (inputs.size() == 1) {
      return inputs[0];
    }

    // get shape from representative element
    auto shape = inputs[0].shape;

    // copy attributes from a root node
    DataType dtype = root_node.attr().at("T").type();

    // add new AddN node
    NodeDef* node = AddEmptyNode(node_name);
    node->set_op("AddN");
    node->set_device(root_node.device());
    (*node->mutable_attr())["T"].set_type(dtype);
    (*node->mutable_attr())["N"].set_i(inputs.size());

    for (const auto& inputAndShape : inputs) {
      ctx_.node_map->AddOutput(inputAndShape.input, node_name);
      node->add_input(inputAndShape.input);
    }

    AddToOptimizedNodes(node);
    return InputAndShape(node_name, shape);
  }

  // Add a single 'Add' node to sum two inputs
  InputAndShape AddAggregatedInputs(const NodeDef& root_node,
                                    const string& node_name,
                                    const InputAndShape& left,
                                    const InputAndShape& right) {
    // copy attributes from a root node
    DataType dtype = root_node.attr().at("T").type();

    // add new Add node
    NodeDef* node = AddEmptyNode(node_name);
    node->set_op("Add");
    node->set_device(root_node.device());
    (*node->mutable_attr())["T"].set_type(dtype);

    ctx_.node_map->AddOutput(left.input, node_name);
    ctx_.node_map->AddOutput(right.input, node_name);

    node->add_input(left.input);
    node->add_input(right.input);

    AddToOptimizedNodes(node);
    return InputAndShape(
        node_name, TensorShapeProto());  // shape is not important at this point
  }
};

// Use the distributive property of multiplication and division over addition,
// along with commutativity of the former, to hoist common factors/denominators
// out of aggregate nodes where ALL the inputs are Mul/Div nodes.
// This pattern occurs frequently in regularization terms for the gradients
// during training.
//
// For example, we can rewrite an expression of the form:
//   AddN(Mul(x, y1), Mul(y2, x), Mul(x, y3), ... Mul(x, yn))
// to the following:
//   Mul(x, AddN(y1, y2, y3, ... yn))
// For division, we can rewrite
//   AddN(Div(y1, x), Div(y2, x), Div(y3, x), ... Div(yn, x))
// to:
//   Div(AddN(y1, y2, y3, ... yn), x)
class HoistCommonFactorOutOfAggregation : public ArithmeticOptimizerStage {
 public:
  explicit HoistCommonFactorOutOfAggregation(
      const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("HoistCommonFactor", ctx, ctx_ext) {}
  ~HoistCommonFactorOutOfAggregation() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsAggregate(*node) && NumNonControlInputs(*node) > 1 &&
           !IsRewritten(node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    TF_RETURN_IF_ERROR(EnsureNodeIsSupported(node));

    bool common_factor_is_denominator = false;
    std::set<string> common_factors;
    std::vector<string> ctrl_deps;
    TF_RETURN_IF_ERROR(GetCommonFactors(
        node, &common_factors, &common_factor_is_denominator, &ctrl_deps));

    if (common_factors.size() == 1) {
      const string& common_factor = *common_factors.begin();

      // Gather up the non-shared factors
      bool shapes_match = true;
      std::vector<string> unique_factors;
      TF_RETURN_IF_ERROR(GetUniqueFactors(node, common_factor,
                                          common_factor_is_denominator,
                                          &shapes_match, &unique_factors));

      if (shapes_match) {
        NodeDef* input_0;
        TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &input_0));

        // Use a copy of the first node for the outer multiplication/division.
        NodeDef* new_outer_node = AddCopyNode(
            OuterNodeName(node, common_factor_is_denominator), input_0);
        // And a copy of aggregation node as one of the inner operands
        NodeDef* new_add_node = AddCopyNode(InnerAddNodeName(node), node);

        new_outer_node->set_device(node->device());
        if (common_factor_is_denominator) {
          new_outer_node->set_input(0, new_add_node->name());
          new_outer_node->set_input(1, common_factor);
        } else {
          new_outer_node->set_input(0, common_factor);
          new_outer_node->set_input(1, new_add_node->name());
        }

        ctx_.node_map->AddOutput(common_factor, new_outer_node->name());
        ctx_.node_map->AddOutput(new_add_node->name(), new_outer_node->name());

        // Hoist non-shared factors up into the new AddN node.
        for (int i = 0; i < unique_factors.size(); ++i) {
          const string& unique_factor_i = unique_factors[i];
          new_add_node->set_input(i, unique_factor_i);
          ctx_.node_map->AddOutput(unique_factor_i, new_add_node->name());
        }

        // Add control deps on add node
        for (const string& ctrl_dep : ctrl_deps) {
          *new_add_node->add_input() = ctrl_dep;
          ctx_.node_map->AddOutput(NodeName(ctrl_dep), new_add_node->name());
        }

        // optimize new inner aggregation node
        AddToOptimizationQueue(new_add_node);
        // do not optimize the same node twice
        rewritten_nodes_.insert(node->name());
        *simplified_node_name = new_outer_node->name();
      }
    }
    return Status::OK();
  }

 private:
  // Get a name for new outer node
  string OuterNodeName(const NodeDef* node, bool is_div) const {
    auto scope_and_name = ParseNodeScopeAndName(node->name());
    return is_div ? OptimizedNodeName(scope_and_name, "Div")
                  : OptimizedNodeName(scope_and_name, "Mul");
  }

  // Get a name new inner Add node
  string InnerAddNodeName(const NodeDef* node) const {
    auto scope_and_name = ParseNodeScopeAndName(node->name());
    return OptimizedNodeName(scope_and_name, "Add");
  }

  // Determine the set of common factors if the input nodes are all Mul or
  // Div nodes.
  Status GetCommonFactors(const NodeDef* node, std::set<string>* common_factors,
                          bool* common_factor_is_denominator,
                          std::vector<string>* ctrl_deps) const {
    CHECK(common_factors->empty());
    CHECK_NOTNULL(common_factor_is_denominator);
    *common_factor_is_denominator = false;

    bool has_mul = false;
    bool has_div = false;
    for (int i = 0; i < node->input_size(); ++i) {
      if (i > 0 && common_factors->empty()) break;
      if (IsControlInput(node->input(i))) {
        ctrl_deps->push_back(node->input(i));
        continue;
      }
      NodeDef* input;
      TF_RETURN_IF_ERROR(GetInputNode(node->input(i), &input));

      if ((!IsMul(*input) && !IsAnyDiv(*input)) || (IsMul(*input) && has_div) ||
          (IsAnyDiv(*input) && has_mul)) {
        // Break if input is neither a Mul or Div, or if there are both Mul &
        // Div Ops.
        common_factors->clear();
        break;
      } else if (IsAnyDiv(*input)) {
        has_div = true;
        // In case of possible common dividers, we avoid hoisting out if any
        // input is not float/double, since integer division is not distributive
        // over addition.
        OpInfo::TensorProperties properties0, properties1;
        TF_RETURN_IF_ERROR(GetTensorProperties(input->input(0), &properties0));
        TF_RETURN_IF_ERROR(GetTensorProperties(input->input(1), &properties1));
        if (properties0.dtype() != DT_FLOAT &&
            properties0.dtype() != DT_DOUBLE &&
            properties1.dtype() != DT_FLOAT &&
            properties1.dtype() != DT_DOUBLE) {
          common_factors->clear();
          break;
        }
      } else if (IsMul(*input)) {
        has_mul = true;
      }

      // We only focus on common factors from denominators if any Op is a
      // Div.
      std::set<string> factors_i =
          has_mul ? std::set<string>{input->input(0), input->input(1)}
                  : std::set<string>{input->input(1)};
      if (i == 0) {
        std::swap(*common_factors, factors_i);
      } else {
        std::set<string> intersection;
        std::set_intersection(
            factors_i.begin(), factors_i.end(), common_factors->begin(),
            common_factors->end(),
            std::inserter(intersection, intersection.begin()));
        std::swap(*common_factors, intersection);
      }
      for (int i = 2; i < input->input_size(); ++i) {
        ctrl_deps->push_back(input->input(i));
      }
    }

    *common_factor_is_denominator = has_div;
    return Status::OK();
  }

  // Gather up the non-shared factors (the y's in the example).
  // Unless the aggregation is Add, we have to make sure that all the y's
  // have the same shape since the other aggregation ops do not support
  // broadcasting.
  Status GetUniqueFactors(const NodeDef* node, const string& common_factor,
                          const bool common_factor_is_denominator,
                          bool* shapes_match,
                          std::vector<string>* unique_factors) const {
    *shapes_match = true;
    unique_factors->reserve(node->input_size());

    for (int i = 0; i < node->input_size() && shapes_match; ++i) {
      const string& input = node->input(i);
      if (IsControlInput(input)) {
        break;
      }
      NodeDef* inner_node;
      TF_RETURN_IF_ERROR(GetInputNode(input, &inner_node));
      const int unique_factor_index =
          common_factor_is_denominator
              ? 0
              : (inner_node->input(0) == common_factor ? 1 : 0);
      unique_factors->push_back(inner_node->input(unique_factor_index));
      if (i > 0 && !IsAdd(*node)) {
        OpInfo::TensorProperties lhs;
        OpInfo::TensorProperties rhs;
        TF_RETURN_IF_ERROR(GetTensorProperties(unique_factors->front(), &lhs));
        TF_RETURN_IF_ERROR(GetTensorProperties(unique_factors->back(), &rhs));
        *shapes_match = ShapesSymbolicallyEqual(lhs, rhs);
      }
    }
    return Status::OK();
  }

  bool IsRewritten(const NodeDef* node) const {
    // if graph rewrite happens in multiple passes without graph pruning between
    // them, it's possible that rewritten node already exists in a graph
    return rewritten_nodes_.find(node->name()) != rewritten_nodes_.end() ||
           ctx_.node_map->NodeExists(OuterNodeName(node, false)) ||
           ctx_.node_map->NodeExists(OuterNodeName(node, true));
  }

  // keep names of the nodes that were optimized by this stage
  std::unordered_set<string> rewritten_nodes_;
};

// Binary associative ops can be re-ordered to minimize the number of broadcasts
// and the size of a temporary tensors.
//
// Example: [a, c] - scalars, [b, d] - matrices
//   @ - binary associative op (Add or Mul)
//   @* - broadcast
//
//           @                      @*
//        /     \                /      \
//      @*       @*      ->     @        @
//    /   \    /   \          /   \    /   \
//   a     b  c     d        a     c  b     d
class MinimizeBroadcasts : public ArithmeticNodesGroupOptimizerStage {
 public:
  explicit MinimizeBroadcasts(const GraphOptimizerContext& ctx,
                              const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticNodesGroupOptimizerStage("MinimizeBroadcasts", ctx, ctx_ext) {
  }
  ~MinimizeBroadcasts() override = default;

  bool IsSupported(const NodeDef* node) const override {
    if (!IsBinaryAssociative(*node)) return false;

    // has a symbolically defined shape with broadcastable inputs
    OpInfo::TensorProperties properties;
    Status has_properties = GetTensorProperties(node->name(), &properties);
    return has_properties.ok() && ShapeIsSymbolicallyDefined(properties) &&
           HasAllInputsBroadcastableToShape(*node, properties);
  }

 protected:
  bool IsBinaryAssociative(const NodeDef& node) const {
    return IsMul(node) || IsAdd(node);
  }

  bool IsSameOp(const OptimizedNodesGroup& group, const NodeDef& node) const {
    return group.root_node->op() == node.op();
  }

  // Check if a node can be absorbed by current OptimizedNodesGroup
  bool IsAbsorbableByOptimizedNodesGroup(const OptimizedNodesGroup& group,
                                         const string& input) const override {
    NodeDef* node;
    Status node_status = GetInputNode(input, &node);
    if (!node_status.ok()) return false;

    if (!IsSameOp(group, *node)) {
      return false;
    }
    if (IsInPreserveSet(*node) || IsAlreadyOptimized(*node)) {
      return false;
    }
    if (IsDrivenByControlDependency(*node) || DrivesControlDependency(*node)) {
      return false;
    }
    if (!IsOnTheSameDevice(group, *node)) {
      return false;
    }
    // Optimized nodes updated in place, and that would break the graph, if the
    // node has multiple output consumers
    if (NumNonControlOutputs(*node, *ctx_.node_map) != 1) {
      return false;
    }
    // All input shapes must be broadcastable to the node shape
    OpInfo::TensorProperties properties;
    Status has_properties = GetTensorProperties(input, &properties);
    return has_properties.ok() &&
           HasAllInputsBroadcastableToShape(*node, properties);
  }

  std::size_t CountUniqueShapes(const std::vector<InputAndShape>& inputs) {
    std::set<string> sigs;
    for (const auto& ias : inputs) {
      sigs.insert(ShapeSignature(ias.shape));
    }
    return sigs.size();
  }

  string RewriteOptimizedNodesGroup(const OptimizedNodesGroup& group) override {
    if (CountUniqueShapes(group.inputs) <= 1) {
      // nothing to optimize when all shapes are the same
      return group.root_node->name();
    }

    auto num_nodes = /*root*/ 1 + group.optimized_nodes.size();
    auto num_inputs = group.inputs.size();
    CHECK_EQ(num_nodes, num_inputs - 1)
        << "Can't build a tree with " << num_inputs << " inputs, using "
        << num_nodes << "binary op nodes.";

    std::deque<InputAndShape> add_ops(group.inputs.begin(), group.inputs.end());
    std::deque<NodeDef*> optimized_nodes(group.optimized_nodes.begin(),
                                         group.optimized_nodes.end());

    // sort inputs by it's shape from smallest to largest
    std::stable_sort(add_ops.begin(), add_ops.end(),
                     [](const InputAndShape& lhs, const InputAndShape& rhs) {
                       return CompareSymbolicallyShapedTensorSizes(lhs.shape,
                                                                   rhs.shape);
                     });

    // If there is an odd number of inputs, last one is the largest, and we want
    // to attach it to the root node, to build a well balanced tree.
    std::deque<InputAndShape> add_ops_leftover;
    if (add_ops.size() % 2 != 0) {
      add_ops_leftover.push_back(add_ops.back());
      add_ops.pop_back();
    }

    // At this point it's guaranteed that add_ops have even number of inputs.
    do {
      const InputAndShape lhs = add_ops.front();
      add_ops.pop_front();
      const InputAndShape rhs = add_ops.front();
      add_ops.pop_front();

      NodeDef* node;
      if (!optimized_nodes.empty()) {
        // re-purpose optimized nodes to build a new tree
        node = optimized_nodes.front();
        optimized_nodes.pop_front();
      } else {
        // or use root node if none optimized nodes left
        node = group.root_node;
      }
      InputAndShape updated_node = UpdateInputs(lhs.input, rhs.input, node);

      // Pushing updated node to the back of a deque will create a wide and
      // short tree, pushing to the front will create a tall tree. We prefer to
      // get a wide tree, it minimizes the potential number of temporary tensors
      // required to keep in memory, though sometimes we can go up to prevent
      // propagating a brodcast from leaves to the root. Example:
      //
      // inputs: [s, s, s, M] (s - scalar, M - matrix)
      // @* - op with broadcast
      //
      //  (only push_back)           @*     (push_front first op)
      //                            /  \
      //       @*                  @    M
      //     /   \                / \
      //    @     @*      ->     @   s
      //   / \   / \            / \
      //  s   s s   M          s   s
      if (add_ops.size() >= 2 &&
          CompareSymbolicallyShapedTensorSizes(add_ops.at(0).shape,
                                               add_ops.at(1).shape)) {
        add_ops.push_front(updated_node);
      } else {
        add_ops.push_back(updated_node);
      }
    } while (add_ops.size() > 1);
    CHECK_EQ(1, add_ops.size());

    // attach the largest tensor to the root op
    if (!add_ops_leftover.empty()) {
      const InputAndShape lhs = add_ops.front();
      add_ops.pop_front();
      const InputAndShape rhs = add_ops_leftover.front();
      InputAndShape updated_node =
          UpdateInputs(lhs.input, rhs.input, group.root_node);
      add_ops.push_back(updated_node);
    }

    return add_ops.front().input;
  }

  InputAndShape UpdateInputs(const string& input_0, const string& input_1,
                             NodeDef* node) {
    string old_input_0 = node->input(0);
    string old_input_1 = node->input(1);

    // Update inputs only if they changed
    if (old_input_0 != input_0 || old_input_1 != input_1) {
      node->set_input(0, input_0);
      node->set_input(1, input_1);
      // Invalidate node properties (shape)
      ctx_.graph_properties->ClearOutputProperties(node->name());
      ctx_.graph_properties->ClearInputProperties(node->name());
      // Update the node map
      ctx_.node_map->RemoveOutput(NodeName(old_input_0), node->name());
      ctx_.node_map->RemoveOutput(NodeName(old_input_1), node->name());
      ctx_.node_map->AddOutput(NodeName(input_0), node->name());
      ctx_.node_map->AddOutput(NodeName(input_1), node->name());
      // Add updated node to optimization queue
      AddToOptimizationQueue(node);
    }

    // Do not add updated node to any other group
    AddToOptimizedNodes(node);

    TensorShapeProto shape;  // shape is not important at this point
    return InputAndShape(node->name(), shape);
  }
};

// Removes inverse transpose nodes
class RemoveIdentityTranspose : public ArithmeticOptimizerStage {
 public:
  explicit RemoveIdentityTranspose(const GraphOptimizerContext& ctx,
                                   const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveIdentityTranspose", ctx, ctx_ext) {}
  ~RemoveIdentityTranspose() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsTranspose(*node) || IsConjugateTranspose(*node);
  }

  // TODO(rmlarsen): Forward control dependencies on the bypassed
  // transpose nodes.
  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    TF_RETURN_IF_ERROR(EnsureNodeIsSupported(node));

    NodeDef* input;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &input));
    NodeDef* node_perm;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &node_perm));
    if (!IsConstant(*node_perm)) {
      return Status::OK();
    }
    std::vector<int64> node_perm_values;
    TF_RETURN_IF_ERROR(GetPermutation(*node_perm, &node_perm_values));
    if (input->op() == node->op()) {
      // Remove pairs of transposes that cancel each other.
      NodeDef* input_perm;
      TF_RETURN_IF_ERROR(GetInputNode(input->input(1), &input_perm));
      if (!IsConstant(*input_perm)) {
        return Status::OK();
      }
      std::vector<int64> input_perm_values;
      TF_RETURN_IF_ERROR(GetPermutation(*input_perm, &input_perm_values));
      if (AreInversePermutations(node_perm_values, input_perm_values)) {
        *simplified_node_name = input->input(0);
      }
    } else {
      // Remove simple identity transposes.
      if (IsIdentityPermutation(node_perm_values)) {
        *simplified_node_name = node->input(0);
      }
    }
    return Status::OK();
  }

 private:
  Status GetPermutation(const NodeDef& node_perm,
                        std::vector<int64>* perm64) const {
    std::vector<int> perm32;
    if (ValuesFromConstNode(node_perm, &perm32)) {
      perm64->reserve(perm32.size());
      for (int val : perm32) {
        perm64->push_back(static_cast<int64>(val));
      }
      return Status::OK();
    }
    if (ValuesFromConstNode(node_perm, perm64)) {
      return Status::OK();
    }
    return errors::InvalidArgument("Couldn't extract permutation from ",
                                   node_perm.name());
  }

  bool AreInversePermutations(const std::vector<int64>& a,
                              const std::vector<int64>& b) {
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

  bool IsIdentityPermutation(const std::vector<int64>& perm) {
    for (int64 i = 0; i < perm.size(); ++i) {
      if (i != perm[i]) {
        return false;
      }
    }
    return true;
  }
};

// Remove redundant Bitcasts.
// 1) Remove Bitcast whose source type and destination type are equal
// 2) Rewrite Bitcast(Bitcast(x, type1), type2) => Bitcast(x, type2)
class RemoveRedundantBitcastStage : public ArithmeticOptimizerStage {
 public:
  explicit RemoveRedundantBitcastStage(
      const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveRedundantBitcast", ctx, ctx_ext) {}
  ~RemoveRedundantBitcastStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsBitcast(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    TF_RETURN_IF_ERROR(EnsureNodeIsSupported(node));

    // Bypass Bitcast whose source type and destination type are equal.
    if (GetSourceDataType(*node) == GetDestinationDataType(*node)) {
      *simplified_node_name = node->input(0);
      return Status::OK();
    }

    NodeDef* bitcast;
    TF_RETURN_IF_ERROR(GetInputNode(node->name(), &bitcast));
    NodeDef* operand;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &operand));

    if (IsBitcast(*operand)) {
      // Bitcast(Bitcast(x, type1), type2) => Bitcast(x, type2)
      bitcast->set_input(0, operand->input(0));
      SetSourceDataType(GetSourceDataType(*operand), bitcast);
      ctx_.node_map->UpdateInput(bitcast->name(), bitcast->input(0),
                                 operand->input(0));
      AddToOptimizationQueue(bitcast);
      *simplified_node_name = bitcast->name();
    }

    return Status::OK();
  }
};

// Remove Casts whose source type and destination type are equal.
class RemoveRedundantCastStage : public ArithmeticOptimizerStage {
 public:
  explicit RemoveRedundantCastStage(const GraphOptimizerContext& ctx,
                                    const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveRedundantCast", ctx, ctx_ext) {}
  ~RemoveRedundantCastStage() override = default;

  bool IsSupported(const NodeDef* node) const override { return IsCast(*node); }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    TF_RETURN_IF_ERROR(EnsureNodeIsSupported(node));

    // Bypass Cast whose source type and destination type are equal.
    if (GetSourceDataType(*node) == GetDestinationDataType(*node)) {
      *simplified_node_name = node->input(0);
    }
    return Status::OK();
  }
};

class RemoveNegationStage : public ArithmeticOptimizerStage {
 public:
  explicit RemoveNegationStage(const GraphOptimizerContext& ctx,
                               const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveNegation", ctx, ctx_ext) {}
  ~RemoveNegationStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsAdd(*node) || IsSub(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    const string node_name = node->name();
    NodeDef* x;
    NodeDef* y;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &x));
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &y));
    bool updated = false;
    if (IsAdd(*node)) {
      if (IsNeg(*x)) {
        // (-a) + b = b - a
        node->set_op("Sub");
        node->mutable_input()->SwapElements(0, 1);
        node->set_input(1, x->input(0));
        node->add_input(AsControlDependency(x->name()));
        ctx_.node_map->AddOutput(NodeName(x->input(0)), node_name);
        updated = true;
      } else if (IsNeg(*y)) {
        // a + (-b) = a - b
        node->set_op("Sub");
        node->set_input(1, y->input(0));
        node->add_input(AsControlDependency(y->name()));
        ctx_.node_map->AddOutput(NodeName(y->input(0)), node_name);
        updated = true;
      }
    } else if (IsSub(*node)) {
      if (IsNeg(*y)) {
        // a - (-b) = a + b
        node->set_op("Add");
        node->set_input(1, y->input(0));
        node->add_input(AsControlDependency(y->name()));
        ctx_.node_map->AddOutput(NodeName(y->input(0)), node_name);
        updated = true;
      }
    }
    if (updated) {
      AddToOptimizationQueue(node);
    }
    return Status::OK();
  }
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
  if (node1.attr().size() != node2.attr().size()) {
    return false;
  }
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

namespace {

bool FeedsInPlaceOp(const SimpleGraphView& graph_view, const NodeDef& node) {
  const std::unordered_set<string> op_types_to_traverse = {
      node.op(),    "Identity", "IdentityN", "Reshape",
      "ExpandDims", "Enter",    "Switch",    "Merge"};
  int node_idx = graph_view.index(node.name());
  std::set<int> node_fanout;
  graph_view.DepthFirstSearch(op_types_to_traverse, node_idx, &node_fanout);
  for (int fanout : node_fanout) {
    if (ModifiesInputsInPlace(graph_view.graph()->node(fanout))) {
      return true;
    }
  }
  return false;
}

}  // namespace

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
  SimpleGraphView graph_view;
  if (!graph_view.Initialize(*optimized_graph_).ok()) {
    LOG(WARNING) << "Failed to build SimpleGraphView.";
    return;
  }
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
      // If either node feeds an inplace op, deduping them may cause data races.
      // For example: If we dedup nodes initializing two independent inplace
      // accumulations, they will write to the same buffer, clobbering each
      // other's results.
      if (FeedsInPlaceOp(graph_view, *rep) ||
          FeedsInPlaceOp(graph_view, *node)) {
        continue;
      }
      const std::set<NodeDef*>& fanouts = node_map_->GetOutputs(node->name());
      for (NodeDef* fanout : fanouts) {
        for (int i = 0; i < fanout->input_size(); ++i) {
          string* name = fanout->mutable_input(i);
          int position;
          const string nodename = ParseNodeName(*name, &position);
          if (nodename == node->name()) {
            // Update name in-place.
            if (position > 0) {
              *name = StrCat(rep->name(), ":", position);
            } else if (position == 0) {
              *name = rep->name();
            } else {
              *name = StrCat("^", rep->name());
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

void ArithmeticOptimizer::ForwardControlDependencies(
    NodeDef* target_node, const std::vector<const NodeDef*>& src_nodes) {
  for (const auto& src : src_nodes) {
    for (int i = src->input_size() - 1; i >= 0; --i) {
      if (IsControlInput(src->input(i))) {
        *target_node->add_input() = src->input(i);
        node_map_->AddOutput(NodeName(src->input(i)), target_node->name());
      } else {
        break;
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
    NodeDef* reshape = const_cast<NodeDef*>(node);
    int output_pos = 0;
    string input_node_name = ParseNodeName(reshape->input(0), &output_pos);
    const NodeDef* input = node_map_->GetNode(input_node_name);
    if (input->op() == "Reshape" && !HasControlInputs(*input)) {
      reshape->set_input(0, input->input(0));
      node_map_->UpdateInput(reshape->name(), input->name(), input->input(0));
      nodes_to_simplify->PushBack(reshape);
      return reshape->name();
    }

    // If the reshape is a no-op, forward its input to its consumers, unless it
    // anchors a control dependency since we want to make sure that control
    // dependency is triggered.
    if (ReshapeIsIdentity(*reshape, *input, output_pos, *graph_properties_) &&
        !HasControlInputs(*reshape)) {
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
        (str_util::StrContains(device, DEVICE_CPU) ||
         str_util::StrContains(device, DEVICE_GPU))) {
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
          ForwardControlDependencies(new_transpose, {cast, node});
          return new_cast->name();
        }
      }
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
            ForwardControlDependencies(scaled_weights, {source});

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
    const DataType type = GetDataTypeFromAttr(*node, "T");
    bool is_complex = (type == DT_COMPLEX64) || (type == DT_COMPLEX128);
    string dontcare;
    string device;
    bool is_on_cpu =
        DeviceNameUtils::SplitDeviceName(node->device(), &dontcare, &device) &&
        str_util::StrContains(device, DEVICE_CPU);
    if (!is_complex || is_on_cpu) {
      NodeDef* new_square_node = AddNode(*node, "square", /*copy_node=*/true);
      new_square_node->set_op("Square");
      for (int i = 1; i < new_square_node->input_size(); ++i) {
        new_square_node->set_input(i - 1, new_square_node->input(i));
      }
      new_square_node->mutable_input()->RemoveLast();
      return new_square_node->name();
    }
  }

  if (IsAggregate(*node) && NumNonControlInputs(*node) > 0) {
    // Discard aggregate nodes with a single input and no control dependencies.
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
      MaybeAddControlInput(NodeName(node->input(0)), new_const_node,
                           optimized_graph_, node_map_.get());
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

      ForwardControlDependencies(new_mul_node, {node});
      return new_mul_node->name();
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
      }
      if (b_is_foldable) {
        const string attr_b =
            node->op() == "BatchMatMul" ? "adj_y" : "transpose_b";
        FlipBooleanAttr(attr_b, new_op);
        new_op->set_input(1, b->input(0));
        node_map_->UpdateInput(new_op->name(), b->name(), b->input(0));
      }
      std::vector<const NodeDef*> deps_to_forward({node});
      if (a_is_foldable) {
        deps_to_forward.push_back(a);
      }
      if (b_is_foldable) {
        deps_to_forward.push_back(b);
      }
      ForwardControlDependencies(new_op, deps_to_forward);
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
      ForwardControlDependencies(new_op, {node, input});
      return new_op->name();
    }
  }

  return "";
}

Status ArithmeticOptimizer::SimplifyArithmeticOps(bool can_use_shapes) {
  SetVector<NodeDef*> nodes_to_simplify;
  nodes_to_simplify.Reserve(optimized_graph_->node_size());
  for (int i = 0; i < optimized_graph_->node_size(); ++i) {
    nodes_to_simplify.PushBack(optimized_graph_->mutable_node(i));
  }

  const GraphOptimizerContext ctx(&nodes_to_preserve_, optimized_graph_,
                                  graph_properties_.get(), node_map_.get());
  const ArithmeticOptimizerContext ctx_ext(&nodes_to_simplify);

  // Stop pipeline after first stage returning non-empty simplified tensor name.
  const auto stop = [](const string& result) { return !result.empty(); };
  GraphOptimizerStagePipeline<string> pipeline(stop);

  if (options_.combine_add_to_addn && can_use_shapes)
    pipeline.AddStage<AddOpsRewriteStage>(ctx, ctx_ext);
  if (options_.hoist_common_factor_out_of_aggregation && can_use_shapes)
    pipeline.AddStage<HoistCommonFactorOutOfAggregation>(ctx, ctx_ext);
  if (options_.minimize_broadcasts && can_use_shapes)
    pipeline.AddStage<MinimizeBroadcasts>(ctx, ctx_ext);
  if (options_.remove_identity_transpose && can_use_shapes)
    pipeline.AddStage<RemoveIdentityTranspose>(ctx, ctx_ext);
  if (options_.remove_redundant_bitcast)
    pipeline.AddStage<RemoveRedundantBitcastStage>(ctx, ctx_ext);
  if (options_.remove_redundant_cast)
    pipeline.AddStage<RemoveRedundantCastStage>(ctx, ctx_ext);
  if (options_.remove_negation)
    pipeline.AddStage<RemoveNegationStage>(ctx, ctx_ext);

  VLOG(1) << "Simplify arithmetic ops using " << pipeline.NumStages()
          << " arithmetic optimization stages";

  while (!nodes_to_simplify.Empty()) {
    NodeDef* node = nodes_to_simplify.PopBack();

    // TODO(ezhulenev): move all rewrites into separate stages
    string simplified_tensor = "";
    if (options_.enable_try_simplify_and_replace) {
      simplified_tensor = TrySimplifyAndReplaceUses(node, &nodes_to_simplify);
    }

    // if it was not simplified try to run it through all configured stages
    if (!stop(simplified_tensor)) {
      bool optimized = pipeline.PassThroughAllStages(node, &simplified_tensor);
      if (!optimized) {
        continue;
      }
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
  // Set up helper data structures.
  nodes_to_preserve_ = item.NodesToPreserve();
  fetch_nodes_known_ = !item.fetch.empty();
  *optimized_graph = item.graph;
  optimized_graph_ = optimized_graph;
  node_map_.reset(new NodeMap(optimized_graph_));

  DedupComputations();

  // Perform topological sort on the graph in order to help AddOpsRewrite to
  // optimize larger subgraphs starting from the roots with more inputs.
  TF_RETURN_IF_ERROR(TopologicalSort(optimized_graph_));

  GrapplerItem optimized_item(item, optimized_graph);
  optimized_graph_ = &optimized_item.graph;
  graph_properties_.reset(new GraphProperties(optimized_item));
  const Status status = graph_properties_->InferStatically(false);
  const bool can_use_shapes = status.ok();
  if (!can_use_shapes) {
    VLOG(1) << "Shape inference failed." << status.error_message();
  }

  // Perform the optimizations.
  TF_RETURN_IF_ERROR(SimplifyArithmeticOps(can_use_shapes));

  optimized_graph->Swap(optimized_graph_);
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
