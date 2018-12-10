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
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer_stage.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/symbolic_shapes.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/strided_slice_op.h"

using tensorflow::strings::StrCat;

namespace tensorflow {
namespace grappler {
namespace {

// Mark nodes created or optimized by a stage with a tag.
constexpr char kAddOpsRewriteTag[] =
    "_grappler:ArithmeticOptimizer:AddOpsRewriteStage";
constexpr char kMinimizeBroadcastsTag[] =
    "_grappler:ArithmeticOptimizer:MinimizeBroadcasts";

// Extract values from a Const op to `values`. Returns true if succeeds.
template <typename T>
bool ValuesFromConstNode(const NodeDef& node, std::vector<T>* values) {
  if (node.op() != "Const") {
    return false;
  }

  if (node.attr().count("dtype") == 0 || node.attr().count("value") == 0 ||
      node.attr().at("dtype").type() != DataTypeToEnum<T>::value) {
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

void SetDataTypeToAttr(DataType dtype, const string& attr_name, NodeDef* node) {
  (*node->mutable_attr())[attr_name].set_type(dtype);
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

NodeDef* GetTailOfIdempotentChain(
    const NodeDef& node, const NodeMap& node_map,
    const std::unordered_set<string>& nodes_to_preserve) {
  auto is_idempotent_non_branching = [&](const NodeDef& node) {
    return nodes_to_preserve.find(node.name()) == nodes_to_preserve.end() &&
           IsIdempotent(node) && NumNonControlOutputs(node, node_map) == 1;
  };
  return GetTailOfChain(node, node_map, /*follow_control_input=*/false,
                        is_idempotent_non_branching);
}

// GetElementUnexhaustive tries to get the value of an element in a tensor and
// turn it into complex128 type. It only check for a limited number of data
// types, so it's unexhaustive.
bool GetElementUnexhaustive(const Tensor& t, int i, const std::set<int>& dtypes,
                            complex128* element) {
  if (dtypes.find(t.dtype()) == dtypes.end()) return false;
  switch (t.dtype()) {
    case DT_BFLOAT16:
      *element = complex128(t.flat<bfloat16>()(i));
      return true;
    case DT_HALF:
      *element = complex128(static_cast<double>(t.flat<Eigen::half>()(i)), 0);
      return true;
    case DT_INT32:
      *element = complex128(t.flat<int32>()(i));
      return true;
    case DT_INT64:
      *element = complex128(t.flat<int64>()(i));
      return true;
    case DT_FLOAT:
      *element = complex128(t.flat<float>()(i));
      return true;
    case DT_DOUBLE:
      *element = complex128(t.flat<double>()(i));
      return true;
    case DT_COMPLEX64:
      *element = complex128(t.flat<complex64>()(i));
      return true;
    case DT_COMPLEX128:
      *element = t.flat<complex128>()(i);
      return true;
    default:
      return false;
  }
}

// Graph optimizer context extension specific to ArithmeticOptimizer.
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
  ~ArithmeticOptimizerStage() override = default;

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
          ctx().node_map->AddOutput(NodeName(src->input(i)),
                                    target_node->name());
        } else {
          break;
        }
      }
    }
    DedupControlInputs(target_node);
  }

  bool IsInPreserveSet(const NodeDef& node) const {
    return ctx().nodes_to_preserve->find(node.name()) !=
           ctx().nodes_to_preserve->end();
  }

  // TODO(ezhulenev): move to GraphOptimizerStage?
  bool IsDrivenByControlDependency(const NodeDef& node) const {
    return std::any_of(
        node.input().begin(), node.input().end(),
        [](const string& input) { return IsControlInput(input); });
  }

  // TODO(ezhulenev): move to GraphOptimizerStage?
  bool DrivesControlDependency(const NodeDef& node) const {
    for (const NodeDef* output : ctx().node_map->GetOutputs(node.name())) {
      for (int i = 0; i < output->input_size(); ++i) {
        const TensorId tensor = ParseTensorName(output->input(i));
        if (tensor.node() == node.name() && tensor.index() < 0) {
          return true;
        }
      }
    }
    return false;
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
      : ArithmeticOptimizerStage(name, ctx, ctx_ext) {}
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
      const OptimizedNodesGroup& group, const NodeDef& node) const = 0;

  Status AbsorbInputByOptimizedNodesGroup(const string& input,
                                          OptimizedNodesGroup* group) const {
    std::deque<const string*> input_tensors;
    input_tensors.push_front(&input);

    while (!input_tensors.empty()) {
      const string* input_tensor = input_tensors.front();
      input_tensors.pop_front();

      // Get a node for the input tensor.
      NodeDef* input_node;
      TF_RETURN_IF_ERROR(GetInputNode(*input_tensor, &input_node));

      if (IsAbsorbableByOptimizedNodesGroup(*group, *input_node)) {
        group->optimized_nodes.push_back(input_node);
        for (int i = input_node->input_size() - 1; i >= 0; --i) {
          const string& absorbed_node_input = input_node->input(i);
          // TODO(ezhulenev): support control inputs
          if (IsControlInput(absorbed_node_input)) continue;
          input_tensors.push_front(&absorbed_node_input);
        }
      } else {
        // If input node can't be absorbed, add it to OptimizedNodesGroup input.
        OpInfo::TensorProperties properties;
        TF_RETURN_IF_ERROR(GetTensorProperties(*input_tensor, &properties));
        group->inputs.emplace_back(*input_tensor, properties.shape());
      }
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
      // TODO(ezhulenev): add support for control inputs
      if (IsControlInput(input_i)) continue;
      TF_RETURN_IF_ERROR(AbsorbInputByOptimizedNodesGroup(input_i, group));
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

  string ShapeSignature(const TensorShapeProto& shape) const {
    string signature = strings::StrCat("rank:", shape.dim_size(), ":dim");
    for (int i = 0; i < shape.dim_size(); ++i)
      strings::StrAppend(&signature, ":", shape.dim(i).size());
    return signature;
  }

  void MarkWithTag(const StringPiece tag, NodeDef* node) {
    AddNodeAttr(tag, true, node);
  }

  void MarkAllMembersWithTag(const OptimizedNodesGroup& group,
                             const StringPiece tag) const {
    AddNodeAttr(tag, true, group.root_node);
    for (NodeDef* optimized_node : group.optimized_nodes) {
      AddNodeAttr(tag, true, optimized_node);
    }
  }

  bool IsOnTheSameDevice(const OptimizedNodesGroup& group,
                         const NodeDef& node) const {
    return group.root_node->device() == node.device();
  }

  bool IsInPreserveSet(const NodeDef& node) const {
    return ctx().nodes_to_preserve->find(node.name()) !=
           ctx().nodes_to_preserve->end();
  }

  bool IsMarkedWithTag(const NodeDef& node, const StringPiece tag) const {
    return HasNodeAttr(node, tag);
  }

  bool IsMarkedWithAnyTag(const NodeDef& node, const StringPiece tag1,
                          const StringPiece tag2) const {
    return IsMarkedWithTag(node, tag1) || IsMarkedWithTag(node, tag2);
  }
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
    if (!CanOptimize(*node)) return false;

    // shape must be symbolically defined and all inputs compatible with it
    OpInfo::TensorProperties properties;
    Status has_properties = GetTensorProperties(node->name(), &properties);
    return has_properties.ok() && ShapeIsSymbolicallyDefined(properties) &&
           HasAllInputsBroadcastableToShape(*node, properties);
  }

 protected:
  // Check if a node can be absorbed by current OptimizedNodesGroup
  bool IsAbsorbableByOptimizedNodesGroup(const OptimizedNodesGroup& group,
                                         const NodeDef& node) const override {
    if (!CanOptimize(node)) return false;

    if (!IsOnTheSameDevice(group, node)) {
      return false;
    }
    // with a single output data consumer (presumably if we reach this node from
    // previously absorbed or a root node, it means that this node is not used
    // as an input to any other op, outside of the group)
    if (NumNonControlDataOutputs(node, *ctx().node_map) != 1) {
      return false;
    }
    // All input shapes must be broadcastable to the node shape
    OpInfo::TensorProperties properties;
    Status has_properties = GetTensorProperties(node.name(), &properties);
    return has_properties.ok() &&
           HasAllInputsBroadcastableToShape(node, properties);
  }

  // Node requirements both for a root node and an absorbed node
  bool CanOptimize(const NodeDef& node) const {
    // TODO(ezhulenev): check if AccumulateNV2 can be supported too
    if (!IsAdd(node) && !IsAddN(node)) {
      return false;
    }
    if (IsInPreserveSet(node) || IsMarkedWithTag(node, kAddOpsRewriteTag)) {
      return false;
    }
    // TODO(ezhulenev): relax this condition for root node
    return !(IsDrivenByControlDependency(node) ||
             DrivesControlDependency(node));
  }

  // Rewrite a group of add ops into a single AddN if all input shapes are
  // symbolically equal. If not, create AddN for equal shapes first, and then
  // build an Add tree, minimizing the cost of broadcasts.
  string RewriteOptimizedNodesGroup(const OptimizedNodesGroup& group) override {
    VLOG(2) << "Collapse Add/AddN: root=" << group.root_node->name()
            << " op=" << group.root_node->op()
            << " num_optimized_nodes=" << group.optimized_nodes.size()
            << " num_inputs=" << group.inputs.size();

    // Do not optimize any of the nodes that are part of this group.
    MarkAllMembersWithTag(group, kAddOpsRewriteTag);

    // All new nodes will be placed under the scope of a root node.
    auto root_scope_and_name = ParseNodeScopeAndName(group.root_node->name());

    // Find what shapes are present in the inputs of absorbed nodes.
    std::unordered_map<string, std::vector<InputAndShape>> shape_sig_to_inputs;
    for (const auto& input : group.inputs) {
      shape_sig_to_inputs[ShapeSignature(input.shape)].push_back(input);
    }

    using SigKV = decltype(shape_sig_to_inputs)::value_type;
    VLOG(3) << "Add/AddN group has " << shape_sig_to_inputs.size()
            << " unique shapes: "
            << str_util::Join(shape_sig_to_inputs, ", ",
                              [](string* out, SigKV p) {
                                strings::StrAppend(out, p.first);
                              });

    // Collect all the shapes from representative elements.
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
    if (inputs.size() == 1 || root_node.attr().count("T") == 0) {
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
      ctx().node_map->AddOutput(inputAndShape.input, node_name);
      node->add_input(inputAndShape.input);
    }

    MarkWithTag(kAddOpsRewriteTag, node);
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
    node->add_input(left.input);
    node->add_input(right.input);

    ctx().node_map->AddOutput(left.input, node_name);
    ctx().node_map->AddOutput(right.input, node_name);

    MarkWithTag(kAddOpsRewriteTag, node);
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

        ctx().node_map->AddOutput(common_factor, new_outer_node->name());
        ctx().node_map->AddOutput(new_add_node->name(), new_outer_node->name());

        // Hoist non-shared factors up into the new AddN node.
        for (int i = 0; i < unique_factors.size(); ++i) {
          const string& unique_factor_i = unique_factors[i];
          new_add_node->set_input(i, unique_factor_i);
          ctx().node_map->AddOutput(unique_factor_i, new_add_node->name());
        }

        // Add control deps on add node
        for (const string& ctrl_dep : ctrl_deps) {
          *new_add_node->add_input() = ctrl_dep;
          ctx().node_map->AddOutput(NodeName(ctrl_dep), new_add_node->name());
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
           ctx().node_map->NodeExists(OuterNodeName(node, false)) ||
           ctx().node_map->NodeExists(OuterNodeName(node, true));
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

    if (IsMarkedWithAnyTag(*node, kMinimizeBroadcastsTag, kAddOpsRewriteTag))
      return false;

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
                                         const NodeDef& node) const override {
    if (!IsSameOp(group, node)) {
      return false;
    }
    if (IsInPreserveSet(node)) {
      return false;
    }
    // Nodes optimized by AddOpsRewrite already have optimal broadcasts.
    if (IsMarkedWithAnyTag(node, kMinimizeBroadcastsTag, kAddOpsRewriteTag)) {
      return false;
    }
    if (IsDrivenByControlDependency(node) || DrivesControlDependency(node)) {
      return false;
    }
    if (!IsOnTheSameDevice(group, node)) {
      return false;
    }
    // Optimized nodes updated in place, and that would break the graph, if the
    // node has multiple output consumers
    if (NumNonControlOutputs(node, *ctx().node_map) != 1) {
      return false;
    }
    // All input shapes must be broadcastable to the node shape
    OpInfo::TensorProperties properties;
    Status has_properties = GetTensorProperties(node.name(), &properties);
    return has_properties.ok() &&
           HasAllInputsBroadcastableToShape(node, properties);
  }

  std::size_t CountUniqueShapes(const std::vector<InputAndShape>& inputs) {
    std::set<string> sigs;
    for (const auto& ias : inputs) {
      sigs.insert(ShapeSignature(ias.shape));
    }
    return sigs.size();
  }

  string RewriteOptimizedNodesGroup(const OptimizedNodesGroup& group) override {
    VLOG(2) << "Minimize broadcast: root=" << group.root_node->name()
            << " op=" << group.root_node->op()
            << " num_optimized_nodes=" << group.optimized_nodes.size();

    // Do not optimize any of the nodes that are part of this group.
    MarkAllMembersWithTag(group, kMinimizeBroadcastsTag);

    if (CountUniqueShapes(group.inputs) <= 1) {
      VLOG(3) << "Skip min-bcast group with single unique shape";
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
        node = optimized_nodes.back();
        optimized_nodes.pop_back();
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
      ctx().graph_properties->ClearOutputProperties(node->name());
      ctx().graph_properties->ClearInputProperties(node->name());
      // Update the node map
      ctx().node_map->RemoveOutput(NodeName(old_input_0), node->name());
      ctx().node_map->RemoveOutput(NodeName(old_input_1), node->name());
      ctx().node_map->AddOutput(NodeName(input_0), node->name());
      ctx().node_map->AddOutput(NodeName(input_1), node->name());
      // Add updated node to optimization queue
      AddToOptimizationQueue(node);
    }

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

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    TF_RETURN_IF_ERROR(EnsureNodeIsSupported(node));
    NodeDef* tail = node;
    tail = GetTailOfIdempotentChain(*tail, *ctx().node_map,
                                    *ctx().nodes_to_preserve);
    NodeDef* first_transpose;
    TF_RETURN_IF_ERROR(GetInputNode(tail->input(0), &first_transpose));

    NodeDef* node_perm;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &node_perm));
    if (!IsConstant(*node_perm)) {
      return Status::OK();
    }
    std::vector<int64> node_perm_values;
    TF_RETURN_IF_ERROR(GetPermutation(*node_perm, &node_perm_values));
    if (first_transpose->op() == node->op()) {
      // Remove pairs of transposes that cancel each other.
      NodeDef* first_transpose_perm;
      TF_RETURN_IF_ERROR(
          GetInputNode(first_transpose->input(1), &first_transpose_perm));
      if (!IsConstant(*first_transpose_perm)) {
        return Status::OK();
      }
      std::vector<int64> first_transpose_perm_values;
      TF_RETURN_IF_ERROR(
          GetPermutation(*first_transpose_perm, &first_transpose_perm_values));
      if (AreInversePermutations(node_perm_values,
                                 first_transpose_perm_values)) {
        if (tail == node) {
          // Bypass adjacent pair.
          *simplified_node_name = first_transpose->input(0);
        } else {
          // Bypass pair connected through chain.
          tail->set_input(0, first_transpose->input(0));
          ctx().node_map->UpdateInput(tail->name(), first_transpose->name(),
                                      first_transpose->input(0));
          ForwardControlDependencies(tail, {first_transpose});
          *simplified_node_name = node->input(0);
        }
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

// An involution is an element-wise function f(x) that is its own inverse,
// i.e. f(f(x)) = x. If we can find a chain of ops
//   f->op1->op2->...opn->f
// where op1 through opn preserve the values of their inputs, we can remove
// the two instances of the involution from the graph, since they cancel
// each other.
class RemoveInvolution : public ArithmeticOptimizerStage {
 public:
  explicit RemoveInvolution(const GraphOptimizerContext& ctx,
                            const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveInvolution", ctx, ctx_ext) {}
  ~RemoveInvolution() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsInvolution(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    NodeDef* tail = GetTailOfValuePreservingChain(*node, *ctx().node_map,
                                                  *ctx().nodes_to_preserve);

    NodeDef* involution;
    TF_RETURN_IF_ERROR(GetInputNode(tail->input(0), &involution));

    if (involution->op() == node->op()) {
      // Skip both *node and *involution since they cancel each other.
      if (tail == node) {
        // The two nodes to eliminate are adjacent.
        *simplified_node_name = involution->input(0);
      } else {
        tail->set_input(0, involution->input(0));
        ctx().node_map->UpdateInput(tail->name(), involution->name(),
                                    involution->input(0));
        *simplified_node_name = node->input(0);
      }
    }

    return Status::OK();
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
    AttrSlice attrs(*node);
    DataType input_type;
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "T", &input_type));
    DataType output_type;
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "type", &output_type));
    if (input_type == output_type) {
      *simplified_node_name = node->input(0);
      return Status::OK();
    }

    NodeDef* bitcast;
    TF_RETURN_IF_ERROR(GetInputNode(node->name(), &bitcast));
    NodeDef* operand;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &operand));

    if (IsBitcast(*operand)) {
      AttrSlice operand_attrs(*operand);
      DataType operand_input_type;
      TF_RETURN_IF_ERROR(GetNodeAttr(operand_attrs, "T", &operand_input_type));
      // Bitcast(Bitcast(x, type1), type2) => Bitcast(x, type2)
      bitcast->set_input(0, operand->input(0));
      SetDataTypeToAttr(operand_input_type, "T", bitcast);
      ctx().node_map->UpdateInput(bitcast->name(), bitcast->input(0),
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
    AttrSlice attrs(*node);
    DataType input_type;
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "SrcT", &input_type));
    DataType output_type;
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "DstT", &output_type));
    if (input_type == output_type) {
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
    NodeDef* x;
    NodeDef* y;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &x));
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &y));
    bool updated = false;
    if (IsNeg(*y)) {
      // a - (-b) = a + b or  a + (-b) = a - b
      ForwardControlDependencies(node, {y});
      ctx().node_map->UpdateInput(node->name(), node->input(1), y->input(0));
      node->set_op(IsAdd(*node) ? "Sub" : "Add");
      node->set_input(1, y->input(0));
      updated = true;
    } else if (IsAdd(*node) && IsNeg(*x)) {
      // (-a) + b = b - a
      ForwardControlDependencies(node, {x});
      ctx().node_map->UpdateInput(node->name(), node->input(0), x->input(0));
      node->set_op("Sub");
      node->mutable_input()->SwapElements(0, 1);
      node->set_input(1, x->input(0));
      updated = true;
    }
    if (updated) {
      AddToOptimizationQueue(node);
    }
    return Status::OK();
  }
};

class RemoveLogicalNotStage : public ArithmeticOptimizerStage {
 public:
  explicit RemoveLogicalNotStage(const GraphOptimizerContext& ctx,
                                 const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveLogicalNot", ctx, ctx_ext) {}
  ~RemoveLogicalNotStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsLogicalNot(*node) && !IsInPreserveSet(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    const string node_name = node->name();
    NodeDef* input;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &input));
    if (IsInPreserveSet(*input) ||
        NumNonControlOutputs(*input, *ctx().node_map) > 1) {
      return Status::OK();
    }
    string new_op;
    if (IsEqual(*input)) {
      new_op = "NotEqual";
    } else if (IsNotEqual(*input)) {
      new_op = "Equal";
    } else if (IsLess(*input)) {
      new_op = "GreaterEqual";
    } else if (IsLessEqual(*input)) {
      new_op = "Greater";
    } else if (IsGreater(*input)) {
      new_op = "LessEqual";
    } else if (IsGreaterEqual(*input)) {
      new_op = "Less";
    }
    if (!new_op.empty()) {
      input->set_op(new_op);
      *simplified_node_name = input->name();
    }
    return Status::OK();
  }
};

// This optimization hoists the common prefix of unary ops of the inputs to
// concat out of the concat, for example:
//    Concat([Exp(Sin(x)), Exp(Sin(y)), Exp(Sin(z))])
// becomes
//    Exp(Sin(Concat([x, y, z]))).
// Similarly, it will hoist the common postfix of unary ops into Split or
// SplitV nodes, for example:
//    [Exp(Sin(y)) for y in Split(x)]
// becomes
//    [y for y in Split(Exp(Sin(x))]
//
// TODO(rmlarsen): Support casting. We would have to change the type attribute
// on the concat/split node.
// TODO(rmlarsen): Handle Enter/Exit.
class HoistCWiseUnaryChainsStage : public ArithmeticOptimizerStage {
 public:
  explicit HoistCWiseUnaryChainsStage(const GraphOptimizerContext& ctx,
                                      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("", ctx, ctx_ext) {}

  ~HoistCWiseUnaryChainsStage() override = default;

  struct ChainLink {
    ChainLink() = default;
    ChainLink(NodeDef* _node, int _port_origin)
        : node(_node), port_origin(_port_origin) {}
    NodeDef* node;    // Node in a chain.
    int port_origin;  // Port on concat/split node from which this chain
                      // originates.

    bool operator<(const ChainLink& other) const {
      if (port_origin < other.port_origin) {
        return true;
      } else if (port_origin > other.port_origin) {
        return false;
      } else {
        return node->name() < other.node->name();
      }
    }
  };

  // We use an ordinary set sorted on port and node name, so the order, and
  // hence the node name used for the hoisted chain, will be deterministic.
  using ChainLinkSet = std::set<ChainLink>;

  bool IsSupported(const NodeDef* node) const override {
    if (IsInPreserveSet(*node)) return false;
    if (IsConcat(*node) && node->attr().count("N") != 0) {
      const int n = node->attr().at("N").i();
      return n > 1;
    } else if ((IsSplit(*node) || IsSplitV(*node)) &&
               node->attr().count("num_split") != 0) {
      const int num_split = node->attr().at("num_split").i();
      if (NumNonControlOutputs(*node, *ctx().node_map) > num_split) {
        // TODO(rmlarsen): Remove this constraint when we have optimizations
        // in place for merging slices into splits.
        return false;
      }
      return num_split > 1 && !IsAlreadyOptimized(*node);
    }
    return false;
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    node_is_concat_ = IsConcat(*node);
    int prefix_length;
    std::set<string> ctrl_inputs;
    ChainLinkSet tails;
    TF_RETURN_IF_ERROR(
        FindCommonUnaryOpChain(*node, &prefix_length, &tails, &ctrl_inputs));
    if (prefix_length > 0 && !tails.empty()) {
      TF_RETURN_IF_ERROR(
          HoistUnaryOpChain(prefix_length, tails, &ctrl_inputs, node));
    }
    return Status::OK();
  }

 private:
  // Returns the length of the common unary chain of ops that can be
  // hoisted to the other side of concat or split.
  Status FindCommonUnaryOpChain(const NodeDef& root_node, int* prefix_length,
                                ChainLinkSet* tails,
                                std::set<string>* ctrl_inputs) const {
    *prefix_length = 0;
    // Follow the chains starting at each concat input or split output as long
    // as all the following conditions hold:
    //   1. The ops in all chains are the same.
    //   2. The ops are unary elemenwise op.
    //   3. The op output has only a single consumer (concat only).
    ChainLinkSet cur_tails;
    TF_RETURN_IF_ERROR(InitializeChains(root_node, &cur_tails));
    if (cur_tails.size() < 2) {
      return Status::OK();
    }
    ctrl_inputs->clear();
    bool stop = false;
    while (!stop && !cur_tails.empty() &&
           OpsAreSafeToHoist(root_node, cur_tails)) {
      // We found one more link that can be hoisted.
      ++(*prefix_length);
      tails->swap(cur_tails);
      GatherControlInputs(ctrl_inputs, *tails);

      // Advance tail pointers to the next level.
      TF_RETURN_IF_ERROR(AdvanceTails(*tails, &cur_tails, &stop));
    }
    return Status::OK();
  }

  // Hoists the chains to the other side of concat or split and attaches the
  // control inputs gathered from them to the concat or split node.
  Status HoistUnaryOpChain(const int prefix_length, const ChainLinkSet& tails,
                           std::set<string>* ctrl_inputs, NodeDef* root_node) {
    if (tails.empty()) {
      return Status::OK();
    }
    AddToOptimizationQueue(root_node);
    optimized_nodes_.insert(root_node->name());
    if (node_is_concat_) {
      AddControlInputs(ctrl_inputs, root_node);
      return HoistChainForConcat(prefix_length, tails, root_node);
    } else {
      return HoistChainForSplit(prefix_length, tails, ctrl_inputs, root_node);
    }
  }

  void GatherControlInputs(std::set<string>* ctrl_inputs,
                           const ChainLinkSet& ops) const {
    for (const auto& link : ops) {
      const NodeDef* node = link.node;
      for (int i = node->input_size() - 1; i >= 0; --i) {
        const string& input = node->input(i);
        if (!IsControlInput(input)) break;
        ctrl_inputs->insert(input);
      }
    }
  }

  void AddControlInputs(std::set<string>* new_ctrl_inputs,
                        NodeDef* node) const {
    for (int i = node->input_size() - 1; i >= 0; --i) {
      const string& existing_input = node->input(i);
      if (!IsControlInput(existing_input)) break;
      new_ctrl_inputs->erase(existing_input);
    }
    for (const string& new_input : *new_ctrl_inputs) {
      ctx().node_map->AddOutput(NodeName(new_input), node->name());
      node->add_input(new_input);
    }
  }

  Status InitializeChains(const NodeDef& node, ChainLinkSet* tails) const {
    if (node_is_concat_) {
      // Handle concat nodes by looking backwards in the graph.
      TF_RETURN_IF_ERROR(CheckAttrExists(node, "N"));
      const int n = node.attr().at("N").i();
      const int start = node.op() == "Concat" ? 1 : 0;
      const int end = start + n;
      // Set up tail pointers to point to the immediate inputs to Concat.
      for (int input_port = start; input_port < end; ++input_port) {
        if (IsControlInput(node.input(input_port))) {
          return errors::FailedPrecondition(
              "Got control input ", node.input(input_port),
              " where normal input was expected.");
        }
        NodeDef* tail;
        TF_RETURN_IF_ERROR(GetInputNode(node.input(input_port), &tail));
        tails->insert(ChainLink(tail, input_port));
      }
      return Status::OK();
    } else {
      // Handle split nodes by looking forwards in the graph.
      const auto& outputs = ctx().node_map->GetOutputs(node.name());
      for (NodeDef* output : outputs) {
        if (IsControlInput(output->input(0))) continue;
        TensorId tensor_id = ParseTensorName(output->input(0));
        if (tensor_id.node() == node.name()) {
          tails->insert(ChainLink(output, tensor_id.index()));
        } else {
          // This output node has a non-control input other than the split node,
          // abort.
          tails->clear();
          return Status::OK();
        }
      }
    }
    return Status::OK();
  }

  bool OpsAreSafeToHoist(const NodeDef& root_node,
                         const ChainLinkSet& ops) const {
    if (ops.empty()) return true;
    const NodeDef* op0 = ops.begin()->node;
    if (ModifiesFrameInfo(*op0) || !IsUnaryElementWise(*op0)) return false;
    for (const auto& link : ops) {
      const NodeDef* op = link.node;
      if (op->device() != root_node.device() || op->op() != op0->op() ||
          IsInPreserveSet(*op)) {
        return false;
      }
      if (ctx().node_map->GetOutputs(op->name()).size() > 1) {
        // TODO(rmlarsen): Allow outgoing control edges.
        return false;
      }
    }
    return true;
  }

  Status AdvanceTails(const ChainLinkSet& tails, ChainLinkSet* new_tails,
                      bool* stop) const {
    *stop = true;
    new_tails->clear();
    for (const auto& link : tails) {
      const NodeDef* tail = link.node;
      if (node_is_concat_) {
        if (tail->input_size() == 0 || IsControlInput(tail->input(0))) {
          return Status::OK();
        }
        NodeDef* new_tail;
        TF_RETURN_IF_ERROR(GetInputNode(tail->input(0), &new_tail));
        // Remember original port.
        new_tails->insert(ChainLink(new_tail, link.port_origin));
      } else {
        for (NodeDef* new_tail : ctx().node_map->GetOutputs(tail->name())) {
          const TensorId tensor = ParseTensorName(new_tail->input(0));
          if (tensor.node() != tail->name()) {
            return Status::OK();
          }
          // Skip control outputs.
          if (tensor.index() >= 0) {
            // Remember original port.
            new_tails->insert(ChainLink(new_tail, link.port_origin));
          }
        }
      }
    }
    *stop = false;
    return Status::OK();
  }

  Status HoistChainForConcat(const int prefix_length, const ChainLinkSet& tails,
                             NodeDef* concat_node) {
    const string& concat_name = concat_node->name();
    const int first_input = concat_node->op() == "Concat" ? 1 : 0;
    for (const auto& link : tails) {
      NodeDef* tail = CHECK_NOTNULL(link.node);
      const int concat_port = link.port_origin;
      CHECK_GE(concat_port, 0);
      CHECK_LT(concat_port, concat_node->input_size());
      const string concat_input = concat_node->input(concat_port);
      // Hook the node following tail directly into the concat node.
      const string tail_input = tail->input(0);
      concat_node->set_input(concat_port, tail_input);
      ctx().node_map->UpdateInput(concat_name, concat_input, tail_input);

      if (concat_port == first_input) {
        // Update the consumers of concat to consume the end of the chain
        // instead.
        UpdateConsumers(concat_node, concat_input);
        // Reuse nodes in the first chain to process output of concat.
        tail->set_input(0, concat_name);
        ctx().node_map->UpdateInput(tail->name(), tail_input, concat_name);
      }
    }
    return Status::OK();
  }

  Status HoistChainForSplit(const int prefix_length, const ChainLinkSet& tails,
                            std::set<string>* ctrl_inputs,
                            NodeDef* split_node) {
    // Create a new chain before the split node to process the input tensor.
    const string& split_name = split_node->name();
    auto root_scope_and_name = ParseNodeScopeAndName(split_name);

    // We use the first tail node in the set as a template to get the list of
    // ops to apply (starting from the end).
    NodeDef* cur_tail = tails.begin()->node;
    NodeDef* cur_copy = AddCopyNode(
        OptimizedNodeName(root_scope_and_name, cur_tail->name()), cur_tail);
    cur_copy->clear_input();

    // Update the split to take its input from the tail of the new chain.
    const int value_slot = split_node->op() == "SplitV" ? 0 : 1;
    const string orig_input = split_node->input(value_slot);
    split_node->set_input(value_slot, cur_copy->name());
    ctx().node_map->UpdateInput(split_node->name(), orig_input,
                                cur_copy->name());
    TF_RETURN_IF_ERROR(GetInputNode(cur_tail->input(0), &cur_tail));

    // Now walk backwards creating the rest of the chain.
    while (cur_tail != split_node) {
      NodeDef* new_copy = AddCopyNode(
          OptimizedNodeName(root_scope_and_name, cur_tail->name()), cur_tail);
      new_copy->clear_input();
      cur_copy->add_input(new_copy->name());
      ctx().node_map->AddOutput(new_copy->name(), cur_copy->name());
      cur_copy = new_copy;
      TF_RETURN_IF_ERROR(GetInputNode(cur_tail->input(0), &cur_tail));
    }
    // Connect the original input to the head of the new chain.
    cur_copy->add_input(orig_input);
    ctx().node_map->UpdateOutput(NodeName(orig_input), split_name,
                                 cur_copy->name());
    // Make sure all the control inputs are satisfied before running the first
    // node in the new chain.
    AddControlInputs(ctrl_inputs, cur_copy);

    // Connect all consumers of the tail nodes directly to the
    // output port of Split from which the chain started.
    for (const auto& link : tails) {
      UpdateConsumers(link.node,
                      link.port_origin == 0
                          ? split_name
                          : strings::StrCat(split_name, ":", link.port_origin));
    }
    return Status::OK();
  }

  // Update consumers of node to take new_input as input instead.
  void UpdateConsumers(NodeDef* node, const string& new_input) {
    const string& node_name = node->name();
    const std::set<NodeDef*> consumers = ctx().node_map->GetOutputs(node_name);
    for (NodeDef* consumer : consumers) {
      for (int i = 0; i < consumer->input_size(); ++i) {
        if (consumer->input(i) == node_name) {
          consumer->set_input(i, new_input);
          ctx().node_map->UpdateInput(consumer->name(), node_name, new_input);
        }
      }
      AddToOptimizationQueue(consumer);
    }
  }

  bool IsAlreadyOptimized(const NodeDef& node) const {
    return optimized_nodes_.find(node.name()) != optimized_nodes_.end();
  }

 private:
  bool node_is_concat_;
  std::unordered_set<string> optimized_nodes_;
};

class RemoveIdempotentStage : public ArithmeticOptimizerStage {
 public:
  explicit RemoveIdempotentStage(const GraphOptimizerContext& ctx,
                                 const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveIdempotent", ctx, ctx_ext) {}
  ~RemoveIdempotentStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return node->input_size() == 1 && IsIdempotent(*node) &&
           !IsInPreserveSet(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    NodeDef* input;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &input));
    if (input->op() == node->op() && input->device() == node->device()) {
      *simplified_node_name = node->input(0);
    }
    return Status::OK();
  }
};

// Performs the conversion:
// Div(x, Sqrt(y)) => Mul(x, Rsqrt(y))
// TODO(srjoglekar): Generalize to optimize cases like (x / pow(y, z)).
class SqrtDivToRsqrtMulStage : public ArithmeticOptimizerStage {
 public:
  explicit SqrtDivToRsqrtMulStage(const GraphOptimizerContext& ctx,
                                  const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("SqrtDivToRsqrtMul", ctx, ctx_ext) {}
  ~SqrtDivToRsqrtMulStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsAnyDiv(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    NodeDef* y;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &y));
    // Optimize only if divisor is a Sqrt whose output is not being consumed
    // elsewhere.
    if (IsSqrt(*y) && !IsInPreserveSet(*y) &&
        (NumNonControlOutputs(*y, *ctx().node_map) == 1)) {
      // a / sqrt(b) = a * rsqrt(b)
      node->set_op("Mul");
      y->set_op("Rsqrt");
      AddToOptimizationQueue(node);
      AddToOptimizationQueue(y);
    }
    return Status::OK();
  }
};

// Bypass redundant reshape nodes:
//
//   Reshape                    Reshape  <-+
//      ^                                  |
//      |                                  |
//   Reshape       becomes      Reshape    |
//      ^                                  |
//      |                                  |
//    input                      input  ---+
class RemoveRedundantReshape : public ArithmeticOptimizerStage {
 public:
  explicit RemoveRedundantReshape(const GraphOptimizerContext& ctx,
                                  const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveRedundantReshape", ctx, ctx_ext) {}
  ~RemoveRedundantReshape() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsReshape(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    NodeDef* input;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &input));

    // 1. Bypass reshape followed by reshape.
    if (IsReshape(*input) && !HasControlInputs(*input)) {
      node->set_input(0, input->input(0));
      ctx().node_map->UpdateInput(node->name(), input->name(), input->input(0));
      *simplified_node_name = node->name();
      AddToOptimizationQueue(node);
      return Status::OK();
    }

    // 2. If the reshape is a no-op, forward its input to its consumers, unless
    // it anchors a control dependency since we want to make sure that control
    // dependency is triggered.
    if (ReshapeIsIdentity(*node) && !HasControlInputs(*node)) {
      *simplified_node_name = node->input(0);
      return Status::OK();
    }

    return Status::OK();
  }

 private:
  // Returns whether `reshape` is an identity op.
  bool ReshapeIsIdentity(const NodeDef& reshape) {
    OpInfo::TensorProperties reshape_props;
    OpInfo::TensorProperties input_props;

    if (!GetTensorProperties(reshape.name(), &reshape_props).ok() ||
        !GetTensorProperties(reshape.input(0), &input_props).ok()) {
      return false;
    }

    return ShapesSymbolicallyEqual(input_props.shape(), reshape_props.shape());
  }
};

// Reorder casting and value-preserving ops if beneficial.
//
// Original motivation: A common pattern after the layout optimizer is
// casting an uint8 NHWC image to float before transposing it to NCHW. It
// is beneficial to reorder the cast and the transpose to make the transpose
// process smaller amount of data. More generally, this optimization converts
//   Op(Cast(tensor, dst_type))
// to
//   Cast(Op(tensor), dst_type)
// when sizeof(tensor.type) < sizeof(dst_type), and Op is any value-preserving
// Op, i.e. an op that only reorders the elements in its first input. Similarly,
// this optimization converts
//   Cast(Op(tensor), dst_type)
// to
//   Op(Cast(tensor, dst_type))
// when sizeof(tensor.type) > sizeof(dst_type)
//
class ReorderCastLikeAndValuePreserving : public ArithmeticOptimizerStage {
 public:
  explicit ReorderCastLikeAndValuePreserving(
      const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("ReorderCastLikeAndValuePreserving", ctx,
                                 ctx_ext) {}
  ~ReorderCastLikeAndValuePreserving() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return (IsValuePreserving(*node) || IsCastLike(*node)) &&
           !IsCheckNumerics(*node) && NodeIsOnCpuOrGpu(node) &&
           !IsControlFlow(*node) && !IsInPreserveSet(*node);
  }

  Status TrySimplify(NodeDef* consumer, string* simplified_node_name) override {
    NodeDef* producer;
    TF_RETURN_IF_ERROR(GetInputNode(consumer->input(0), &producer));
    const bool producer_is_cast = IsCastLike(*producer);
    const bool can_optimize =
        !IsCheckNumerics(*producer) &&
        ((producer_is_cast && IsValuePreserving(*consumer)) ||
         (IsValuePreserving(*producer) && IsCastLike(*consumer)));
    if (!can_optimize || IsControlFlow(*producer) ||
        producer->device() != consumer->device()) {
      return Status::OK();
    }

    const NodeDef* cast_like_node = producer_is_cast ? producer : consumer;
    const OpDef* cast_like_op_def = nullptr;
    TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(cast_like_node->op(),
                                                         &cast_like_op_def));
    DataType cast_src_type;
    TF_RETURN_IF_ERROR(InputTypeForNode(*cast_like_node, *cast_like_op_def, 0,
                                        &cast_src_type));
    DataType cast_dst_type;
    TF_RETURN_IF_ERROR(OutputTypeForNode(*cast_like_node, *cast_like_op_def, 0,
                                         &cast_dst_type));
    if (!IsFixedSizeType(cast_src_type) || !IsFixedSizeType(cast_dst_type)) {
      return Status::OK();
    } else if (producer_is_cast &&
               DataTypeSize(cast_dst_type) <= DataTypeSize(cast_src_type)) {
      return Status::OK();
    } else if (!producer_is_cast &&
               DataTypeSize(cast_dst_type) >= DataTypeSize(cast_src_type)) {
      return Status::OK();
    }

    // Check that nodes were not already optimized.
    const string optimized_producer_name = OptimizedNodeName(
        ParseNodeScopeAndName(producer->name()), DataTypeString(cast_dst_type));
    const string optimized_consumer_name = OptimizedNodeName(
        ParseNodeScopeAndName(consumer->name()), DataTypeString(cast_src_type));
    const bool is_already_optimized =
        ctx().node_map->NodeExists(optimized_consumer_name) ||
        ctx().node_map->NodeExists(optimized_producer_name);
    if (is_already_optimized) {
      return Status::OK();
    }

    // Add copies of consumer and producer in reverse order.
    NodeDef* input;
    TF_RETURN_IF_ERROR(GetInputNode(producer->input(0), &input));
    // Create new producer node.
    NodeDef* new_producer = AddCopyNode(optimized_consumer_name, consumer);
    new_producer->set_input(0, producer->input(0));
    ctx().node_map->AddOutput(input->name(), new_producer->name());

    // Create new consumer node.
    NodeDef* new_consumer = AddCopyNode(optimized_producer_name, producer);
    new_consumer->set_input(0, new_producer->name());

    NodeDef* new_value_preserving =
        producer_is_cast ? new_producer : new_consumer;
    const DataType new_input_type =
        producer_is_cast ? cast_src_type : cast_dst_type;
    // Update the input type of the value-preserving node. The input and
    // output types of the cast-like nodes remain the same.
    TF_RETURN_IF_ERROR(SetInputType(new_input_type, new_value_preserving));
    // Make sure there is a kernel registered for the value preserving op
    // with the new input type.
    TF_RETURN_IF_ERROR(IsKernelRegisteredForNode(*new_value_preserving));
    ctx().node_map->AddOutput(new_producer->name(), new_consumer->name());

    AddToOptimizationQueue(new_producer);
    *simplified_node_name = new_consumer->name();

    return Status::OK();
  }

 private:
  // Sets the type of the first input to dtype.
  Status SetInputType(DataType dtype, NodeDef* node) {
    const OpDef* op_def = nullptr;
    TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(node->op(), &op_def));
    const OpDef::ArgDef& input_arg = op_def->input_arg(0);
    const string& type_attr_name = input_arg.type_attr();
    if (type_attr_name.empty()) {
      if (input_arg.type() == DT_INVALID || input_arg.type() != dtype) {
        return errors::InvalidArgument("Could not set input type of ",
                                       node->op(), " op to ",
                                       DataTypeString(dtype));
      } else {
        // Op has fixed input type that already matches dtype.
        return Status::OK();
      }
    }
    SetDataTypeToAttr(dtype, type_attr_name, node);
    return Status::OK();
  }
  // This optimization can be dangerous on devices other than CPU and
  // GPU. The transpose might not be implemented for image.type, or
  // might be slower with image.type than with cast_dst_type.
  bool NodeIsOnCpuOrGpu(const NodeDef* node) const {
    using str_util::StrContains;

    string task;
    string device;

    return DeviceNameUtils::SplitDeviceName(node->device(), &task, &device) &&
           (StrContains(device, DEVICE_CPU) || StrContains(device, DEVICE_GPU));
  }

  bool IsFixedSizeType(DataType dtype) {
    return dtype != DT_STRING && dtype != DT_VARIANT && dtype != DT_RESOURCE &&
           !kQuantizedTypes.Contains(dtype);
  }
};

// Fold a multiply of a scalar into the following convolution. This folding
// can jump across nodes that merely reorders data (such as reshape and
// transpose). For example, we can optimize
//
//
//         Conv2D                             Conv2D
//        /      \                           /      \
//    Transpose  weights*       ->     Transpose    Mul
//       |                                |        /   \
//      Mul                               |    weights  scale
//     /   \                              |
//   input  scale**                     input
//
//  *) weights must be a const
// **) scale must be a const scalar
//
// When `weights` and `scale` are constant, `Mul` in the optimized graph can be
// constant-folded, also weights tend to be smaller than the activations.
//
// TODO(jingyue): Fold scalar multiplies to Conv?DBackpropFilter and
// Conv?DBackpropInput.
class FoldMultiplyIntoConv : public ArithmeticOptimizerStage {
 public:
  explicit FoldMultiplyIntoConv(const GraphOptimizerContext& ctx,
                                const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("FoldMultiplyIntoConv", ctx, ctx_ext) {}
  ~FoldMultiplyIntoConv() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsConv2D(*node) || IsConv3D(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
#define TF_RETURN_IF_TRUE(...) \
  if ((__VA_ARGS__)) return Status::OK()

    NodeDef* conv = node;

    NodeDef* weights;
    TF_RETURN_IF_ERROR(GetInputNode(conv->input(1), &weights));

    // Fold the multiply to conv only when the weights are constant, so the
    // multiply can be constant-folded.
    //
    // TODO(jingyue): When the weights aren't constant, this should also help
    // performance a bit and memory usage a lot, since the weights tend to be
    // smaller than the activations.
    TF_RETURN_IF_TRUE(!IsConstant(*weights));

    // Verify that this node was not already optimized.
    const string scaled_weights_node_name =
        OptimizedNodeName(ParseNodeScopeAndName(weights->name()),
                          strings::StrCat("scaled", "_", conv->name()));

    TF_RETURN_IF_TRUE(ctx().node_map->NodeExists(scaled_weights_node_name));

    // Find the tail of value preserving chain entering the Conv node.
    NodeDef* tail = GetTailOfValuePreservingChain(*conv, *ctx().node_map,
                                                  *ctx().nodes_to_preserve);

    NodeDef* source;
    TF_RETURN_IF_ERROR(GetInputNode(tail->input(0), &source));

    // Check that value preserving chain is the only consumer of the Mul output.
    TF_RETURN_IF_TRUE(!IsMul(*source));
    TF_RETURN_IF_TRUE(NumNonControlOutputs(*source, *ctx().node_map) != 1);

    const NodeDef* mul = source;

    // TODO(jingyue): handle the case where `scale` is 0-th operand.
    NodeDef* scale;  // scalar multiplier fot the input tensor
    NodeDef* input;
    TF_RETURN_IF_ERROR(GetInputNode(mul->input(1), &scale));
    TF_RETURN_IF_ERROR(GetInputNode(mul->input(0), &input));

    // Check that 'scale * weight' can be const folded.
    TF_RETURN_IF_TRUE(!IsConstant(*scale));
    TF_RETURN_IF_ERROR(CheckAttrsExist(*scale, {"dtype", "value"}));
    TF_RETURN_IF_ERROR(CheckAttrExists(*weights, "dtype"));
    TF_RETURN_IF_TRUE(scale->attr().at("dtype").type() !=
                      weights->attr().at("dtype").type());

    // Check that `scale` is a scalar.
    const TensorProto& scale_tensor = scale->attr().at("value").tensor();
    bool scale_is_a_scalar = scale_tensor.has_tensor_shape() &&
                             scale_tensor.tensor_shape().dim_size() == 0;
    TF_RETURN_IF_TRUE(!scale_is_a_scalar);

    // At this point all preconditions are met, and we safely do the rewrite.
    VLOG(3) << "Fold multiply into conv: conv=" << conv->name()
            << " mul=" << mul->name() << " weights=" << weights->name();

    // Create new node `scaled_weights`.
    NodeDef* scaled_weights = AddEmptyNode(scaled_weights_node_name);
    scaled_weights->set_op("Mul");
    scaled_weights->set_device(weights->device());
    (*scaled_weights->mutable_attr())["T"] = weights->attr().at("dtype");
    AddToOptimizationQueue(scaled_weights);

    // Link in its inputs.
    scaled_weights->add_input(conv->input(1));
    ctx().node_map->AddOutput(weights->name(), scaled_weights->name());
    scaled_weights->add_input(mul->input(1));
    ctx().node_map->AddOutput(scale->name(), scaled_weights->name());
    ForwardControlDependencies(scaled_weights, {source});

    // Update `conv`'s weights to `scaled_weights`.
    conv->set_input(1, scaled_weights->name());
    ctx().node_map->UpdateInput(conv->name(), weights->name(),
                                scaled_weights->name());
    AddToOptimizationQueue(conv);

    // Update `tail` node to bypass `mul` because it's folded to the weights.
    tail->set_input(0, mul->input(0));
    ctx().node_map->UpdateInput(tail->name(), mul->name(), input->name());
    AddToOptimizationQueue(tail);
    *simplified_node_name = conv->name();

    return Status::OK();
#undef TF_RETURN_IF_TRUE
  }
};

// Fold Transpose into matrix multiplication.
class FoldTransposeIntoMatMul : public ArithmeticOptimizerStage {
 public:
  explicit FoldTransposeIntoMatMul(const GraphOptimizerContext& ctx,
                                   const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("FoldTransposeIntoMatMul", ctx, ctx_ext) {}
  ~FoldTransposeIntoMatMul() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsMatMul(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    const NodeScopeAndName matmul = ParseNodeScopeAndName(node->name());
    const string optimized_node_name = OptimizedNodeName(matmul);
    if (ctx().node_map->NodeExists(optimized_node_name)) return Status::OK();

    NodeDef* a;
    NodeDef* b;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &a));
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &b));

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
                               IsInnerMatrixTransposeNode(*a, ctx().node_map);
    const bool b_is_foldable = foldable_transpose_ops.count(b->op()) > 0 &&
                               IsInnerMatrixTransposeNode(*b, ctx().node_map);
    if (!a_is_foldable && !b_is_foldable) return Status::OK();

    NodeDef* new_op = AddCopyNode(optimized_node_name, node);

    if (a_is_foldable) {
      const string attr_a =
          node->op() == "BatchMatMul" ? "adj_x" : "transpose_a";
      FlipBooleanAttr(attr_a, new_op);
      new_op->set_input(0, a->input(0));
      ctx().node_map->UpdateInput(new_op->name(), a->name(), a->input(0));
    }

    if (b_is_foldable) {
      const string attr_b =
          node->op() == "BatchMatMul" ? "adj_y" : "transpose_b";
      FlipBooleanAttr(attr_b, new_op);
      new_op->set_input(1, b->input(0));
      ctx().node_map->UpdateInput(new_op->name(), b->name(), b->input(0));
    }

    std::vector<const NodeDef*> deps_to_forward = {node};
    if (a_is_foldable) deps_to_forward.push_back(a);
    if (b_is_foldable) deps_to_forward.push_back(b);
    ForwardControlDependencies(new_op, deps_to_forward);

    return Status::OK();
  }

 private:
  void FlipBooleanAttr(const string& attr_name, NodeDef* node) {
    const bool old_value =
        !node->attr().count(attr_name) ? false : node->attr().at(attr_name).b();
    (*node->mutable_attr())[attr_name].set_b(!old_value);
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
};

// Fold Transpose into matrix multiplication.
class FoldConjugateIntoTranspose : public ArithmeticOptimizerStage {
 public:
  explicit FoldConjugateIntoTranspose(const GraphOptimizerContext& ctx,
                                      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("FoldConjugateIntoTranspose", ctx, ctx_ext) {}
  ~FoldConjugateIntoTranspose() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsConj(*node) || IsTranspose(*node) || IsConjugateTranspose(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    const NodeScopeAndName matmul = ParseNodeScopeAndName(node->name());
    const string optimized_node_name = OptimizedNodeName(matmul);
    if (ctx().node_map->NodeExists(optimized_node_name)) return Status::OK();

    NodeDef* input;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &input));

    const NodeDef* transpose_op = node->op() == "Conj" ? input : node;
    const NodeDef* conj_op = node->op() == "Conj" ? node : input;

    if ((IsTranspose(*transpose_op) || IsConjugateTranspose(*transpose_op)) &&
        IsConj(*conj_op)) {
      NodeDef* new_op = AddCopyNode(optimized_node_name, transpose_op);

      // Flip the type of transpose op to absorb the conjugation.
      new_op->set_op(transpose_op->op() == "Transpose" ? "ConjugateTranspose"
                                                       : "Transpose");
      new_op->set_input(0, input->input(0));
      ctx().node_map->UpdateInput(new_op->name(), node->name(),
                                  input->input(0));
      ForwardControlDependencies(new_op, {node, input});
      *simplified_node_name = new_op->name();
    }

    return Status::OK();
  }
};

// Replace Mul node with identical inputs with a Square.
class ReplaceMulWithSquare : public ArithmeticOptimizerStage {
 public:
  explicit ReplaceMulWithSquare(const GraphOptimizerContext& ctx,
                                const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("ReplaceMulWithSquare", ctx, ctx_ext) {}
  ~ReplaceMulWithSquare() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsMul(*node) && node->input(0) == node->input(1);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    const NodeScopeAndName mul = ParseNodeScopeAndName(node->name());
    const string optimized_node_name = OptimizedNodeName(mul);
    if (ctx().node_map->NodeExists(optimized_node_name)) return Status::OK();

    const DataType type = GetDataTypeFromAttr(*node, "T");
    bool is_complex = (type == DT_COMPLEX64) || (type == DT_COMPLEX128);

    string task;
    string device;
    bool is_on_cpu =
        DeviceNameUtils::SplitDeviceName(node->device(), &task, &device) &&
        str_util::StrContains(device, DEVICE_CPU);

    if (!is_complex || is_on_cpu) {
      NodeDef* new_square_node = AddCopyNode(optimized_node_name, node);
      new_square_node->set_op("Square");
      for (int i = 1; i < new_square_node->input_size(); ++i) {
        new_square_node->set_input(i - 1, new_square_node->input(i));
      }
      new_square_node->mutable_input()->RemoveLast();
      for (const string& input : new_square_node->input()) {
        ctx().node_map->AddOutput(NodeName(input), new_square_node->name());
      }
      *simplified_node_name = new_square_node->name();
    }

    return Status::OK();
  }
};

// Simplify aggregation (e.g. AddN) nodes:
//
// 1. Discard aggregate nodes with a single input and no control dependencies.
//
// 2. Try to rewrite aggregations of N >= 2 identical terms (possibly due to
//    deduping or other rewrites) so we can get rid of the sum entirely.
//
//    The expression (using AddN as an example of an aggregate op):
//      AddN(x, x, x, ... ,x)
//           <-- N terms -->
//    can be rewritten to:
//      Mul(Const(N), x))
//
class SimplifyAggregation : public ArithmeticOptimizerStage {
 public:
  explicit SimplifyAggregation(const GraphOptimizerContext& ctx,
                               const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("SimplifyAggregation", ctx, ctx_ext) {}
  ~SimplifyAggregation() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsAggregate(*node) && NumNonControlInputs(*node) > 0 &&
           GetDataTypeFromAttr(*node, "T") !=
               DT_VARIANT;  // TODO(b/119787146): Enable for variants.
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    // 1. Discard aggregate nodes with a single input and no control deps.
    if (node->input_size() == 1) {
      *simplified_node_name = node->input(0);
      return Status::OK();
    }

    // 2. Rewrite aggregations of N >= 2 identical terms.

    // All non-control inputs must be identical.
    bool all_equal = true;
    int num_inputs = 1;
    for (int i = 1; i < node->input_size(); ++i) {
      if (IsControlInput(node->input(i))) break;
      ++num_inputs;
      if (node->input(i) != node->input(0)) {
        all_equal = false;
        break;
      }
    }
    if (!all_equal) return Status::OK();

    // And node should not be optimized earlier.
    const NodeScopeAndName node_scope_and_name =
        ParseNodeScopeAndName(node->name());
    const string optimized_const_name =
        OptimizedNodeName(node_scope_and_name, "Const");
    const string optimized_mul_name =
        OptimizedNodeName(node_scope_and_name, "Mul");

    bool is_already_optimized =
        ctx().node_map->NodeExists(optimized_const_name) ||
        ctx().node_map->NodeExists(optimized_mul_name);

    if (is_already_optimized) return Status::OK();

    // At this point all preconditions are met, and we safely do the rewrite.
    VLOG(3) << "Simplify aggregation with identical inputs: node="
            << node->name() << " num_inputs=" << num_inputs;

    // 1. Create constant node with value N.
    const auto type = GetDataTypeFromAttr(*node, "T");
    Tensor t(type, TensorShape({}));
    Status status = SetTensorValue(type, num_inputs, &t);
    if (!status.ok()) {
      return errors::Internal("Failed to create const node: ",
                              status.error_message());
    }

    TensorValue value(&t);
    NodeDef* new_const_node = AddEmptyNode(optimized_const_name);
    status = ConstantFolding::CreateNodeDef(new_const_node->name(), value,
                                            new_const_node);
    if (!status.ok()) {
      return errors::Internal("Failed to create const node: ",
                              status.error_message());
    }
    new_const_node->set_device(node->device());
    MaybeAddControlInput(NodeName(node->input(0)), new_const_node,
                         ctx().optimized_graph, ctx().node_map);
    AddToOptimizationQueue(new_const_node);

    // 2. Replace the aggregate node with Mul(Const(N), x).
    NodeDef* new_mul_node = AddEmptyNode(optimized_mul_name);
    new_mul_node->set_op("Mul");
    new_mul_node->set_device(node->device());
    SetDataTypeToAttr(type, "T", new_mul_node);
    new_mul_node->add_input(new_const_node->name());
    ctx().node_map->AddOutput(new_const_node->name(), new_mul_node->name());
    new_mul_node->add_input(node->input(0));
    ctx().node_map->AddOutput(node->input(0), new_mul_node->name());

    ForwardControlDependencies(new_mul_node, {node});
    *simplified_node_name = new_mul_node->name();

    return Status::OK();
  }
};

class ConvertPowStage : public ArithmeticOptimizerStage {
 public:
  explicit ConvertPowStage(const GraphOptimizerContext& ctx,
                           const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("ConvertPow", ctx, ctx_ext) {}

  bool IsSupported(const NodeDef* node) const override {
    return IsPow(*node) &&
           ctx().graph_properties->GetInputProperties(node->name()).size() == 2;
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    const auto& pow_props =
        ctx().graph_properties->GetInputProperties(node->name())[1];
    PartialTensorShape shape(pow_props.shape());
    if (!shape.IsFullyDefined()) {
      // skip if p is not fully defined.
      return Status::OK();
    }
    if (TensorShape::IsValid(pow_props.shape()) && pow_props.has_value()) {
      Tensor pow(pow_props.dtype(), pow_props.shape());
      if (!pow.FromProto(pow_props.value())) {
        return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                       pow_props.value().DebugString());
      }

      complex128 prev, curr;
      for (int i = 0; i < pow.NumElements(); ++i) {
        if (!GetElementUnexhaustive(pow, i, {pow_props.dtype()}, &curr)) {
          // input data type is not supported by Pow. Skip.
          return Status::OK();
        }
        if (i != 0 && curr != prev) {
          // pow has different values on different elements. Skip.
          return Status::OK();
        }
        prev = curr;
      }
      NodeDef *x, *y;
      TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &x));
      TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &y));
      const auto& value_props =
          ctx().graph_properties->GetInputProperties(node->name())[0];
      const TensorShapeProto& output_shape =
          ctx().graph_properties->GetOutputProperties(node->name())[0].shape();
      if (curr == complex128(2, 0)) {
        node->set_op("Square");
        node->set_input(1, AsControlDependency(y->name()));
        AddToOptimizationQueue(node);
        AddToOptimizationQueue(y);
      } else if (curr == complex128(1, 0) &&
                 ShapesSymbolicallyEqual(value_props.shape(), output_shape)) {
        // Pow could be used to broadcast, so make sure the shapes of the two
        // arguments are identical before replacing Pow with Identity.
        node->set_op("Identity");
        node->set_input(1, AsControlDependency(y->name()));
        AddToOptimizationQueue(node);
        AddToOptimizationQueue(y);
      } else if (curr == complex128(0.5, 0)) {
        node->set_op("Sqrt");
        node->set_input(1, AsControlDependency(y->name()));
        AddToOptimizationQueue(node);
        AddToOptimizationQueue(y);
      } else if (curr == complex128(0, 0) &&
                 ShapesSymbolicallyEqual(value_props.shape(), output_shape)) {
        PartialTensorShape shape(value_props.shape());
        if (!shape.IsFullyDefined()) {
          // skip if b is not fully defined.
          return Status::OK();
        }
        if (TensorShape::IsValid(value_props.shape()) &&
            value_props.has_value()) {
          Tensor base(value_props.dtype(), value_props.shape());
          if (!base.FromProto(value_props.value())) {
            return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                           value_props.value().DebugString());
          }
          node->set_op("Const");
          Tensor c(base.dtype(), base.shape());
          for (int i = 0; i < c.NumElements(); ++i) {
            TF_RETURN_IF_ERROR(SetElementToOne(i, &c));
          }
          (*node->mutable_attr())["dtype"].set_type(base.dtype());
          c.AsProtoTensorContent(
              (*node->mutable_attr())["value"].mutable_tensor());
          node->mutable_attr()->erase("T");
          node->set_input(0, AsControlDependency(x->name()));
          node->set_input(1, AsControlDependency(y->name()));
          AddToOptimizationQueue(node);
          AddToOptimizationQueue(x);
          AddToOptimizationQueue(y);
        }
      } else if (curr == complex128(-0.5, 0)) {
        node->set_op("Rsqrt");
        node->set_input(1, AsControlDependency(y->name()));
        AddToOptimizationQueue(node);
        AddToOptimizationQueue(y);
      } else if (curr == complex128(-1, 0)) {
        node->set_op("Reciprocal");
        node->set_input(1, AsControlDependency(y->name()));
        AddToOptimizationQueue(node);
        AddToOptimizationQueue(y);
      }
    }
    return Status::OK();
  }

 private:
  Status SetElementToOne(int i, Tensor* t) {
    switch (t->dtype()) {
      case DT_INT32:
        t->flat<int32>()(i) = 1;
        return Status::OK();
      case DT_INT64:
        t->flat<int64>()(i) = 1L;
        return Status::OK();
      case DT_FLOAT:
        t->flat<float>()(i) = 1.0f;
        return Status::OK();
      case DT_DOUBLE:
        t->flat<double>()(i) = 1.0;
        return Status::OK();
      case DT_COMPLEX64:
        t->flat<complex64>()(i) = complex64(1);
        return Status::OK();
      case DT_COMPLEX128:
        t->flat<complex128>()(i) = complex128(1);
        return Status::OK();
      default:
        return errors::InvalidArgument("Invalid data type: ", t->dtype());
    }
  }
};

class ConvertLog1pStage : public ArithmeticOptimizerStage {
 public:
  explicit ConvertLog1pStage(const GraphOptimizerContext& ctx,
                             const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("ConvertLog1p", ctx, ctx_ext) {}
  ~ConvertLog1pStage() override = default;

  bool IsSupported(const NodeDef* node) const override { return IsLog(*node); }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    NodeDef* input;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &input));
    if (!IsAdd(*input)) {
      return Status::OK();
    }

    if (ctx().graph_properties->GetInputProperties(input->name()).size() < 2) {
      return Status::OK();
    }

    bool modified = false;
    TF_RETURN_IF_ERROR(TrySimplifyInternal(node, input, 0, 1, &modified));
    if (!modified) {
      TF_RETURN_IF_ERROR(TrySimplifyInternal(node, input, 1, 0, &modified));
    }
    if (modified) {
      *simplified_node_name = node->name();
    }
    return Status::OK();
  }

 private:
  Status TrySimplifyInternal(NodeDef* node, NodeDef* input, int i, int j,
                             bool* modified) {
    const auto& t =
        ctx().graph_properties->GetInputProperties(input->name())[i];
    const auto& c =
        ctx().graph_properties->GetInputProperties(input->name())[j];
    for (int k = 0; k < c.shape().dim_size(); ++k) {
      // Skip if c shape is not fully determined.
      if (c.shape().dim(k).size() < 0) {
        return Status::OK();
      }
    }
    TensorShapeProto broadcast_shape;
    if (!ShapeAfterBroadcast(t.shape(), c.shape(), &broadcast_shape)) {
      return Status::OK();
    }
    if (!ShapesSymbolicallyEqual(t.shape(), broadcast_shape)) {
      // skip if the non-constant tensor doesn't have the same shape after
      // broadcast.
      return Status::OK();
    }
    if (TensorShape::IsValid(c.shape()) && c.has_value()) {
      Tensor constant(c.dtype(), c.shape());
      if (!constant.FromProto(c.value())) {
        return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                       c.value().DebugString());
      }
      complex128 element;
      for (int k = 0; k < constant.NumElements(); ++k) {
        if (!GetElementUnexhaustive(constant, k,
                                    {DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE,
                                     DT_COMPLEX64, DT_COMPLEX128},
                                    &element)) {
          // input data type is not supported by log1p. Skip.
          return Status::OK();
        }
        if (element != complex128(1)) {
          // current element is not 1. Skip.
          return Status::OK();
        }
      }
      NodeDef *x, *y;
      TF_RETURN_IF_ERROR(GetInputNode(input->input(i), &x));
      TF_RETURN_IF_ERROR(GetInputNode(input->input(j), &y));
      node->set_op("Log1p");
      node->set_input(0, input->input(i));
      node->add_input(AsControlDependency(y->name()));
      ForwardControlDependencies(node, {input});

      AddToOptimizationQueue(node);
      AddToOptimizationQueue(input);
      AddToOptimizationQueue(x);
      AddToOptimizationQueue(y);
      *modified = true;
    }
    return Status::OK();
  }
};

class ConvertExpm1Stage : public ArithmeticOptimizerStage {
 public:
  explicit ConvertExpm1Stage(const GraphOptimizerContext& ctx,
                             const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("ConvertExpm1", ctx, ctx_ext) {}
  ~ConvertExpm1Stage() override = default;

  bool IsSupported(const NodeDef* node) const override {
    if (!IsSub(*node)) return false;

    NodeDef* input;
    if (!GetInputNode(node->input(0), &input).ok()) return false;

    return IsExp(*input);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    if (ctx().graph_properties->GetInputProperties(node->name()).size() < 2) {
      return Status::OK();
    }

    NodeDef* exp;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &exp));
    if (!IsExp(*exp)) {
      return Status::OK();
    }

    if (ctx().graph_properties->GetInputProperties(exp->name()).empty()) {
      return Status::OK();
    }

    const auto& t = ctx().graph_properties->GetInputProperties(exp->name())[0];
    const auto& c = ctx().graph_properties->GetInputProperties(node->name())[1];
    for (int k = 0; k < c.shape().dim_size(); ++k) {
      // Skip if c shape is not fully determined.
      if (c.shape().dim(k).size() < 0) {
        return Status::OK();
      }
    }
    TensorShapeProto broadcast_shape;
    if (!ShapeAfterBroadcast(t.shape(), c.shape(), &broadcast_shape)) {
      return Status::OK();
    }
    if (!ShapesSymbolicallyEqual(t.shape(), broadcast_shape)) {
      // skip if the non-constant tensor doesn't have the same shape after
      // broadcast.
      return Status::OK();
    }
    if (TensorShape::IsValid(c.shape()) && c.has_value()) {
      Tensor constant(c.dtype(), c.shape());
      if (!constant.FromProto(c.value())) {
        return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                       c.value().DebugString());
      }
      complex128 element;
      for (int k = 0; k < constant.NumElements(); ++k) {
        if (!GetElementUnexhaustive(constant, k,
                                    {DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE,
                                     DT_COMPLEX64, DT_COMPLEX128},
                                    &element)) {
          // input data type is not supported by expm1. Skip.
          return Status::OK();
        }
        if (element != complex128(1)) {
          // current element is not 1. Skip.
          return Status::OK();
        }
      }
      NodeDef *exp_input, *ones;
      TF_RETURN_IF_ERROR(GetInputNode(exp->input(0), &exp_input));
      TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &ones));
      node->set_op("Expm1");
      node->set_input(0, exp->input(0));
      node->set_input(1, AsControlDependency(ones->name()));
      ForwardControlDependencies(node, {exp});

      AddToOptimizationQueue(node);
      AddToOptimizationQueue(exp);
      AddToOptimizationQueue(exp_input);
      AddToOptimizationQueue(ones);
    }
    return Status::OK();
  }
};

// Performs conversions like:
// Max(Sqrt(x)) => Sqrt(Max(x))
// Checks for a max/min reduction over element-wise monotonic functions, such
// as Sqrt, Sigmoid, Tanh, etc.
class OptimizeMaxOrMinOfMonotonicStage : public ArithmeticOptimizerStage {
 public:
  explicit OptimizeMaxOrMinOfMonotonicStage(
      const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("OptimizeMaxOrMinOfMonotonicStage", ctx,
                                 ctx_ext) {}
  ~OptimizeMaxOrMinOfMonotonicStage() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsMax(*node) || IsMin(*node);
  }

  Status TrySimplify(NodeDef* reduction_node,
                     string* simplified_node_name) override {
    NodeDef* inner_function;
    TF_RETURN_IF_ERROR(GetInputNode(reduction_node->input(0), &inner_function));
    // Optimize only if:
    // 0. inner_function is not in the preserve set,
    // 1. inner_function's Op is element-wise monotonic
    // 2. inner_function's output is not being consumed elsewhere.
    bool is_non_decreasing = false;
    if (!IsInPreserveSet(*inner_function) &&
        IsElementWiseMonotonic(*inner_function, &is_non_decreasing) &&
        ctx().node_map->GetOutputs(inner_function->name()).size() == 1) {
      // Swap the first inputs of the inner function Op & the reduction Op.
      NodeDef* inner_input;
      TF_RETURN_IF_ERROR(GetInputNode(inner_function->input(0), &inner_input));
      reduction_node->set_input(0, inner_input->name());
      ctx().node_map->UpdateInput(reduction_node->name(),
                                  inner_function->name(), inner_input->name());
      inner_function->set_input(0, reduction_node->name());
      UpdateConsumers(reduction_node, inner_function->name());
      ctx().node_map->UpdateInput(inner_function->name(), inner_input->name(),
                                  reduction_node->name());
      if (!is_non_decreasing) {
        // Flip Min<->Max if the function is non-increasing, e.g.
        // Max(Neg(x)) = Neg(Min(x)).
        const string opposite = IsMax(*reduction_node) ? "Min" : "Max";
        reduction_node->set_op(opposite);
      }
      AddToOptimizationQueue(reduction_node);
      AddToOptimizationQueue(inner_function);
      AddToOptimizationQueue(inner_input);
    }
    return Status::OK();
  }

  void UpdateConsumers(NodeDef* node, const string& new_input) {
    const string& node_name = node->name();
    const std::set<NodeDef*> consumers = ctx().node_map->GetOutputs(node_name);
    for (NodeDef* consumer : consumers) {
      for (int i = 0; i < consumer->input_size(); ++i) {
        if (consumer->input(i) == node_name && consumer->name() != new_input) {
          consumer->set_input(i, new_input);
          ctx().node_map->UpdateInput(consumer->name(), node_name, new_input);
        }
      }
      AddToOptimizationQueue(consumer);
    }
  }
};

// Replace a chain of type&shape preserving unary ops with a
// '_UnaryOpsComposition' node.
// TODO(ezhulenev): It should be a part of remapper optimizer because it doesn't
// have to do much with arithmetic (together with FoldMultiplyIntoConv stage?).
class UnaryOpsComposition : public ArithmeticOptimizerStage {
 public:
  explicit UnaryOpsComposition(const GraphOptimizerContext& ctx,
                               const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("UnaryOpsComposition", ctx, ctx_ext) {
    // WARN: This should be consistent with unary_ops_composition.cc.
    // clang-format off
    supported_ops_ = {// Ops defined via Eigen scalar ops.
                      {"Abs",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Acos",       {DT_FLOAT,          DT_DOUBLE}},
                      {"Acosh",      {DT_FLOAT,          DT_DOUBLE}},
                      {"Asin",       {DT_FLOAT,          DT_DOUBLE}},
                      {"Asinh",      {DT_FLOAT,          DT_DOUBLE}},
                      {"Atan",       {DT_FLOAT,          DT_DOUBLE}},
                      {"Atanh",      {DT_FLOAT,          DT_DOUBLE}},
                      {"Ceil",       {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Cos",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Cosh",       {DT_FLOAT,          DT_DOUBLE}},
                      {"Expm1",      {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Exp",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Floor",      {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Inv",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Log",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Log1p",      {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Neg",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Reciprocal", {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Rint",       {DT_FLOAT,          DT_DOUBLE}},
                      {"Round",      {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Rsqrt",      {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Sigmoid",    {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Sin",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Sinh",       {DT_FLOAT,          DT_DOUBLE}},
                      {"Sqrt",       {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Square",     {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Tan",        {DT_FLOAT,          DT_DOUBLE}},
                      {"Tanh",       {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      // Additional ops that are not part of the Eigen.
                      {"Elu",        {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Relu",       {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Relu6",      {DT_FLOAT, DT_HALF, DT_DOUBLE}},
                      {"Selu",       {DT_FLOAT, DT_HALF, DT_DOUBLE}}};
    // clang-format on
  }
  ~UnaryOpsComposition() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return CanOptimize(*node) &&
           // Check that this node was not already a root of a fused chain. If
           // graph optimization runs twice without pruning in between,
           // fused_nodes_ will not have this information.
           !ctx().node_map->NodeExists(OptimizedNodeName(*node));
  }

  Status TrySimplify(NodeDef* root, string* simplified_node_name) override {
    TF_RETURN_IF_ERROR(CheckAttrExists(*root, "T"));
    DataType dtype = root->attr().at("T").type();

    // Keep a trace of all supported input nodes that can be fused together.
    std::vector<string> op_nodes = {root->name()};
    std::vector<string> op_names = {root->op()};

    // Check if we should follow input(0) while building an op composition.
    const auto predicate_fn = [&](const NodeDef& input) {
      if (input.name() == root->name()) return true;

      bool follow_input_node =
          dtype == GetDataTypeFromAttr(input, "T") &&
          NumNonControlDataOutputs(input, *ctx().node_map) == 1 &&
          CanOptimize(input);

      if (follow_input_node) {
        op_nodes.push_back(input.name());
        op_names.push_back(input.op());
      }

      return follow_input_node;
    };

    NodeDef* last_op = GetTailOfChain(
        *root, *ctx().node_map, /*follow_control_input*/ false, predicate_fn);

    // We were not able to find a chain that can be replaced.
    if (op_names.size() == 1) return Status::OK();

    // Do not add fused nodes to any other chain.
    std::for_each(op_nodes.begin(), op_nodes.end(),
                  [this](const string& name) { AddToFusedNodes(name); });

    // Reverse the trace to get correct composition computation order.
    std::reverse(op_names.begin(), op_names.end());

    VLOG(2) << "Fuse unary ops: root=" << root->name() << " op_names=["
            << str_util::Join(op_names, ", ") << "]";

    NodeDef* composition_node = ctx().optimized_graph->add_node();
    composition_node->set_name(OptimizedNodeName(*root));
    composition_node->set_op("_UnaryOpsComposition");
    composition_node->add_input(last_op->input(0));
    composition_node->set_device(root->device());

    auto attr = composition_node->mutable_attr();
    SetAttrValue(dtype, &(*attr)["T"]);
    SetAttrValue(op_names, &(*attr)["op_names"]);

    ctx().node_map->AddNode(composition_node->name(), composition_node);
    ctx().node_map->AddOutput(NodeName(last_op->input(0)),
                              composition_node->name());

    *simplified_node_name = composition_node->name();

    return Status::OK();
  }

 private:
  bool CanOptimize(const NodeDef& node) const {
    DataType dtype = GetDataTypeFromAttr(node, "T");
    if (!IsSupported(node.op(), dtype)) {
      return false;
    }
    if (IsInPreserveSet(node)) {
      return false;
    }
    if (!NodeIsOnCpu(node)) {
      return false;
    }
    if (NodeIsAlreadyFused(node)) {
      return false;
    }
    return !(IsDrivenByControlDependency(node) ||
             DrivesControlDependency(node));
  }

  // UnaryOpsComposition is defined only for CPU.
  bool NodeIsOnCpu(const NodeDef& node) const {
    using str_util::StartsWith;

    string task;
    string device;

    return DeviceNameUtils::SplitDeviceName(node.device(), &task, &device) &&
           StartsWith(device, DEVICE_CPU);
  }

  bool NodeIsAlreadyFused(const NodeDef& node) const {
    return fused_nodes_.count(node.name()) > 0;
  }

  string OptimizedNodeName(const NodeDef& node) const {
    return strings::StrCat(node.name(), "/unary_ops_composition");
  }

  void AddToFusedNodes(const string& name) { fused_nodes_.insert(name); }

  // Check if an op is supported by the _UnaryOpsComposition for the given type.
  bool IsSupported(const string& op_name, DataType dtype) const {
    const auto it = supported_ops_.find(op_name);
    return it != supported_ops_.end() && it->second.count(dtype) > 0;
  }

  std::unordered_map<string, std::set<DataType>> supported_ops_;
  std::unordered_set<string> fused_nodes_;
};

// Replace operations of the form:
//    x = stack((a_0, a_1, ..., a_{n-1}), axis=k)[:,...,i,...]
// with
//    a_i
// when the strided slice index `i` is applied in the k'th axis.
//
// Similarly, replace operations of the form:
//    x = stack((a_0, a_1, ..., a_{n-1}), axis=k)[:,...,i:i+1,...]
// with
//    expand_dims(a_i, axis=k)
//
// TODO(ebrevdo): Extend to also replace operations of the form
//    concat((a_0, a_1, ..., ), axis=k)[:, ..., s_i:s_{i+1}, ...]
// with
//    a_i,
// when
//    s_i = cumsum(shape(a)[k] for a in (a_0, ...,))[i]
// and slicing is in the k'th axis.
class RemoveStackStridedSliceSameAxis : public ArithmeticOptimizerStage {
 public:
  explicit RemoveStackStridedSliceSameAxis(
      const GraphOptimizerContext& ctx,
      const ArithmeticOptimizerContext& ctx_ext)
      : ArithmeticOptimizerStage("RemoveStackStridedSliceSameAxis", ctx,
                                 ctx_ext) {}
  ~RemoveStackStridedSliceSameAxis() override = default;

  bool IsSupported(const NodeDef* node) const override {
    return IsStridedSlice(*node);
  }

  Status TrySimplify(NodeDef* node, string* simplified_node_name) override {
    // *node is a StridedSlice NodeDef.
    NodeDef* pack;

    // Get the input and see if it's a Pack op.
    TF_RETURN_IF_ERROR(GetInputNode(node->input(0), &pack));
    if (!IsPack(*pack)) return Status::OK();

    bool return_early;
    PartialTensorShape pack_output_shape;
    int pack_axis;
    TF_RETURN_IF_ERROR(
        CheckInputs(node, pack, &pack_output_shape, &pack_axis, &return_early));
    if (return_early) return Status::OK();

    int slice_start_value;
    bool found;
    TF_RETURN_IF_ERROR(GetSliceAxis(node, pack, pack_output_shape, pack_axis,
                                    &slice_start_value, &found));
    if (!found) return Status::OK();

    return RewriteGraph(node, pack, slice_start_value, pack_axis,
                        simplified_node_name);
  }

 protected:
  bool IsReallyConstant(const NodeDef& node) const {
    if (!IsConstant(node)) {
      return false;
    }
    // If the node is fed it's not constant anymore.
    return ctx().feed_nodes->find(node.name()) == ctx().feed_nodes->end();
  }

  bool GetConstantAsInt64(const NodeDef& node, DataType dtype,
                          std::vector<int64>* values) {
    if (dtype == DT_INT32) {
      std::vector<int32> values_int32;
      if (!ValuesFromConstNode(node, &values_int32)) {
        return false;
      }
      std::copy(values_int32.begin(), values_int32.end(),
                std::inserter(*values, values->begin()));
      return true;
    } else {
      return ValuesFromConstNode(node, values);
    }
  }

  Status CheckInputs(const NodeDef* node, const NodeDef* pack,
                     PartialTensorShape* pack_output_shape, int* pack_axis,
                     bool* return_early) {
    *return_early = true;
    TF_RETURN_IF_ERROR(CheckAttrExists(*pack, "axis"));

    *pack_axis = pack->attr().at("axis").i();
    auto slice_properties =
        ctx().graph_properties->GetInputProperties(node->name());
    if (slice_properties.empty() ||
        slice_properties[0].shape().unknown_rank()) {
      return Status::OK();
    }
    *pack_output_shape = slice_properties[0].shape();
    const int pack_input_rank = pack_output_shape->dims() - 1;
    if (*pack_axis < 0) {
      // The ndims of any input into Pack op is its output ndims - 1.
      *pack_axis += pack_input_rank;
    }
    if (*pack_axis < 0 || *pack_axis >= pack_input_rank) {
      return errors::InvalidArgument(
          "Pack node (", pack->name(),
          ") axis attribute is out of bounds: ", pack->attr().at("axis").i());
    }
    *return_early = false;
    return Status::OK();
  }

  Status GetSliceAxis(const NodeDef* node, const NodeDef* pack,
                      const PartialTensorShape& pack_output_shape,
                      int pack_axis, int* slice_start_value, bool* found) {
    *found = false;
    TF_RETURN_IF_ERROR(
        CheckAttrsExist(*node, {"begin_mask", "end_mask", "ellipsis_mask",
                                "new_axis_mask", "shrink_axis_mask"}));

    const int begin_mask = node->attr().at("begin_mask").i();
    const int end_mask = node->attr().at("end_mask").i();
    const int ellipsis_mask = node->attr().at("ellipsis_mask").i();
    const int new_axis_mask = node->attr().at("new_axis_mask").i();
    const int shrink_axis_mask = node->attr().at("shrink_axis_mask").i();

    // Check that the StridedSlice is one of these at pack_axis:
    //   [..., i, ...]
    //   [..., i:i+1, ...]
    //   [..., :1, ...]
    //   [..., -1:, ...]
    ///  [..., s_{pack_axis}-1:, ...]
    NodeDef* slice_begin;
    NodeDef* slice_end;
    NodeDef* slice_strides;
    TF_RETURN_IF_ERROR(GetInputNode(node->input(1), &slice_begin));
    TF_RETURN_IF_ERROR(GetInputNode(node->input(2), &slice_end));
    TF_RETURN_IF_ERROR(GetInputNode(node->input(3), &slice_strides));

    for (const auto* n : {slice_begin, slice_end, slice_strides}) {
      if (!IsReallyConstant(*n)) return Status::OK();
    }

    Tensor slice_begin_t;
    Tensor slice_end_t;
    Tensor slice_strides_t;

    TF_RETURN_IF_ERROR(CheckAttrExists(*slice_begin, "value"));
    if (!slice_begin_t.FromProto(slice_begin->attr().at("value").tensor())) {
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(CheckAttrExists(*slice_end, "value"));
    if (!slice_end_t.FromProto(slice_end->attr().at("value").tensor())) {
      return Status::OK();
    }
    TF_RETURN_IF_ERROR(CheckAttrExists(*slice_strides, "value"));
    if (!slice_strides_t.FromProto(
            slice_strides->attr().at("value").tensor())) {
      return Status::OK();
    }
    TensorShape processing_shape;
    TensorShape final_shape;
    bool is_identity;
    bool is_simple_slice;
    bool slice_dim0;
    gtl::InlinedVector<int64, 4> slice_begin_vec;
    gtl::InlinedVector<int64, 4> slice_end_vec;
    gtl::InlinedVector<int64, 4> slice_strides_vec;
    TF_RETURN_IF_ERROR(ValidateStridedSliceOp(
        &slice_begin_t, &slice_end_t, slice_strides_t, pack_output_shape,
        begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask,
        &processing_shape, &final_shape, &is_identity, &is_simple_slice,
        &slice_dim0, &slice_begin_vec, &slice_end_vec, &slice_strides_vec));

    if (!is_simple_slice) return Status::OK();

    int begin_index = -1;
    int64 begin_value = 0;
    for (int i = 0; i < slice_begin_vec.size(); ++i) {
      const int64 v = slice_begin_vec[i];
      if (v != 0) {
        if (begin_index != -1) {
          // At least two start values that are nonzero.
          return Status::OK();
        }
        begin_index = i;
        begin_value = v;
      }
    }

    int end_index = -1;
    int64 end_value = 0;
    for (int i = 0; i < slice_end_vec.size(); ++i) {
      const int64 v = slice_end_vec[i];
      if (v != pack_output_shape.dim_size(i)) {
        if (end_index != -1) {
          // At least two end values that are nonzero.
          return Status::OK();
        }
        end_index = i;
        end_value = v;
      }
    }

    if (begin_index == -1 && end_index == -1) return Status::OK();
    if (begin_index != -1 && end_index != -1 && begin_index != end_index) {
      // Somehow received different axes for begin/end slicing
      return Status::OK();
    }
    const int slice_axis = (begin_index == -1) ? end_index : begin_index;
    if (slice_axis != pack_axis) {
      // Not slicing on the same axis as the Pack op.
      return Status::OK();
    }
    *slice_start_value = (begin_index == -1) ? 0 : begin_value;
    const int64 slice_end_value =
        (end_index == -1) ? pack_output_shape.dim_size(slice_axis) : end_value;
    if (slice_end_value != *slice_start_value + 1) {
      // Not slicing a single value out.
      return Status::OK();
    }

    if (*slice_start_value < 0 || *slice_start_value >= pack->input_size()) {
      return errors::InvalidArgument(
          "Node ", node->name(), " requested invalid slice index ",
          *slice_start_value, " on axis ", slice_axis,
          " from tensor of shape: ", pack_output_shape.DebugString());
    }

    *found = true;  // slice_start_value is valid.
    return Status::OK();
  }

  Status RewriteGraph(const NodeDef* node, const NodeDef* pack,
                      int slice_start_value, int pack_axis,
                      string* simplified_node_name) {
    OpInfo::TensorProperties input_slice_properties;
    NodeDef* input_slice;
    TF_RETURN_IF_ERROR(
        GetInputNode(pack->input(slice_start_value), &input_slice));
    TF_RETURN_IF_ERROR(GetTensorProperties(pack->input(slice_start_value),
                                           &input_slice_properties));
    PartialTensorShape input_slice_shape(input_slice_properties.shape());

    OpInfo::TensorProperties output_properties;
    TF_RETURN_IF_ERROR(GetTensorProperties(
        strings::StrCat(node->name(), ":", 0), &output_properties));
    PartialTensorShape output_shape(output_properties.shape());
    NodeDef* output =
        AddEmptyNode(OptimizedNodeName(ParseNodeScopeAndName(node->name())));
    if (input_slice_shape.IsCompatibleWith(output_shape)) {
      output->set_op("Identity");
      output->set_device(node->device());
      SetDataTypeToAttr(output_properties.dtype(), "T", output);
      output->add_input(input_slice->name());
    } else {
      NodeDef* axis = AddEmptyNode(
          OptimizedNodeName(ParseNodeScopeAndName(node->name()), "Axis"));
      axis->set_op("Const");
      axis->set_device(node->device());
      auto axis_attr = axis->mutable_attr();
      SetDataTypeToAttr(DT_INT32, "dtype", axis);
      auto* axis_t = (*axis_attr)["value"].mutable_tensor();
      axis_t->set_dtype(DT_INT32);
      axis_t->add_int_val(pack_axis);
      AddToOptimizationQueue(axis);
      output->set_op("ExpandDims");
      output->set_device(node->device());
      SetDataTypeToAttr(output_properties.dtype(), "T", output);
      output->add_input(input_slice->name());
      output->add_input(axis->name());
    }

    // Copy dependencies over.
    ForwardControlDependencies(output, {node, pack});
    AddToOptimizationQueue(output);
    *simplified_node_name = output->name();

    return Status::OK();
  }
};

}  // namespace

class UniqueNodes {
 public:
  NodeDef* FindOrAddRepresentative(NodeDef* node) {
    uint64 sig = ComputeSignature(*node);
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
  uint64 ComputeSignature(const NodeDef& node) const;
  bool SameNode(const NodeDef& node1, const NodeDef& node2) const;

  std::unordered_map<uint64, std::vector<NodeDef*>> rep_;
};

uint64 UniqueNodes::ComputeSignature(const NodeDef& node) const {
  uint64 h = Hash64(node.op());
  h = Hash64Combine(Hash64(node.device()), h);

  for (const auto& input : node.input()) {
    const TensorId input_tensor = ParseTensorName(input);
    h = Hash64CombineUnordered(
        Hash64(input_tensor.node().data(), input_tensor.node().size()), h);
    h = Hash64CombineUnordered(std::hash<int>()(input_tensor.index()), h);
  }
  for (const auto& attr : node.attr()) {
    h = Hash64CombineUnordered(Hash64(attr.first), h);
    h = Hash64CombineUnordered(FastAttrValueHash(attr.second), h);
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
    if (it == node2.attr().end()) return false;
    if (!FastAreAttrValuesEqual(attr1.second, it->second)) return false;
  }

  return true;
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
  // Populate feed_inplace_op;
  std::unordered_set<NodeDef*> feeds_inplace_op;
  for (int i = 0; i < optimized_graph_->node_size(); ++i) {
    if (FeedsInPlaceOp(graph_view, optimized_graph_->node(i))) {
      feeds_inplace_op.insert(optimized_graph_->mutable_node(i));
    }
  }
  do {
    stop = true;
    UniqueNodes nodes;
    for (int i = 0; i < optimized_graph_->node_size(); ++i) {
      if (duplicates.find(i) != duplicates.end()) {
        continue;
      }
      NodeDef* node = optimized_graph_->mutable_node(i);
      if (!CanDedup(*node) ||
          feeds_inplace_op.find(node) != feeds_inplace_op.end()) {
        continue;
      }
      NodeDef* rep = nodes.FindOrAddRepresentative(node);
      if (rep == node) {
        continue;
      }
      // If either node or rep feeds an inplace op, deduping them may cause data
      // races. For example: If we dedup nodes initializing two independent
      // inplace accumulations, they will write to the same buffer, clobbering
      // each other's results.
      if (feeds_inplace_op.find(rep) != feeds_inplace_op.end()) {
        continue;
      }
      VLOG(3) << "Remove duplicated node: node=" << node->name()
              << " representative=" << rep->name();
      const std::set<NodeDef*>& tmp = node_map_->GetOutputs(node->name());
      std::vector<NodeDef*> fanouts(tmp.begin(), tmp.end());
      for (NodeDef* fanout : fanouts) {
        for (int i = 0; i < fanout->input_size(); ++i) {
          string* fanout_input = fanout->mutable_input(i);
          const int position =
              NodePositionIfSameNode(*fanout_input, node->name());
          // Update name in-place.
          if (position < -1) {
            continue;
          } else if (position > 0) {
            *fanout_input = StrCat(rep->name(), ":", position);
          } else if (position == 0) {
            *fanout_input = rep->name();
          } else {
            *fanout_input = StrCat("^", rep->name());
          }
          node_map_->AddOutput(rep->name(), fanout->name());
        }
      }
      duplicates.insert(i);
      stop = false;
    }
  } while (!stop);

  // Delete duplicates
  if (fetch_nodes_known_ && !duplicates.empty()) {
    EraseNodesFromGraph(duplicates, optimized_graph_);
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
  DedupControlInputs(target_node);
}

Status ArithmeticOptimizer::SimplifyArithmeticOps(bool can_use_shapes) {
  SetVector<NodeDef*> nodes_to_simplify;
  nodes_to_simplify.Reserve(optimized_graph_->node_size());
  for (int i = 0; i < optimized_graph_->node_size(); ++i) {
    nodes_to_simplify.PushBack(optimized_graph_->mutable_node(i));
  }

  const GraphOptimizerContext ctx(&nodes_to_preserve_, optimized_graph_,
                                  graph_properties_.get(), node_map_.get(),
                                  &feed_nodes_, opt_level_);
  const ArithmeticOptimizerContext ctx_ext(&nodes_to_simplify);

  // Stop pipeline after first stage returning non-empty simplified tensor name.
  const auto stop = [](const string& result) { return !result.empty(); };
  GraphOptimizerStagePipeline<string> pipeline(stop);

  if (options_.combine_add_to_addn && can_use_shapes)
    pipeline.AddStage<AddOpsRewriteStage>(ctx, ctx_ext);
  if (options_.fold_conjugate_into_transpose)
    pipeline.AddStage<FoldConjugateIntoTranspose>(ctx, ctx_ext);
  if (options_.fold_multiply_into_conv)
    pipeline.AddStage<FoldMultiplyIntoConv>(ctx, ctx_ext);
  if (options_.fold_transpose_into_matmul)
    pipeline.AddStage<FoldTransposeIntoMatMul>(ctx, ctx_ext);
  if (options_.hoist_common_factor_out_of_aggregation && can_use_shapes)
    pipeline.AddStage<HoistCommonFactorOutOfAggregation>(ctx, ctx_ext);
  if (options_.minimize_broadcasts && can_use_shapes)
    pipeline.AddStage<MinimizeBroadcasts>(ctx, ctx_ext);
  if (options_.remove_identity_transpose && can_use_shapes)
    pipeline.AddStage<RemoveIdentityTranspose>(ctx, ctx_ext);
  if (options_.remove_involution)
    pipeline.AddStage<RemoveInvolution>(ctx, ctx_ext);
  if (options_.remove_redundant_bitcast)
    pipeline.AddStage<RemoveRedundantBitcastStage>(ctx, ctx_ext);
  if (options_.remove_redundant_cast)
    pipeline.AddStage<RemoveRedundantCastStage>(ctx, ctx_ext);
  if (options_.remove_redundant_reshape)
    pipeline.AddStage<RemoveRedundantReshape>(ctx, ctx_ext);
  if (options_.remove_negation)
    pipeline.AddStage<RemoveNegationStage>(ctx, ctx_ext);
  if (options_.replace_mul_with_square)
    pipeline.AddStage<ReplaceMulWithSquare>(ctx, ctx_ext);
  if (options_.remove_logical_not)
    pipeline.AddStage<RemoveLogicalNotStage>(ctx, ctx_ext);
  if (options_.reorder_cast_like_and_value_preserving)
    pipeline.AddStage<ReorderCastLikeAndValuePreserving>(ctx, ctx_ext);
  if (options_.simplify_aggregation)
    pipeline.AddStage<SimplifyAggregation>(ctx, ctx_ext);
  if (options_.hoist_cwise_unary_chains)
    pipeline.AddStage<HoistCWiseUnaryChainsStage>(ctx, ctx_ext);
  if (options_.convert_sqrt_div_to_rsqrt_mul)
    pipeline.AddStage<SqrtDivToRsqrtMulStage>(ctx, ctx_ext);
  if (options_.remove_idempotent)
    pipeline.AddStage<RemoveIdempotentStage>(ctx, ctx_ext);
  if (options_.convert_pow) pipeline.AddStage<ConvertPowStage>(ctx, ctx_ext);
  if (options_.convert_log1p)
    pipeline.AddStage<ConvertLog1pStage>(ctx, ctx_ext);
  if (options_.optimize_max_or_min_of_monotonic)
    pipeline.AddStage<OptimizeMaxOrMinOfMonotonicStage>(ctx, ctx_ext);
  if (options_.convert_expm1)
    pipeline.AddStage<ConvertExpm1Stage>(ctx, ctx_ext);
  if (options_.unary_ops_composition)
    pipeline.AddStage<UnaryOpsComposition>(ctx, ctx_ext);
  if (options_.remove_stack_strided_slice_same_axis)
    pipeline.AddStage<RemoveStackStridedSliceSameAxis>(ctx, ctx_ext);

  VLOG(1) << "Run " << pipeline.NumStages() << " arithmetic optimizer stages: "
          << str_util::Join(pipeline.StageNames(), ", ");

  while (!nodes_to_simplify.Empty()) {
    GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
    NodeDef* node = nodes_to_simplify.PopBack();

    string simplified_tensor = "";
    bool optimized = pipeline.PassThroughAllStages(node, &simplified_tensor);

    // If the node was not optimized by any of the stages, go to the next one.
    if (!optimized) continue;

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
  GrapplerItem optimized_item(item);
  optimized_graph_ = &optimized_item.graph;
  node_map_.reset(new NodeMap(optimized_graph_));

  for (const auto& feed : item.feed) {
    feed_nodes_.insert(NodeName(feed.first));
  }

  // Disable restricted graph rewrites.
  options_.unary_ops_composition &=
      item.allowed_optimizations().non_differentiable_rewrites;

  if (options_.dedup_computations) {
    DedupComputations();
  }
  GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();

  // Perform topological sort on the graph in order to help AddOpsRewrite to
  // optimize larger subgraphs starting from the roots with more inputs.
  TF_RETURN_IF_ERROR(TopologicalSort(optimized_graph_));

  graph_properties_.reset(new GraphProperties(optimized_item));
  const bool assume_valid_feeds = opt_level_ == RewriterConfig::AGGRESSIVE;
  const Status status = graph_properties_->InferStatically(assume_valid_feeds);
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

}  // namespace grappler
}  // namespace tensorflow
