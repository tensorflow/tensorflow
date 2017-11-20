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
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"

namespace tensorflow {
namespace grappler {
namespace {

template <typename T>
bool SafeSetTensorValue(double value, Tensor* tensor) {
  using RealType = typename Eigen::NumTraits<T>::Real;
  if (value > std::numeric_limits<RealType>::max() ||
      value < std::numeric_limits<RealType>::min()) {
    return false;
  }
  tensor->flat<T>()(0) = static_cast<T>(value);
  return true;
}

#define HANDLE_CASE(DTYPE)                                          \
  case DTYPE:                                                       \
    if (!SafeSetTensorValue<EnumToDataType<DTYPE>::Type>(           \
            static_cast<double>(value), tensor)) {                  \
      return errors::InvalidArgument("Cannot store value ", value,  \
                                     " in tensor of type " #DTYPE); \
    }                                                               \
    break

Status SetTensorValue(DataType dtype, int value, Tensor* tensor) {
  switch (dtype) {
    //    HANDLE_CASE(DT_HALF);
    HANDLE_CASE(DT_FLOAT);
    HANDLE_CASE(DT_DOUBLE);
    HANDLE_CASE(DT_UINT8);
    HANDLE_CASE(DT_INT8);
    HANDLE_CASE(DT_UINT16);
    HANDLE_CASE(DT_INT16);
    HANDLE_CASE(DT_INT32);
    HANDLE_CASE(DT_INT64);
    HANDLE_CASE(DT_COMPLEX64);
    HANDLE_CASE(DT_COMPLEX128);
    default:
      return errors::InvalidArgument("Unexpected type ", DataTypeString(dtype));
  }
  return Status::OK();
}

bool IsInvolution(const NodeDef& node) {
  const std::unordered_set<string> involution_ops = {
      "Conj", "Reciprocal", "Invert", "Neg", "LogicalNot"};
  return involution_ops.count(node.op()) > 0;
}

// Returns true if the op in node only rearranges the order of elements in an
// input tensor, or more specifically, if it commutes with all element-wise
// operations on the values.
bool IsValuePreserving(const NodeDef& node) {
  const std::unordered_set<string> value_preserving_ops = {
      "Transpose",  "Reshape",      "Identity",        "InvertPermutation",
      "Reverse",    "StopGradient", "PreventGradient", "CheckNumerics",
      "ExpandDims", "Squeeze"};
  return value_preserving_ops.count(node.op()) > 0;
}

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

bool SimplyReordersData(const NodeDef& node) {
  return node.op() == "Transpose";
}

// Follow a chain (through input(0)) of ops starting at `source->input(0)` as
// long as they
//  1. preserve the values of their first input,
//  2. have a single (non-control) output,
//  3. are not in nodes_to_preserve.
// Returns the last node in the chain satisfying these properties or source
// itself if a chain of length zero was found.
//
// source <- vp <- vp <- vp <- non_vp
//                       ^^
//                   return value
NodeDef* GetTailOfValuePreservingChain(
    const NodeDef* source, const NodeMap* node_map,
    const std::unordered_set<string>& nodes_to_preserve) {
  const NodeDef* source_parent = source;
  if (!IsControlInput(source->input(0))) {
    source = node_map->GetNode(source->input(0));
    while (IsValuePreserving(*source) &&
           node_map->GetOutputs(source->name()).size() == 1 &&
           // Do not skip over preserved nodes, because folding will change
           // the results of these skipped data-reordering nodes.
           // TODO(jingyue): A more elegant way is to copy this chain of
           // data-reordering nodes and modify only the copy.
           !nodes_to_preserve.count(source->name())) {
      source_parent = source;
      if (IsControlInput(source->input(0))) {
        break;
      }
      source = node_map->GetNode(source->input(0));
    }
  }
  return const_cast<NodeDef*>(source_parent);
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

// Returns the data type in attribute `attr_name` of `node`. If that attribute
// doesn't exist, returns DT_INVALID.
DataType GetDataTypeFromAttr(const NodeDef& node, const string& attr_name) {
  if (!node.attr().count(attr_name)) {
    return DT_INVALID;
  }
  const auto& attr = node.attr().at(attr_name);
  if (attr.value_case() != AttrValue::kType) {
    return DT_INVALID;
  }
  return attr.type();
}

bool IsCommutative(const NodeDef& node) {
  if (node.op() == "Add" && node.input_size() > 0) {
    // Workaround for "Add" not being marked is_commutative and is_aggregate.
    // (See cl/173915048).
    const auto type = GetDataTypeFromAttr(node, "T");
    return type != DT_INVALID && type != DT_STRING;
  }
  const OpDef* op_def = nullptr;
  const Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  return status.ok() && op_def->is_commutative();
}

bool IsAggregate(const NodeDef& node) {
  if (node.op() == "Add" && node.input_size() > 0) {
    // Workaround for "Add" not being marked is_commutative and is_aggregate.
    // (See cl/173915048).
    const auto type = GetDataTypeFromAttr(node, "T");
    return type != DT_INVALID && type != DT_STRING;
  }
  const OpDef* op_def = nullptr;
  const Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  return status.ok() && op_def->is_aggregate();
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

bool IsNumberType(DataType dtype) {
  DataTypeVector number_types = NumberTypes();
  return std::find(number_types.begin(), number_types.end(), dtype) !=
         number_types.end();
}

const char kOutputShapesAttr[] = "_output_shapes";

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

// Fix frame dependencies by adding control dependencies from old_input to nodes
// in new_nodes_for_control_dep, and update frame_map for all nodes in
// new_nodes.
void AddFrameControlDeps(const NodeDef* old_node,
                         const std::vector<NodeDef*>& new_nodes,
                         const string& source_for_ctrl_dep,
                         const std::vector<NodeDef*>& sinks_for_control_dep,
                         GraphDef* graph, NodeMap* node_map,
                         FrameMap* frame_map) {
  const auto frame_it = frame_map->find(old_node);
  if (frame_it != frame_map->end()) {
    for (auto node : new_nodes) {
      frame_map->emplace(node, frame_it->second);
    }
    if (!source_for_ctrl_dep.empty() && !sinks_for_control_dep.empty()) {
      const string ctrl_dep = ConstantFolding::AddControlDependency(
          source_for_ctrl_dep, graph, node_map);
      for (auto node : sinks_for_control_dep) {
        node->add_input(ctrl_dep);
      }
    }
  }
}

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

// static
bool ArithmeticOptimizer::CanDedup(
    const NodeDef& node, const std::unordered_set<string>& nodes_to_preserve) {
  if (nodes_to_preserve.find(node.name()) != nodes_to_preserve.end()) {
    return false;
  }
  if (IsEnter(node) || IsExit(node) || IsPlaceholder(node)) {
    return false;
  }
  if (node.device().find("SPU") != string::npos) {
    return false;
  }
  const OpDef* op_def = nullptr;
  Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (!status.ok()) {
    return false;
  }
  if (op_def->is_stateful()) {
    return false;
  }
  // Don't consolidate ops such as AssignAdd
  for (const auto& input : op_def->input_arg()) {
    if (input.is_ref()) {
      return false;
    }
  }
  return true;
}

void ArithmeticOptimizer::DedupComputations(GraphDef* optimized_graph) const {
  NodeMap map(optimized_graph);
  bool stop = true;
  std::set<int> duplicates;
  do {
    stop = true;
    UniqueNodes nodes;
    for (int i = 0; i < optimized_graph->node_size(); ++i) {
      if (duplicates.find(i) != duplicates.end()) {
        continue;
      }
      NodeDef* node = optimized_graph->mutable_node(i);
      if (!CanDedup(*node, nodes_to_preserve_)) {
        continue;
      }
      NodeDef* rep = nodes.FindOrAddRepresentative(node);
      if (rep == node) {
        continue;
      }
      const std::set<NodeDef*>& fanouts = map.GetOutputs(node->name());
      for (NodeDef* fanout : fanouts) {
        for (string& name : *fanout->mutable_input()) {
          int position;
          string nodename = ParseNodeName(name, &position);
          if (nodename == node->name()) {
            if (position > 0) {
              name = strings::StrCat(rep->name(), ":", position);
            } else if (position == 0) {
              name = rep->name();
            } else {
              name = strings::StrCat("^", rep->name());
            }
            map.AddOutput(rep->name(), fanout->name());
          }
        }
      }
      duplicates.insert(i);
      stop = false;
    }
  } while (!stop);

  // Delete duplicates
  if (!duplicates.empty()) {
    int last = optimized_graph->node_size() - 1;
    for (auto it = duplicates.rbegin(); it != duplicates.rend(); ++it) {
      int index = *it;
      optimized_graph->mutable_node()->SwapElements(index, last);
      last--;
    }
    optimized_graph->mutable_node()->DeleteSubrange(last + 1,
                                                    duplicates.size());
  }
}

string ArithmeticOptimizer::TrySimplifyAndReplaceUses(
    const NodeDef* node, GraphDef* graph_def, NodeMap* node_map,
    std::vector<const NodeDef*>* new_nodes, FrameMap* frame_map) const {
  // Remove involutions applied twice.
  if (IsInvolution(*node)) {
    // An involution is an element-wise function f(x) that is its own inverse,
    // i.e. f(f(x)) = x. If we can find a chain of ops
    //   f->op1->op2->...opn->f
    // where op1 through opn preserve the values of their inputs, we can remove
    // the two instances of the involution from the graph, since they cancel
    // each other.
    NodeDef* tail =
        GetTailOfValuePreservingChain(node, node_map, nodes_to_preserve_);
    NodeDef* involution = node_map->GetNode(tail->input(0));
    if (involution->op() == node->op()) {
      // Skip both *node and *involution since they cancel each other.
      if (tail == node) {
        // The two nodes to eliminate are adjacent.
        return involution->input(0);
      } else {
        tail->set_input(0, involution->input(0));
        node_map->UpdateInput(tail->name(), involution->name(),
                              involution->input(0));
        return node->input(0);
      }
    }
  }

  // Remove inverse transposes.
  if (node->op() == "Transpose" || node->op() == "ConjugateTranspose") {
    const NodeDef* input = node_map->GetNode(node->input(0));
    if (input->op() == node->op()) {
      const NodeDef* node_perm = node_map->GetNode(node->input(1));
      const NodeDef* input_perm = node_map->GetNode(input->input(1));
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
    NodeDef* reshape = node_map->GetNode(node->name());
    int output_pos = 0;
    string input_node_name = ParseNodeName(node->input(0), &output_pos);
    const NodeDef* input = node_map->GetNode(input_node_name);
    if (input->op() == "Reshape") {
      reshape->set_input(0, input->input(0));
      node_map->UpdateInput(reshape->name(), input->name(), input->input(0));
      new_nodes->push_back(reshape);
      return reshape->name();
    }

    // If the reshape is a no-op, forward its input to its consumers. This is
    // considered aggressive and turned off by default, because users may state
    // that the placeholder outputs tensors of shape [M, N] while feeding it
    // with tensors of shape [M*N] (or worse). The reshape nodes are then
    // necessary to update the tensor metadata to the required shape.
    if (opt_level_ == RewriterConfig::AGGRESSIVE &&
        ReshapeIsIdentity(*reshape, *input, output_pos)) {
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
      const NodeDef* cast = node_map->GetNode(transpose->input(0));
      if (cast->op() == "Cast") {
        const NodeDef* input = node_map->GetNode(cast->input(0));
        const DataType src_type = GetSourceDataType(*cast);
        const DataType dst_type = GetDestinationDataType(*cast);
        if (IsNumberType(src_type) && IsNumberType(dst_type) &&
            DataTypeSize(src_type) < DataTypeSize(dst_type)) {
          NodeDef* new_transpose = graph_def->add_node();
          *new_transpose = *transpose;
          new_transpose->set_name(transpose->name() + "_" +
                                  DataTypeString(src_type));
          (*new_transpose->mutable_attr())["T"].set_type(src_type);
          node_map->AddNode(new_transpose->name(), new_transpose);

          new_transpose->set_input(0, cast->input(0));
          node_map->AddOutput(input->name(), new_transpose->name());
          node_map->AddOutput(NodeName(new_transpose->input(1)),
                              new_transpose->name());

          NodeDef* new_cast = graph_def->add_node();
          *new_cast = *cast;
          new_cast->set_name(cast->name() + "_new");
          node_map->AddNode(new_cast->name(), new_cast);

          new_cast->set_input(0, new_transpose->name());
          node_map->AddOutput(new_transpose->name(), new_cast->name());

          new_nodes->push_back(new_transpose);
          //  Add frame dependencies that the original node might have had.
          AddFrameControlDeps(node, {new_transpose, new_cast},
                              new_transpose->input(0), {new_transpose},
                              graph_def, node_map, frame_map);

          return new_cast->name();
        }
      }
    }
  }

  if (node->op() == "Bitcast") {
    NodeDef* bitcast = node_map->GetNode(node->name());
    // Bypass bitcasts whose source type and destination type are equal.
    if (GetSourceDataType(*bitcast) == GetDestinationDataType(*bitcast)) {
      return bitcast->input(0);
    }

    const NodeDef* operand = node_map->GetNode(bitcast->input(0));
    if (operand->op() == bitcast->op()) {
      // Bitcast(Bitcast(x, type1), type2) => Bitcast(x, type2)
      bitcast->set_input(0, operand->input(0));
      SetSourceDataType(GetSourceDataType(*operand), bitcast);
      node_map->UpdateInput(bitcast->name(), bitcast->input(0),
                            operand->input(0));
      new_nodes->push_back(bitcast);
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
    const NodeDef* weights = node_map->GetNode(NodeName(conv->input(1)));
    // Fold the multiply to conv only when the weights are constant, so the
    // multiply can be constant-folded. TODO(jingyue): When the weights aren't
    // constant, this should also help performance a bit and memory usage a lot,
    // since the weights tend to be smaller than the activations.
    if (weights->op() == "Const") {
      const NodeDef* source = node_map->GetNode(
          GetTailOfValuePreservingChain(node, node_map, nodes_to_preserve_)
              ->input(0));
      if (source->op() == "Mul" &&
          node_map->GetOutputs(source->name()).size() == 1) {
        const NodeDef* mul = source;
        // `scale` is the scalar multiplier, and `other` is the other operand.
        // TODO(jingyue): handle the case where `scale` is 0-th operand.
        const NodeDef* scale = node_map->GetNode(mul->input(1));
        const NodeDef* other = node_map->GetNode(mul->input(0));
        if (scale->op() == "Const" && scale->attr().at("dtype").type() ==
                                          weights->attr().at("dtype").type()) {
          const TensorProto& scale_tensor = scale->attr().at("value").tensor();
          // Test whether `scale` is a scalar.
          if (scale_tensor.has_tensor_shape() &&
              scale_tensor.tensor_shape().dim_size() == 0) {
            // Create new node `scaled_weights`.
            NodeDef* scaled_weights = graph_def->add_node();
            scaled_weights->set_name(weights->name() + "_scaled_" +
                                     conv->name());
            scaled_weights->set_op("Mul");
            scaled_weights->set_device(weights->device());
            (*scaled_weights->mutable_attr())["T"] =
                weights->attr().at("dtype");
            node_map->AddNode(scaled_weights->name(), scaled_weights);
            new_nodes->push_back(scaled_weights);

            // Link in its inputs.
            scaled_weights->add_input(conv->input(1));
            node_map->AddOutput(weights->name(), scaled_weights->name());
            scaled_weights->add_input(mul->input(1));
            node_map->AddOutput(scale->name(), scaled_weights->name());
            AddFrameControlDeps(node, {scaled_weights}, "", {}, graph_def,
                                node_map, frame_map);

            // Update `conv`'s weights to `scaled_weights`.
            conv->set_input(1, scaled_weights->name());
            node_map->UpdateInput(conv->name(), weights->name(),
                                  scaled_weights->name());
            new_nodes->push_back(conv);

            // Update `mul`'s consumer to bypass `mul` because it's folded to
            // the weights.
            CHECK_EQ(node_map->GetOutputs(mul->name()).size(), 1);
            NodeDef* consumer_of_mul =
                *node_map->GetOutputs(mul->name()).begin();
            consumer_of_mul->set_input(0, mul->input(0));
            node_map->UpdateInput(consumer_of_mul->name(), mul->name(),
                                  other->name());
            new_nodes->push_back(consumer_of_mul);
            return conv->name();
          }
        }
      }
    }
  }

  if (node->input_size() > 0 && IsAggregate(*node)) {
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
    if (all_equal && node_map->GetNode(node->name() + "_const") == nullptr) {
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
      NodeDef* new_const_node = graph_def->add_node();
      *new_const_node =
          ConstantFolding::CreateNodeDef(node->name() + "_const", value);
      new_const_node->set_device(node->device());
      node_map->AddNode(new_const_node->name(), new_const_node);
      new_nodes->push_back(new_const_node);

      // 2. Replace the aggregate node with Mul(Const(N), x).
      NodeDef* new_mul_node = graph_def->add_node();
      new_mul_node->set_name(node->name() + "_mul");
      new_mul_node->set_op("Mul");
      new_mul_node->set_device(node->device());
      SetDataTypeToAttr(type, "T", new_mul_node);
      node_map->AddNode(new_mul_node->name(), new_mul_node);
      new_mul_node->add_input(new_const_node->name());
      node_map->AddOutput(new_const_node->name(), new_mul_node->name());
      new_mul_node->add_input(node->input(0));
      node_map->AddOutput(node->input(0), new_mul_node->name());

      CopyControlInputs(*node, new_mul_node, graph_def, node_map);
      AddFrameControlDeps(node, {new_const_node, new_mul_node}, node->input(0),
                          {new_const_node}, graph_def, node_map, frame_map);
      return new_mul_node->name();
    }
  }

  // Use the commutativity and (left- and right-) distributive property of
  // multiplication over addition to hoist common factors out of aggregate nodes
  // where all the inputs are Mul nodes. This pattern occurs frequently in
  // regularization terms for the gradients during training.
  if (node->input_size() > 1 && IsAggregate(*node) &&
      node_map->GetNode(node->name() + "_hoist_add") == nullptr) {
    // Determine the set of common factors if the input nodes are all Mul nodes.
    std::set<string> common_factors;
    int i = 0;
    while (i < node->input_size() && (i == 0 || !common_factors.empty()) &&
           !IsControlInput(node->input(i))) {
      const NodeDef* input = node_map->GetNode(node->input(i));
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
        break;
      }
      ++i;
    }
    if (common_factors.size() == 1) {
      // In this case we have an expression of the form
      //   AddN(Mul(x, y1), Mul(y2, x), Mul(x, y3), ... Mul(x, yn))
      // that can be rewritten as
      //   Mul(x, AddN(y1, y2, y3, ... yn))
      // 1. Hoist non-shared factors up into AddN node.
      const string& common_factor = *common_factors.begin();
      NodeDef* new_mul_node = graph_def->add_node();
      NodeDef* new_add_node = graph_def->add_node();
      *new_add_node = *node;
      new_add_node->set_name(node->name() + "_hoist_add");
      new_nodes->push_back(new_add_node);
      node_map->AddNode(new_add_node->name(), new_add_node);
      for (int i = 0; i < node->input_size(); ++i) {
        const string& input = node->input(i);
        if (IsControlInput(input)) {
          MaybeAddControlInput(input, new_add_node, graph_def, node_map);
          continue;
        }
        NodeDef* mul_node = node_map->GetNode(input);
        int unique_factor_index = mul_node->input(0) == common_factor ? 1 : 0;
        const string unique_factor = mul_node->input(unique_factor_index);
        new_add_node->set_input(i, unique_factor);
        // 2. Use a copy of the first Mul node for the outer multiplication.
        if (i == 0) {
          *new_mul_node = *mul_node;
          new_mul_node->set_device(node->device());
          new_mul_node->set_name(node->name() + "_hoist_mul");
          new_mul_node->set_input(0, common_factor);
          new_mul_node->set_input(1, new_add_node->name());
          node_map->AddNode(new_mul_node->name(), new_mul_node);
        }
      }

      // 3. Add frame dependencies that the original node might have had.
      AddFrameControlDeps(node, {new_add_node, new_mul_node}, common_factor,
                          {new_add_node}, graph_def, node_map, frame_map);

      return new_mul_node->name();
    }
  }

  // Fold Transpose into matrix multiplication.
  if ((node->op() == "MatMul" || node->op() == "SparseMatMul" ||
       node->op() == "BatchMatMul") &&
      node_map->GetNode(node->name() + "_fused") == nullptr) {
    const NodeDef* a = node_map->GetNode(node->input(0));
    const NodeDef* b = node_map->GetNode(node->input(1));
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
                               IsInnerMatrixTransposeNode(*a, node_map);
    const bool b_is_foldable = foldable_transpose_ops.count(b->op()) > 0 &&
                               IsInnerMatrixTransposeNode(*b, node_map);
    if (a_is_foldable || b_is_foldable) {
      NodeDef* new_op = graph_def->add_node();
      *new_op = *node;
      new_op->set_name(node->name() + "_fused");
      node_map->AddNode(new_op->name(), new_op);
      if (a_is_foldable) {
        const string attr_a =
            node->op() == "BatchMatMul" ? "adj_x" : "transpose_a";
        FlipBooleanAttr(attr_a, new_op);
        new_op->set_input(0, a->input(0));
        node_map->UpdateInput(new_op->name(), a->name(), a->input(0));
        AddFrameControlDeps(node, {new_op}, a->input(0), {new_op}, graph_def,
                            node_map, frame_map);
      }
      if (b_is_foldable) {
        const string attr_b =
            node->op() == "BatchMatMul" ? "adj_y" : "transpose_b";
        FlipBooleanAttr(attr_b, new_op);
        new_op->set_input(1, b->input(0));
        node_map->UpdateInput(new_op->name(), b->name(), b->input(0));
        if (!a_is_foldable) {
          AddFrameControlDeps(node, {new_op}, b->input(0), {new_op}, graph_def,
                              node_map, frame_map);
        }
      }
    }
  }

  // Fold Conj into Transpose or ConjugateTranspose.
  if ((node->op() == "Conj" || node->op() == "Transpose" ||
       node->op() == "ConjugateTranspose") &&
      node_map->GetNode(node->name() + "_fused") == nullptr) {
    const NodeDef* input = node_map->GetNode(node->input(0));
    const NodeDef* transpose_op = node->op() == "Conj" ? input : node;
    const NodeDef* conj_op = node->op() == "Conj" ? node : input;

    if ((transpose_op->op() == "Transpose" ||
         transpose_op->op() == "ConjugateTranspose") &&
        conj_op->op() == "Conj") {
      NodeDef* new_op = graph_def->add_node();
      *new_op = *transpose_op;
      new_op->set_name(node->name() + "_fused");
      // Flip the type of transpose op to absorb the conjugation.
      new_op->set_op(transpose_op->op() == "Transpose" ? "ConjugateTranspose"
                                                       : "Transpose");
      new_op->set_input(0, input->input(0));
      node_map->AddNode(new_op->name(), new_op);
      node_map->UpdateInput(new_op->name(), node->name(), input->input(0));
      AddFrameControlDeps(node, {new_op}, "", {}, graph_def, node_map,
                          frame_map);
      return new_op->name();
    }
  }

  return "";
}

namespace {
// A vector with a set. The set stores the same elements as the vector, and
// quickly answers whether a value is in the vector. Duplicated elements are not
// allowed for now.
template <class T>
class SetVector {
 public:
  // Returns false if value already existed in the set, true otherwise.
  bool PushBack(const T& value) {
    if (!set_.insert(value).second) {
      VLOG(2) << "Value " << value << " is already in the set.";
      return false;
    }
    vector_.push_back(value);
    return true;
  }

  T PopBack() {
    T back = vector_.back();
    set_.erase(back);
    vector_.pop_back();
    return back;
  }

  bool Exists(const T& value) const { return set_.count(value); }

  bool Empty() const { return vector_.empty(); }

 private:
  std::unordered_set<T> set_;
  std::vector<T> vector_;
};
}  // namespace

Status ArithmeticOptimizer::SimplifyArithmeticOps(
    GraphDef* optimized_graph) const {
  NodeMap node_map(optimized_graph);
  FrameMap frame_map;
  int num_frames;
  TF_RETURN_IF_ERROR(IdentifyFramesWithNodeMap(*optimized_graph, node_map,
                                               &frame_map, &num_frames));
  SetVector<const NodeDef*> nodes_to_simplify;
  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    nodes_to_simplify.PushBack(optimized_graph->mutable_node()->Mutable(i));
  }
  while (!nodes_to_simplify.Empty()) {
    const NodeDef* node = nodes_to_simplify.PopBack();
    std::vector<const NodeDef*> new_nodes;
    const string simplified_tensor = TrySimplifyAndReplaceUses(
        node, optimized_graph, &node_map, &new_nodes, &frame_map);
    if (simplified_tensor.empty()) {
      continue;
    }

    if (NodeName(simplified_tensor) != node->name()) {
      // Always consider simplified_tensor for further optimizations.
      const NodeDef* simplified_node = node_map.GetNode(simplified_tensor);
      if (simplified_node != nullptr) {
        nodes_to_simplify.PushBack(simplified_node);
      }
      // When `node` is simplified to another node rather than in-place, the
      // consumers of `node` are already redirected to `simplified_tensor`.
      // Re-push the consumers into `nodes_to_simplify` for further
      // optimizations.
      std::set<NodeDef*> consumers = node_map.GetOutputs(node->name());
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
          VLOG(2) << "Update input " << consumer->input(i) << " of "
                  << consumer->name() << " to " << simplified_tensor;
        }
        node_map.UpdateInput(consumer->name(), node->name(), simplified_tensor);
        nodes_to_simplify.PushBack(consumer);
      }
    }
    for (const NodeDef* new_node : new_nodes) {
      nodes_to_simplify.PushBack(new_node);
    }
  }
  return Status::OK();
}

Status ArithmeticOptimizer::Optimize(Cluster* /*cluster*/,
                                     const GrapplerItem& item,
                                     GraphDef* optimized_graph) {
  *optimized_graph = item.graph;
  nodes_to_preserve_ = item.NodesToPreserve();
  GraphProperties graph_properties(item);
  TF_RETURN_IF_ERROR(graph_properties.InferStatically());
  TF_RETURN_IF_ERROR(graph_properties.AnnotateOutputShapes(optimized_graph));

  DedupComputations(optimized_graph);
  TF_RETURN_IF_ERROR(SimplifyArithmeticOps(optimized_graph));

  // Clear output shapes.
  for (int i = 0; i < optimized_graph->node_size(); ++i) {
    optimized_graph->mutable_node(i)->mutable_attr()->erase(kOutputShapesAttr);
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
