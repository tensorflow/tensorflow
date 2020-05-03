/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr char kOptimizedSuffix[] = "LayoutOptimizer";
constexpr char kAttrKSize[] = "ksize";
constexpr char kAttrStrides[] = "strides";
constexpr char kAttrDilations[] = "dilations";
constexpr char kAttrExplicitPaddings[] = "explicit_paddings";
constexpr char kAttrDataFormat[] = "data_format";
constexpr char kAttrIsTraining[] = "is_training";
constexpr char kAttrValue[] = "value";
constexpr char kAttrN[] = "N";
constexpr char kAttrT[] = "T";
constexpr char kAttrNumSplit[] = "num_split";
constexpr char kAttrNumOuts[] = "num_outs";
constexpr char kAttrKeepDims[] = "keep_dims";
constexpr char kAttrSqueezeDims[] = "squeeze_dims";
constexpr char kOpTranspose[] = "Transpose";
constexpr char kOpDataFormatVecPermute[] = "DataFormatVecPermute";
constexpr char kOpDataFormatDimMap[] = "DataFormatDimMap";
constexpr char kOpConst[] = "Const";
constexpr char kReshape[] = "Reshape";
constexpr char kReshapeConst[] = "ReshapeConst";
constexpr int kRank = 4;

inline bool AttrDataFormatMatch(const utils::MutableNodeView& node,
                                absl::string_view src_data_format,
                                bool* missing) {
  const auto* attr = node.GetAttr(kAttrDataFormat);
  if (attr != nullptr) {
    return attr->s() == src_data_format;
  }
  *missing = true;
  return false;
}

inline bool AttrDataFormatMatch(const utils::MutableNodeView& node,
                                absl::string_view src_data_format) {
  bool missing = false;
  return AttrDataFormatMatch(node, src_data_format, &missing);
}

// Utils for layout agnostic transposer.

bool IsComparisonOp(const NodeDef& node) {
  bool is_compare = IsApproximateEqual(node) || IsEqual(node) ||
                    IsGreater(node) || IsGreaterEqual(node) || IsLess(node) ||
                    IsLessEqual(node) || IsNotEqual(node);
  return is_compare;
}

std::vector<int> GetRegularFaninPorts(const utils::MutableNodeView& node) {
  const int num_regular_fanins = node.NumRegularFanins();
  std::vector<int> values(num_regular_fanins);
  std::iota(values.begin(), values.end(), 0);
  return values;
}

std::vector<int> GetConcatDataFaninPorts(const utils::MutableNodeView& node) {
  const auto* n_attr = node.GetAttr(kAttrN);
  const int n = n_attr != nullptr ? n_attr->i() : 0;
  const int start = (node.GetOp() == "Concat") ? 1 : 0;
  const int end = start + n;
  std::vector<int> values(end - start);
  std::iota(values.begin(), values.end(), start);
  return values;
}

struct ComparatorByNodeNameAndIndex {
  bool operator()(const utils::MutableFaninView& node1,
                  const utils::MutableFaninView& node2) const {
    auto* node1_view = node1.node_view();
    auto* node2_view = node2.node_view();
    auto name_compare = node1_view->GetName().compare(node2_view->GetName());
    if (name_compare == 0) {
      return node1.index() < node2.index();
    }
    return name_compare < 0;
  }
};

bool IsHostMemory(const NodeDef& node, int output_port) {
  DeviceNameUtils::ParsedName parsed_name;
  if (DeviceNameUtils::ParseFullName(node.device(), &parsed_name)) {
    DeviceType device_type(parsed_name.type);
    Status s = FindKernelDef(device_type, node, nullptr, nullptr);
    if (s.ok()) {
      tensorflow::MemoryTypeVector in_mtypes;
      tensorflow::MemoryTypeVector out_mtypes;
      s = tensorflow::MemoryTypesForNode(OpRegistry::Global(), device_type,
                                         node, &in_mtypes, &out_mtypes);
      if (s.ok()) {
        if (out_mtypes[output_port] == HOST_MEMORY) {
          return true;
        }
      }
    } else {
      return true;
    }
  }
  return false;
}

std::vector<int> GetDimensionIndicesFromLabel(
    const absl::flat_hash_map<char, int>& dim_indices,
    absl::Span<const char> labels) {
  std::vector<int> indices;
  indices.reserve(labels.size());
  for (const auto& label : labels) {
    indices.push_back(dim_indices.at(label));
  }
  return indices;
}

}  // namespace

// TransposeContext.

Status TransposeContext::InitializeTransposeContext(const GrapplerItem& item,
                                                    const Cluster* cluster,
                                                    TransposeContext* context) {
  DCHECK(context != nullptr);
  context->graph_properties = absl::make_unique<GraphProperties>(item);
  TF_RETURN_IF_ERROR(context->graph_properties->InferStatically(false));
  TF_RETURN_IF_ERROR(context->graph_properties->AnnotateOutputShapes(
      &context->graph, /*allow_symbolic_shapes=*/true));
  Status status;
  context->graph_view =
      absl::make_unique<utils::MutableGraphView>(&context->graph, &status);
  TF_RETURN_IF_ERROR(status);
  context->num_nodes = context->graph.node_size();
  const auto& nodes_to_preserve = item.NodesToPreserve();
  context->nodes_to_preserve = absl::flat_hash_set<string>(
      nodes_to_preserve.begin(), nodes_to_preserve.end());
  TF_RETURN_IF_ERROR(context->frames.InferFromGraph(context->graph));
  if (cluster != nullptr) {
    context->virtual_placer =
        absl::make_unique<const VirtualPlacer>(cluster->GetDevices());
  }
  return Status::OK();
}

// Sets data formats to convert from and to for specified device type.
void TransposeContext::AssignDeviceAndDataFormats(
    absl::string_view target_device, absl::string_view src_format,
    absl::string_view dst_format) {
  this->target_device = string(target_device);
  this->src_format = string(src_format);
  this->dst_format = string(dst_format);
  this->src_dim_indices = GetDimensionIndices(src_format);
  this->dst_dim_indices = GetDimensionIndices(dst_format);
  this->src_to_dst = GetPermutation(this->src_dim_indices, dst_format);
  this->dst_to_src = GetPermutation(this->dst_dim_indices, src_format);
}

// Transposer.

bool Transposer::ShouldProcess(const TransposeContext& context,
                               const utils::MutableNodeView& node) const {
  const auto* node_def = node.node();
  const string& device_name =
      GetDeviceName(context.virtual_placer.get(), *node_def);
  string device;
  string task;
  bool is_on_target_device =
      DeviceNameUtils::SplitDeviceName(device_name, &task, &device) &&
      absl::StrContains(absl::AsciiStrToLower(device),
                        absl::AsciiStrToLower(context.target_device));

  // Only checks data format for layout sensitive op.
  bool data_format_match = !IsLayoutSensitiveOp(*node_def) ||
                           AttrDataFormatMatch(node, context.src_format);
  return is_on_target_device && data_format_match &&
         !context.nodes_to_preserve.contains(node_def->name()) &&
         !(node.NumRegularFanouts() == 0 && node.NumControlledFanouts() == 0);
}

Status Transposer::CreateConstPermNode(TransposeContext* context,
                                       absl::string_view node_name,
                                       absl::string_view device,
                                       absl::Span<const int> permutation,
                                       absl::string_view control_node_name,
                                       utils::MutationNewNode* added_node) {
  auto* graph_view = context->graph_view.get();
  DCHECK(!graph_view->HasNode(node_name));

  NodeDef node;
  node.set_name(string(node_name));
  node.set_op(kOpConst);
  node.set_device(string(device));

  if (!control_node_name.empty()) {
    node.add_input(string(control_node_name));
  }

  AttrValue attr_data_type;
  attr_data_type.set_type(DT_INT32);
  node.mutable_attr()->insert({"dtype", attr_data_type});

  AttrValue attr_tensor;
  Tensor tensor(DT_INT32, TensorShape({4}));
  for (int i = 0; i < permutation.size(); i++) {
    tensor.flat<int>()(i) = permutation[i];
  }
  tensor.AsProtoTensorContent(attr_tensor.mutable_tensor());
  node.mutable_attr()->insert({"value", attr_tensor});

  Status status;
  *added_node =
      graph_view->GetMutationBuilder()->AddNode(std::move(node), &status);
  return status;
}

Status Transposer::CreateTransposeNode(
    TransposeContext* context, absl::string_view name_format,
    const DataType& data_type, absl::string_view device,
    TensorShapeProto fanin_shape, absl::Span<const int> permutation,
    absl::string_view control_node_name, utils::MutationNewNode* added_node,
    string* transpose_node_name) {
  const string node_name = absl::Substitute(name_format, kOpTranspose);
  auto* graph_view = context->graph_view.get();
  DCHECK(!graph_view->HasNode(node_name));
  *transpose_node_name = node_name;

  NodeDef node;
  node.set_name(node_name);
  node.set_op(kOpTranspose);
  node.set_device(string(device));

  AttrValue attr_data_type;
  attr_data_type.set_type(data_type);
  node.mutable_attr()->insert({"T", attr_data_type});

  AttrValue attr_data_type_perm;
  attr_data_type_perm.set_type(DT_INT32);
  node.mutable_attr()->insert({"Tperm", attr_data_type_perm});

  if (!fanin_shape.unknown_rank()) {
    TF_RETURN_IF_ERROR(
        PermuteSingle(absl::StrCat("fanin shape in", node.name()), permutation,
                      fanin_shape.mutable_dim()));
    AttrValue attr_output_shape;
    *attr_output_shape.mutable_list()->add_shape() = fanin_shape;
    node.mutable_attr()->insert({kAttrOutputShape, attr_output_shape});
  }

  // Create Const Node
  utils::MutationNewNode const_perm_added_node;
  const string const_perm_node_name =
      absl::Substitute(name_format, "PermConst");
  TF_RETURN_IF_ERROR(CreateConstPermNode(context, const_perm_node_name, device,
                                         permutation, control_node_name,
                                         &const_perm_added_node));
  // Add place holder for 1st input.
  node.add_input("");
  // Connect const_perm_node to 2nd input of transpose_node.
  node.add_input(const_perm_node_name);

  Status status;
  *added_node =
      graph_view->GetMutationBuilder()->AddNode(std::move(node), &status);
  return status;
}

Status Transposer::UpdateFaninEdgesWithOp(TransposeContext* context,
                                          absl::Span<const int> dst_ports,
                                          utils::MutableNodeView* dst_node,
                                          absl::string_view op) {
  const bool is_in_frame = context->frames.IsInFrame(*dst_node->node());
  for (int dst_port : dst_ports) {
    auto& fanin_port = dst_node->GetRegularFanin(dst_port);
    auto* fanin_node_view = fanin_port.node_view();

    TF_RETURN_IF_ERROR(
        UpdateEdge(context,
                   GetFaninNameFormat(dst_node->GetName(), dst_port,
                                      context->src_format, context->dst_format),
                   op, /*input_shape=*/nullptr, /*is_in_frame=*/is_in_frame,
                   /*is_src_format_to_dst_format=*/true, fanin_port.index(),
                   dst_port, fanin_node_view, dst_node));
  }
  return Status::OK();
}

Status Transposer::UpdateFanoutEdgesWithOp(TransposeContext* context,
                                           absl::Span<const int> src_ports,
                                           utils::MutableNodeView* src_node,
                                           absl::string_view op) {
  // Update attr _output_shapes for output ports.
  const auto* output_shape_attr = src_node->GetAttr(kAttrOutputShape);
  AttrValue shape_attr_copy;
  if (op == kOpTranspose && output_shape_attr != nullptr) {
    shape_attr_copy = *output_shape_attr;
    for (int port : src_ports) {
      auto* shape = shape_attr_copy.mutable_list()->mutable_shape(port);
      if (shape->unknown_rank()) continue;
      TF_RETURN_IF_ERROR(
          PermuteSingle(absl::StrCat("output shape attribute at port ", port,
                                     " in", src_node->GetName()),
                        context->src_to_dst, shape->mutable_dim()));
    }
    context->graph_view->GetMutationBuilder()->AddOrUpdateNodeAttr(
        src_node, kAttrOutputShape, shape_attr_copy);
  }

  const bool is_in_frame = context->frames.IsInFrame(*src_node->node());
  // We might modify the output set in the loop. Make a copy first.
  // Use a set with custom comparator to order output nodes by node name,
  // so that we can keep transposer name deterministic.
  for (int src_port : src_ports) {
    const auto& fanouts_src_port = src_node->GetRegularFanout(src_port);
    std::vector<utils::MutableFaninView> sorted_fanouts(
        fanouts_src_port.begin(), fanouts_src_port.end());
    std::sort(sorted_fanouts.begin(), sorted_fanouts.end(),
              ComparatorByNodeNameAndIndex());
    int num_downstream_transposers = 0;
    for (const auto& fanout : sorted_fanouts) {
      TF_RETURN_IF_ERROR(UpdateEdge(
          context,
          GetFanoutNameFormat(src_node->GetName(), src_port,
                              num_downstream_transposers++, context->src_format,
                              context->dst_format),
          op, &shape_attr_copy, /*is_in_frame=*/is_in_frame,
          /*is_src_format_to_dst_format=*/false, src_port, fanout.index(),
          src_node, fanout.node_view()));
    }
  }
  return Status::OK();
}

Status Transposer::CreateDataFormatNode(
    TransposeContext* context, absl::string_view node_name,
    absl::string_view op, absl::string_view device, const DataType& data_type,
    bool is_fanin_on_host, bool is_src_format_to_dst_format,
    utils::MutationNewNode* added_node) {
  auto* graph_view = context->graph_view.get();
  DCHECK(!graph_view->HasNode(node_name));

  // Create the node
  NodeDef node;
  node.set_name(string(node_name));

  // Set up parameters of node.
  node.set_op(string(op));
  node.set_device(string(device));
  AttrValue attr_data_type;
  attr_data_type.set_type(data_type);
  node.mutable_attr()->insert({"T", attr_data_type});

  // The inputs of a DataFormat op could be in host memory for ops such as
  // Reshape. In such cases, run the kernel on the host too.
  if (is_fanin_on_host) {
    AttrValue attr_kernel;
    attr_kernel.set_s("host");
    node.mutable_attr()->insert({"_kernel", attr_kernel});
  }

  AttrValue src_format;
  src_format.set_s(is_src_format_to_dst_format ? context->src_format
                                               : context->dst_format);
  node.mutable_attr()->insert({kAttrSrcFormat, src_format});
  AttrValue dst_format;
  dst_format.set_s(is_src_format_to_dst_format ? context->dst_format
                                               : context->src_format);
  node.mutable_attr()->insert({kAttrDstFormat, dst_format});

  // Add place holder for 1st input field.
  node.add_input("");

  Status status;
  *added_node =
      graph_view->GetMutationBuilder()->AddNode(std::move(node), &status);
  return status;
}

Status Transposer::UpdateEdge(
    TransposeContext* context, absl::string_view name_format,
    absl::string_view op, const AttrValue* input_shape, bool is_in_frame,
    bool is_src_format_to_dst_format, const int src_port, const int dst_port,
    utils::MutableNodeView* src_node, utils::MutableNodeView* dst_node) {
  DCHECK(src_node != nullptr);
  DCHECK(dst_node != nullptr);
  auto* src_node_def = src_node->node();
  auto* dst_node_def = dst_node->node();

  // TODO(lyandy): Minimize device parsing/fetching.
  const string device = GetDeviceName(
      context->virtual_placer.get(),
      is_src_format_to_dst_format ? *dst_node_def : *src_node_def);
  DataType data_type =
      is_src_format_to_dst_format
          ? context->graph_properties
                ->GetInputProperties(dst_node->GetName())[dst_port]
                .dtype()
          : context->graph_properties
                ->GetOutputProperties(src_node->GetName())[src_port]
                .dtype();

  utils::MutationNewNode added_node;
  string added_node_name;
  if (op == kOpTranspose) {
    TensorShapeProto input_shape_proto;
    input_shape_proto.set_unknown_rank(true);
    if (input_shape != nullptr) {
      input_shape_proto = input_shape->list().shape(src_port);
    } else {
      const auto* src_node_shape_attr = src_node->GetAttr(kAttrOutputShape);
      if (src_node_shape_attr != nullptr) {
        input_shape_proto = src_node_shape_attr->list().shape(src_port);
      }
    }
    const string control_node_name =
        is_in_frame ? AsControlDependency(src_node_def->name()) : "";
    const std::vector<int>& permutation =
        is_src_format_to_dst_format ? context->src_to_dst : context->dst_to_src;
    TF_RETURN_IF_ERROR(CreateTransposeNode(
        context, name_format, data_type, device, input_shape_proto, permutation,
        control_node_name, &added_node, &added_node_name));
  } else if (op == kOpDataFormatVecPermute || op == kOpDataFormatDimMap) {
    DeviceNameUtils::ParsedName parsed_name;
    bool is_fanin_on_host =
        DeviceNameUtils::ParseFullName(
            GetDeviceName(context->virtual_placer.get(), *src_node_def),
            &parsed_name) &&
        parsed_name.type != "CPU" && IsHostMemory(*src_node_def, src_port);
    const string node_name = absl::Substitute(name_format, op);
    TF_RETURN_IF_ERROR(CreateDataFormatNode(
        context, node_name, op, device, data_type, is_fanin_on_host,
        is_src_format_to_dst_format, &added_node));
    added_node_name = node_name;
  } else {
    return Status(error::INVALID_ARGUMENT,
                  absl::StrCat("Unsupported op \"", op,
                               "\". Supported ops are Transpose, "
                               "DataFormatVecPerm, DataFormatDimMap."));
  }

  // Connect src_node to 1st input of added_node.
  utils::Mutation* mutation = context->graph_view->GetMutationBuilder();
  mutation->AddOrUpdateRegularFanin(added_node, 0,
                                    {src_node->GetName(), src_port});

  // Connect output of added_node to dst_node:dst_port.
  mutation->AddOrUpdateRegularFanin(dst_node, dst_port, {added_node_name, 0});

  return Status::OK();
}

bool Transposer::IsFanoutPortRankN(const utils::MutableNodeView& node, int port,
                                   int n) const {
  const auto* output_shape_attr = node.GetAttr(kAttrOutputShape);
  if (output_shape_attr == nullptr ||
      output_shape_attr->list().shape_size() <= port) {
    return false;
  }
  const auto& shape = output_shape_attr->list().shape(port);
  return !shape.unknown_rank() && shape.dim_size() == n;
}

bool Transposer::IsFanoutPortsRankN(const utils::MutableNodeView& node,
                                    absl::Span<const int> ports, int n) const {
  for (const auto& port : ports) {
    if (!IsFanoutPortRankN(node, port, n)) {
      return false;
    }
  }
  return true;
}

bool Transposer::IsFaninPortRankN(const utils::MutableNodeView& node, int port,
                                  int n) const {
  if (port < node.NumRegularFanins() && port >= 0) {
    const auto& regular_fanin = node.GetRegularFanin(port);
    return IsFanoutPortRankN(*regular_fanin.node_view(), regular_fanin.index(),
                             n);
  }
  return false;
}

bool Transposer::IsFaninPortDimsNIfConst(const utils::MutableNodeView& node,
                                         int port,
                                         absl::Span<const int> dims) const {
  if (port < node.NumRegularFanins() && port >= 0) {
    const auto& regular_fanin = node.GetRegularFanin(port);
    const auto* fanin_node_view = regular_fanin.node_view();
    if (!IsConstant(*fanin_node_view->node())) {
      return true;
    }
    // If fanin is a Const, check tensor to see if dimensions match.
    const auto* value_attr = fanin_node_view->GetAttr(kAttrValue);
    if (value_attr == nullptr) {
      return false;
    }
    Tensor tensor;
    if (!tensor.FromProto(value_attr->tensor())) {
      return false;
    }
    if (tensor.dims() != dims.size()) {
      return false;
    }
    for (int i = 0; i < dims.size(); ++i) {
      if (tensor.dim_size(i) != dims[i]) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool Transposer::IsFaninPortsDimsNIfConst(const utils::MutableNodeView& node,
                                          absl::Span<const int> ports,
                                          absl::Span<const int> dims) const {
  for (const auto& port : ports) {
    if (!IsFaninPortDimsNIfConst(node, port, dims)) {
      return false;
    }
  }
  return true;
}

bool Transposer::CanProcessNode(const TransposeContext& context,
                                const utils::MutableNodeView& node) const {
  return !context.nodes_to_preserve.contains(node.GetName()) &&
         !(node.NumRegularFanouts() == 0 && node.NumControlledFanouts() == 0);
}

string Transposer::GetFaninNameFormat(absl::string_view node_name, int port,
                                      absl::string_view src_format,
                                      absl::string_view dst_format) {
  return absl::StrCat(node_name, "-", port, "-$0", src_format, "To", dst_format,
                      "-", kOptimizedSuffix);
}

string Transposer::GetFanoutNameFormat(absl::string_view node_name, int port,
                                       int index, absl::string_view src_format,
                                       absl::string_view dst_format) {
  return absl::StrCat(node_name, "-", port, "-", index, "-$0", dst_format, "To",
                      src_format, "-", kOptimizedSuffix);
}

string Transposer::LayoutOptimizerNode(absl::string_view node_name) {
  return absl::StrCat(node_name, "-", kOptimizedSuffix);
}

string Transposer::GetReshapeNodeNameFormat(absl::string_view node_name,
                                            int index,
                                            absl::string_view src_format,
                                            absl::string_view dst_format) {
  return absl::StrCat(node_name, "-", index, "-", kReshape, src_format, "To",
                      dst_format);
}

string Transposer::GetShapeConstNodeNameFormat(absl::string_view node_name,
                                               int index) {
  return absl::StrCat(node_name, "-", index, "-", kReshapeConst);
}

// Layout sensitive transposer.

inline string GetLayoutSensitiveNodeDataFormat(
    const utils::MutableNodeView& node) {
  const auto* attr = node.GetAttr(kAttrDataFormat);
  if (attr != nullptr) {
    return attr->s();
  }
  return "";
}

Status LayoutSensitiveOpTransposer::UpdateNode(TransposeContext* context,
                                               utils::MutableNodeView* node) {
  utils::Mutation* mutation = context->graph_view->GetMutationBuilder();
  AttrValue data_format_attr;
  data_format_attr.set_s(context->dst_format);
  mutation->AddOrUpdateNodeAttr(node, kAttrDataFormat, data_format_attr);

  auto permute_attr = [&context, &node,
                       &mutation](absl::string_view attr_name) {
    const auto* attr = node->GetAttr(attr_name);
    if (attr != nullptr) {
      AttrValue attr_copy(*attr);
      TF_RETURN_IF_ERROR(PermuteSingle(
          absl::StrCat(attr_name, " attribute in", node->GetName()),
          context->src_to_dst, attr_copy.mutable_list()->mutable_i()));
      mutation->AddOrUpdateNodeAttr(node, attr_name, attr_copy);
    }
    return Status::OK();
  };

  // Update attrs.
  TF_RETURN_IF_ERROR(permute_attr(kAttrStrides));
  TF_RETURN_IF_ERROR(permute_attr(kAttrKSize));
  TF_RETURN_IF_ERROR(permute_attr(kAttrDilations));

  const auto* explicit_paddings_attr = node->GetAttr(kAttrExplicitPaddings);
  if (explicit_paddings_attr != nullptr && explicit_paddings_attr->has_list() &&
      explicit_paddings_attr->list().i_size() > 0) {
    AttrValue explicit_paddings_attr_copy(*explicit_paddings_attr);
    TF_RETURN_IF_ERROR(PermuteDouble(
        absl::StrCat("explicit_paddings attribute in", node->GetName()),
        context->src_to_dst,
        explicit_paddings_attr_copy.mutable_list()->mutable_i()));
    mutation->AddOrUpdateNodeAttr(node, kAttrExplicitPaddings,
                                  explicit_paddings_attr_copy);
  }

  return Status::OK();
}

Status DefaultLayoutSensitiveOpTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
  DCHECK(IsDefaultLayoutSensitiveOp(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status AvgPoolGradTransposer::TransposeNode(TransposeContext* context,
                                            utils::MutableNodeView* node) {
  DCHECK(IsAvgPoolGrad(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFaninPortRankN(*node, 1, 4)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {1}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status BiasAddGradTransposer::TransposeNode(TransposeContext* context,
                                            utils::MutableNodeView* node) {
  DCHECK(IsBiasAddGrad(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFaninPortRankN(*node, 0, 4)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  // No need to update output shape, as it is always of shape 1-D with size the
  // feature dimension of `out_backprop`, regardless of whether NCHW or NHWC is
  // used.
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status Conv2DBackpropFilterTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
  DCHECK(IsConv2DBackpropFilter(*node->node()) ||
         IsDepthwiseConv2dNativeBackpropFilter(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0, 2}, node, kOpTranspose));
  // No need to update output shape, as it is always of shape
  // [filter_height, filter_width, in_channels, out_channels], regardless of
  // whether NCHW or NHWC is used.
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status Conv2DBackpropInputTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
  DCHECK(IsConv2DBackpropInput(*node->node()) ||
         IsDepthwiseConv2dNativeBackpropInput(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4)) {
    return Status::OK();
  }

  const auto& fanin = node->GetRegularFanin(0);
  auto* fanin_node = fanin.node_view();
  const auto* output_shape_attr = fanin_node->GetAttr(kAttrOutputShape);
  if (output_shape_attr == nullptr) {
    VLOG(3) << "Cannot compute the shape of " << fanin_node->GetName()
            << " because it is missing attribute " << kAttrOutputShape;
    return Status::OK();
  }
  TensorShapeProto fanin_shape = output_shape_attr->list().shape(fanin.index());
  if (fanin_shape.dim_size() != 1) {
    VLOG(3) << fanin_node->GetName() << " is not a vector.";
    return Status::OK();
  }
  int vector_size = fanin_shape.dim(0).size();
  if (vector_size == -1) {
    VLOG(3) << "The number of elements in " << fanin_node->GetName()
            << " is unknown.";
    return Status::OK();
  }
  if (vector_size != 2 && vector_size != 4) {
    return errors::InvalidArgument(
        fanin_node->GetName(), " must be a vector of size 2 or 4, but found ",
        vector_size);
  }

  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  // Do not permute a input_sizes of size 2 because it represents HW regardless
  // of whether NCHW or NHWC.
  if (vector_size != 2) {
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpDataFormatVecPermute));
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {2}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status FusedBatchNormExTransposer::TransposeNode(TransposeContext* context,
                                                 utils::MutableNodeView* node) {
  DCHECK(IsFusedBatchNormEx(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  if (node->NumRegularFanins() == 6) {
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0, 5}, node, kOpTranspose));
  } else {
    TF_RETURN_IF_ERROR(
        UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  }
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool FusedBatchNormGradTransposer::IsTraining(
    const utils::MutableNodeView& node) const {
  const auto* is_training_attr = node.GetAttr(kAttrIsTraining);
  if (is_training_attr != nullptr) {
    return is_training_attr->b();
  }
  return false;
}

Status FusedBatchNormGradTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
  DCHECK(IsFusedBatchNormGrad(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsTraining(*node)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0, 1}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status MaxPoolV2Transposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
  DCHECK(IsMaxPoolV2(*node->node()));
  // We check data_input's shape instead, because the shape inference of
  // MaxPoolV2 is not able to infer the shape when ksize or strides is not
  // constant.
  const auto& data_fanin = node->GetRegularFanin(0);
  auto* data_fanin_node = data_fanin.node_view();
  if (!ShouldProcess(*context, *node) ||
      !IsFanoutPortRankN(*data_fanin_node, data_fanin.index(), 4)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {1, 2}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status MaxPoolGradTransposer::TransposeNode(TransposeContext* context,
                                            utils::MutableNodeView* node) {
  DCHECK(IsMaxPoolGrad(*node->node()) || IsMaxPoolGradGradV1(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0, 1, 2}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status MaxPoolGradV2Transposer::TransposeNode(TransposeContext* context,
                                              utils::MutableNodeView* node) {
  DCHECK(IsMaxPoolGradV2(*node->node()) || IsMaxPoolGradGradV2(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4)) {
    return Status::OK();
  }
  VLOG(3) << "GenericLayoutOptimizer: transforming node '" << node->GetName()
          << "' with op '" << node->GetOp() << "' from data format '"
          << context->src_format << "' to '" << context->dst_format << "'";
  TF_RETURN_IF_ERROR(UpdateNode(context, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0, 1, 2}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {3, 4}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

// Layout agnostic transposer.

inline bool IsValidConstPermTransposeNode(const utils::MutableNodeView& node,
                                          absl::Span<const int> permutation) {
  Tensor tensor;
  if (!GetValueAttrFromConstInputNode(node, IsTranspose, 1, &tensor)) {
    return false;
  }
  if (tensor.NumElements() != permutation.size()) {
    return false;
  }

  const auto& tensor_data = tensor.unaligned_flat<int32>();
  for (int i = 0; i < permutation.size(); i++) {
    if (permutation[i] != tensor_data(i)) {
      return false;
    }
  }
  return true;
}

inline bool IsValidDataFormatNode(const utils::MutableNodeView& node,
                                  absl::string_view src_format,
                                  absl::string_view dst_format) {
  if (!IsDataFormatOp(node)) {
    return false;
  }
  const auto* src_format_attr = node.GetAttr(kAttrSrcFormat);
  if (src_format_attr == nullptr || src_format_attr->s() != src_format) {
    return false;
  }
  const auto* dst_format_attr = node.GetAttr(kAttrDstFormat);
  if (dst_format_attr == nullptr || dst_format_attr->s() != dst_format) {
    return false;
  }
  return true;
}

inline bool IsLayoutOptimizerAddedDstToSrcTranspose(
    const TransposeContext& context, const utils::MutableNodeView& node) {
  return node.node_index() >= context.num_nodes &&
         IsValidConstPermTransposeNode(node, context.dst_to_src);
}

inline bool IsLayoutOptimizerAddedDstToSrcTransform(
    const TransposeContext& context, const utils::MutableNodeView& node) {
  return node.node_index() >= context.num_nodes &&
         (IsValidConstPermTransposeNode(node, context.dst_to_src) ||
          IsValidDataFormatNode(node, context.dst_format, context.src_format));
}

bool LayoutAgnosticOpTransposer::IsAfterDstToSrcTransform(
    const TransposeContext& context, const utils::MutableNodeView& node) const {
  std::deque<utils::MutableNodeView*> queue;
  absl::flat_hash_set<utils::MutableNodeView*> visited_nodes;
  auto data_node_pos = GetDataFaninPorts(node);
  for (const int pos : data_node_pos) {
    const auto& fanin = node.GetRegularFanin(pos);
    auto* fanin_node = fanin.node_view();
    queue.push_back(fanin_node);
    visited_nodes.insert(fanin_node);
  }
  // The code will exit this while loop in one iteration in most cases, as the
  // graph is already topologically sorted.
  while (!queue.empty()) {
    utils::MutableNodeView* current_node = queue.front();
    queue.pop_front();
    if (IsLayoutOptimizerAddedDstToSrcTransform(context, *current_node)) {
      return true;
    }
    // We only continue searching if the path is connected through
    // format-agnostic nodes.
    if (IsLayoutAgnosticOp(*current_node->node())) {
      auto current_node_pos = GetDataFaninPorts(*current_node);
      for (const auto& pos : current_node_pos) {
        const auto& fanin = current_node->GetRegularFanin(pos);
        auto* fanin_node = fanin.node_view();
        if (visited_nodes.insert(fanin_node).second) {
          queue.push_back(fanin_node);
        }
      }
    }
  }
  return false;
}

std::vector<int> LayoutAgnosticOpTransposer::GetVariadic4DFaninPorts(
    const TransposeContext& context, const utils::MutableNodeView& node) const {
  std::vector<int> ports;
  const int num_regular_fanins = node.NumRegularFanins();
  ports.reserve(num_regular_fanins);
  for (int i = 0; i < num_regular_fanins; ++i) {
    const auto& regular_fanin = node.GetRegularFanin(i);
    auto* regular_fanin_node = regular_fanin.node_view();
    int regular_fanin_port = regular_fanin.index();
    if (IsFanoutPortRankN(*regular_fanin_node, regular_fanin_port, 4) &&
        ((IsAfterDstToSrcTransform(context, *regular_fanin_node) &&
          IsLayoutAgnosticOp(*regular_fanin_node->node())) ||
         IsLayoutOptimizerAddedDstToSrcTranspose(context,
                                                 *regular_fanin_node))) {
      ports.push_back(i);
    }
  }
  return ports;
}

Status DefaultLayoutAgnosticOpTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
  DCHECK(IsDefaultLayoutAgnosticOp(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status AddNTransposer::TransposeNode(TransposeContext* context,
                                     utils::MutableNodeView* node) {
  DCHECK(IsAddN(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, GetDataFaninPorts(*node),
                                            node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool BinaryOpTransposer::IsNDOperateWithMD(const utils::MutableNodeView& node,
                                           int n, int m) {
  return IsFaninPortRankN(node, 0, n) && IsFaninPortRankN(node, 1, m);
}

bool BinaryOpTransposer::IsFaninShapeSupported(
    const utils::MutableNodeView& node) {
  return (IsNDOperateWithMD(node, 4, 0) || IsNDOperateWithMD(node, 4, 1) ||
          IsNDOperateWithMD(node, 4, 4) || IsNDOperateWithMD(node, 0, 4) ||
          IsNDOperateWithMD(node, 1, 4));
}

std::vector<int> BinaryOpTransposer::Get4DDataFaninPorts(
    const utils::MutableNodeView& node) {
  std::vector<int> values;
  if (IsFaninPortRankN(node, 0, 4)) {
    values.push_back(0);
  }
  if (IsFaninPortRankN(node, 1, 4)) {
    values.push_back(1);
  }
  return values;
}

Status BinaryOpTransposer::AddNodeReshape(
    utils::Mutation* mutation, absl::string_view node_name,
    absl::string_view node_device, absl::string_view input_name,
    absl::string_view shape_const_node_name, const DataType& data_type) {
  NodeDef new_node;
  new_node.set_name(string(node_name));
  new_node.add_input(string(input_name));
  new_node.add_input(string(shape_const_node_name));
  new_node.set_op(kReshape);
  new_node.set_device(string(node_device));

  AttrValue attr_type_indices;
  attr_type_indices.set_type(DT_INT32);
  new_node.mutable_attr()->insert({"Tshape", attr_type_indices});

  AttrValue attr_type_params;
  attr_type_params.set_type(data_type);
  new_node.mutable_attr()->insert({"T", attr_type_params});

  Status status;
  mutation->AddNode(std::move(new_node), &status);
  return status;
}

Status BinaryOpTransposer::AddNodeShapeConst(utils::Mutation* mutation,
                                             absl::string_view node_name,
                                             absl::string_view node_device,
                                             bool node_in_frame,
                                             int num_channels,
                                             absl::string_view depended_node) {
  NodeDef new_node;
  new_node.set_name(string(node_name));
  new_node.set_op(kOpConst);
  new_node.set_device(string(node_device));
  AttrValue attr_data_type;
  attr_data_type.set_type(DT_INT32);
  new_node.mutable_attr()->insert({"dtype", attr_data_type});

  AttrValue attr_tensor;
  Tensor tensor(DT_INT32, TensorShape({4}));
  std::vector<int> shape = {1, num_channels, 1, 1};
  for (int i = 0; i < static_cast<int>(shape.size()); i++) {
    tensor.flat<int>()(i) = shape[i];
  }
  tensor.AsProtoTensorContent(attr_tensor.mutable_tensor());
  new_node.mutable_attr()->insert({"value", attr_tensor});
  if (node_in_frame) {
    // This is to ensure the transpose node and the const node are in the same
    // frame.
    // TODO(halehri): Add Test that exercises this condition.
    new_node.add_input(AsControlDependency(string(depended_node)));
  }

  Status status;
  mutation->AddNode(std::move(new_node), &status);
  return status;
}

Status BinaryOpTransposer::MaybeReshapeVectorFanin(
    TransposeContext* context, utils::MutableNodeView* node) {
  int vector_index = -1;
  if (IsNDOperateWithMD(*node, 4, 1)) {
    vector_index = 1;
  } else if (IsNDOperateWithMD(*node, 1, 4)) {
    vector_index = 0;
  }
  if (vector_index != -1) {
    const string& node_name = node->GetName();
    const string& node_device = node->GetDevice();
    string reshape_node_name = LayoutOptimizerNode(GetReshapeNodeNameFormat(
        node_name, vector_index, context->src_format, context->dst_format));
    string shape_const_node_name = LayoutOptimizerNode(
        GetShapeConstNodeNameFormat(node_name, vector_index));
    const auto& fanin = node->GetRegularFanin(vector_index);
    auto* fanin_node = fanin.node_view();
    const auto* output_shape_attr = fanin_node->GetAttr(kAttrOutputShape);
    if (output_shape_attr == nullptr) {
      return errors::InvalidArgument("Missing attribute ", kAttrOutputShape);
    }
    int vector_size =
        output_shape_attr->list().shape(fanin.index()).dim(0).size();
    utils::Mutation* mutation = context->graph_view->GetMutationBuilder();
    TF_RETURN_IF_ERROR(
        AddNodeShapeConst(mutation, shape_const_node_name, node_device,
                          context->frames.IsInFrame(*node->node()), vector_size,
                          fanin_node->GetName()));
    const auto* t_attr = node->GetAttr(kAttrT);
    if (t_attr == nullptr) {
      return errors::InvalidArgument("Missing attribute ", kAttrT);
    }
    TF_RETURN_IF_ERROR(
        AddNodeReshape(mutation, reshape_node_name, node_device,
                       TensorIdToString({fanin_node->GetName(), fanin.index()}),
                       shape_const_node_name, t_attr->type()));
    mutation->AddOrUpdateRegularFanin(node, vector_index,
                                      {reshape_node_name, 0});
  }
  return Status::OK();
}

Status BinaryOpTransposer::TransposeNode(TransposeContext* context,
                                         utils::MutableNodeView* node) {
  DCHECK(IsBinaryOp(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFaninShapeSupported(*node) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, Get4DDataFaninPorts(*node),
                                            node, kOpTranspose));
  TF_RETURN_IF_ERROR(MaybeReshapeVectorFanin(context, node));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status ConcatOpTransposer::TransposeNode(TransposeContext* context,
                                         utils::MutableNodeView* node) {
  DCHECK(IsConcat(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
      context, GetConcatDataFaninPorts(*node), node, kOpTranspose));
  int axis_node = 0;
  if (node->GetOp() == "ConcatV2") {
    const auto* n_attr = node->GetAttr(kAttrN);
    if (n_attr != nullptr) {
      axis_node = n_attr->i();
    }
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {axis_node}, node, kOpDataFormatDimMap));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status FillOpTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
  DCHECK(IsFill(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsFaninPortDimsNIfConst(*node, 0, {4}) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status IdentityNTransposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
  DCHECK(IsIdentityN(*node->node()));
  const auto ports = GetVariadic4DFaninPorts(*context, *node);
  if (!ShouldProcess(*context, *node) || ports.empty()) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, ports, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, ports, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool MergeTransposer::IsEveryFaninAfterDstToSrcTransform(
    const TransposeContext& context, const utils::MutableNodeView& node) const {
  for (const auto& regular_fanin : node.GetRegularFanins()) {
    auto* regular_fanin_node = regular_fanin.node_view();
    if (IsFanoutPortRankN(*regular_fanin_node, regular_fanin.index(), 4) &&
        ((IsAfterDstToSrcTransform(context, *regular_fanin_node) &&
          IsLayoutAgnosticOp(*regular_fanin_node->node())) ||
         IsLayoutOptimizerAddedDstToSrcTranspose(context,
                                                 *regular_fanin_node))) {
      continue;
    }
    return false;
  }
  return true;
}

Status MergeTransposer::TransposeNode(TransposeContext* context,
                                      utils::MutableNodeView* node) {
  DCHECK(IsMerge(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsEveryFaninAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, GetDataFaninPorts(*node),
                                            node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status PadTransposer::TransposeNode(TransposeContext* context,
                                    utils::MutableNodeView* node) {
  DCHECK(IsMirrorPad(*node->node()) || IsMirrorPadGrad(*node->node()) ||
         IsPad(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsFaninPortDimsNIfConst(*node, 1, {4, 2}) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {1}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool ReduceTransposer::KeepDims(const utils::MutableNodeView& node) {
  const auto* keep_dims_attr = node.GetAttr(kAttrKeepDims);
  if (keep_dims_attr != nullptr) {
    return keep_dims_attr->b();
  }
  return false;
}

bool ReduceTransposer::IsAlongAxis(const Tensor& tensor,
                                   absl::Span<const int> axis, int rank) {
  if (tensor.dims() != 1 || tensor.dim_size(0) != axis.size()) {
    return false;
  }
  for (int i = 0; i < axis.size(); ++i) {
    int local_axis = tensor.flat<int>()(i);
    if (local_axis < 0) {
      local_axis += rank;
    }
    bool along_axis = false;
    for (int dim : axis) {
      if (local_axis == dim) {
        along_axis = true;
        break;
      }
    }
    if (!along_axis) {
      return false;
    }
  }
  return true;
}

bool ReduceTransposer::IsReduceAxisSupported(
    const TransposeContext& context, const utils::MutableNodeView& node) {
  if (KeepDims(node)) {
    return true;
  }
  const auto& regular_fanin_1 = node.GetRegularFanin(1);
  auto* axis_node = regular_fanin_1.node_view();
  if (!IsConstant(*axis_node->node())) {
    return false;
  }
  const auto* value_attr = axis_node->GetAttr(kAttrValue);
  if (value_attr == nullptr) {
    return false;
  }
  Tensor tensor;
  if (!tensor.FromProto(value_attr->tensor())) {
    LOG(ERROR) << "Failed to parse TensorProto.";
    return false;
  }
  auto indices = [&context](absl::Span<const char> labels) {
    return GetDimensionIndicesFromLabel(context.src_dim_indices, labels);
  };
  return IsAlongAxis(tensor, indices({'N', 'H', 'W', 'C'}), kRank) ||
         IsAlongAxis(tensor, indices({'H', 'W', 'C'}), kRank) ||
         IsAlongAxis(tensor, indices({'N', 'H', 'W'}), kRank) ||
         IsAlongAxis(tensor, indices({'H', 'W'}), kRank) ||
         IsAlongAxis(tensor, indices({'C'}), kRank);
}

Status ReduceTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
  DCHECK(IsReduceOp(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFaninPortRankN(*node, 0, 4) ||
      !IsReduceAxisSupported(*context, *node) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {1}, node, kOpDataFormatDimMap));
  if (KeepDims(*node)) {
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  }
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status ReverseV2Transposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
  DCHECK(IsReverseV2(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {1}, node, kOpDataFormatDimMap));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool SelectTransposer::IsFaninScalarVector4D(
    const utils::MutableNodeView& fanin, int port) {
  return IsFanoutPortRankN(fanin, port, 0) ||
         IsFanoutPortRankN(fanin, port, 1) || IsFanoutPortRankN(fanin, port, 4);
}

std::vector<int> SelectTransposer::GetFaninPorts(
    const utils::MutableNodeView& fanin, int port) {
  // Input 0 could be a scalar, a vector with size matching the first dimension
  // of input 1 and 2, or must have the same shape as input 1 and 2.
  if (IsFanoutPortRankN(fanin, port, 4)) {
    return {0, 1, 2};
  }
  return {1, 2};
}

Status SelectTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
  DCHECK(IsSelect(*node->node()));
  const auto& regular_fanin_0 = node->GetRegularFanin(0);
  auto* regular_fanin_0_node = regular_fanin_0.node_view();
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsFaninScalarVector4D(*regular_fanin_0_node, regular_fanin_0.index()) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
      context, GetFaninPorts(*regular_fanin_0_node, regular_fanin_0.index()),
      node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status ShapeTransposer::TransposeNode(TransposeContext* context,
                                      utils::MutableNodeView* node) {
  DCHECK(IsShape(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFaninPortRankN(*node, 0, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, {0}, node, kOpDataFormatVecPermute));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status ShapeNTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
  DCHECK(IsShapeN(*node->node()));
  const auto ports = GetVariadic4DFaninPorts(*context, *node);
  if (!ShouldProcess(*context, *node) || ports.empty()) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, ports, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, ports, node, kOpDataFormatVecPermute));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status SliceTransposer::TransposeNode(TransposeContext* context,
                                      utils::MutableNodeView* node) {
  DCHECK(IsSlice(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsFaninPortsDimsNIfConst(*node, {1, 2}, {4}) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {1, 2}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status SplitTransposer::TransposeNode(TransposeContext* context,
                                      utils::MutableNodeView* node) {
  DCHECK(IsSplit(*node->node()));
  const auto ports = GetDataFanoutPorts(*node);
  if (!ShouldProcess(*context, *node) || !IsFanoutPortsRankN(*node, ports, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {1}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0}, node, kOpDataFormatDimMap));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, ports, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status SplitVTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
  DCHECK(IsSplitV(*node->node()));
  const auto ports = GetDataFanoutPorts(*node);
  if (!ShouldProcess(*context, *node) || !IsFanoutPortsRankN(*node, ports, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {2}, node, kOpDataFormatDimMap));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, ports, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool SqueezeTransposer::IsInputConvertible(
    const TransposeContext& context, const utils::MutableNodeView& node) const {
  const auto& regular_fanin_0 = node.GetRegularFanin(0);
  auto* regular_fanin_0_node = regular_fanin_0.node_view();
  const auto* output_shape_attr =
      regular_fanin_0_node->GetAttr(kAttrOutputShape);
  if (output_shape_attr != nullptr) {
    auto& shape = output_shape_attr->list().shape(regular_fanin_0.index());
    if (shape.dim_size() != kRank) {
      return false;
    }
    const int height_dim = context.src_dim_indices.at('H');
    const int width_dim = context.src_dim_indices.at('W');
    if (shape.dim(height_dim).size() == 1 && shape.dim(width_dim).size() == 1) {
      return true;
    }
  }
  return false;
}

bool SqueezeTransposer::IsAlongAxis(const AttrValue& attr,
                                    absl::Span<const int> axis,
                                    int rank) const {
  const auto& list = attr.list();
  // If list is empty, Squeeze op will squeeze all dimensions of size 1.
  if (list.i_size() == 0) {
    return true;
  } else if (list.i_size() != axis.size()) {
    return false;
  }
  for (int i = 0; i < axis.size(); ++i) {
    int local_axis = list.i(i);
    if (local_axis < 0) {
      local_axis += rank;
    }
    bool along_axis = false;
    for (int dim : axis) {
      if (local_axis == dim) {
        along_axis = true;
        break;
      }
    }
    if (!along_axis) {
      return false;
    }
  }
  return true;
}

bool SqueezeTransposer::IsDimsSupported(
    const TransposeContext& context, const utils::MutableNodeView& node) const {
  auto indices = [&context](absl::Span<const char> labels) {
    return GetDimensionIndicesFromLabel(context.src_dim_indices, labels);
  };
  const auto* squeeze_dims_attr = node.GetAttr(kAttrSqueezeDims);
  if (squeeze_dims_attr == nullptr) {
    return false;
  }
  return (IsFanoutPortRankN(node, 0, 2) &&
          IsAlongAxis(*squeeze_dims_attr, indices({'H', 'W'}), kRank)) ||
         (IsFanoutPortRankN(node, 0, 1) &&
          IsAlongAxis(*squeeze_dims_attr, indices({'N', 'H', 'W'}), kRank));
}

Status SqueezeTransposer::UpdateSqueezeDims(TransposeContext* context,
                                            utils::MutableNodeView* node) {
  const auto* squeeze_dims_attr = node->GetAttr(kAttrSqueezeDims);
  if (squeeze_dims_attr == nullptr) {
    return errors::InvalidArgument("Missing attribute ", kAttrSqueezeDims);
  }
  const int max_num_squeeze_dim = context->src_format.length() - 1;
  const int min_squeeze_dim = -(max_num_squeeze_dim + 1);
  std::vector<int> squeeze_dims_mapped;
  const int squeeze_dims_size = squeeze_dims_attr->list().i_size();
  squeeze_dims_mapped.reserve(squeeze_dims_size);
  for (int i = 0; i < squeeze_dims_size; ++i) {
    int dim = squeeze_dims_attr->list().i(i);
    if (dim < min_squeeze_dim || dim >= max_num_squeeze_dim) {
      return errors::InvalidArgument(
          "Attribute '", kAttrSqueezeDims, "' contains out of range index '",
          dim, "', index must be between [", min_squeeze_dim, ", ",
          max_num_squeeze_dim, ")");
    }
    if (dim < 0) {
      dim += max_num_squeeze_dim;
    }
    squeeze_dims_mapped.push_back(context->dst_to_src[dim]);
  }
  std::sort(squeeze_dims_mapped.begin(), squeeze_dims_mapped.end());
  AttrValue squeeze_dims;
  squeeze_dims.mutable_list()->mutable_i()->Reserve(squeeze_dims_size);
  for (const auto& dim : squeeze_dims_mapped) {
    squeeze_dims.mutable_list()->mutable_i()->Add(dim);
  }
  context->graph_view->GetMutationBuilder()->AddOrUpdateNodeAttr(
      node, kAttrSqueezeDims, squeeze_dims);
  return Status::OK();
}

Status SqueezeTransposer::TransposeNode(TransposeContext* context,
                                        utils::MutableNodeView* node) {
  DCHECK(IsSqueeze(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsDimsSupported(*context, *node) ||
      !IsInputConvertible(*context, *node) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateSqueezeDims(context, node));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool StridedSliceTransposer::IsMaskZero(const utils::MutableNodeView& node,
                                        absl::string_view mask) {
  const auto* mask_attr = node.GetAttr(mask);
  if (mask_attr != nullptr) {
    return mask_attr->i() == 0;
  }
  return true;
}

bool StridedSliceTransposer::HasOnlyBeginEndMask(
    const utils::MutableNodeView& node) {
  return IsMaskZero(node, "ellipsis_mask") &&
         IsMaskZero(node, "new_axis_mask") &&
         IsMaskZero(node, "shrink_axis_mask");
}

Status StridedSliceTransposer::PermuteMask(TransposeContext* context,
                                           utils::MutableNodeView* node,
                                           absl::string_view mask) {
  // Computers the permutation of the masks based on the src and dst format.
  // For example:
  // src_format = NHWC
  // dst_format = NCHW
  // src_to_dst permutation = [0, 3, 1, 2].
  // mask : 0010 [Note the bit positions correspond to indexes i.e this is in
  // reverse order of the src format (CWHN)] result : 0100 (WHCN)
  const auto* mask_attr = node->GetAttr(mask);
  const int mask_i = mask_attr != nullptr ? mask_attr->i() : 0;
  if (mask_i < 0 || mask_i > 15) {
    return errors::InvalidArgument("invalid mask value: ", mask_i);
  }
  int result = 0;
  for (int i = 0; i < context->src_to_dst.size(); i++) {
    const int final_pos = context->src_to_dst[i];
    const int position_mask = 1 << final_pos;
    const int bit_i = (mask_i & position_mask) >> final_pos;
    result |= bit_i << i;
  }
  AttrValue new_mask_attr;
  new_mask_attr.set_i(result);
  context->graph_view->GetMutationBuilder()->AddOrUpdateNodeAttr(node, mask,
                                                                 new_mask_attr);
  return Status::OK();
}

Status StridedSliceTransposer::TransposeNode(TransposeContext* context,
                                             utils::MutableNodeView* node) {
  DCHECK(IsStridedSlice(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsFaninPortsDimsNIfConst(*node, {1, 2, 3}, {4}) ||
      !HasOnlyBeginEndMask(*node) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(PermuteMask(context, node, "begin_mask"));
  TF_RETURN_IF_ERROR(PermuteMask(context, node, "end_mask"));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {1, 2, 3}, node,
                                            kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status SwitchTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
  DCHECK(IsSwitch(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFaninPortRankN(*node, 0, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, GetDataFanoutPorts(*node),
                                             node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status TernaryOpTransposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
  DCHECK(IsTernaryOp(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0, 1, 2}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status TileTransposer::TransposeNode(TransposeContext* context,
                                     utils::MutableNodeView* node) {
  DCHECK(IsTile(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsFaninPortDimsNIfConst(*node, 1, {4}) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {1}, node, kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status UnaryGradTransposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
  DCHECK(IsUnaryGrad(*node->node()));
  if (!ShouldProcess(*context, *node) || !IsFanoutPortRankN(*node, 0, 4) ||
      !IsAfterDstToSrcTransform(*context, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, {0, 1}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

// Utils.

string GetDeviceName(const VirtualPlacer* virtual_placer, const NodeDef& node) {
  return (node.device().empty() && virtual_placer != nullptr)
             ? virtual_placer->get_canonical_device_name(node)
             : node.device();
}

bool IsDefaultLayoutSensitiveOp(const NodeDef& node) {
  std::set<string> default_layout_sensitive_ops = {"AvgPool",
                                                   "BiasAdd",
                                                   "Conv2D",
                                                   "DepthwiseConv2dNative",
                                                   "DepthToSpace",
                                                   "FusedBatchNorm",
                                                   "FusedBatchNormV2",
                                                   "FusedBatchNormV3",
                                                   "FusedConv2DBiasActivation",
                                                   "MaxPool",
                                                   "SpaceToDepth"};
  return default_layout_sensitive_ops.find(node.op()) !=
         default_layout_sensitive_ops.end();
}

bool IsLayoutSensitiveOp(const NodeDef& node) {
  return IsDefaultLayoutSensitiveOp(node) || IsAvgPoolGrad(node) ||
         IsBiasAddGrad(node) || IsConv2DBackpropFilter(node) ||
         IsConv2DBackpropInput(node) ||
         IsDepthwiseConv2dNativeBackpropFilter(node) ||
         IsDepthwiseConv2dNativeBackpropInput(node) ||
         IsFusedBatchNormEx(node) || IsFusedBatchNormGrad(node) ||
         IsMaxPoolV2(node) || IsMaxPoolGrad(node) || IsMaxPoolGradV2(node) ||
         IsMaxPoolGradGradV1(node) || IsMaxPoolGradGradV2(node);
}

bool IsDefaultLayoutAgnosticOp(const NodeDef& node) {
  std::set<string> agnostic_nodes = {"Abs",          "Acos",
                                     "Acosh",        "Angle",
                                     "Asin",         "Asinh",
                                     "Atan",         "Atanh",
                                     "Bitcast",      "Cast",
                                     "Ceil",         "CheckNumerics",
                                     "ComplexAbs",   "Conj",
                                     "Cos",          "Cosh",
                                     "Digamma",      "Elu",
                                     "Enter",        "Erf",
                                     "Erfc",         "Exit",
                                     "Exp",          "Expm1",
                                     "Floor",        "GuaranteeConst",
                                     "Identity",     "Imag",
                                     "Inv",          "IsFinite",
                                     "IsInf",        "IsNan",
                                     "Lgamma",       "Log",
                                     "LogicalNot",   "Log1p",
                                     "Neg",          "NextIteration",
                                     "OnesLike",     "PreventGradient",
                                     "Real",         "Reciprocal",
                                     "Relu",         "Relu6",
                                     "Rint",         "Selu",
                                     "Sigmoid",      "Sign",
                                     "Sin",          "Sinh",
                                     "Snapshot",     "Softplus",
                                     "Round",        "Rsqrt",
                                     "Sqrt",         "Square",
                                     "StopGradient", "Tan",
                                     "Tanh",         "ZerosLike"};
  return agnostic_nodes.find(node.op()) != agnostic_nodes.end();
}

bool IsLayoutAgnosticOp(const NodeDef& node) {
  return IsDefaultLayoutAgnosticOp(node) || IsAddN(node) || IsBinaryOp(node) ||
         IsIdentityN(node) || IsMerge(node) || IsMirrorPad(node) ||
         IsMirrorPadGrad(node) || IsPad(node) || IsSelect(node) ||
         IsSwitch(node) || IsTernaryOp(node) || IsUnaryGrad(node) ||
         IsConcat(node) || IsReverseV2(node) || IsTile(node) || IsShape(node) ||
         IsShapeN(node) || IsFill(node) || IsSlice(node) || IsSplit(node) ||
         IsSqueeze(node) || IsSplitV(node) || IsStridedSlice(node) ||
         IsReduceOp(node);
}

bool IsTernaryOp(const NodeDef& node) { return IsBetainc(node); }

bool IsUnaryGrad(const NodeDef& node) {
  bool is_unary_grad =
      IsEluGrad(node) || IsInvGrad(node) || IsReciprocalGrad(node) ||
      IsRelu6Grad(node) || IsReluGrad(node) || IsRsqrtGrad(node) ||
      IsSeluGrad(node) || IsSigmoidGrad(node) || IsSoftplusGrad(node) ||
      IsSoftsignGrad(node) || IsSqrtGrad(node) || IsTanhGrad(node);
  return is_unary_grad;
}

bool IsMaxPoolV2(const NodeDef& node) { return node.op() == "MaxPoolV2"; }

bool IsMaxPoolGradV2(const NodeDef& node) {
  return node.op() == "MaxPoolGradV2";
}

bool IsMaxPoolGradGradV1(const NodeDef& node) {
  return node.op() == "MaxPoolGradGrad";
}

bool IsMaxPoolGradGradV2(const NodeDef& node) {
  return node.op() == "MaxPoolGradGradV2";
}

bool IsBinaryOp(const NodeDef& node) {
  bool is_binary =
      IsAdd(node) || IsAtan2(node) || IsComparisonOp(node) || IsComplex(node) ||
      IsDiv(node) || IsFloorDiv(node) || IsIgamma(node) || IsIgammac(node) ||
      IsLogicalAnd(node) || IsLogicalOr(node) || IsMaximum(node) ||
      IsMinimum(node) || IsMod(node) || IsMul(node) || IsPolygamma(node) ||
      IsPow(node) || IsRealDiv(node) || IsSquaredDifference(node) ||
      IsSub(node) || IsTruncateDiv(node) || IsTruncateMod(node) || IsZeta(node);
  return is_binary;
}

bool IsReduceOp(const NodeDef& node) {
  return IsSum(node) || IsMean(node) || IsProd(node) || IsMax(node) ||
         IsMin(node) || IsAll(node) || IsAny(node);
}

std::vector<int> GetDataFaninPorts(const utils::MutableNodeView& node) {
  const auto* node_def = node.node();
  if (IsAvgPoolGrad(*node_def) || IsSplit(*node_def)) {
    return {1};
  }
  if (IsStridedSliceGrad(*node_def)) {
    return {4};
  }
  if (IsBinaryOp(*node_def) || IsUnaryGrad(*node_def)) {
    return {0, 1};
  }
  if (IsTernaryOp(*node_def) || IsSelect(*node_def) ||
      IsMaxPoolGrad(*node_def) || IsMaxPoolGradV2(*node_def) ||
      IsMaxPoolGradGradV1(*node_def) || IsMaxPoolGradGradV2(*node_def)) {
    return {0, 1, 2};
  }
  if (IsShapeN(*node_def) || IsIdentityN(*node_def) || IsAddN(*node_def) ||
      IsMerge(*node_def)) {
    return GetRegularFaninPorts(node);
  }
  if (IsConcat(*node_def)) {
    return GetConcatDataFaninPorts(node);
  }
  if (node.NumRegularFanins() > 0) {
    return {0};
  }
  return {};
}

std::vector<int> GetDataFanoutPorts(const utils::MutableNodeView& node) {
  const auto* node_def = node.node();
  if (IsIdentityN(*node_def) || IsShape(*node_def) || IsShapeN(*node_def)) {
    return GetDataFaninPorts(node);
  }
  if (IsSplit(*node_def) || IsSplitV(*node_def)) {
    const auto* num_split_attr = node.GetAttr(kAttrNumSplit);
    if (num_split_attr == nullptr) {
      return {0};
    }
    std::vector<int> values(num_split_attr->i());
    std::iota(values.begin(), values.end(), 0);
    return values;
  }
  if (IsSwitch(*node_def)) {
    const auto* num_outs_attr = node.GetAttr(kAttrNumOuts);
    const int num_outs = num_outs_attr != nullptr ? num_outs_attr->i() : 2;
    std::vector<int> values(num_outs);
    std::iota(values.begin(), values.end(), 0);
    return values;
  }
  return {0};
}

bool GetValueAttrFromConstInputNode(
    const utils::MutableNodeView& node,
    const std::function<bool(const NodeDef&)>& predicate, int index,
    Tensor* tensor) {
  if (!predicate(*node.node())) {
    return false;
  }
  const auto& regular_fanin = node.GetRegularFanin(index);
  auto* regular_fanin_node = regular_fanin.node_view();
  if (!IsConstant(*regular_fanin_node->node())) {
    return false;
  }
  const auto* value_attr = regular_fanin_node->GetAttr(kAttrValue);
  if (value_attr == nullptr || value_attr->tensor().dtype() != DT_INT32) {
    return false;
  }
  if (!tensor->FromProto(value_attr->tensor())) {
    return false;
  }

  return true;
}

bool IsDataFormatOp(const utils::MutableNodeView& node) {
  const string& op = node.GetOp();
  return op == kOpDataFormatDimMap || op == kOpDataFormatVecPermute;
}

absl::flat_hash_map<char, int> GetDimensionIndices(
    absl::string_view data_format) {
  const int size = data_format.size();
  absl::flat_hash_map<char, int> index;
  index.reserve(size);
  for (int i = 0; i < size; i++) {
    index[data_format[i]] = i;
  }
  return index;
}

std::vector<int> GetPermutation(
    const absl::flat_hash_map<char, int>& src_dim_indices,
    absl::string_view dst_format) {
  // Generate permutation for transformation between src and dst format.
  // Example:
  // src = NWHC, dst = NCWH
  // index = { N:0 W:1 H:2 C:3 }
  // permutation = [0, 3, 1, 2]
  DCHECK(src_dim_indices.size() == dst_format.size());
  std::vector<int> permutation;
  const int size = dst_format.size();
  permutation.reserve(size);
  for (int i = 0; i < size; i++) {
    permutation.push_back(src_dim_indices.at(dst_format[i]));
  }
  return permutation;
}

}  // namespace grappler
}  // namespace tensorflow
