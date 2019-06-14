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

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
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
constexpr char kAttrOutputShape[] = "_output_shapes";
constexpr char kAttrKSize[] = "ksize";
constexpr char kAttrStrides[] = "strides";
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
constexpr char kGPU[] = "GPU";
constexpr char kNHWC[] = "NHWC";
constexpr char kNCHW[] = "NCHW";

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

}  // namespace

// NodeLayoutContext.

void NodeLayoutContext::Initialize(absl::string_view src_format,
                                   absl::string_view dst_format,
                                   bool is_transposable) {
  src = string(src_format);
  dst = string(dst_format);
  src_to_dst = GetPermutation(src_format, dst_format);
  dst_to_src = GetPermutation(dst_format, src_format);
  transposable = is_transposable;
}

// TransposeContext.

Status TransposeContext::InitializeTransposeContext(const GrapplerItem& item,
                                                    const Cluster* cluster,
                                                    TransposeContext* context) {
  DCHECK(context != nullptr);
  context->graph_properties = absl::make_unique<GraphProperties>(item);
  TF_RETURN_IF_ERROR(context->graph_properties->InferStatically(false));
  TF_RETURN_IF_ERROR(
      context->graph_properties->AnnotateOutputShapes(&context->graph));
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

// Transposer.

string Transposer::GetDeviceName(const VirtualPlacer* virtual_placer,
                                 const NodeDef& node) const {
  return (node.device().empty() && virtual_placer != nullptr)
             ? virtual_placer->get_canonical_device_name(node)
             : node.device();
}

string Transposer::GetDstDataFormatForDevice(
    const DeviceProperties& device) const {
  if (device.type() == kGPU) {
    return kNCHW;
  }
  return "";
}

Status Transposer::CreateConstPermNode(TransposeContext* context,
                                       absl::string_view node_name,
                                       absl::string_view device,
                                       absl::Span<const int> permutation,
                                       utils::MutationNewNode* added_node) {
  auto* graph_view = context->graph_view.get();
  DCHECK(!graph_view->HasNode(node_name));

  NodeDef node;
  node.set_name(string(node_name));
  node.set_op(kOpConst);
  node.set_device(string(device));

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
  node.set_name(string(node_name));
  node.set_op(kOpTranspose);
  node.set_device(string(device));

  AttrValue attr_data_type;
  attr_data_type.set_type(data_type);
  node.mutable_attr()->insert({"T", attr_data_type});

  AttrValue attr_data_type_perm;
  attr_data_type_perm.set_type(DT_INT32);
  node.mutable_attr()->insert({"Tperm", attr_data_type_perm});

  if (!fanin_shape.unknown_rank()) {
    TF_RETURN_IF_ERROR(Permute(permutation, fanin_shape.mutable_dim()));
    AttrValue attr_output_shape;
    *attr_output_shape.mutable_list()->add_shape() = fanin_shape;
    node.mutable_attr()->insert({kAttrOutputShape, attr_output_shape});
  }

  // Create Const Node
  utils::MutationNewNode const_perm_added_node;
  const string const_perm_node_name =
      absl::Substitute(name_format, "PermConst");
  TF_RETURN_IF_ERROR(CreateConstPermNode(context, const_perm_node_name, device,
                                         permutation, &const_perm_added_node));
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
                                          const NodeLayoutContext& layout,
                                          absl::Span<const int> dst_ports,
                                          utils::MutableNodeView* dst_node,
                                          absl::string_view op) {
  for (int dst_port : dst_ports) {
    auto& fanin_port = dst_node->GetRegularFanin(dst_port);
    auto* fanin_node_view = fanin_port.node_view();

    TF_RETURN_IF_ERROR(
        UpdateEdge(context, layout,
                   GetFaninNameFormat(dst_node->GetName(), dst_port, layout.src,
                                      layout.dst),
                   op,
                   /*input_shape=*/nullptr,
                   /*is_src_format_to_dst_format=*/true, fanin_port.index(),
                   dst_port, fanin_node_view, dst_node));
  }
  return Status::OK();
}

Status Transposer::UpdateFanoutEdgesWithOp(TransposeContext* context,
                                           const NodeLayoutContext& layout,
                                           absl::Span<const int> src_ports,
                                           utils::MutableNodeView* src_node,
                                           absl::string_view op) {
  // Update attr _output_shapes for output ports.
  const auto* output_shape_attr = src_node->GetAttr(kAttrOutputShape);
  AttrValue shape_attr_copy;
  if (op == kOpTranspose && output_shape_attr != nullptr) {
    shape_attr_copy = *output_shape_attr;
    for (int port : src_ports) {
      TF_RETURN_IF_ERROR(Permute(
          layout.src_to_dst,
          shape_attr_copy.mutable_list()->mutable_shape(port)->mutable_dim()));
    }
    context->graph_view->GetMutationBuilder()->AddOrUpdateNodeAttr(
        src_node, kAttrOutputShape, shape_attr_copy);
  }

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
      TF_RETURN_IF_ERROR(
          UpdateEdge(context, layout,
                     GetFanoutNameFormat(src_node->GetName(), src_port,
                                         num_downstream_transposers++,
                                         layout.src, layout.dst),
                     op, &shape_attr_copy,
                     /*is_src_format_to_dst_format=*/false, src_port,
                     fanout.index(), src_node, fanout.node_view()));
    }
  }
  return Status::OK();
}

Status Transposer::CreateDataFormatNode(
    TransposeContext* context, const NodeLayoutContext& layout,
    absl::string_view node_name, absl::string_view op, absl::string_view device,
    const DataType& data_type, bool is_fanin_on_host,
    bool is_src_format_to_dst_format, utils::MutationNewNode* added_node) {
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
  src_format.set_s(is_src_format_to_dst_format ? layout.src : layout.dst);
  node.mutable_attr()->insert({kAttrSrcFormat, src_format});
  AttrValue dst_format;
  dst_format.set_s(is_src_format_to_dst_format ? layout.dst : layout.src);
  node.mutable_attr()->insert({kAttrDstFormat, dst_format});

  // Add place holder for 1st input field.
  node.add_input("");

  Status status;
  *added_node =
      graph_view->GetMutationBuilder()->AddNode(std::move(node), &status);
  return status;
}

Status Transposer::UpdateEdge(
    TransposeContext* context, const NodeLayoutContext& layout,
    absl::string_view name_format, absl::string_view op,
    const AttrValue* input_shape, bool is_src_format_to_dst_format,
    const int src_port, const int dst_port, utils::MutableNodeView* src_node,
    utils::MutableNodeView* dst_node) {
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
    if (input_shape != nullptr) {
      input_shape_proto = input_shape->list().shape(src_port);
    } else {
      const auto* src_node_shape_attr = src_node->GetAttr(kAttrOutputShape);
      if (src_node_shape_attr != nullptr) {
        input_shape_proto = src_node_shape_attr->list().shape(src_port);
      }
    }
    const string control_node_name =
        (context->frames.IsInFrame(*src_node_def) &&
         context->frames.IsInFrame(*dst_node_def))
            ? AsControlDependency(src_node_def->name())
            : "";
    const std::vector<int>& permutation =
        is_src_format_to_dst_format ? layout.src_to_dst : layout.dst_to_src;
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
        context, layout, node_name, op, device, data_type, is_fanin_on_host,
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

bool Transposer::IsFanoutPortDimsN(const utils::MutableNodeView& node, int port,
                                   int n) const {
  const auto* output_shape_attr = node.GetAttr(kAttrOutputShape);
  if (output_shape_attr == nullptr ||
      output_shape_attr->list().shape_size() <= port) {
    return false;
  }
  const auto& shape = output_shape_attr->list().shape(port);
  return !shape.unknown_rank() && shape.dim_size() == n;
}

bool Transposer::IsFanoutPortsDimsN(const utils::MutableNodeView& node,
                                    absl::Span<const int> ports, int n) const {
  for (auto port : ports) {
    if (!IsFanoutPortDimsN(node, port, n)) {
      return false;
    }
  }
  return true;
}

bool Transposer::IsFaninPortDimsN(const utils::MutableNodeView& node, int port,
                                  int n) const {
  if (port < node.NumRegularFanins() && port >= 0) {
    const auto& regular_fanin = node.GetRegularFanin(port);
    return IsFanoutPortDimsN(*regular_fanin.node_view(), regular_fanin.index(),
                             n);
  }
  return false;
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

const NodeLayoutContext& LayoutSensitiveOpTransposer::CheckNodeTransposable(
    TransposeContext* context, const utils::MutableNodeView& node) const {
  const auto* node_def = node.node();

  string src_format = GetLayoutSensitiveNodeDataFormat(node);
  if (src_format.empty()) {
    src_format = kNHWC;
  }

  NodeLayoutContext layout;
  DeviceProperties device;
  string dst_format;
  if (context->virtual_placer != nullptr) {
    device = context->virtual_placer->get_device(*node_def);
    dst_format = GetDstDataFormatForDevice(device);
  }

  if (dst_format.empty() || src_format.length() != dst_format.length() ||
      src_format == dst_format || !CanProcessNode(*context, node)) {
    layout.Initialize(src_format, src_format,
                      /*is_transposable=*/false);
    return context->sensitive_node_layouts.emplace(node.node_index(), layout)
        .first->second;
  }

  layout.Initialize(src_format, dst_format, /*is_transposable=*/true);
  return context->sensitive_node_layouts.emplace(node.node_index(), layout)
      .first->second;
}

Status LayoutSensitiveOpTransposer::UpdateNode(TransposeContext* context,
                                               const NodeLayoutContext& layout,
                                               utils::MutableNodeView* node) {
  utils::Mutation* mutation = context->graph_view->GetMutationBuilder();
  AttrValue data_format_attr;
  data_format_attr.set_s(layout.dst);
  mutation->AddOrUpdateNodeAttr(node, kAttrDataFormat, data_format_attr);

  // Update attrs strides and ksize.
  const auto* strides_attr = node->GetAttr(kAttrStrides);
  if (strides_attr != nullptr) {
    AttrValue strides_attr_copy(*strides_attr);
    TF_RETURN_IF_ERROR(Permute(layout.src_to_dst,
                               strides_attr_copy.mutable_list()->mutable_i()));
    mutation->AddOrUpdateNodeAttr(node, kAttrStrides, strides_attr_copy);
  }

  const auto* ksize_attr = node->GetAttr(kAttrKSize);
  if (ksize_attr != nullptr) {
    AttrValue ksize_attr_copy(*ksize_attr);
    TF_RETURN_IF_ERROR(Permute(layout.src_to_dst,
                               ksize_attr_copy.mutable_list()->mutable_i()));
    mutation->AddOrUpdateNodeAttr(node, kAttrKSize, ksize_attr_copy);
  }
  return Status::OK();
}

Status LayoutSensitiveOpTransposer::CommitMutation(
    TransposeContext* context, const utils::MutableNodeView& node) {
  Status status = context->graph_view->GetMutationBuilder()->Apply();
  if (!status.ok()) {
    // If mutation failed, update node layout to reflect current state in graph.
    auto& it = context->sensitive_node_layouts[node.node_index()];
    it.dst = it.src;
    it.transposable = false;
  }
  return status;
}

Status DefaultLayoutSensitiveOpTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
  DCHECK(IsDefaultLayoutSensitiveOp(*node->node()));
  const NodeLayoutContext& layout = CheckNodeTransposable(context, *node);
  if (!layout.transposable || !IsFanoutPortDimsN(*node, 0, 4)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateNode(context, layout, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return CommitMutation(context, *node);
}

Status BiasAddGradTransposer::TransposeNode(TransposeContext* context,
                                            utils::MutableNodeView* node) {
  DCHECK(IsBiasAddGrad(*node->node()));
  const NodeLayoutContext& layout = CheckNodeTransposable(context, *node);
  if (!layout.transposable || !IsFaninPortDimsN(*node, 0, 4)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateNode(context, layout, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  // No need to update output shape, as it is always of shape 1-D with size the
  // feature dimension of `out_backprop`, regardless of whether NCHW or NHWC is
  // used.
  return CommitMutation(context, *node);
}

Status Conv2DBackpropFilterTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
  DCHECK(IsConv2DBackpropFilter(*node->node()));
  const NodeLayoutContext& layout = CheckNodeTransposable(context, *node);
  if (!layout.transposable || !IsFanoutPortDimsN(*node, 0, 4)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateNode(context, layout, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0, 2}, node, kOpTranspose));
  // No need to update output shape, as it is always of shape
  // [filter_height, filter_width, in_channels, out_channels], regardless of
  // whether NCHW or NHWC is used.
  return CommitMutation(context, *node);
}

Status Conv2DBackpropInputTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
  DCHECK(IsConv2DBackpropInput(*node->node()));
  const NodeLayoutContext& layout = CheckNodeTransposable(context, *node);
  if (!layout.transposable || !IsFanoutPortDimsN(*node, 0, 4)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateNode(context, layout, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {2}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, layout, {0}, node,
                                            kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return CommitMutation(context, *node);
}

Status FusedBatchNormGradTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
  DCHECK(IsFusedBatchNormGrad(*node->node()));
  const NodeLayoutContext& layout = CheckNodeTransposable(context, *node);
  if (!layout.transposable || !IsFanoutPortDimsN(*node, 0, 4) ||
      !IsTraining(*node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateNode(context, layout, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0, 1}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return CommitMutation(context, *node);
}

bool FusedBatchNormGradTransposer::IsTraining(
    const utils::MutableNodeView& node) const {
  const auto* is_training_attr = node.GetAttr(kAttrIsTraining);
  if (is_training_attr != nullptr) {
    return is_training_attr->b();
  }
  return false;
}

Status MaxPoolV2Transposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
  DCHECK(IsMaxPoolV2(*node->node()));
  // We check data_input's shape instead, because the shape inference of
  // MaxPoolV2 is not able to infer the shape when ksize or strides is not
  // constant.
  const auto& data_fanin = node->GetRegularFanin(0);
  auto* data_fanin_node = data_fanin.node_view();
  NodeLayoutContext layout = CheckNodeTransposable(context, *node);
  if (!layout.transposable ||
      !IsFanoutPortDimsN(*data_fanin_node, data_fanin.index(), 4)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateNode(context, layout, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, layout, {1, 2}, node,
                                            kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return CommitMutation(context, *node);
}

Status MaxPoolGradTransposer::TransposeNode(TransposeContext* context,
                                            utils::MutableNodeView* node) {
  DCHECK(IsMaxPoolGrad(*node->node()));
  const NodeLayoutContext& layout = CheckNodeTransposable(context, *node);
  if (!layout.transposable || !IsFanoutPortDimsN(*node, 0, 4)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateNode(context, layout, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0, 1, 2}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return CommitMutation(context, *node);
}

Status MaxPoolGradV2Transposer::TransposeNode(TransposeContext* context,
                                              utils::MutableNodeView* node) {
  DCHECK(IsMaxPoolGradV2(*node->node()));
  const NodeLayoutContext& layout = CheckNodeTransposable(context, *node);
  if (!layout.transposable || !IsFanoutPortDimsN(*node, 0, 4)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateNode(context, layout, node));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0, 1, 2}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, layout, {3, 4}, node,
                                            kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return CommitMutation(context, *node);
}

// Layout agnostic transposer.

// Checks if fanin is a Transpose or DataFormatDimMap/DataFormatVecPermute added
// by the transposer.
inline bool IsTransposerAddedOp(const utils::MutableFanoutView& fanin,
                                int num_nodes) {
  if (fanin.node_index() < num_nodes) {
    return false;
  }
  const auto* fanin_node = fanin.node_view();
  if (IsDataFormatOp(*fanin_node)) {
    return true;
  }
  const auto* fanin_node_def = fanin_node->node();
  if (IsTranspose(*fanin_node_def)) {
    const auto& regular_fanin_1 = fanin_node->GetRegularFanin(1);
    return regular_fanin_1.node_index() >= num_nodes &&
           IsConstant(*regular_fanin_1.node_view()->node());
  }
  return false;
}

// Check if fanin is of a processed layout sensitive or agnostic node and return
// its data layout that was used to transform it.
NodeLayoutContext GetTransposerAddedOpDataFormat(
    const TransposeContext& context, const utils::MutableFanoutView& fanin) {
  NodeLayoutContext layout;
  if (!IsTransposerAddedOp(fanin, context.num_nodes)) {
    return layout;
  }
  const auto* fanin_node = fanin.node_view();
  const auto& regular_fanin_0 = fanin_node->GetRegularFanin(0);
  const auto* node_def = regular_fanin_0.node_view()->node();
  if (IsLayoutSensitiveOp(*node_def)) {
    auto it = context.sensitive_node_layouts.find(regular_fanin_0.node_index());
    if (it != context.sensitive_node_layouts.end()) {
      layout = it->second;
    }
  } else if (IsLayoutAgnosticOp(*node_def)) {
    auto it = context.agnostic_node_layouts.find(regular_fanin_0.node_index());
    if (it != context.agnostic_node_layouts.end()) {
      layout = it->second.fanout_layouts[regular_fanin_0.index()];
    }
  }
  return layout;
}

// Resolve fanout data layouts from fanin data layouts for a given node.
void SetFanoutDataLayouts(const TransposeContext& context,
                          const utils::MutableNodeView& node,
                          absl::Span<const NodeLayoutContext> fanin_layouts,
                          absl::Span<const int> fanin_ports,
                          std::vector<NodeLayoutContext>* fanout_layouts,
                          absl::Span<const int> fanout_ports,
                          bool* transposable) {
  const auto* node_def = node.node();
  DeviceProperties device;
  if (context.virtual_placer != nullptr) {
    device = context.virtual_placer->get_device(*node_def);
  }
  if (IsMerge(*node_def)) {
    // Check if both fanins of Merge are being transposed in the same manner.
    // TODO(lyandy): Determine if the restriction of both fanins needing to be
    // transposed is necessary.
    const auto& fanin_layout_0 = fanin_layouts[0];
    if (fanin_layout_0.transposable && fanin_layout_0 == fanin_layouts[1]) {
      (*fanout_layouts)[0] = fanin_layout_0;
      *transposable = true;
    }
  } else if (IsIdentityN(*node_def) || IsShapeN(*node_def)) {
    // Simply forward transposed fanins to fanouts as fanins of IdentityN/ShapeN
    // are independent from one another.
    for (const auto& port : fanin_ports) {
      const auto& layout = fanin_layouts[port];
      if (layout.transposable) {
        (*fanout_layouts)[port] = layout;
        *transposable = true;
      }
    }
  } else {
    // Check if all fanins that are transposed have been transposed from and to
    // the same data format layouts.
    // TODO(lyandy): Implement a better selection of data format if only some
    // fanins are transposed.
    bool is_transposable = true;
    bool found_one_transposable = false;
    NodeLayoutContext layout;
    for (const auto& port : fanin_ports) {
      if (fanin_layouts[port].transposable) {
        if (!found_one_transposable) {
          found_one_transposable = true;
          layout = fanin_layouts[port];
        } else if (layout != fanin_layouts[port]) {
          is_transposable = false;
          break;
        }
      }
    }
    if (is_transposable && found_one_transposable) {
      for (const auto& port : fanout_ports) {
        (*fanout_layouts)[port] = layout;
      }
      *transposable = true;
    }
  }
}

void LayoutAgnosticOpTransposer::ProcessFanoutDataLayouts(
    TransposeContext* context, const utils::MutableNodeView& node) {
  const int node_index = node.node_index();
  auto it = context->agnostic_node_layouts.emplace(node_index,
                                                   LayoutAgnosticNodeFanouts());
  auto fanout_ports = GetDataFanoutPorts(node);
  auto& node_layout = it.first->second;
  if (!fanout_ports.empty()) {
    node_layout.fanout_layouts.resize(fanout_ports[fanout_ports.size() - 1] +
                                      1);
  }
  auto fanin_ports = GetDataFaninPorts(node);
  if (!fanin_ports.empty()) {
    node_layout.fanin_layouts.resize(fanin_ports[fanin_ports.size() - 1] + 1);
  }
  // Copy over fanout data layouts for each fanin.
  for (const auto& port : fanin_ports) {
    const auto& fanin = node.GetRegularFanin(port);
    if (IsLayoutAgnosticOp(*fanin.node_view()->node())) {
      node_layout.fanin_layouts[port] =
          context->agnostic_node_layouts[fanin.node_index()]
              .fanout_layouts[fanin.index()];
    } else {
      node_layout.fanin_layouts[port] =
          GetTransposerAddedOpDataFormat(*context, fanin);
    }
  }
  SetFanoutDataLayouts(*context, node, node_layout.fanin_layouts, fanin_ports,
                       &node_layout.fanout_layouts, fanout_ports,
                       &node_layout.transposable);
}

// Performs a DFS traversal processing layout agnostic parent nodes before
// layout agnostic children nodes to propagate data layouts from processed
// layout sensitive nodes and layout agnostic nodes. Nodes are only visited once
// and results are cached.
void LayoutAgnosticOpTransposer::FetchNodeDataLayouts(
    TransposeContext* context, const utils::MutableNodeView& node) {
  const int num_nodes = context->graph_view->NumNodes();
  std::vector<bool> visited(num_nodes);
  for (const auto& it : context->agnostic_node_layouts) {
    visited[it.first] = true;
  }
  if (visited[node.node_index()]) {
    return;
  }

  // TODO(lyandy): Pull out DFS traversal into a GraphView util.
  enum RecursionStackState : bool { ENTER, EXIT };

  struct RecursionStackEntry {
    RecursionStackEntry(int node_index, RecursionStackState recursion_state)
        : node_index(node_index), recursion_state(recursion_state) {}

    const int node_index;
    const RecursionStackState recursion_state;
  };

  std::vector<RecursionStackEntry> recursion_stack;
  recursion_stack.push_back({node.node_index(), ENTER});
  visited[node.node_index()] = true;

  while (!recursion_stack.empty()) {
    auto curr_entry = recursion_stack.back();
    recursion_stack.pop_back();
    const auto* curr_node = context->graph_view->GetNode(curr_entry.node_index);
    if (curr_entry.recursion_state == ENTER) {
      recursion_stack.push_back({curr_entry.node_index, EXIT});
      auto fanin_ports = GetDataFaninPorts(*curr_node);
      for (const auto& port : fanin_ports) {
        const auto& fanin = curr_node->GetRegularFanin(port);
        const auto* fanin_node = fanin.node_view();
        // Traverse up chains of layout agnostic nodes.
        if (IsLayoutAgnosticOp(*fanin_node->node()) &&
            !visited[fanin.node_index()]) {
          // Unprocessed layout agnostic node fanin.
          recursion_stack.push_back({fanin.node_index(), ENTER});
          visited[fanin.node_index()] = true;
        } else if (IsTransposerAddedOp(fanin, context->num_nodes)) {
          const auto& fanin_fanin_0 = fanin_node->GetRegularFanin(0);
          if (IsLayoutAgnosticOp(*fanin_fanin_0.node_view()->node()) &&
              !visited[fanin_fanin_0.node_index()]) {
            // Processed layout agnostic node fanin.
            recursion_stack.push_back({fanin_fanin_0.node_index(), ENTER});
            visited[fanin_fanin_0.node_index()] = true;
          }
        }
      }
    } else {
      ProcessFanoutDataLayouts(context, *curr_node);
    }
  }
}

const LayoutAgnosticNodeFanouts&
LayoutAgnosticOpTransposer::CheckNodeTransposable(
    TransposeContext* context, const utils::MutableNodeView& node) {
  FetchNodeDataLayouts(context, node);
  if (!CanProcessNode(*context, node)) {
    return context->unknown_agnostic_node_layout;
  }
  auto it = context->agnostic_node_layouts.find(node.node_index());
  if (it != context->agnostic_node_layouts.end()) {
    return it->second;
  }
  return context->unknown_agnostic_node_layout;
}

const NodeLayoutContext& LayoutAgnosticOpTransposer::GetDataFormatLayoutForNode(
    TransposeContext* context, const utils::MutableNodeView& node,
    int fanout_port,
    std::function<bool(const utils::MutableNodeView& node)> node_check) {
  if (!node_check(node)) {
    return context->unknown_node_layout;
  }
  const LayoutAgnosticNodeFanouts& layout =
      CheckNodeTransposable(context, node);
  if (!layout.transposable) {
    return context->unknown_node_layout;
  }
  return layout.fanout_layouts[fanout_port];
}

Status DefaultLayoutAgnosticOpTransposer::TransposeNode(
    TransposeContext* context, utils::MutableNodeView* node) {
  DCHECK(IsDefaultLayoutAgnosticOp(*node->node()));
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFanoutPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status AddNTransposer::TransposeNode(TransposeContext* context,
                                     utils::MutableNodeView* node) {
  DCHECK(IsAddN(*node->node()));
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFanoutPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
      context, layout, GetDataFaninPorts(*node), node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool BinaryOpTransposer::IsNDOperateWithMD(const utils::MutableNodeView& node,
                                           int n, int m) {
  return IsFaninPortDimsN(node, 0, n) && IsFaninPortDimsN(node, 1, m);
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
  if (IsFaninPortDimsN(node, 0, 4)) {
    values.push_back(0);
  }
  if (IsFaninPortDimsN(node, 1, 4)) {
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
    TransposeContext* context, const NodeLayoutContext& layout,
    utils::MutableNodeView* node) {
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
        node_name, vector_index, layout.src, layout.dst));
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
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFaninShapeSupported(node);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
      context, layout, Get4DDataFaninPorts(*node), node, kOpTranspose));
  TF_RETURN_IF_ERROR(MaybeReshapeVectorFanin(context, layout, node));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status ConcatOpTransposer::TransposeNode(TransposeContext* context,
                                         utils::MutableNodeView* node) {
  DCHECK(IsConcat(*node->node()));
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFanoutPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
      context, layout, GetConcatDataFaninPorts(*node), node, kOpTranspose));
  int axis_node = 0;
  if (node->GetOp() == "ConcatV2") {
    const auto* n_attr = node->GetAttr(kAttrN);
    if (n_attr != nullptr) {
      axis_node = n_attr->i();
    }
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, layout, {axis_node}, node,
                                            kOpDataFormatDimMap));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status FillOpTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
  DCHECK(IsFill(*node->node()));
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFanoutPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, layout, {0}, node,
                                            kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status IdentityNTransposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
  DCHECK(IsIdentityN(*node->node()));
  const LayoutAgnosticNodeFanouts& layout =
      CheckNodeTransposable(context, *node);
  if (!layout.transposable) {
    return Status::OK();
  }
  bool transposed = false;
  const int num_layout_ports = layout.fanout_layouts.size();
  for (int i = 0; i < num_layout_ports; ++i) {
    const auto& layout_port = layout.fanout_layouts[i];
    if (layout_port.transposable && IsFanoutPortDimsN(*node, i, 4)) {
      TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, layout_port, {i}, node,
                                                kOpTranspose));
      TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, layout_port, {i},
                                                 node, kOpTranspose));
      transposed = true;
    }
  }
  if (transposed) {
    return context->graph_view->GetMutationBuilder()->Apply();
  }
  return Status::OK();
}

Status MergeTransposer::TransposeNode(TransposeContext* context,
                                      utils::MutableNodeView* node) {
  DCHECK(IsMerge(*node->node()));
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFanoutPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
      context, layout, GetDataFaninPorts(*node), node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status PadTransposer::TransposeNode(TransposeContext* context,
                                    utils::MutableNodeView* node) {
  DCHECK(IsMirrorPad(*node->node()) || IsMirrorPadGrad(*node->node()) ||
         IsPad(*node->node()));
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFanoutPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, layout, {1}, node,
                                            kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status ReduceTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
  DCHECK(IsReduceOp(*node->node()));
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFaninPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable || !IsReduceAxisSupported(layout, *node)) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {1}, node, kOpDataFormatDimMap));
  if (KeepDims(*node)) {
    TF_RETURN_IF_ERROR(
        UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  }
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool ReduceTransposer::KeepDims(const utils::MutableNodeView& node) {
  const auto* keep_dims_attr = node.GetAttr(kAttrKeepDims);
  if (keep_dims_attr != nullptr) {
    return keep_dims_attr->b();
  }
  return false;
}

bool ReduceTransposer::IsAlongAxis(const utils::MutableNodeView& axis_node,
                                   absl::Span<const int> axis) {
  if (!IsConstant(*axis_node.node())) {
    return false;
  }
  const auto* value_attr = axis_node.GetAttr(kAttrValue);
  if (value_attr != nullptr) {
    Tensor tensor;
    if (!tensor.FromProto(value_attr->tensor())) {
      LOG(ERROR) << "Failed to parse TensorProto.";
    }
    if (tensor.dims() == 1 && tensor.dim_size(0) == axis.size()) {
      bool along_axis = true;
      for (int i = 0; i < axis.size(); i++) {
        along_axis = along_axis && (tensor.flat<int>()(i) == axis[i]);
      }
      if (along_axis) return true;
    }
  }
  return false;
}

bool ReduceTransposer::IsReduceAxisSupported(
    const NodeLayoutContext& layout, const utils::MutableNodeView& node) {
  const auto& regular_fanin_1 = node.GetRegularFanin(1);
  auto* axis_node = regular_fanin_1.node_view();
  // TODO(lyandy): Generalize this for other data format conversions.
  return KeepDims(node) ||
         (layout.src == kNHWC && layout.dst == kNCHW &&
          (IsAlongAxis(*axis_node, {0, 1, 2, 3}) ||
           IsAlongAxis(*axis_node, {1, 2, 3}) ||
           IsAlongAxis(*axis_node, {0, 1, 2}) ||
           IsAlongAxis(*axis_node, {1, 2}) || IsAlongAxis(*axis_node, {3})));
}

Status ReverseV2Transposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
  DCHECK(IsReverseV2(*node->node()));
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFanoutPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {1}, node, kOpDataFormatDimMap));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status SelectTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
  DCHECK(IsSelect(*node->node()));
  const auto& regular_fanin_0 = node->GetRegularFanin(0);
  auto* regular_fanin_0_node = regular_fanin_0.node_view();
  if (!IsFaninScalarVector4D(*regular_fanin_0_node, regular_fanin_0.index())) {
    return Status::OK();
  }
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFanoutPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(
      context, layout,
      GetFaninPorts(*regular_fanin_0_node, regular_fanin_0.index()), node,
      kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool SelectTransposer::IsFaninScalarVector4D(
    const utils::MutableNodeView& fanin, int port) {
  return IsFanoutPortDimsN(fanin, port, 0) ||
         IsFanoutPortDimsN(fanin, port, 1) || IsFanoutPortDimsN(fanin, port, 4);
}

std::vector<int> SelectTransposer::GetFaninPorts(
    const utils::MutableNodeView& fanin, int port) {
  // Input 0 could be a scalar, a vector with size matching the first dimension
  // of input 1 and 2, or must have the same shape as input 1 and 2.
  if (IsFanoutPortDimsN(fanin, port, 4)) {
    return {0, 1, 2};
  }
  return {1, 2};
}

Status ShapeTransposer::TransposeNode(TransposeContext* context,
                                      utils::MutableNodeView* node) {
  DCHECK(IsShape(*node->node()));
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFaninPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, layout, {0}, node,
                                             kOpDataFormatVecPermute));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status ShapeNTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
  DCHECK(IsShapeN(*node->node()));
  const LayoutAgnosticNodeFanouts& layout =
      CheckNodeTransposable(context, *node);
  if (!layout.transposable) {
    return Status::OK();
  }
  bool transposed = false;
  const int num_layout_ports = layout.fanout_layouts.size();
  for (int i = 0; i < num_layout_ports; ++i) {
    const auto& layout_port = layout.fanout_layouts[i];
    if (layout_port.transposable && IsFaninPortDimsN(*node, i, 4)) {
      TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, layout_port, {i}, node,
                                                kOpTranspose));
      TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(
          context, layout_port, {i}, node, kOpDataFormatVecPermute));
      transposed = true;
    }
  }
  if (transposed) {
    return context->graph_view->GetMutationBuilder()->Apply();
  }
  return Status::OK();
}

Status SliceTransposer::TransposeNode(TransposeContext* context,
                                      utils::MutableNodeView* node) {
  DCHECK(IsSlice(*node->node()));
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFanoutPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, layout, {1, 2}, node,
                                            kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status SplitTransposer::TransposeNode(TransposeContext* context,
                                      utils::MutableNodeView* node) {
  DCHECK(IsSplit(*node->node()));
  auto output_ports = GetDataFanoutPorts(*node);
  const NodeLayoutContext& layout = GetDataFormatLayoutForNode(
      context, *node, /*fanout_port=*/0,
      [this, output_ports](const utils::MutableNodeView& node) {
        return IsFanoutPortsDimsN(node, output_ports, 4);
      });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {1}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpDataFormatDimMap));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, layout, output_ports,
                                             node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status SplitVTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
  DCHECK(IsSplitV(*node->node()));
  auto output_ports = GetDataFanoutPorts(*node);
  const NodeLayoutContext& layout = GetDataFormatLayoutForNode(
      context, *node, /*fanout_port=*/0,
      [this, output_ports](const utils::MutableNodeView& node) {
        return IsFanoutPortsDimsN(node, output_ports, 4);
      });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {2}, node, kOpDataFormatDimMap));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(context, layout, output_ports,
                                             node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status SqueezeTransposer::TransposeNode(TransposeContext* context,
                                        utils::MutableNodeView* node) {
  DCHECK(IsSqueeze(*node->node()));
  const NodeLayoutContext& layout = GetDataFormatLayoutForNode(
      context, *node, /*fanout_port=*/0,
      [this](const utils::MutableNodeView& node) {
        return IsDimsSupported(node) && IsInputConvertible(node);
      });
  // TODO(lyandy): Generalize this for other data format conversions.
  if (!layout.transposable || layout.src != kNHWC || layout.dst != kNCHW) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateSqueezeDims(context, node));
  return context->graph_view->GetMutationBuilder()->Apply();
}

bool SqueezeTransposer::IsInputConvertible(
    const utils::MutableNodeView& node) const {
  const auto& regular_fanin_0 = node.GetRegularFanin(0);
  auto* regular_fanin_0_node = regular_fanin_0.node_view();
  const auto* output_shape_attr =
      regular_fanin_0_node->GetAttr(kAttrOutputShape);
  if (output_shape_attr != nullptr) {
    auto& shape = output_shape_attr->list().shape(regular_fanin_0.index());
    if (shape.dim_size() != 4) {
      return false;
    }
    if (shape.dim(1).size() == 1 && shape.dim(2).size() == 1) {
      return true;
    }
  }
  return false;
}

bool SqueezeTransposer::IsAlongAxis(const utils::MutableNodeView& node,
                                    absl::Span<const int> axis) const {
  const auto* squeeze_dims_attr = node.GetAttr(kAttrSqueezeDims);
  if (squeeze_dims_attr != nullptr) {
    auto& list = squeeze_dims_attr->list();
    // If list is empty, Squeeze op will squeeze all dimensions of size 1.
    if (list.i_size() == 0) return true;
    if (list.i_size() == axis.size()) {
      bool along_axis = true;
      for (int i = 0; i < axis.size(); i++) {
        along_axis = along_axis && (list.i(i) == axis[i]);
      }
      if (along_axis) return true;
    }
  }
  return false;
}

bool SqueezeTransposer::IsAlongHW(const utils::MutableNodeView& node) const {
  return IsAlongAxis(node, {1, 2});
}

bool SqueezeTransposer::IsAlongNHW(const utils::MutableNodeView& node) const {
  return IsAlongAxis(node, {0, 1, 2});
}

bool SqueezeTransposer::IsDimsSupported(
    const utils::MutableNodeView& node) const {
  return (IsFanoutPortDimsN(node, 0, 2) && IsAlongHW(node)) ||
         (IsFanoutPortDimsN(node, 0, 1) && IsAlongNHW(node));
}

Status SqueezeTransposer::UpdateSqueezeDims(TransposeContext* context,
                                            utils::MutableNodeView* node) {
  const auto* squeeze_dims_attr = node->GetAttr(kAttrSqueezeDims);
  if (squeeze_dims_attr == nullptr) {
    return errors::InvalidArgument("Missing attribute ", kAttrSqueezeDims);
  }
  AttrValue squeeze_dims_copy(*squeeze_dims_attr);
  if (squeeze_dims_copy.list().i_size() == 2) {
    squeeze_dims_copy.mutable_list()->set_i(0, 2);
    squeeze_dims_copy.mutable_list()->set_i(1, 3);
  } else if (squeeze_dims_copy.list().i_size() == 3) {
    squeeze_dims_copy.mutable_list()->set_i(1, 2);
    squeeze_dims_copy.mutable_list()->set_i(2, 3);
  }
  context->graph_view->GetMutationBuilder()->AddOrUpdateNodeAttr(
      node, kAttrSqueezeDims, squeeze_dims_copy);
  return Status::OK();
}

Status StridedSliceTransposer::TransposeNode(TransposeContext* context,
                                             utils::MutableNodeView* node) {
  DCHECK(IsStridedSlice(*node->node()));
  const NodeLayoutContext& layout = GetDataFormatLayoutForNode(
      context, *node, /*fanout_port=*/0,
      [this](const utils::MutableNodeView& node) {
        return IsFanoutPortDimsN(node, 0, 4) && HasOnlyBeginEndMask(node);
      });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(PermuteMask(context, layout, node, "begin_mask"));
  TF_RETURN_IF_ERROR(PermuteMask(context, layout, node, "end_mask"));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, layout, {1, 2, 3}, node,
                                            kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
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
                                           const NodeLayoutContext& layout,
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
  for (int i = 0; i < layout.src_to_dst.size(); i++) {
    const int final_pos = layout.src_to_dst[i];
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

Status SwitchTransposer::TransposeNode(TransposeContext* context,
                                       utils::MutableNodeView* node) {
  DCHECK(IsSwitch(*node->node()));
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFaninPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFanoutEdgesWithOp(
      context, layout, GetDataFanoutPorts(*node), node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status TernaryOpTransposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
  DCHECK(IsTernaryOp(*node->node()));
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFanoutPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0, 1, 2}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status TileTransposer::TransposeNode(TransposeContext* context,
                                     utils::MutableNodeView* node) {
  DCHECK(IsTile(*node->node()));
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFanoutPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(UpdateFaninEdgesWithOp(context, layout, {1}, node,
                                            kOpDataFormatVecPermute));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

Status UnaryGradTransposer::TransposeNode(TransposeContext* context,
                                          utils::MutableNodeView* node) {
  DCHECK(IsUnaryGrad(*node->node()));
  const NodeLayoutContext& layout =
      GetDataFormatLayoutForNode(context, *node, /*fanout_port=*/0,
                                 [this](const utils::MutableNodeView& node) {
                                   return IsFanoutPortDimsN(node, 0, 4);
                                 });
  if (!layout.transposable) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      UpdateFaninEdgesWithOp(context, layout, {0, 1}, node, kOpTranspose));
  TF_RETURN_IF_ERROR(
      UpdateFanoutEdgesWithOp(context, layout, {0}, node, kOpTranspose));
  return context->graph_view->GetMutationBuilder()->Apply();
}

// Utils.

bool IsDefaultLayoutSensitiveOp(const NodeDef& node) {
  std::set<string> default_layout_sensitive_ops = {"AvgPool",
                                                   "BiasAdd",
                                                   "Conv2D",
                                                   "DepthToSpace",
                                                   "FusedBatchNorm",
                                                   "FusedBatchNormV2",
                                                   "FusedConv2DBiasActivation",
                                                   "MaxPool",
                                                   "SpaceToDepth"};
  return default_layout_sensitive_ops.find(node.op()) !=
         default_layout_sensitive_ops.end();
}

bool IsLayoutSensitiveOp(const NodeDef& node) {
  return IsDefaultLayoutSensitiveOp(node) || IsBiasAddGrad(node) ||
         IsConv2DBackpropInput(node) || IsConv2DBackpropFilter(node) ||
         IsFusedBatchNormGrad(node) || IsMaxPoolGrad(node) ||
         IsMaxPoolGradV2(node) || IsMaxPoolV2(node);
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
  if (IsSplit(*node_def)) {
    return {1};
  }
  if (IsStridedSliceGrad(*node_def)) {
    return {4};
  }
  if (IsBinaryOp(*node_def) || IsUnaryGrad(*node_def)) {
    return {0, 1};
  }
  if (IsTernaryOp(*node_def) || IsSelect(*node_def) ||
      IsMaxPoolGrad(*node_def) || IsMaxPoolGradV2(*node_def)) {
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

bool GetValueAttrIfConstPermTransposeNode(const utils::MutableNodeView& node,
                                          Tensor* tensor) {
  if (!IsTranspose(*node.node())) {
    return false;
  }
  const auto& regular_fanin_1 = node.GetRegularFanin(1);
  auto* regular_fanin_1_node = regular_fanin_1.node_view();
  if (!IsConstant(*regular_fanin_1_node->node())) {
    return false;
  }
  const auto* value_attr = regular_fanin_1_node->GetAttr(kAttrValue);
  if (value_attr == nullptr || value_attr->tensor().dtype() != DT_INT32) {
    return false;
  }
  if (!tensor->FromProto(value_attr->tensor())) {
    return false;
  }

  return true;
}

bool IsValidConstPermTransposeNode(const utils::MutableNodeView& node,
                                   absl::Span<const int> permutation) {
  Tensor tensor;
  if (!GetValueAttrIfConstPermTransposeNode(node, &tensor)) {
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

bool IsDataFormatOp(const utils::MutableNodeView& node) {
  const string& op = node.GetOp();
  return op == kOpDataFormatDimMap || op == kOpDataFormatVecPermute;
}

bool IsValidDataFormatNode(const utils::MutableNodeView& node,
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

std::vector<int> GetPermutation(absl::string_view src_format,
                                absl::string_view dst_format) {
  // Generate permutation for transformation between src and dst format.
  // Example:
  // src = NWHC, dst = NCWH
  // index = { N:0 W:1 H:2 C:3 }
  // permutation = [0, 3, 1, 2]
  DCHECK(src_format.size() == dst_format.size())
      << "src format \"" << src_format
      << "\" is not compatible with dst format \"" << dst_format << "\".";
  std::vector<int> permuataion;
  const int size = src_format.size();
  absl::flat_hash_map<char, int> index;
  for (int i = 0; i < size; i++) {
    index[src_format[i]] = i;
  }
  permuataion.reserve(size);
  for (int i = 0; i < size; i++) {
    permuataion.push_back(index[dst_format[i]]);
  }
  return permuataion;
}

}  // namespace grappler
}  // namespace tensorflow
