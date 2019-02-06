/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/remapper.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/symbolic_shapes.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr char kFusedConv2D[] = "_FusedConv2D";

constexpr char kDataFormat[] = "data_format";
constexpr char kIsTraining[] = "is_training";

// TODO(b/119765980): Upgrade upstream Eigen to set `m_can_use_xsmm=false` for
// contractions with non-default contraction output kernels.
bool EigenSupportsContractionOutputKernel() {
#if defined(EIGEN_USE_LIBXSMM)
  return false;
#endif
  return true;
}

struct RemapperContext {
  explicit RemapperContext(const GrapplerItem& item)
      : nodes_to_preserve(item.NodesToPreserve()),
        graph_view(&item.graph),
        graph_properties(item),
        inferred_graph_properties(false) {}

  std::unordered_set<string> nodes_to_preserve;
  GraphView graph_view;
  GraphProperties graph_properties;
  bool inferred_graph_properties;
};

// FusedBatchNorm that can be replaced with a cheaper set of primitives.
struct FusedBatchNorm {
  FusedBatchNorm() = default;
  explicit FusedBatchNorm(const NodeDef* fused_batch_norm)
      : fused_batch_norm(fused_batch_norm) {}

  const NodeDef* fused_batch_norm = nullptr;
};

// Conv2D node followed by a BiasAdd.
struct Conv2DWithBiasAdd {
  Conv2DWithBiasAdd() = default;
  Conv2DWithBiasAdd(const NodeDef* conv2d, const NodeDef* bias_add)
      : conv2d(conv2d), bias_add(bias_add) {}

  const NodeDef* conv2d = nullptr;
  const NodeDef* bias_add = nullptr;
};

// Conv2D node followed by a BiasAdd and Relu.
struct Conv2DWithBiasAddAndRelu {
  Conv2DWithBiasAddAndRelu() = default;
  Conv2DWithBiasAddAndRelu(const NodeDef* conv2d, const NodeDef* bias_add,
                           const NodeDef* relu)
      : conv2d(conv2d), bias_add(bias_add), relu(relu) {}

  const NodeDef* conv2d = nullptr;
  const NodeDef* bias_add = nullptr;
  const NodeDef* relu = nullptr;
};

// Conv2D node followed by a Squeeze and BiasAdd.
struct Conv2DWithSqueezeAndBiasAdd {
  Conv2DWithSqueezeAndBiasAdd() = default;
  Conv2DWithSqueezeAndBiasAdd(const NodeDef* conv2d, const NodeDef* squeeze,
                              const NodeDef* bias_add)
      : conv2d(conv2d), squeeze(squeeze), bias_add(bias_add) {}

  const NodeDef* conv2d = nullptr;
  const NodeDef* squeeze = nullptr;
  const NodeDef* bias_add = nullptr;
};

// Conv2D node followed by a FusedBatchNorm.
struct Conv2DWithBatchNorm {
  Conv2DWithBatchNorm() = default;
  Conv2DWithBatchNorm(const NodeDef* conv2d, const NodeDef* fused_batch_norm,
                      float epsilon = 0.0)
      : conv2d(conv2d), fused_batch_norm(fused_batch_norm), epsilon(epsilon) {}

  const NodeDef* conv2d = nullptr;
  const NodeDef* fused_batch_norm = nullptr;
  float epsilon = 0.0;
};

// Conv2D node followed by a FusedBatchNorm and Relu.
struct Conv2DWithBatchNormAndRelu {
  Conv2DWithBatchNormAndRelu() = default;
  Conv2DWithBatchNormAndRelu(const NodeDef* conv2d,
                             const NodeDef* fused_batch_norm,
                             const NodeDef* relu, float epsilon = 0.0)
      : conv2d(conv2d),
        fused_batch_norm(fused_batch_norm),
        relu(relu),
        epsilon(epsilon) {}

  const NodeDef* conv2d = nullptr;
  const NodeDef* fused_batch_norm = nullptr;
  const NodeDef* relu = nullptr;
  float epsilon = 0.0;
};

bool IsInPreserveSet(const RemapperContext& ctx, const NodeDef* node) {
  return ctx.nodes_to_preserve.count(node->name()) > 0;
}

bool HaveSameDataType(const NodeDef* lhs, const NodeDef* rhs,
                      const string& type_attr = "T") {
  DataType lhs_attr = GetDataTypeFromAttr(*lhs, type_attr);
  DataType rhs_attr = GetDataTypeFromAttr(*rhs, type_attr);

  return lhs_attr != DT_INVALID && rhs_attr != DT_INVALID &&
         lhs_attr == rhs_attr;
}

bool HasDataType(const NodeDef* node, const DataType& expected,
                 const string& type_attr = "T") {
  DataType dtype = GetDataTypeFromAttr(*node, type_attr);
  return dtype == expected;
}

bool IsCpuCompatibleDataType(const NodeDef* node,
                             const string& type_attr = "T") {
  DataType dtype = GetDataTypeFromAttr(*node, type_attr);
  return dtype == DT_FLOAT || dtype == DT_DOUBLE;
}

bool IsGpuCompatibleDataType(const NodeDef* node,
                             const string& type_attr = "T") {
  DataType dtype = GetDataTypeFromAttr(*node, type_attr);
  return dtype == DT_FLOAT;
}

bool IsCpuCompatibleDataFormat(const NodeDef* conv2d) {
  DCHECK(IsConv2D(*conv2d)) << "Expected Conv2D op";
  const string& data_format = conv2d->attr().at(kDataFormat).s();
  return data_format == "NHWC";
}

bool IsGpuCompatibleDataFormat(const NodeDef* conv2d) {
  DCHECK(IsConv2D(*conv2d)) << "Expected Conv2D op";
  const string& data_format = conv2d->attr().at(kDataFormat).s();
  return data_format == "NHWC" || data_format == "NCHW";
}

bool IsCpuCompatibleConv2D(const NodeDef* conv2d) {
  DCHECK(IsConv2D(*conv2d)) << "Expected Conv2D op";
  return NodeIsOnCpu(conv2d) && IsCpuCompatibleDataType(conv2d) &&
         IsCpuCompatibleDataFormat(conv2d);
}

bool IsGpuCompatibleConv2D(const NodeDef* conv2d) {
  DCHECK(IsConv2D(*conv2d)) << "Expected Conv2D op";
  return NodeIsOnGpu(conv2d) && IsGpuCompatibleDataType(conv2d) &&
         IsGpuCompatibleDataFormat(conv2d);
}

// Checks if we can rewrite a pattern to the `_FusedConv2D` on CPU device.
template <typename Pattern>
bool IsCpuCompatible(const Pattern& matched) {
  return IsCpuCompatibleConv2D(matched.conv2d);
}

// Checks if we can rewrite a pattern to the `_FusedConv2D` on GPU device.
bool IsGpuCompatible(const RemapperContext& ctx,
                     const Conv2DWithBiasAddAndRelu& matched) {
  const std::vector<OpInfo::TensorProperties>& input_props =
      ctx.graph_properties.GetInputProperties(matched.conv2d->name());
  const TensorShapeProto& filter_shape =
      input_props.size() >= 2 ? input_props[1].shape() : TensorShapeProto();

  // FusedConv2D on GPU with 1x1 convolution is marginally faster than
  // in-graph computation in micro benchmarks (see kernels/conv_ops_test.cc),
  // and significantly slower in large scale benchmarks.
  bool is_spatial_conv = Rank(filter_shape) == 4 &&          //
                         IsKnown(filter_shape.dim(1)) &&     //
                         IsKnown(filter_shape.dim(2)) &&     //
                         filter_shape.dim(1).size() != 1 &&  //
                         filter_shape.dim(2).size() != 1;

  return is_spatial_conv && IsGpuCompatibleConv2D(matched.conv2d);
}
bool IsGpuCompatible(const RemapperContext& ctx,
                     const Conv2DWithBiasAdd& matched) {
  return false;
}
bool IsGpuCompatible(const RemapperContext& ctx,
                     const Conv2DWithSqueezeAndBiasAdd& matched) {
  return false;
}

// Returns true if the given pattern is supported on the assigned device.
template <typename Pattern>
bool IsDeviceCompatible(const RemapperContext& ctx, Pattern& matched) {
  return IsCpuCompatible(matched) || IsGpuCompatible(ctx, matched);
}

bool FindConv2DWithBias(const RemapperContext& ctx, const NodeDef* bias_add,
                        Conv2DWithBiasAdd* matched,
                        bool check_device_compatible = true) {
  if (!EigenSupportsContractionOutputKernel()) return false;

  // Root of the pattern must be a BiasAdd.
  if (bias_add == nullptr || !IsBiasAdd(*bias_add) ||
      HasControlFaninOrFanout(ctx.graph_view, bias_add))
    return false;

  // Input to the BiasAdd must be a Conv2D.
  const auto input_port = GraphView::InputPort(bias_add, 0);
  const auto conv2d = ctx.graph_view.GetRegularFanin(input_port);

  if (!conv2d.node || !IsConv2D(*conv2d.node) ||
      !HaveSameDataType(bias_add, conv2d.node) ||
      HasControlFaninOrFanout(ctx.graph_view, conv2d.node) ||
      !HasSingleFanoutNode(ctx.graph_view, conv2d.node) ||
      IsInPreserveSet(ctx, conv2d.node))
    return false;

  // Check that data type and data format are supported on assigned device.
  const Conv2DWithBiasAdd pattern{conv2d.node, bias_add};
  if (check_device_compatible && !IsDeviceCompatible(ctx, pattern)) {
    return false;
  }

  // We successfully found a Conv2D+BiasAdd pattern.
  *matched = pattern;

  return true;
}

bool FindConv2DWithBiasAndRelu(const RemapperContext& ctx, const NodeDef* relu,
                               Conv2DWithBiasAddAndRelu* matched) {
  if (!EigenSupportsContractionOutputKernel()) return false;

  // Root of the pattern must be a Relu.
  if (!relu || !IsRelu(*relu) || HasControlFaninOrFanout(ctx.graph_view, relu))
    return false;

  // And input to Relu must match Conv2DWithBiasAdd pattern.
  const auto input_port = GraphView::InputPort(relu, 0);
  const auto bias_add = ctx.graph_view.GetRegularFanin(input_port);

  Conv2DWithBiasAdd base;
  if (!FindConv2DWithBias(ctx, bias_add.node, &base,
                          /*check_device_compatible=*/false) ||
      !HasSingleFanoutNode(ctx.graph_view, base.bias_add) ||
      !HaveSameDataType(relu, base.bias_add) ||
      IsInPreserveSet(ctx, base.bias_add))
    return false;

  // Check that data type and data format are supported on assigned device.
  const Conv2DWithBiasAddAndRelu pattern{base.conv2d, base.bias_add, relu};
  if (!IsDeviceCompatible(ctx, pattern)) return false;

  // We successfully found a Conv2D+BiasAdd+Relu pattern.
  *matched = pattern;

  return true;
}

bool FindConv2DWithSqueezeAndBias(const RemapperContext& ctx,
                                  const NodeDef* bias_add,
                                  Conv2DWithSqueezeAndBiasAdd* matched) {
  if (!EigenSupportsContractionOutputKernel()) return false;

  // Root of the pattern must be a BiasAdd.
  if (!bias_add || !IsBiasAdd(*bias_add) ||
      HasControlFaninOrFanout(ctx.graph_view, bias_add))
    return false;

  // Input to the BiasAdd must be a Squeeze.
  const auto bias_input_port = GraphView::InputPort(bias_add, 0);
  const auto squeeze = ctx.graph_view.GetRegularFanin(bias_input_port);

  if (!squeeze.node || !IsSqueeze(*squeeze.node) ||
      !HaveSameDataType(bias_add, squeeze.node, "T") ||
      HasControlFaninOrFanout(ctx.graph_view, squeeze.node) ||
      !HasSingleFanoutNode(ctx.graph_view, squeeze.node) ||
      IsInPreserveSet(ctx, squeeze.node))
    return false;

  // Squeeze must not squeeze output channel dimension.
  std::vector<int32> dims;
  if (!GetNodeAttr(*squeeze.node, "squeeze_dims", &dims).ok()) return false;
  for (auto dim : dims) {
    if (dim == 3) return false;
  }

  // Input to the Squeeze must be a Conv2D.
  const auto squeeze_input_port = GraphView::InputPort(squeeze.node, 0);
  const auto conv2d = ctx.graph_view.GetRegularFanin(squeeze_input_port);

  if (!conv2d.node || !IsConv2D(*conv2d.node) ||
      !HaveSameDataType(bias_add, conv2d.node, "T") ||
      HasControlFaninOrFanout(ctx.graph_view, conv2d.node) ||
      !HasSingleFanoutNode(ctx.graph_view, conv2d.node) ||
      IsInPreserveSet(ctx, conv2d.node))
    return false;

  // Check that data type and data format are supported on assigned device.
  const Conv2DWithSqueezeAndBiasAdd pattern{conv2d.node, squeeze.node,
                                            bias_add};
  if (!IsDeviceCompatible(ctx, pattern)) return false;

  // We successfully found a Conv2D+Squeeze+BiasAdd pattern.
  *matched = pattern;

  return true;
}

bool FindConv2DWithBatchNorm(const RemapperContext& ctx,
                             const NodeDef* batch_norm,
                             Conv2DWithBatchNorm* matched) {
  if (!EigenSupportsContractionOutputKernel()) return false;

  // Root of the pattern must be a FusedBatchNorm or a FusedBatchNormV2.
  if (!batch_norm || !IsFusedBatchNorm(*batch_norm)) return false;

  // V2 has a separate data type for the scale/offset/mean/variance inputs.
  if (batch_norm->op() == "FusedBatchNormV2" &&
      !HasDataType(batch_norm, DT_FLOAT, "U"))
    return false;

  // Check that batch normalization is in inference mode.
  const auto& attr = batch_norm->attr();
  if (attr.count(kIsTraining) > 0 && attr.at(kIsTraining).b()) return false;

  // Check that only 0th output is consumed by other nodes.
  if (HasControlFaninOrFanout(ctx.graph_view, batch_norm) ||
      HasFanouts(ctx.graph_view, batch_norm, 1) ||  // batch_mean
      HasFanouts(ctx.graph_view, batch_norm, 2) ||  // batch_variance
      HasFanouts(ctx.graph_view, batch_norm, 3) ||  // reserve_space_1
      HasFanouts(ctx.graph_view, batch_norm, 4))    // reserve_space_2
    return false;

  // Input to the FusedBatchNorm must be a Conv2D.
  const auto input_port = GraphView::InputPort(batch_norm, 0);
  const auto conv2d = ctx.graph_view.GetRegularFanin(input_port);

  if (!conv2d.node || !IsConv2D(*conv2d.node) ||               //
      !NodeIsOnCpu(conv2d.node) ||                             //
      !HaveSameDataType(batch_norm, conv2d.node) ||            //
      !IsCpuCompatibleDataType(conv2d.node) ||                 //
      !IsCpuCompatibleDataFormat(conv2d.node) ||               //
      HasControlFaninOrFanout(ctx.graph_view, conv2d.node) ||  //
      !HasSingleFanoutNode(ctx.graph_view, conv2d.node) ||     //
      IsInPreserveSet(ctx, conv2d.node))
    return false;

  // We successfully found a Conv2D+FusedBatchNorm pattern.
  matched->conv2d = conv2d.node;
  matched->fused_batch_norm = batch_norm;
  if (!GetNodeAttr(*batch_norm, "epsilon", &matched->epsilon).ok())
    return false;

  return true;
}

bool FindConv2DWithBatchNormAndRelu(const RemapperContext& ctx,
                                    const NodeDef* node,
                                    Conv2DWithBatchNormAndRelu* matched) {
  if (!EigenSupportsContractionOutputKernel()) return false;

  // Root of the pattern must be a Relu.
  if (!node || !IsRelu(*node) || HasControlFaninOrFanout(ctx.graph_view, node))
    return false;

  // And input to Relu must match Conv2DWithBatchNorm pattern.
  const auto input_port = GraphView::InputPort(node, 0);
  const auto batch_norm = ctx.graph_view.GetRegularFanin(input_port);

  Conv2DWithBatchNorm base;
  if (!FindConv2DWithBatchNorm(ctx, batch_norm.node, &base) ||
      !HasSingleFanoutNode(ctx.graph_view, base.fused_batch_norm) ||
      !HaveSameDataType(node, base.fused_batch_norm) ||
      IsInPreserveSet(ctx, base.fused_batch_norm))
    return false;

  // We successfully found a Conv2D+FusedBatchNorm+Relu pattern.
  matched->conv2d = base.conv2d;
  matched->fused_batch_norm = base.fused_batch_norm;
  matched->relu = node;
  matched->epsilon = base.epsilon;

  return true;
}

// Check that given node meets some basic FusedBatchNorm optimization
// preconditions. We use this check to lazily infer graph properties which is
// rather expensive.
bool IsFusedBatchNormCandidate(const NodeDef& node) {
  if (!IsFusedBatchNorm(node)) return false;
  if (GetDataTypeFromAttr(node, "T") != DT_FLOAT) return false;

  // Check that the node is in inference mode.
  const auto& attr = node.attr();
  if (attr.count(kIsTraining) > 0 && attr.at(kIsTraining).b()) return false;

  return true;
}

bool FindFusedBatchNorm(const RemapperContext& ctx, const NodeDef* node,
                        FusedBatchNorm* matched) {
  if (!IsFusedBatchNormCandidate(*node)) return false;

  const auto& props = ctx.graph_properties.GetInputProperties(node->name());

  // a. Scaling factor can be const folded:
  //      scaling_factor = (variance + epsilon).rsqrt() * scale
  bool const_scaling_factor =
      props.size() == 5 &&     // [x, scale, offset, mean, variance]
      props[1].has_value() &&  // scale
      props[4].has_value();    // variance aka estimated variance

  // b. Or input can be const folded into some other expression.
  auto const_inputs = std::count_if(
      props.begin(), props.end(),
      [](const OpInfo::TensorProperties& props) { return props.has_value(); });

  // TODO(bsteiner): use the cost model to compare the cost of fused batch
  // norm against that of the optimized form.
  bool can_remap = const_scaling_factor || const_inputs >= 4;
  if (!can_remap) return false;

  // The optimized version only generates the first output.
  for (GraphView::Edge edge : ctx.graph_view.GetFanoutEdges(*node, false)) {
    if (edge.src.port_id != 0) return false;
  }

  // We found a fused batch norm node that can be replaced with primitive ops.
  matched->fused_batch_norm = node;
  return true;
}

void CopyConv2DAttributes(const NodeDef* conv2d, NodeDef* fused_conv2d) {
  auto* attr = fused_conv2d->mutable_attr();
  auto src_attr = conv2d->attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["strides"] = src_attr.at("strides");
  (*attr)["padding"] = src_attr.at("padding");
  (*attr)["dilations"] = src_attr.at("dilations");
  (*attr)["data_format"] = src_attr.at("data_format");
  (*attr)["use_cudnn_on_gpu"] = src_attr.at("use_cudnn_on_gpu");
}

void SetFusedConv2DAttributes(
    NodeDef* fused_conv2d, const absl::Span<const absl::string_view> fused_ops,
    int num_args = 1, float epsilon = 0.0) {
  auto* attr = fused_conv2d->mutable_attr();
  SetAttrValue(fused_ops, &(*attr)["fused_ops"]);
  SetAttrValue(num_args, &(*attr)["num_args"]);
  SetAttrValue(epsilon, &(*attr)["epsilon"]);  // required only for BatchNorm
}

void AddFusedConv2DNode(
    const RemapperContext& ctx, const Conv2DWithBiasAdd& matched,
    GraphDef* optimized_graph,
    absl::flat_hash_set<const NodeDef*>* invalidated_nodes) {
  DCHECK(IsDeviceCompatible(ctx, matched))
      << "Unsupported fused Conv2D pattern";

  VLOG(2) << "Fuse Conv2D with BiasAdd: "
          << " bias_add=" << matched.bias_add->name()
          << " conv2d=" << matched.conv2d->name();

  NodeDef* fused_conv2d = optimized_graph->add_node();
  fused_conv2d->set_op(kFusedConv2D);
  fused_conv2d->set_name(matched.bias_add->name());
  fused_conv2d->set_device(matched.conv2d->device());
  fused_conv2d->add_input(matched.conv2d->input(0));    // 0: input
  fused_conv2d->add_input(matched.conv2d->input(1));    // 1: filter
  fused_conv2d->add_input(matched.bias_add->input(1));  // 2: bias

  CopyConv2DAttributes(matched.conv2d, fused_conv2d);
  SetFusedConv2DAttributes(fused_conv2d, {"BiasAdd"});

  invalidated_nodes->insert(matched.bias_add);
  invalidated_nodes->insert(matched.conv2d);
}

void AddFusedConv2DNode(
    const RemapperContext& ctx, const Conv2DWithBiasAddAndRelu& matched,
    GraphDef* optimized_graph,
    absl::flat_hash_set<const NodeDef*>* invalidated_nodes) {
  DCHECK(IsDeviceCompatible(ctx, matched))
      << "Unsupported fused Conv2D pattern";

  VLOG(2) << "Fuse Conv2D with BiasAdd and Relu: "
          << " relu=" << matched.relu->name()
          << " bias_add=" << matched.bias_add->name()
          << " conv2d=" << matched.conv2d->name();

  NodeDef* fused_conv2d = optimized_graph->add_node();
  fused_conv2d->set_name(matched.relu->name());
  fused_conv2d->set_op(kFusedConv2D);
  fused_conv2d->set_device(matched.conv2d->device());
  fused_conv2d->add_input(matched.conv2d->input(0));    // 0: input
  fused_conv2d->add_input(matched.conv2d->input(1));    // 1: filter
  fused_conv2d->add_input(matched.bias_add->input(1));  // 2: bias

  CopyConv2DAttributes(matched.conv2d, fused_conv2d);
  SetFusedConv2DAttributes(fused_conv2d, {"BiasAdd", "Relu"});

  invalidated_nodes->insert(matched.relu);
  invalidated_nodes->insert(matched.bias_add);
  invalidated_nodes->insert(matched.conv2d);
}

void AddFusedConv2DNode(
    const RemapperContext& ctx, const Conv2DWithSqueezeAndBiasAdd& matched,
    GraphDef* optimized_graph,
    absl::flat_hash_set<const NodeDef*>* invalidated_nodes) {
  DCHECK(IsDeviceCompatible(ctx, matched))
      << "Unsupported fused Conv2D pattern";

  VLOG(2) << "Fuse Conv2D with Squeeze and BiasAdd: "
          << " bias_add=" << matched.bias_add->name()
          << " squeeze=" << matched.squeeze->name()
          << " conv2d=" << matched.conv2d->name();

  // Replace Conv2D node with a fused Conv2D. Matched pattern guarantees that it
  // has single consumer (only the squeeze node).
  NodeDef* fused_conv2d = optimized_graph->add_node();
  fused_conv2d->set_name(matched.conv2d->name());
  fused_conv2d->set_op(kFusedConv2D);
  fused_conv2d->set_device(matched.conv2d->device());
  fused_conv2d->add_input(matched.conv2d->input(0));    // 0: input
  fused_conv2d->add_input(matched.conv2d->input(1));    // 1: filter
  fused_conv2d->add_input(matched.bias_add->input(1));  // 2: bias

  CopyConv2DAttributes(matched.conv2d, fused_conv2d);
  SetFusedConv2DAttributes(fused_conv2d, {"BiasAdd"});

  // Replace BiasAdd node with a Squeeze.
  NodeDef* remapped_squeeze = optimized_graph->add_node();
  *remapped_squeeze = *matched.squeeze;
  remapped_squeeze->set_name(matched.bias_add->name());
  remapped_squeeze->set_input(0, fused_conv2d->name());

  invalidated_nodes->insert(matched.squeeze);
  invalidated_nodes->insert(matched.bias_add);
  invalidated_nodes->insert(matched.conv2d);
}

void AddFusedConv2DNode(
    const Conv2DWithBatchNorm& matched, GraphDef* optimized_graph,
    absl::flat_hash_set<const NodeDef*>* invalidated_nodes) {
  VLOG(2) << "Fuse Conv2D with BatchNorm: batch_norm="
          << matched.fused_batch_norm->name()
          << " conv2d=" << matched.conv2d->name();

  NodeDef* fused_conv2d = optimized_graph->add_node();
  fused_conv2d->set_name(matched.fused_batch_norm->name());
  fused_conv2d->set_op(kFusedConv2D);
  fused_conv2d->set_device(matched.conv2d->device());
  fused_conv2d->add_input(matched.conv2d->input(0));            // 0: input
  fused_conv2d->add_input(matched.conv2d->input(1));            // 1: filter
  fused_conv2d->add_input(matched.fused_batch_norm->input(1));  // 2: scale
  fused_conv2d->add_input(matched.fused_batch_norm->input(2));  // 3: offset
  fused_conv2d->add_input(matched.fused_batch_norm->input(3));  // 4: mean
  fused_conv2d->add_input(matched.fused_batch_norm->input(4));  // 5: variance

  CopyConv2DAttributes(matched.conv2d, fused_conv2d);
  SetFusedConv2DAttributes(fused_conv2d, {"FusedBatchNorm"},
                           /*num_args=*/4, /*epsilon=*/matched.epsilon);

  invalidated_nodes->insert(matched.fused_batch_norm);
  invalidated_nodes->insert(matched.conv2d);
}

void AddFusedConv2DNode(
    const Conv2DWithBatchNormAndRelu& matched, GraphDef* optimized_graph,
    absl::flat_hash_set<const NodeDef*>* invalidated_nodes) {
  VLOG(2) << "Fuse Conv2D with BatchNorm and Relu: relu="
          << matched.relu->name()
          << " batch_norm=" << matched.fused_batch_norm->name()
          << " conv2d=" << matched.conv2d->name();

  NodeDef* fused_conv2d = optimized_graph->add_node();
  fused_conv2d->set_name(matched.relu->name());
  fused_conv2d->set_op(kFusedConv2D);
  fused_conv2d->set_device(matched.conv2d->device());
  fused_conv2d->add_input(matched.conv2d->input(0));            // 0: input
  fused_conv2d->add_input(matched.conv2d->input(1));            // 1: filter
  fused_conv2d->add_input(matched.fused_batch_norm->input(1));  // 2: scale
  fused_conv2d->add_input(matched.fused_batch_norm->input(2));  // 3: offset
  fused_conv2d->add_input(matched.fused_batch_norm->input(3));  // 4: mean
  fused_conv2d->add_input(matched.fused_batch_norm->input(4));  // 5: variance

  CopyConv2DAttributes(matched.conv2d, fused_conv2d);
  SetFusedConv2DAttributes(fused_conv2d, {"FusedBatchNorm", "Relu"},
                           /*num_args=*/4, /*epsilon=*/matched.epsilon);

  invalidated_nodes->insert(matched.relu);
  invalidated_nodes->insert(matched.fused_batch_norm);
  invalidated_nodes->insert(matched.conv2d);
}

void AddBatchNormNodes(const FusedBatchNorm& matched,
                       GraphDef* optimized_graph) {
  const NodeDef& fused_node = *matched.fused_batch_norm;
  VLOG(2) << "Optimizing fused batch norm node "
          << SummarizeNodeDef(fused_node);

  const string& x = fused_node.input(0);
  string scale = fused_node.input(1);
  string offset = fused_node.input(2);
  string mean = fused_node.input(3);
  string variance = fused_node.input(4);

  if (fused_node.attr().at(kDataFormat).s() == "NCHW") {
    // Need to reshape the last 4 inputs
    NodeDef* new_shape = optimized_graph->add_node();
    new_shape->set_name(AddPrefixToNodeName("NCHWShape", fused_node.name()));
    new_shape->set_op("Const");
    new_shape->set_device(fused_node.device());
    *new_shape->add_input() = AsControlDependency(scale);
    (*new_shape->mutable_attr())["dtype"].set_type(DT_INT32);
    Tensor t(DT_INT32, {4});
    t.flat<int32>()(0) = 1;
    t.flat<int32>()(1) = -1;
    t.flat<int32>()(2) = 1;
    t.flat<int32>()(3) = 1;
    t.AsProtoTensorContent(
        (*new_shape->mutable_attr())["value"].mutable_tensor());

    NodeDef* reshaped_scale = optimized_graph->add_node();
    reshaped_scale->set_name(
        AddPrefixToNodeName("NCHWShapedScale", fused_node.name()));
    reshaped_scale->set_op("Reshape");
    reshaped_scale->set_device(fused_node.device());
    *reshaped_scale->add_input() = scale;
    *reshaped_scale->add_input() = new_shape->name();
    (*reshaped_scale->mutable_attr())["T"] = fused_node.attr().at("T");
    (*reshaped_scale->mutable_attr())["Tshape"].set_type(DT_INT32);
    scale = reshaped_scale->name();

    NodeDef* reshaped_offset = optimized_graph->add_node();
    reshaped_offset->set_name(
        AddPrefixToNodeName("NCHWShapedOffset", fused_node.name()));
    reshaped_offset->set_op("Reshape");
    reshaped_offset->set_device(fused_node.device());
    *reshaped_offset->add_input() = offset;
    *reshaped_offset->add_input() = new_shape->name();
    (*reshaped_offset->mutable_attr())["T"] = fused_node.attr().at("T");
    (*reshaped_offset->mutable_attr())["Tshape"].set_type(DT_INT32);
    offset = reshaped_offset->name();

    NodeDef* reshaped_mean = optimized_graph->add_node();
    reshaped_mean->set_name(
        AddPrefixToNodeName("NCHWShapedMean", fused_node.name()));
    reshaped_mean->set_op("Reshape");
    reshaped_mean->set_device(fused_node.device());
    *reshaped_mean->add_input() = mean;
    *reshaped_mean->add_input() = new_shape->name();
    (*reshaped_mean->mutable_attr())["T"] = fused_node.attr().at("T");
    (*reshaped_mean->mutable_attr())["Tshape"].set_type(DT_INT32);
    mean = reshaped_mean->name();

    NodeDef* reshaped_variance = optimized_graph->add_node();
    reshaped_variance->set_name(
        AddPrefixToNodeName("NCHWShapedVariance", fused_node.name()));
    reshaped_variance->set_op("Reshape");
    reshaped_variance->set_device(fused_node.device());
    *reshaped_variance->add_input() = variance;
    *reshaped_variance->add_input() = new_shape->name();
    (*reshaped_variance->mutable_attr())["T"] = fused_node.attr().at("T");
    (*reshaped_variance->mutable_attr())["Tshape"].set_type(DT_INT32);
    variance = reshaped_variance->name();
  }

  float epsilon = 0.0f;
  if (fused_node.attr().count("epsilon")) {
    epsilon = fused_node.attr().at("epsilon").f();
  }
  DataType dtype = fused_node.attr().at("T").type();
  Tensor value(dtype, TensorShape());
  value.scalar<float>()() = epsilon;
  NodeDef* variance_epsilon = optimized_graph->add_node();
  TF_CHECK_OK(ConstantFolding::CreateNodeDef(
      AddPrefixToNodeName("Const", fused_node.name()), &value,
      variance_epsilon));
  variance_epsilon->set_device(fused_node.device());

  NodeDef* variance_plus_epsilon = optimized_graph->add_node();
  variance_plus_epsilon->set_name(
      AddPrefixToNodeName("VarPlusEpsilon", fused_node.name()));
  variance_plus_epsilon->set_op("Add");
  (*variance_plus_epsilon->mutable_attr())["T"].set_type(dtype);
  variance_plus_epsilon->set_device(fused_node.device());
  *variance_plus_epsilon->add_input() = variance;
  *variance_plus_epsilon->add_input() = variance_epsilon->name();

  NodeDef* inv = optimized_graph->add_node();
  inv->set_name(AddPrefixToNodeName("Inv", fused_node.name()));
  inv->set_op("Rsqrt");
  inv->set_device(fused_node.device());
  (*inv->mutable_attr())["T"].set_type(dtype);
  *inv->add_input() = variance_plus_epsilon->name();

  NodeDef* scaled = optimized_graph->add_node();
  scaled->set_name(AddPrefixToNodeName("Scaled", fused_node.name()));
  scaled->set_op("Mul");
  scaled->set_device(fused_node.device());
  (*scaled->mutable_attr())["T"].set_type(dtype);
  *scaled->add_input() = inv->name();
  *scaled->add_input() = scale;

  NodeDef* a = optimized_graph->add_node();
  a->set_name(AddPrefixToNodeName("Mul", fused_node.name()));
  a->set_op("Mul");
  a->set_device(fused_node.device());
  (*a->mutable_attr())["T"].set_type(dtype);
  *a->add_input() = x;
  *a->add_input() = scaled->name();

  NodeDef* b = optimized_graph->add_node();
  b->set_name(AddPrefixToNodeName("Mul2", fused_node.name()));
  b->set_op("Mul");
  b->set_device(fused_node.device());
  (*b->mutable_attr())["T"].set_type(dtype);
  *b->add_input() = mean;
  *b->add_input() = scaled->name();

  NodeDef* c = optimized_graph->add_node();
  c->set_name(AddPrefixToNodeName("Offset", fused_node.name()));
  c->set_op("Sub");
  c->set_device(fused_node.device());
  (*c->mutable_attr())["T"].set_type(dtype);
  *c->add_input() = offset;
  *c->add_input() = b->name();

  NodeDef* r = optimized_graph->add_node();
  r->set_name(fused_node.name());
  r->set_op("Add");
  r->set_device(fused_node.device());
  (*r->mutable_attr())["T"].set_type(dtype);
  *r->add_input() = a->name();
  *r->add_input() = c->name();
}
}  // namespace

Status Remapper::Optimize(Cluster* /*cluster*/, const GrapplerItem& item,
                          GraphDef* optimized_graph) {
  // Supported graph patterns.
  // clang-format off
  FusedBatchNorm              fused_batch_norm;
  Conv2DWithBiasAdd           conv2d_with_bias;
  Conv2DWithBiasAddAndRelu    conv2d_with_bias_and_relu;
  Conv2DWithBatchNorm         conv2d_with_batch_norm;
  Conv2DWithBatchNormAndRelu  conv2d_with_batch_norm_and_relu;
  Conv2DWithSqueezeAndBiasAdd conv2d_with_squeeze_and_bias;
  // clang-format on

  // Processing graph in reverse-topological sorted order allows to remap
  // longer chains of dependent ops in one pass.
  GraphDef topo_sorted_graph = item.graph;
  TF_RETURN_IF_ERROR(TopologicalSort(&topo_sorted_graph));
  std::reverse(topo_sorted_graph.mutable_node()->begin(),
               topo_sorted_graph.mutable_node()->end());

  GrapplerItem topo_sorted_item = item.WithGraph(std::move(topo_sorted_graph));
  RemapperContext ctx(topo_sorted_item);

  // Skip nodes that were invalidated by a remapper, e.g. do not process BiasAdd
  // and Relu nodes that were fused into a Conv2D node.
  absl::flat_hash_set<const NodeDef*> invalidated_nodes;

  optimized_graph->mutable_node()->Reserve(topo_sorted_item.graph.node_size());
  for (const NodeDef& node : topo_sorted_item.graph.node()) {
    // Check if node was invalidated by one of the previous remaps.
    if (invalidated_nodes.count(&node) > 0) continue;

    // Remap Conv2D+BiasAdd into the _FusedConv2D.
    if (FindConv2DWithBias(ctx, &node, &conv2d_with_bias)) {
      AddFusedConv2DNode(ctx, conv2d_with_bias, optimized_graph,
                         &invalidated_nodes);
      continue;
    }

    // Remap Conv2D+BiasAdd+Relu into the _FusedConv2D.
    if (FindConv2DWithBiasAndRelu(ctx, &node, &conv2d_with_bias_and_relu)) {
      AddFusedConv2DNode(ctx, conv2d_with_bias_and_relu, optimized_graph,
                         &invalidated_nodes);
      continue;
    }

// TODO(penporn):
// Remove this once TF-MKL supports _FusedConv2D with these operations.
#ifndef INTEL_MKL
    // Remap Conv2D+Squeeze+BiasAdd into the _FusedConv2D+Squeeze.
    if (FindConv2DWithSqueezeAndBias(ctx, &node,
                                     &conv2d_with_squeeze_and_bias)) {
      AddFusedConv2DNode(ctx, conv2d_with_squeeze_and_bias, optimized_graph,
                         &invalidated_nodes);
      continue;
    }

    // Remap Conv2D+FusedBatchNorm into the _FusedConv2D;
    if (FindConv2DWithBatchNorm(ctx, &node, &conv2d_with_batch_norm)) {
      AddFusedConv2DNode(conv2d_with_batch_norm, optimized_graph,
                         &invalidated_nodes);
      continue;
    }

    // Remap Conv2D+FusedBatchNorm+Relu into the _FusedConv2D;
    if (FindConv2DWithBatchNormAndRelu(ctx, &node,
                                       &conv2d_with_batch_norm_and_relu)) {
      AddFusedConv2DNode(conv2d_with_batch_norm_and_relu, optimized_graph,
                         &invalidated_nodes);
      continue;
    }
#endif  // !INTEL_MKL

    // Infer properties lazily in case they are not needed.
    if (!ctx.inferred_graph_properties && IsFusedBatchNormCandidate(node)) {
      TF_RETURN_IF_ERROR(ctx.graph_properties.InferStatically(false));
      ctx.inferred_graph_properties = true;
    }

    // During inference, most of the inputs to FusedBatchNorm are constant, and
    // we can therefore replace the op with a much cheaper set of primitives.
    if (FindFusedBatchNorm(ctx, &node, &fused_batch_norm)) {
      AddBatchNormNodes(fused_batch_norm, optimized_graph);
      continue;
    }

    // If we didn't match a node to any pattern copy it to the optimized graph.
    *optimized_graph->add_node() = node;
  }

  *optimized_graph->mutable_library() = topo_sorted_item.graph.library();
  *optimized_graph->mutable_versions() = topo_sorted_item.graph.versions();

  return Status::OK();
}

void Remapper::Feedback(Cluster* /*cluster*/, const GrapplerItem& /*item*/,
                        const GraphDef& /*optimized_graph*/,
                        double /*result*/) {
  // Nothing to do for RemapperOptimizer.
}

}  // namespace grappler
}  // namespace tensorflow
