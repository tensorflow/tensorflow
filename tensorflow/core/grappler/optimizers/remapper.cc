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
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/grappler/utils/symbolic_shapes.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {
namespace grappler {

// Supported patterns:
//
// Conv2D + ... -> _FusedConv2D
//   (1) Conv2D + BiasAdd + <Activation>
//   (2) Conv2D + FusedBatchNorm + <Activation>
//   (3) Conv2D + Squeeze + BiasAdd
//
// MatMul + ... -> _FusedMatMul:
//   (1) MatMul + BiasAdd + <Activation>
//
// FusedBatchNorm[$is_training] + ... -> _FusedBatchNormEx[$is_training]
//   (1) FusedBatchNorm + <Activation>
//   (2) FusedBatchNorm + SideInput + <Activation>
//
// Both Conv2D and MatMul implemented as Tensor contraction (on CPU), so all the
// patterns are "ContractionWith...".
namespace {

constexpr char kFusedConv2D[] = "_FusedConv2D";
constexpr char kFusedMatMul[] = "_FusedMatMul";
constexpr char kFusedDepthwiseConv2dNative[] = "_FusedDepthwiseConv2dNative";
constexpr char kFusedBatchNormEx[] = "_FusedBatchNormEx";

constexpr char kDataFormat[] = "data_format";
constexpr char kIsTraining[] = "is_training";

constexpr int kMissingIndex = -1;

struct RemapperContext {
  explicit RemapperContext(GrapplerItem* item, Status* status)
      : nodes_to_preserve(item->NodesToPreserve()),
        graph_view(&item->graph, status),
        graph_properties(*item),
        inferred_graph_properties(false) {}

  std::unordered_set<string> nodes_to_preserve;
  utils::MutableGraphView graph_view;
  GraphProperties graph_properties;
  bool inferred_graph_properties;
};

// FusedBatchNorm that can be replaced with a cheaper set of primitives.
struct FusedBatchNorm {
  FusedBatchNorm() = default;
  explicit FusedBatchNorm(int fused_batch_norm)
      : fused_batch_norm(fused_batch_norm) {}

  int fused_batch_norm = kMissingIndex;
};

// FusedBatchNorm[$is_training] with fused side input and/or activation.
struct FusedBatchNormEx {
  FusedBatchNormEx() = default;

  int fused_batch_norm = kMissingIndex;
  int side_input = kMissingIndex;
  int activation = kMissingIndex;
  // Add node that will be invalidated by fusing side input and fused batch norm
  int invalidated = kMissingIndex;
};

// Contraction node followed by a BiasAdd.
struct ContractionWithBiasAdd {
  ContractionWithBiasAdd() = default;
  ContractionWithBiasAdd(int contraction, int bias_add)
      : contraction(contraction), bias_add(bias_add) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
};

// Contraction node followed by a BiasAdd and Activation.
struct ContractionWithBiasAddAndActivation {
  ContractionWithBiasAddAndActivation() = default;
  ContractionWithBiasAddAndActivation(int contraction, int bias_add,
                                      int activation)
      : contraction(contraction), bias_add(bias_add), activation(activation) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int activation = kMissingIndex;
};

// Contraction node followed by a Squeeze and BiasAdd.
struct ContractionWithSqueezeAndBiasAdd {
  ContractionWithSqueezeAndBiasAdd() = default;
  ContractionWithSqueezeAndBiasAdd(int contraction, int squeeze, int bias_add)
      : contraction(contraction), squeeze(squeeze), bias_add(bias_add) {}

  int contraction = kMissingIndex;
  int squeeze = kMissingIndex;
  int bias_add = kMissingIndex;
};

// Contraction node followed by a FusedBatchNorm.
struct ContractionWithBatchNorm {
  ContractionWithBatchNorm() = default;
  ContractionWithBatchNorm(int contraction, int fused_batch_norm,
                           float epsilon = 0.0)
      : contraction(contraction),
        fused_batch_norm(fused_batch_norm),
        epsilon(epsilon) {}

  int contraction = kMissingIndex;
  int fused_batch_norm = kMissingIndex;
  float epsilon = 0.0;
};

// Contraction node followed by a FusedBatchNorm and Activation.
struct ContractionWithBatchNormAndActivation {
  ContractionWithBatchNormAndActivation() = default;
  ContractionWithBatchNormAndActivation(int contraction, int fused_batch_norm,
                                        int activation, float epsilon = 0.0)
      : contraction(contraction),
        fused_batch_norm(fused_batch_norm),
        activation(activation),
        epsilon(epsilon) {}

  int contraction = kMissingIndex;
  int fused_batch_norm = kMissingIndex;
  int activation = kMissingIndex;
  float epsilon = 0.0;
};

#ifdef INTEL_MKL
// Contraction node followed by a BiasAdd and Add.
struct ContractionWithBiasAddAndAdd {
  ContractionWithBiasAddAndAdd() = default;
  ContractionWithBiasAddAndAdd(int contraction, int bias_add, int add,
                               int port_id)
      : contraction(contraction),
        bias_add(bias_add),
        add(add),
        port_id(port_id) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int add = kMissingIndex;
  int port_id = 0;
};

// Contraction node followed by a BiasAdd, Add and Relu.
struct ContractionWithBiasAndAddActivation {
  ContractionWithBiasAndAddActivation() = default;
  ContractionWithBiasAndAddActivation(int contraction, int bias_add, int add,
                                      int port_id, int activation)
      : contraction(contraction),
        bias_add(bias_add),
        add(add),
        port_id(port_id),
        activation(activation) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int add = kMissingIndex;
  int port_id = 0;
  int activation = kMissingIndex;
};
#endif  // INTEL_MKL

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

bool IsCpuCompatibleDataType(const NodeDef* contraction,
                             const string& type_attr = "T") {
  DataType dtype = GetDataTypeFromAttr(*contraction, type_attr);
#if defined(INTEL_MKL)
#if defined(ENABLE_INTEL_MKL_BFLOAT16)
  if (IsConv2D(*contraction) || IsDepthwiseConv2dNative(*contraction) ||
      IsMatMul(*contraction)) {
    return dtype == DT_FLOAT || dtype == DT_BFLOAT16;
#else
  if (IsConv2D(*contraction) || IsDepthwiseConv2dNative(*contraction) ||
      IsMatMul(*contraction)) {
    return dtype == DT_FLOAT;
#endif  // ENABLE_INTEL_MKL_BFLOAT16
#else
  if (IsConv2D(*contraction)) {
    return dtype == DT_FLOAT || dtype == DT_DOUBLE;
  } else if (IsMatMul(*contraction)) {
    return dtype == DT_FLOAT;
#endif  // INTEL_MKL
  } else {
    return false;
  }
}

bool IsGpuCompatibleDataType(const NodeDef* contraction,
                             const string& type_attr = "T") {
  DataType dtype = GetDataTypeFromAttr(*contraction, type_attr);
  if (IsConv2D(*contraction)) {
    return dtype == DT_FLOAT;
  } else {
    return false;
  }
}

bool IsCpuCompatibleDataFormat(const NodeDef* conv2d) {
  DCHECK(IsConv2D(*conv2d)) << "Expected Conv2D op";
  const string& data_format = conv2d->attr().at(kDataFormat).s();
#ifndef INTEL_MKL
  return data_format == "NHWC";
#else
  return data_format == "NHWC" || data_format == "NCHW";
#endif  // !INTEL_MKL
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

bool IsCpuCompatibleMatMul(const NodeDef* matmul) {
  DCHECK(IsMatMul(*matmul)) << "Expected MatMul op";
  return NodeIsOnCpu(matmul) && IsCpuCompatibleDataType(matmul);
}

bool IsCpuCompatibleDepthwiseConv2dNative(const NodeDef* dw_conv2d) {
  DCHECK(IsDepthwiseConv2dNative(*dw_conv2d))
      << "Expected DepthwiseConv2dNative op";
  return NodeIsOnCpu(dw_conv2d) && IsCpuCompatibleDataType(dw_conv2d);
}

// Checks if we can rewrite a pattern to the `_Fused{Conv2D,MatMul}` on CPU.
template <typename Pattern>
bool IsCpuCompatible(const RemapperContext& ctx, const Pattern& matched) {
  const NodeDef& node = ctx.graph_view.graph()->node(matched.contraction);
  if (IsConv2D(node)) {
    return IsCpuCompatibleConv2D(&node);
  } else if (IsDepthwiseConv2dNative(node)) {
#ifdef INTEL_MKL
    return IsCpuCompatibleDepthwiseConv2dNative(&node);
#else
    return false;
#endif  // INTEL_MKL
  } else if (IsMatMul(node)) {
    return IsCpuCompatibleMatMul(&node);
  } else {
    return false;
  }
}

// Checks if we can rewrite a pattern to the `_FusedConv2D` on GPU device.
bool IsGpuCompatible(const RemapperContext& ctx,
                     const ContractionWithBiasAddAndActivation& matched) {
#if TENSORFLOW_USE_ROCM
  // ROCm does not support _FusedConv2D
  return false;
#endif
  const GraphDef* graph = ctx.graph_view.graph();
  const NodeDef& contraction_node = graph->node(matched.contraction);
  if (!IsConv2D(contraction_node)) return false;

  const std::vector<OpInfo::TensorProperties>& input_props =
      ctx.graph_properties.GetInputProperties(contraction_node.name());
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

  // We rely on cuDNN for fused convolution, and it currently supports only Relu
  // activation.
  const NodeDef& activation_node = graph->node(matched.activation);
  bool is_relu = IsRelu(activation_node);

  return is_relu && is_spatial_conv && IsGpuCompatibleConv2D(&contraction_node);
}
bool IsGpuCompatible(const RemapperContext& ctx,
                     const ContractionWithBiasAdd& matched) {
  return false;
}
bool IsGpuCompatible(const RemapperContext& ctx,
                     const ContractionWithSqueezeAndBiasAdd& matched) {
  return false;
}

// Returns true if the given pattern is supported on the assigned device.
template <typename Pattern>
bool IsDeviceCompatible(const RemapperContext& ctx, Pattern& matched) {
  return IsCpuCompatible(ctx, matched) || IsGpuCompatible(ctx, matched);
}

bool IsSupportedActivation(const NodeDef& node) {
  return IsRelu(node) || IsRelu6(node) || IsElu(node);
}

inline bool HasControlFaninOrFanout(const utils::MutableNodeView& node_view) {
  return node_view.NumControllingFanins() > 0 ||
         node_view.NumControlledFanouts() > 0;
}

// Returns true if at most one fanout reads output at port 0 (output used once).
inline bool HasAtMostOneFanoutAtPort0(const utils::MutableNodeView& node_view) {
  return node_view.GetRegularFanout(0).size() <= 1;
}

// Returns true if at most one fanout reads actual tensor data at output port 0
// (output used once for any data computation).
inline bool HasAtMostOneDataFanoutAtPort0(
    const utils::MutableNodeView& node_view) {
  const auto predicate = [](const auto& fanout) -> bool {
    const NodeDef* node = fanout.node_view()->node();
    return !IsShape(*node) && !IsRank(*node);
  };
  return absl::c_count_if(node_view.GetRegularFanout(0), predicate) <= 1;
}

bool FindContractionWithBias(const RemapperContext& ctx, int node_index,
                             ContractionWithBiasAdd* matched,
                             bool check_device_compatible = true) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be a BiasAdd.
  // TODO(lyandy): Forward controls for patterns with control dependencies.
  if (HasControlFaninOrFanout(*node_view)) return false;

  const auto* node_def = node_view->node();
  if (!IsBiasAdd(*node_def)) return false;

  // Input to the BiasAdd must be a Conv2D or a MatMul.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* contraction_node_view = regular_fanin_0.node_view();
  const auto* contraction_node_def = contraction_node_view->node();

  // Conv2D, MatMul or DepthwiseConv2D
  bool is_contraction = IsConv2D(*contraction_node_def) ||
                        IsMatMul(*contraction_node_def) ||
                        IsDepthwiseConv2dNative(*contraction_node_def);

  if (!is_contraction || !HaveSameDataType(node_def, contraction_node_def) ||
      HasControlFaninOrFanout(*contraction_node_view) ||
      !HasAtMostOneFanoutAtPort0(*contraction_node_view) ||
      IsInPreserveSet(ctx, contraction_node_def))
    return false;

  // Check that data type and data format are supported on assigned device.
  const ContractionWithBiasAdd pattern{contraction_node_view->node_index(),
                                       node_index};
  if (check_device_compatible && !IsDeviceCompatible(ctx, pattern))
    return false;

  // We successfully found a {Conv2D, MatMul}+BiasAdd pattern.
  *matched = pattern;

  return true;
}

bool FindContractionWithBiasAndActivation(
    const RemapperContext& ctx, int node_index,
    ContractionWithBiasAddAndActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be an activation node.
  // TODO(lyandy): Forward controls for patterns with control dependencies.
  if (HasControlFaninOrFanout(*node_view)) return false;

  const auto* node_def = node_view->node();
  if (!IsSupportedActivation(*node_def)) return false;

  // And input to the activation node must match ContractionWithBiasAdd pattern.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* bias_add_node_view = regular_fanin_0.node_view();
  const auto* bias_add_node_def = bias_add_node_view->node();

  ContractionWithBiasAdd base;
  if (!FindContractionWithBias(ctx, bias_add_node_view->node_index(), &base,
                               /*check_device_compatible=*/false) ||
      !HasAtMostOneFanoutAtPort0(*bias_add_node_view) ||
      !HaveSameDataType(node_def, bias_add_node_def) ||
      IsInPreserveSet(ctx, bias_add_node_def))
    return false;

  // Check that data type and data format are supported on assigned device.
  const ContractionWithBiasAddAndActivation pattern{base.contraction,
                                                    base.bias_add, node_index};
  if (!IsDeviceCompatible(ctx, pattern)) return false;

  // We successfully found a {Conv2D, MatMul}+BiasAdd+Activation pattern.
  *matched = pattern;

  return true;
}

bool FindConv2DWithSqueezeAndBias(const RemapperContext& ctx, int node_index,
                                  ContractionWithSqueezeAndBiasAdd* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // TODO(lyandy): Forward controls for patterns with control dependencies.
  if (HasControlFaninOrFanout(*node_view)) return false;

  // Root of the pattern must be a BiasAdd.
  const auto* node_def = node_view->node();
  if (!IsBiasAdd(*node_def)) return false;

  // Input to the BiasAdd must be a Squeeze.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* squeeze_node_view = regular_fanin_0.node_view();
  const auto* squeeze_node_def = squeeze_node_view->node();

  if (!IsSqueeze(*squeeze_node_def) ||
      !HaveSameDataType(node_def, squeeze_node_def, "T") ||
      HasControlFaninOrFanout(*squeeze_node_view) ||
      !HasAtMostOneFanoutAtPort0(*squeeze_node_view) ||
      IsInPreserveSet(ctx, squeeze_node_def))
    return false;

  // Squeeze must not squeeze output channel dimension.
  std::vector<int32> dims;
  if (!TryGetNodeAttr(*squeeze_node_def, "squeeze_dims", &dims)) return false;
  for (auto dim : dims) {
    if (dim == 3) return false;
  }

  // Input to the Squeeze must be a Conv2D.
  if (squeeze_node_view->NumRegularFanins() < 1) return false;
  const auto& squeeze_regular_fanin_0 = squeeze_node_view->GetRegularFanin(0);
  const auto* conv2d_node_view = squeeze_regular_fanin_0.node_view();
  const auto* conv2d_node_def = conv2d_node_view->node();

  if (!IsConv2D(*conv2d_node_def) ||
      !HaveSameDataType(node_def, conv2d_node_def, "T") ||
      HasControlFaninOrFanout(*conv2d_node_view) ||
      !HasAtMostOneFanoutAtPort0(*conv2d_node_view) ||
      IsInPreserveSet(ctx, conv2d_node_def))
    return false;

  // Check that data type and data format are supported on assigned device.
  const ContractionWithSqueezeAndBiasAdd pattern{
      conv2d_node_view->node_index(), squeeze_node_view->node_index(),
      node_index};
  if (!IsDeviceCompatible(ctx, pattern)) return false;

  // We successfully found a Conv2D+Squeeze+BiasAdd pattern.
  *matched = pattern;

  return true;
}

bool FindConv2DWithBatchNorm(const RemapperContext& ctx, int node_index,
                             ContractionWithBatchNorm* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  // Root of the pattern must be a FusedBatchNorm.
  if (!IsFusedBatchNorm(*node_def)) return false;

  // FusedBatchNormV2 and V3 have an extra type parameter.
  if (node_view->GetOp() != "FusedBatchNorm" &&
      !HasDataType(node_def, DT_FLOAT, "U"))
    return false;

  // Check that batch normalization is in inference mode.
  const auto* training_attr = node_view->GetAttr(kIsTraining);
  if (training_attr != nullptr && training_attr->b()) return false;

  // Check that only 0th output is consumed by other nodes.
  // TODO(lyandy): Forward controls for patterns with control dependencies.
  if (HasControlFaninOrFanout(*node_view) ||
      !node_view->GetRegularFanout(1).empty() ||  // batch_mean
      !node_view->GetRegularFanout(2).empty() ||  // batch_variance
      !node_view->GetRegularFanout(3).empty() ||  // reserve_space_1
      !node_view->GetRegularFanout(4).empty())    // reserve_space_2
    return false;

  // Input to the FusedBatchNorm must be a Conv2D.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* conv2d_node_view = regular_fanin_0.node_view();
  const auto* conv2d_node_def = conv2d_node_view->node();

  if (!IsConv2D(*conv2d_node_def) || !NodeIsOnCpu(conv2d_node_def) ||
      !HaveSameDataType(node_def, conv2d_node_def) ||
      !IsCpuCompatibleDataType(conv2d_node_def) ||
      !IsCpuCompatibleDataFormat(conv2d_node_def) ||
      HasControlFaninOrFanout(*conv2d_node_view) ||
      !HasAtMostOneFanoutAtPort0(*conv2d_node_view) ||
      IsInPreserveSet(ctx, conv2d_node_def))
    return false;

  // We successfully found a Conv2D+FusedBatchNorm pattern.
  matched->contraction = conv2d_node_view->node_index();
  matched->fused_batch_norm = node_index;
  if (!TryGetNodeAttr(*node_def, "epsilon", &matched->epsilon)) return false;

  return true;
}

bool FindConv2DWithBatchNormAndActivation(
    const RemapperContext& ctx, int node_index,
    ContractionWithBatchNormAndActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // TODO(lyandy): Forward controls for patterns with control dependencies.
  if (HasControlFaninOrFanout(*node_view)) return false;

  // Root of the pattern must be an activation node.
  const auto* node_def = node_view->node();
  if (!IsSupportedActivation(*node_def)) return false;

  // And input to the activation node must match Conv2DWithBatchNorm pattern.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* batch_norm_node_view = regular_fanin_0.node_view();

  ContractionWithBatchNorm base;
  if (!FindConv2DWithBatchNorm(ctx, batch_norm_node_view->node_index(), &base))
    return false;

  const auto* fused_batch_norm_node_view =
      ctx.graph_view.GetNode(base.fused_batch_norm);
  const auto* fused_batch_norm_node_def = fused_batch_norm_node_view->node();
  if (!HasAtMostOneFanoutAtPort0(*fused_batch_norm_node_view) ||
      !HaveSameDataType(node_def, fused_batch_norm_node_def) ||
      IsInPreserveSet(ctx, fused_batch_norm_node_def))
    return false;

  // We successfully found a Conv2D+FusedBatchNorm+Activation pattern.
  matched->contraction = base.contraction;
  matched->fused_batch_norm = base.fused_batch_norm;
  matched->activation = node_index;
  matched->epsilon = base.epsilon;

  return true;
}

#ifdef INTEL_MKL
// As AddN has multiple inputs, this function tries to find Conv2D + Bias
// pattern in specific input port.
bool FindContractionWithBiasInPort(const RemapperContext& ctx,
                                   const utils::MutableNodeView& add_node_view,
                                   const NodeDef& add_node_def, int port_id,
                                   ContractionWithBiasAdd* base) {
  // Input to AddN must match ContractionWithBiasAdd pattern.
  if (add_node_view.NumRegularFanins() < port_id + 1) return false;
  const auto& bias_add_node_view =
      add_node_view.GetRegularFanin(port_id).node_view();
  if (bias_add_node_view == nullptr) return false;
  const auto* bias_add_node_def = bias_add_node_view->node();

  if (!FindContractionWithBias(ctx, bias_add_node_view->node_index(), base,
                               /*check_device_compatible=*/false))
    return false;
  if (!HasAtMostOneFanoutAtPort0(*bias_add_node_view) ||
      !HaveSameDataType(&add_node_def, bias_add_node_def) ||
      IsInPreserveSet(ctx, bias_add_node_def))
    return false;
  return true;
}

bool IsAddWithNoBroadcast(const RemapperContext& ctx, const NodeDef& node) {
  if (!IsAdd(node)) return false;

  // Check if this is case of broadcasting - Add node supports broadcasting.
  const auto& props = ctx.graph_properties.GetInputProperties(node.name());
  if (props.size() == 2 &&
      ShapesSymbolicallyEqual(props[0].shape(), props[1].shape())) {
    return true;
  }
  return false;
}

bool FindContractionWithBiasAddAndAdd(const RemapperContext& ctx,
                                      const utils::MutableNodeView& node_view,
                                      ContractionWithBiasAddAndAdd* matched) {
  // Fusion with AddN is supported only when it has two inputs.
  // TODO(lyandy): Forward controls for patterns with control dependencies.
  if (HasControlFaninOrFanout(node_view) || node_view.NumRegularFanins() != 2)
    return false;

  // Root of the pattern must be a AddN or Add with same input shapes
  // (no broadcasting).
  const auto* node_def = node_view.node();
  if (!IsAddN(*node_def) && !IsAddWithNoBroadcast(ctx, *node_def)) return false;

#ifdef ENABLE_INTEL_MKL_BFLOAT16
  // MKL AddN ops only support float and bfloat16 data types.
  if (!HasDataType(node_def, DT_FLOAT) && !HasDataType(node_def, DT_BFLOAT16))
    return false;
#else
  // MKL AddN ops only support float data type.
  if (!HasDataType(node_def, DT_FLOAT)) return false;
#endif  // ENABLE_INTEL_MKL_BFLOAT16

  ContractionWithBiasAdd base;
  matched->port_id = 0;

  // Find the conv+bias pattern in specific port.
  if (!FindContractionWithBiasInPort(ctx, node_view, *node_def,
                                     matched->port_id, &base)) {
    matched->port_id = 1;
    if (!FindContractionWithBiasInPort(ctx, node_view, *node_def,
                                       matched->port_id, &base)) {
      return false;
    }
  }

  // We successfully found a Conv2D+BiasAdd+{AddN,Add} pattern.
  matched->contraction = base.contraction;
  matched->bias_add = base.bias_add;
  matched->add = node_view.node_index();

  return true;
}

bool FindContractionWithBiasAddAndAdd(const RemapperContext& ctx,
                                      int node_index,
                                      ContractionWithBiasAddAndAdd* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  return FindContractionWithBiasAddAndAdd(ctx, *node_view, matched);
}

bool FindContractionWithBiasAndAddActivation(
    const RemapperContext& ctx, int node_index,
    ContractionWithBiasAndAddActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // TODO(lyandy): Forward controls for patterns with control dependencies.
  if (HasControlFaninOrFanout(*node_view)) return false;

  // Root of the pattern must be an activation node.
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;
  if (!IsSupportedActivation(*node_def)) return false;

#ifdef ENABLE_INTEL_MKL_BFLOAT16
  // MKL activation op only supports float and bfloat16 data types.
  if (!HasDataType(node_def, DT_FLOAT) && !HasDataType(node_def, DT_BFLOAT16))
    return false;
#else
  // MKL activation op only supports float data type.
  if (!HasDataType(node_def, DT_FLOAT)) return false;
#endif  // ENABLE_INTEL_MKL_BFLOAT16

  // And input to activation must match ContractionWithBiasAddAndAdd pattern.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* add_node_view = regular_fanin_0.node_view();

  ContractionWithBiasAddAndAdd base;

  if (!FindContractionWithBiasAddAndAdd(ctx, *add_node_view, &base)) {
    return false;
  }

  // We successfully found a Conv2D+BiasAdd+AddN+activation pattern.
  const ContractionWithBiasAndAddActivation pattern{
      base.contraction, base.bias_add, base.add, base.port_id, node_index};
  *matched = pattern;

  return true;
}
#endif

bool FindFusedBatchNorm(const RemapperContext& ctx, int node_index,
                        FusedBatchNorm* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (!IsFusedBatchNorm(*node_def)) return false;
  if (GetDataTypeFromAttr(*node_def, "T") != DT_FLOAT) return false;

  // Check that the node is in inference mode.
  bool is_training = true;
  if (!TryGetNodeAttr(*node_def, kIsTraining, &is_training)) return false;
  if (is_training) return false;

  const auto& props = ctx.graph_properties.GetInputProperties(node_def->name());

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
  if (node_view->GetRegularFanouts().size() > 1) {
    return false;
  }

  // We found a fused batch norm node that can be replaced with primitive ops.
  matched->fused_batch_norm = node_index;

  return true;
}

// NOTE(ezhulenev): See `BatchnormSpatialPersistentEnabled` documentation in the
// `tensorflow/stream_executor/cuda/cuda_dnn.cc` for details.
bool BatchnormSpatialPersistentEnabled() {
#if CUDNN_VERSION >= 7402
  static bool is_enabled = [] {
    bool is_enabled = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar(
        "TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT",
        /*default_val=*/false, &is_enabled));
    return is_enabled;
  }();
  return is_enabled;
#else
  return false;
#endif
}

bool FindFusedBatchNormEx(const RemapperContext& ctx, int node_index,
                          FusedBatchNormEx* matched) {
  // Root of the pattern must be a Relu.
  // TODO(ezhulenev): Forward control dependencies.
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  // TODO(lyandy): Forward controls for patterns with control dependencies.
  if (!IsRelu(*node_def) || HasControlFaninOrFanout(*node_view)) return false;

  // Returns true iff the node is a compatible FusedBatchNorm node.
  const auto valid_batch_norm =
      [&](const utils::MutableNodeView& fused_batch_norm) -> bool {
    const auto* fused_batch_norm_node_def = fused_batch_norm.node();
    if (!IsFusedBatchNorm(*fused_batch_norm_node_def)) return false;

    // We fuse FusedBatchNorm only on GPU, because on CPU we fuse it with
    // contraction (MatMul or Conv2D node).
    if (!NodeIsOnGpu(fused_batch_norm_node_def)) return false;

    DataType t_dtype = GetDataTypeFromAttr(*fused_batch_norm_node_def, "T");
    if (t_dtype != DT_FLOAT && t_dtype != DT_HALF) return false;

    // Get the FusedBatchNorm training mode.
    bool is_training;
    if (!GetNodeAttr(*fused_batch_norm_node_def, kIsTraining, &is_training)
             .ok())
      return false;

    // In training mode we rely on cuDNN for computing FusedBatchNorm with side
    // inputs and activation, and it has its own limitations. In inference mode
    // we have a custom CUDA kernel that doesn't not have these constraints.
    if (is_training) {
      // cuDNN only supports NHWC data layout.
      string data_format;
      if (!GetNodeAttr(*fused_batch_norm_node_def, kDataFormat, &data_format)
               .ok())
        return false;
      if (data_format != "NHWC") return false;

      // Data type must be DT_HALF.
      if (t_dtype != DT_HALF) return false;

      // Channel dimension must be a multiple of 4.
      const auto& props = ctx.graph_properties.GetInputProperties(
          fused_batch_norm_node_def->name());

      const bool valid_channel_dim = !props.empty() &&
                                     props[0].shape().dim_size() == 4 &&
                                     props[0].shape().dim(3).size() % 4 == 0;
      if (!valid_channel_dim) return false;

      // cuDNN must support CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode.
      if (!BatchnormSpatialPersistentEnabled()) return false;
    }

    // FusedBatchNormV2 and V3 have an extra type parameter.
    if ((fused_batch_norm_node_def->op() != "FusedBatchNorm") &&
        !HasDataType(fused_batch_norm_node_def, DT_FLOAT, "U"))
      return false;

    // Check that only one node consumes the 0-th output of a FusedBatchNorm.
    if (HasControlFaninOrFanout(fused_batch_norm) ||
        !HasAtMostOneDataFanoutAtPort0(fused_batch_norm) ||
        IsInPreserveSet(ctx, fused_batch_norm_node_def))
      return false;

    return true;
  };

  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* relu_fanin_0_node_view = regular_fanin_0.node_view();
  const auto* relu_fanin_0_node_def = relu_fanin_0_node_view->node();

  // Input to a Relu can be a FusedBatchNorm.
  if (valid_batch_norm(*relu_fanin_0_node_view)) {
    matched->activation = node_index;
    matched->fused_batch_norm = regular_fanin_0.node_index();
    return true;
  }

  // Input to a Relu can be an Add node with FusedBatchNorm as one of the inputs
  if (IsAdd(*relu_fanin_0_node_def)) {
    // Check that only Relu node consumes the output of an Add node.
    if (HasControlFaninOrFanout(*relu_fanin_0_node_view) ||
        !HasAtMostOneFanoutAtPort0(*relu_fanin_0_node_view) ||
        IsInPreserveSet(ctx, relu_fanin_0_node_def))
      return false;

    // Add node supports broadcasting, FusedBatchNormEx does not.
    const auto& props =
        ctx.graph_properties.GetInputProperties(relu_fanin_0_node_def->name());
    if (props.size() < 2 ||
        !ShapesSymbolicallyEqual(props[0].shape(), props[1].shape()))
      return false;

    if (relu_fanin_0_node_view->NumRegularFanins() < 2) return false;
    const auto& add_regular_fanin_0 =
        relu_fanin_0_node_view->GetRegularFanin(0);
    const auto& add_regular_fanin_1 =
        relu_fanin_0_node_view->GetRegularFanin(1);

    if (valid_batch_norm(*add_regular_fanin_0.node_view())) {
      matched->activation = node_index;
      matched->side_input = add_regular_fanin_1.node_index();
      matched->fused_batch_norm = add_regular_fanin_0.node_index();
      matched->invalidated = regular_fanin_0.node_index();
      return true;
    }

    if (valid_batch_norm(*add_regular_fanin_1.node_view())) {
      matched->activation = node_index;
      matched->side_input = add_regular_fanin_0.node_index();
      matched->fused_batch_norm = add_regular_fanin_1.node_index();
      matched->invalidated = regular_fanin_0.node_index();
      return true;
    }
  }

  return false;
}

void CopyConv2DAttributes(const NodeDef& conv2d, NodeDef* fused_conv2d) {
  DCHECK(IsConv2D(conv2d)) << "Input node must be a Conv2D";

  auto* attr = fused_conv2d->mutable_attr();
  auto& src_attr = conv2d.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["strides"] = src_attr.at("strides");
  (*attr)["padding"] = src_attr.at("padding");
  (*attr)["explicit_paddings"] = src_attr.at("explicit_paddings");
  (*attr)["dilations"] = src_attr.at("dilations");
  (*attr)["data_format"] = src_attr.at("data_format");
  (*attr)["use_cudnn_on_gpu"] = src_attr.at("use_cudnn_on_gpu");
}

void CopyDepthwiseConv2dNativeAttributes(const NodeDef& dw_conv2d,
                                         NodeDef* fused_dw_conv2d) {
  DCHECK(IsDepthwiseConv2dNative(dw_conv2d))
      << "Input node must be a DepthwiseConv2dNative";

  auto* attr = fused_dw_conv2d->mutable_attr();
  auto& src_attr = dw_conv2d.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["strides"] = src_attr.at("strides");
  (*attr)["padding"] = src_attr.at("padding");
  (*attr)["dilations"] = src_attr.at("dilations");
  (*attr)["data_format"] = src_attr.at("data_format");
}

void CopyFusedBatchNormAttributes(const NodeDef& fused_batch_norm,
                                  NodeDef* fused_batch_norm_ex) {
  DCHECK(IsFusedBatchNorm(fused_batch_norm))
      << "Input node must be a FusedBatchNorm";

  auto* attr = fused_batch_norm_ex->mutable_attr();
  auto src_attr = fused_batch_norm.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["is_training"] = src_attr.at("is_training");
  (*attr)["data_format"] = src_attr.at("data_format");
  (*attr)["epsilon"] = src_attr.at("epsilon");

  // FusedBatchNormV2 and V3 have an extra type parameter.
  if (fused_batch_norm.op() != "FusedBatchNorm") {
    (*attr)["U"] = src_attr.at("U");
  } else {
    (*attr)["U"] = src_attr.at("T");
  }
}

void CopyMatMulAttributes(const NodeDef& matmul, NodeDef* fused_matmul) {
  DCHECK(IsMatMul(matmul)) << "Input node must be a MatMul";

  auto* attr = fused_matmul->mutable_attr();
  auto& src_attr = matmul.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["transpose_a"] = src_attr.at("transpose_a");
  (*attr)["transpose_b"] = src_attr.at("transpose_b");
}

void SetFusedOpAttributes(NodeDef* fused,
                          const absl::Span<const absl::string_view> fused_ops,
                          int num_args = 1, float epsilon = 0.0) {
  auto* attr = fused->mutable_attr();
  SetAttrValue(fused_ops, &(*attr)["fused_ops"]);
  SetAttrValue(num_args, &(*attr)["num_args"]);
  SetAttrValue(epsilon, &(*attr)["epsilon"]);  // required only for BatchNorm
}

Status AddFusedContractionNode(RemapperContext* ctx,
                               const ContractionWithBiasAdd& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  DCHECK(IsDeviceCompatible(*ctx, matched)) << "Unsupported fusion pattern";

  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  VLOG(2) << "Fuse " << contraction.op() << " with BiasAdd: "
          << " bias_add=" << bias_add.name()
          << " contraction=" << contraction.name();

  NodeDef fused_op;
  fused_op.set_name(bias_add.name());
  fused_op.set_device(contraction.device());
  fused_op.add_input(contraction.input(0));  // 0: input
  fused_op.add_input(contraction.input(1));  // 1: filter
  fused_op.add_input(bias_add.input(1));     // 2: bias

  if (IsConv2D(contraction)) {
    fused_op.set_op(kFusedConv2D);
    CopyConv2DAttributes(contraction, &fused_op);
  } else if (IsDepthwiseConv2dNative(contraction)) {
    fused_op.set_op(kFusedDepthwiseConv2dNative);
    CopyDepthwiseConv2dNativeAttributes(contraction, &fused_op);
  } else if (IsMatMul(contraction)) {
    fused_op.set_op(kFusedMatMul);
    CopyMatMulAttributes(contraction, &fused_op);
  }

  SetFusedOpAttributes(&fused_op, {"BiasAdd"});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return Status::OK();
}

Status AddFusedContractionNode(
    RemapperContext* ctx, const ContractionWithBiasAddAndActivation& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  DCHECK(IsDeviceCompatible(*ctx, matched)) << "Unsupported fusion pattern";

  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& activation = graph->node(matched.activation);
  VLOG(2) << "Fuse " << contraction.op() << " with BiasAdd and "
          << activation.op() << ":"
          << " activation=" << activation.name()
          << " bias_add=" << bias_add.name()
          << " contraction=" << contraction.name();

  NodeDef fused_op;
  fused_op.set_name(activation.name());
  fused_op.set_device(contraction.device());
  fused_op.add_input(contraction.input(0));  // 0: input
  fused_op.add_input(contraction.input(1));  // 1: filter
  fused_op.add_input(bias_add.input(1));     // 2: bias

  if (IsConv2D(contraction)) {
    fused_op.set_op(kFusedConv2D);
    CopyConv2DAttributes(contraction, &fused_op);
  } else if (IsDepthwiseConv2dNative(contraction)) {
    fused_op.set_op(kFusedDepthwiseConv2dNative);
    CopyDepthwiseConv2dNativeAttributes(contraction, &fused_op);
  } else if (IsMatMul(contraction)) {
    fused_op.set_op(kFusedMatMul);
    CopyMatMulAttributes(contraction, &fused_op);
  }

  SetFusedOpAttributes(&fused_op, {"BiasAdd", activation.op()});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return Status::OK();
}

Status AddFusedConv2DNode(RemapperContext* ctx,
                          const ContractionWithSqueezeAndBiasAdd& matched,
                          std::vector<bool>* invalidated_nodes,
                          std::vector<bool>* nodes_to_delete) {
  DCHECK(IsDeviceCompatible(*ctx, matched)) << "Unsupported fusion pattern";

  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  DCHECK(IsConv2D(contraction)) << "Only Conv2D supported for now";

  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& squeeze = graph->node(matched.squeeze);
  VLOG(2) << "Fuse Conv2D with Squeeze and BiasAdd: "
          << " bias_add=" << bias_add.name() << " squeeze=" << squeeze.name()
          << " conv2d=" << contraction.name();

  // Replace Conv2D node with a fused Conv2D. Matched pattern guarantees that it
  // has single consumer (only the squeeze node).
  NodeDef fused_conv2d;
  fused_conv2d.set_name(contraction.name());
  fused_conv2d.set_op(kFusedConv2D);
  fused_conv2d.set_device(contraction.device());
  fused_conv2d.add_input(contraction.input(0));  // 0: input
  fused_conv2d.add_input(contraction.input(1));  // 1: filter
  fused_conv2d.add_input(bias_add.input(1));     // 2: bias

  CopyConv2DAttributes(contraction, &fused_conv2d);
  SetFusedOpAttributes(&fused_conv2d, {"BiasAdd"});

  // Replace BiasAdd node with a Squeeze.
  NodeDef remapped_squeeze = squeeze;
  remapped_squeeze.set_name(bias_add.name());
  remapped_squeeze.set_input(0, contraction.name());

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_conv2d), &status);
  TF_RETURN_IF_ERROR(status);
  mutation->AddNode(std::move(remapped_squeeze), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.contraction] = true;
  (*invalidated_nodes)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.squeeze] = true;

  return Status::OK();
}

Status AddFusedConv2DNode(RemapperContext* ctx,
                          const ContractionWithBatchNorm& matched,
                          std::vector<bool>* invalidated_nodes,
                          std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  DCHECK(IsConv2D(contraction)) << "Only Conv2D supported for now";
  const NodeDef& fused_batch_norm = graph->node(matched.fused_batch_norm);
  VLOG(2) << "Fuse Conv2D with BatchNorm: batch_norm="
          << fused_batch_norm.name() << " conv2d=" << contraction.name();

  NodeDef fused_conv2d;
  fused_conv2d.set_name(fused_batch_norm.name());
  fused_conv2d.set_op(kFusedConv2D);
  fused_conv2d.set_device(contraction.device());
  fused_conv2d.add_input(contraction.input(0));       // 0: input
  fused_conv2d.add_input(contraction.input(1));       // 1: filter
  fused_conv2d.add_input(fused_batch_norm.input(1));  // 2: scale
  fused_conv2d.add_input(fused_batch_norm.input(2));  // 3: offset
  fused_conv2d.add_input(fused_batch_norm.input(3));  // 4: mean
  fused_conv2d.add_input(fused_batch_norm.input(4));  // 5: variance

  CopyConv2DAttributes(contraction, &fused_conv2d);
  SetFusedOpAttributes(&fused_conv2d, {"FusedBatchNorm"},
                       /*num_args=*/4, /*epsilon=*/matched.epsilon);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_conv2d), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.fused_batch_norm] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return Status::OK();
}

Status AddFusedConv2DNode(RemapperContext* ctx,
                          const ContractionWithBatchNormAndActivation& matched,
                          std::vector<bool>* invalidated_nodes,
                          std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);

  DCHECK(IsConv2D(contraction)) << "Only Conv2D supported for now";

  const NodeDef& activation = graph->node(matched.activation);
  const NodeDef& fused_batch_norm = graph->node(matched.fused_batch_norm);
  VLOG(2) << "Fuse Conv2D with BatchNorm and " << activation.op()
          << ": activation=" << activation.name()
          << " batch_norm=" << fused_batch_norm.name()
          << " conv2d=" << contraction.name();

  NodeDef fused_conv2d;
  fused_conv2d.set_name(activation.name());
  fused_conv2d.set_op(kFusedConv2D);
  fused_conv2d.set_device(contraction.device());
  fused_conv2d.add_input(contraction.input(0));       // 0: input
  fused_conv2d.add_input(contraction.input(1));       // 1: filter
  fused_conv2d.add_input(fused_batch_norm.input(1));  // 2: scale
  fused_conv2d.add_input(fused_batch_norm.input(2));  // 3: offset
  fused_conv2d.add_input(fused_batch_norm.input(3));  // 4: mean
  fused_conv2d.add_input(fused_batch_norm.input(4));  // 5: variance

  CopyConv2DAttributes(contraction, &fused_conv2d);
  SetFusedOpAttributes(&fused_conv2d, {"FusedBatchNorm", activation.op()},
                       /*num_args=*/4, /*epsilon=*/matched.epsilon);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_conv2d), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.activation] = true;
  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.fused_batch_norm] = true;

  return Status::OK();
}

#ifdef INTEL_MKL
Status AddFusedContractionNode(RemapperContext* ctx,
                               const ContractionWithBiasAddAndAdd& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);

  // MKL version only support fusion for Conv2D
  DCHECK(IsConv2D(contraction));

  NodeDef fused_conv2d;
  const NodeDef& add = graph->node(matched.add);
  fused_conv2d.set_name(add.name());
  fused_conv2d.set_op(kFusedConv2D);
  fused_conv2d.set_device(contraction.device());
  fused_conv2d.add_input(contraction.input(0));  // 0: input
  fused_conv2d.add_input(contraction.input(1));  // 1: filter
  fused_conv2d.add_input(bias_add.input(1));     // 2: bias

  // Add OP has two inputs, one is conv+bias pattern matched previously,
  // the other input to add is fused here.
  fused_conv2d.add_input(add.input(1 - matched.port_id));

  CopyConv2DAttributes(contraction, &fused_conv2d);
  SetFusedOpAttributes(&fused_conv2d, {"BiasAdd", "Add"}, 2);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_conv2d), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.add] = true;
  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.bias_add] = true;

  return Status::OK();
}

Status AddFusedContractionNode(
    RemapperContext* ctx, const ContractionWithBiasAndAddActivation& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  // MKL version only support fusion for Conv2D
  const NodeDef& contraction = graph->node(matched.contraction);
  DCHECK(IsConv2D(contraction));
  const NodeDef& activation = graph->node(matched.activation);

  NodeDef fused_conv2d;
  fused_conv2d.set_name(activation.name());
  fused_conv2d.set_op(kFusedConv2D);
  fused_conv2d.set_device(contraction.device());
  fused_conv2d.add_input(contraction.input(0));  // 0: input
  fused_conv2d.add_input(contraction.input(1));  // 1: filter
  const NodeDef& bias_add = graph->node(matched.bias_add);
  fused_conv2d.add_input(bias_add.input(1));  // 2: bias

  // Add OP has two inputs, one is conv+bias pattern matched previously,
  // the other input to add is fused here.
  const NodeDef& add = graph->node(matched.add);
  fused_conv2d.add_input(add.input(1 - matched.port_id));

  CopyConv2DAttributes(contraction, &fused_conv2d);
  SetFusedOpAttributes(&fused_conv2d, {"BiasAdd", "Add", "Relu"}, 2);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_conv2d), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.activation] = true;
  (*nodes_to_delete)[matched.add] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return Status::OK();
}
#endif

Status AddFusedBatchNormExNode(RemapperContext* ctx,
                               const FusedBatchNormEx& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& fused_batch_norm = graph->node(matched.fused_batch_norm);
  const NodeDef& activation = graph->node(matched.activation);

  VLOG(2) << "Fuse " << activation.op() << " with FusedBatchNorm:"
          << " activation=" << activation.name() << " side_input="
          << (matched.side_input != kMissingIndex
                  ? graph->node(matched.side_input).name()
                  : "<none>")
          << " invalidated="
          << (matched.invalidated != kMissingIndex
                  ? graph->node(matched.invalidated).name()
                  : "<none>")
          << " fused_batch_norm=" << fused_batch_norm.name();

  // Replace FusedBatchNorm with _FusedBatchNormEx + <SideInput> + <Activation>.
  NodeDef fused_op;
  fused_op.set_op(kFusedBatchNormEx);
  fused_op.set_name(fused_batch_norm.name());
  fused_op.set_device(fused_batch_norm.device());

  fused_op.add_input(fused_batch_norm.input(0));  // 0: input
  fused_op.add_input(fused_batch_norm.input(1));  // 1: scale
  fused_op.add_input(fused_batch_norm.input(2));  // 2: offset
  fused_op.add_input(fused_batch_norm.input(3));  // 3: estimated_mean
  fused_op.add_input(fused_batch_norm.input(4));  // 4: estimated_var

  CopyFusedBatchNormAttributes(fused_batch_norm, &fused_op);

  auto* attrs = fused_op.mutable_attr();
  SetAttrValue(activation.op(), &(*attrs)["activation_mode"]);

  if (matched.side_input != kMissingIndex) {
    SetAttrValue(1, &(*attrs)["num_side_inputs"]);
    const NodeDef& side_input = graph->node(matched.side_input);
    fused_op.add_input(side_input.name());  // 5: side_input
  } else {
    SetAttrValue(0, &(*attrs)["num_side_inputs"]);
  }

  // Turn activation node into Identity node.
  NodeDef identity_op;
  identity_op.set_op("Identity");
  identity_op.set_name(activation.name());
  identity_op.set_device(fused_batch_norm.device());
  identity_op.add_input(fused_batch_norm.name());
  (*identity_op.mutable_attr())["T"] = attrs->at("T");

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  mutation->AddNode(std::move(identity_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.fused_batch_norm] = true;
  (*invalidated_nodes)[matched.activation] = true;
  if (matched.side_input != kMissingIndex) {
    (*nodes_to_delete)[matched.invalidated] = true;
  }

  return Status::OK();
}

Status AddBatchNormNodes(RemapperContext* ctx, const FusedBatchNorm& matched) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& fused_node = graph->node(matched.fused_batch_norm);
  VLOG(2) << "Optimizing fused batch norm node "
          << SummarizeNodeDef(fused_node);

  const string& x = fused_node.input(0);
  string scale = fused_node.input(1);
  string offset = fused_node.input(2);
  string mean = fused_node.input(3);
  string variance = fused_node.input(4);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;

  if (fused_node.attr().at(kDataFormat).s() == "NCHW") {
    // Need to reshape the last 4 inputs
    NodeDef new_shape;
    const string new_shape_name =
        AddPrefixToNodeName("NCHWShape", fused_node.name());
    new_shape.set_name(new_shape_name);
    new_shape.set_op("Const");
    new_shape.set_device(fused_node.device());
    *new_shape.add_input() = AsControlDependency(scale);
    (*new_shape.mutable_attr())["dtype"].set_type(DT_INT32);
    Tensor t(DT_INT32, {4});
    t.flat<int32>()(0) = 1;
    t.flat<int32>()(1) = -1;
    t.flat<int32>()(2) = 1;
    t.flat<int32>()(3) = 1;
    t.AsProtoTensorContent(
        (*new_shape.mutable_attr())["value"].mutable_tensor());
    mutation->AddNode(std::move(new_shape), &status);
    TF_RETURN_IF_ERROR(status);

    NodeDef reshaped_scale;
    reshaped_scale.set_name(
        AddPrefixToNodeName("NCHWShapedScale", fused_node.name()));
    reshaped_scale.set_op("Reshape");
    reshaped_scale.set_device(fused_node.device());
    *reshaped_scale.add_input() = scale;
    *reshaped_scale.add_input() = new_shape_name;
    (*reshaped_scale.mutable_attr())["T"] = fused_node.attr().at("T");
    (*reshaped_scale.mutable_attr())["Tshape"].set_type(DT_INT32);
    scale = reshaped_scale.name();
    mutation->AddNode(std::move(reshaped_scale), &status);
    TF_RETURN_IF_ERROR(status);

    NodeDef reshaped_offset;
    reshaped_offset.set_name(
        AddPrefixToNodeName("NCHWShapedOffset", fused_node.name()));
    reshaped_offset.set_op("Reshape");
    reshaped_offset.set_device(fused_node.device());
    *reshaped_offset.add_input() = offset;
    *reshaped_offset.add_input() = new_shape_name;
    (*reshaped_offset.mutable_attr())["T"] = fused_node.attr().at("T");
    (*reshaped_offset.mutable_attr())["Tshape"].set_type(DT_INT32);
    offset = reshaped_offset.name();
    mutation->AddNode(std::move(reshaped_offset), &status);
    TF_RETURN_IF_ERROR(status);

    NodeDef reshaped_mean;
    reshaped_mean.set_name(
        AddPrefixToNodeName("NCHWShapedMean", fused_node.name()));
    reshaped_mean.set_op("Reshape");
    reshaped_mean.set_device(fused_node.device());
    *reshaped_mean.add_input() = mean;
    *reshaped_mean.add_input() = new_shape_name;
    (*reshaped_mean.mutable_attr())["T"] = fused_node.attr().at("T");
    (*reshaped_mean.mutable_attr())["Tshape"].set_type(DT_INT32);
    mean = reshaped_mean.name();
    mutation->AddNode(std::move(reshaped_mean), &status);
    TF_RETURN_IF_ERROR(status);

    NodeDef reshaped_variance;
    reshaped_variance.set_name(
        AddPrefixToNodeName("NCHWShapedVariance", fused_node.name()));
    reshaped_variance.set_op("Reshape");
    reshaped_variance.set_device(fused_node.device());
    *reshaped_variance.add_input() = variance;
    *reshaped_variance.add_input() = new_shape_name;
    (*reshaped_variance.mutable_attr())["T"] = fused_node.attr().at("T");
    (*reshaped_variance.mutable_attr())["Tshape"].set_type(DT_INT32);
    variance = reshaped_variance.name();
    mutation->AddNode(std::move(reshaped_variance), &status);
    TF_RETURN_IF_ERROR(status);
  }

  float epsilon = 0.0f;
  if (fused_node.attr().count("epsilon")) {
    epsilon = fused_node.attr().at("epsilon").f();
  }
  DataType dtype = fused_node.attr().at("T").type();
  Tensor value(dtype, TensorShape());
  value.scalar<float>()() = epsilon;
  NodeDef variance_epsilon;
  const string variance_epsilon_name =
      AddPrefixToNodeName("Const", fused_node.name());
  TF_RETURN_IF_ERROR(ConstantFolding::CreateNodeDef(
      variance_epsilon_name, TensorValue(&value), &variance_epsilon));
  variance_epsilon.set_device(fused_node.device());
  mutation->AddNode(std::move(variance_epsilon), &status);
  TF_RETURN_IF_ERROR(status);

  NodeDef variance_plus_epsilon;
  const string variance_plus_epsilon_name =
      AddPrefixToNodeName("VarPlusEpsilon", fused_node.name());
  variance_plus_epsilon.set_name(variance_plus_epsilon_name);
  variance_plus_epsilon.set_op("Add");
  (*variance_plus_epsilon.mutable_attr())["T"].set_type(dtype);
  variance_plus_epsilon.set_device(fused_node.device());
  *variance_plus_epsilon.add_input() = variance;
  *variance_plus_epsilon.add_input() = variance_epsilon_name;
  mutation->AddNode(std::move(variance_plus_epsilon), &status);
  TF_RETURN_IF_ERROR(status);

  NodeDef inv;
  const string inv_name = AddPrefixToNodeName("Inv", fused_node.name());
  inv.set_name(inv_name);
  inv.set_op("Rsqrt");
  inv.set_device(fused_node.device());
  (*inv.mutable_attr())["T"].set_type(dtype);
  *inv.add_input() = variance_plus_epsilon_name;
  mutation->AddNode(std::move(inv), &status);
  TF_RETURN_IF_ERROR(status);

  NodeDef scaled;
  const string scaled_name = AddPrefixToNodeName("Scaled", fused_node.name());
  scaled.set_name(scaled_name);
  scaled.set_op("Mul");
  scaled.set_device(fused_node.device());
  (*scaled.mutable_attr())["T"].set_type(dtype);
  *scaled.add_input() = inv_name;
  *scaled.add_input() = scale;
  mutation->AddNode(std::move(scaled), &status);
  TF_RETURN_IF_ERROR(status);

  NodeDef a;
  const string a_name = AddPrefixToNodeName("Mul", fused_node.name());
  a.set_name(a_name);
  a.set_op("Mul");
  a.set_device(fused_node.device());
  (*a.mutable_attr())["T"].set_type(dtype);
  *a.add_input() = x;
  *a.add_input() = scaled_name;
  mutation->AddNode(std::move(a), &status);
  TF_RETURN_IF_ERROR(status);

  NodeDef b;
  const string b_name = AddPrefixToNodeName("Mul2", fused_node.name());
  b.set_name(b_name);
  b.set_op("Mul");
  b.set_device(fused_node.device());
  (*b.mutable_attr())["T"].set_type(dtype);
  *b.add_input() = mean;
  *b.add_input() = scaled_name;
  mutation->AddNode(std::move(b), &status);
  TF_RETURN_IF_ERROR(status);

  NodeDef c;
  const string c_name = AddPrefixToNodeName("Offset", fused_node.name());
  c.set_name(c_name);
  c.set_op("Sub");
  c.set_device(fused_node.device());
  (*c.mutable_attr())["T"].set_type(dtype);
  *c.add_input() = offset;
  *c.add_input() = b_name;
  mutation->AddNode(std::move(c), &status);
  TF_RETURN_IF_ERROR(status);

  NodeDef r;
  r.set_name(fused_node.name());
  r.set_op("Add");
  r.set_device(fused_node.device());
  (*r.mutable_attr())["T"].set_type(dtype);
  *r.add_input() = a_name;
  *r.add_input() = c_name;
  mutation->AddNode(std::move(r), &status);
  TF_RETURN_IF_ERROR(status);

  return mutation->Apply();
}

#ifdef INTEL_MKL
bool IsConv2DWithAdd(const RemapperContext& ctx, int node_index) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  // Candidate for Conv2D + Add or Conv2D + BiasAdd + Add fusion.
  auto is_supported_add_input = [](const auto* node_view) -> bool {
    if (IsConv2D(*node_view->node())) return true;
    if (IsBiasAdd(*node_view->node())) {
      if (node_view->NumRegularFanins() < 2) return false;
      const auto& bias_add_fanin_0 = node_view->GetRegularFanin(0);
      const auto& bias_add_fanin_1 = node_view->GetRegularFanin(1);
      return IsConv2D(*bias_add_fanin_0.node_view()->node()) ||
             IsConv2D(*bias_add_fanin_1.node_view()->node());
    }
    return false;
  };

  auto is_supported_add = [&](const auto* node_view) -> bool {
    const auto* node_def = node_view->node();
    if (IsAdd(*node_def)) {
      if (node_view->NumRegularFanins() < 2) return false;
      const auto& add_fanin_0 = node_view->GetRegularFanin(0);
      const auto& add_fanin_1 = node_view->GetRegularFanin(1);
      return is_supported_add_input(add_fanin_0.node_view()) ||
             is_supported_add_input(add_fanin_1.node_view());
    }
    return false;
  };

  bool ret = false;
  for (int i = 0; i < node_view->NumRegularFanins(); i++) {
    const auto& fanin_i = node_view->GetRegularFanin(i);
    ret = is_supported_add(fanin_i.node_view());
    if (ret) break;
  }

  return ret;
}
#endif

// Check if a node is a candidate to one of the patterns that require inferred
// shapes:
//   (1) Splitting FusedBatchNorm into primitives.
//   (2) Fusing side input and/or activation into FusedBatchNorm.
//   (3) INTEL_MKL specific: Conv2D -> Add or Conv2D -> BiasAdd -> Add.
bool RequiresInferredShapes(const RemapperContext& ctx, int node_index) {
  // Candidate for a FusedBatchNorm splitting.
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  const auto is_batch_norm_candidate = [&]() -> bool {
    if (!IsFusedBatchNorm(*node_def)) return false;
    if (GetDataTypeFromAttr(*node_def, "T") != DT_FLOAT) return false;

    bool is_training = true;
    if (!TryGetNodeAttr(*node_def, kIsTraining, &is_training)) return false;
    if (is_training) return false;

    return true;
  };

  // Candidate for a FusedBatchNorm fusion.
  const auto is_batch_norm_fusion_candidate = [&]() -> bool {
    if (!IsRelu(*node_def)) return false;

    if (node_view->NumRegularFanins() < 1) return false;
    const auto& relu_fanin_0 = node_view->GetRegularFanin(0);
    const auto* relu_fanin_0_node_view = relu_fanin_0.node_view();
    const auto* relu_fanin_0_node_def = relu_fanin_0_node_view->node();

    if (IsFusedBatchNorm(*relu_fanin_0_node_def)) {
      // FusedBatchNorm + Relu.
      return true;

    } else if (IsAdd(*relu_fanin_0_node_def)) {
      // FusedBatchNorm + Add + Relu.

      if (relu_fanin_0_node_view->NumRegularFanins() < 2) return false;
      const auto& add_regular_fanin_0 =
          relu_fanin_0_node_view->GetRegularFanin(0);
      if (IsFusedBatchNorm(*add_regular_fanin_0.node_view()->node()))
        return true;
      const auto& add_regular_fanin_1 =
          relu_fanin_0_node_view->GetRegularFanin(1);
      if (IsFusedBatchNorm(*add_regular_fanin_1.node_view()->node()))
        return true;
    }

    return false;
  };

#ifdef INTEL_MKL
  return is_batch_norm_candidate() || is_batch_norm_fusion_candidate() ||
         IsConv2DWithAdd(ctx, node_index);
#else
  return is_batch_norm_candidate() || is_batch_norm_fusion_candidate();
#endif  // INTEL_MKL
}

}  // namespace

Status Remapper::Optimize(Cluster* cluster, const GrapplerItem& item,
                          GraphDef* optimized_graph) {
  GrapplerItem mutable_item = item;
  Status status;
  RemapperContext ctx(&mutable_item, &status);
  TF_RETURN_IF_ERROR(status);
  // Processing graph in reverse-topological sorted order allows to remap
  // longer chains of dependent ops in one pass.
  TF_RETURN_IF_ERROR(
      ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));

  const int num_nodes = item.graph.node_size();
  // Skip nodes that were invalidated by a remapper, e.g. do not process BiasAdd
  // and Activation nodes that were fused into a Conv2D node.
  std::vector<bool> invalidated_nodes(num_nodes);
  std::vector<bool> nodes_to_delete(num_nodes);

  // _Fused{...} kernels do not have registered gradient function, so we must
  // not perform rewrite if the graph will be differentiated later.
  bool allow_non_differentiable_rewrites =
      item.optimization_options().allow_non_differentiable_rewrites;

  for (int i = num_nodes - 1; i >= 0; --i) {
    // Check if node was invalidated by one of the previous remaps.
    if (invalidated_nodes[i] || nodes_to_delete[i]) {
      continue;
    }

    // Infer properties lazily in case they are not needed.
    if (!ctx.inferred_graph_properties && RequiresInferredShapes(ctx, i)) {
      const bool assume_valid_feeds = opt_level_ == RewriterConfig::AGGRESSIVE;
      TF_RETURN_IF_ERROR(ctx.graph_properties.InferStatically(
          assume_valid_feeds,
          /*aggressive_shape_inference=*/false,
          /*include_input_tensor_values=*/true,
          /*include_output_tensor_values=*/false));
      ctx.inferred_graph_properties = true;
    }

#ifdef INTEL_MKL
    ContractionWithBiasAddAndAdd contract_with_bias_and_add;
    ContractionWithBiasAndAddActivation contract_with_bias_and_add_activation;

    if (!item.optimization_options().is_eager_mode) {
      // Remap Conv2D+BiasAdd+Add+relu into the _FusedConv2D.
      if (FindContractionWithBiasAndAddActivation(
              ctx, i, &contract_with_bias_and_add_activation)) {
        TF_RETURN_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_bias_and_add_activation,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap Conv2D+BiasAdd+Add into the _FusedConv2D.
      if (FindContractionWithBiasAddAndAdd(ctx, i,
                                           &contract_with_bias_and_add)) {
        TF_RETURN_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_bias_and_add,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }
    }
#endif  //! INTEL_MKL

    // Remap {Conv2D,DepthwiseConv2D,MatMul}+BiasAdd into the
    // _Fused{Conv2D,DepthwiseConv2dNative,MatMul}
    ContractionWithBiasAdd contract_with_bias;
    if (allow_non_differentiable_rewrites &&
        FindContractionWithBias(ctx, i, &contract_with_bias)) {
      TF_RETURN_IF_ERROR(AddFusedContractionNode(
          &ctx, contract_with_bias, &invalidated_nodes, &nodes_to_delete));
      continue;
    }

    // Remap {Conv2D,DepthwiseConv2D,MatMul}+BiasAdd+Activation into the
    // _Fused{Conv2D,DepthwiseConv2dNative,MatMul}.
    ContractionWithBiasAddAndActivation contract_with_bias_and_activation;
    if (allow_non_differentiable_rewrites &&
        FindContractionWithBiasAndActivation(
            ctx, i, &contract_with_bias_and_activation)) {
      TF_RETURN_IF_ERROR(
          AddFusedContractionNode(&ctx, contract_with_bias_and_activation,
                                  &invalidated_nodes, &nodes_to_delete));
      continue;
    }

// NOTE: We can only fuse BatchNorm into Conv2D nodes. In theory we can do
// it for MatMul as well, but in practice this pattern does not appear in
// real Tensorflow graphs.

// TODO(penporn):
// Remove this once TF-MKL supports _FusedConv2D with these operations.
#ifndef INTEL_MKL
    // Remap Conv2D+Squeeze+BiasAdd into the _FusedConv2D+Squeeze.
    ContractionWithSqueezeAndBiasAdd contract_with_squeeze_and_bias;
    if (allow_non_differentiable_rewrites &&
        FindConv2DWithSqueezeAndBias(ctx, i, &contract_with_squeeze_and_bias)) {
      TF_RETURN_IF_ERROR(
          AddFusedConv2DNode(&ctx, contract_with_squeeze_and_bias,
                             &invalidated_nodes, &nodes_to_delete));
      continue;
    }

    // Remap Conv2D+FusedBatchNorm into the _FusedConv2D;
    ContractionWithBatchNorm contract_with_batch_norm;
    if (allow_non_differentiable_rewrites &&
        FindConv2DWithBatchNorm(ctx, i, &contract_with_batch_norm)) {
      TF_RETURN_IF_ERROR(AddFusedConv2DNode(&ctx, contract_with_batch_norm,
                                            &invalidated_nodes,
                                            &nodes_to_delete));
      continue;
    }

    // Remap Conv2D+FusedBatchNorm+Activation into the _FusedConv2D;
    ContractionWithBatchNormAndActivation
        contract_with_batch_norm_and_activation;
    if (allow_non_differentiable_rewrites &&
        FindConv2DWithBatchNormAndActivation(
            ctx, i, &contract_with_batch_norm_and_activation)) {
      TF_RETURN_IF_ERROR(
          AddFusedConv2DNode(&ctx, contract_with_batch_norm_and_activation,
                             &invalidated_nodes, &nodes_to_delete));
      continue;
    }
#endif  // !INTEL_MKL

    // Remap FusedBatchNorm+<SideInput>+<Activation> into the _FusedBatchNormEx.
    FusedBatchNormEx fused_batch_norm_ex;
    if (allow_non_differentiable_rewrites &&
        FindFusedBatchNormEx(ctx, i, &fused_batch_norm_ex)) {
      TF_RETURN_IF_ERROR(AddFusedBatchNormExNode(
          &ctx, fused_batch_norm_ex, &invalidated_nodes, &nodes_to_delete));
      continue;
    }

    // During inference, most of the inputs to FusedBatchNorm are constant, and
    // we can therefore replace the op with a much cheaper set of primitives.
    FusedBatchNorm fused_batch_norm;
    if (FindFusedBatchNorm(ctx, i, &fused_batch_norm)) {
      TF_RETURN_IF_ERROR(AddBatchNormNodes(&ctx, fused_batch_norm));
      continue;
    }
  }

  // Remove invalidated nodes.
  utils::Mutation* mutation = ctx.graph_view.GetMutationBuilder();
  for (int i = 0; i < num_nodes; ++i) {
    if (nodes_to_delete[i]) {
      mutation->RemoveNode(ctx.graph_view.GetNode(i));
    }
  }
  TF_RETURN_IF_ERROR(mutation->Apply());

  *optimized_graph = std::move(mutable_item.graph);

  return Status::OK();
}

void Remapper::Feedback(Cluster* cluster, const GrapplerItem& item,
                        const GraphDef& optimized_graph, double result) {
  // Nothing to do for RemapperOptimizer.
}

}  // namespace grappler
}  // namespace tensorflow
