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
constexpr char kFusedBatchNormEx[] = "_FusedBatchNormEx";

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

// FusedBatchNorm[$is_training] with fused side input and/or activation.
struct FusedBatchNormEx {
  FusedBatchNormEx() = default;

  const NodeDef* fused_batch_norm = nullptr;
  const NodeDef* side_input = nullptr;
  const NodeDef* activation = nullptr;
  // Add node that will be invalidated by fusing side input and fused batch norm
  const NodeDef* invalidated = nullptr;
};

// Contraction node followed by a BiasAdd.
struct ContractionWithBiasAdd {
  ContractionWithBiasAdd() = default;
  ContractionWithBiasAdd(const NodeDef* contraction, const NodeDef* bias_add)
      : contraction(contraction), bias_add(bias_add) {}

  const NodeDef* contraction = nullptr;
  const NodeDef* bias_add = nullptr;
};

// Contraction node followed by a BiasAdd and Activation.
struct ContractionWithBiasAddAndActivation {
  ContractionWithBiasAddAndActivation() = default;
  ContractionWithBiasAddAndActivation(const NodeDef* contraction,
                                      const NodeDef* bias_add,
                                      const NodeDef* activation)
      : contraction(contraction), bias_add(bias_add), activation(activation) {}

  const NodeDef* contraction = nullptr;
  const NodeDef* bias_add = nullptr;
  const NodeDef* activation = nullptr;
};

// Contraction node followed by a Squeeze and BiasAdd.
struct ContractionWithSqueezeAndBiasAdd {
  ContractionWithSqueezeAndBiasAdd() = default;
  ContractionWithSqueezeAndBiasAdd(const NodeDef* contraction,
                                   const NodeDef* squeeze,
                                   const NodeDef* bias_add)
      : contraction(contraction), squeeze(squeeze), bias_add(bias_add) {}

  const NodeDef* contraction = nullptr;
  const NodeDef* squeeze = nullptr;
  const NodeDef* bias_add = nullptr;
};

// Contraction node followed by a FusedBatchNorm.
struct ContractionWithBatchNorm {
  ContractionWithBatchNorm() = default;
  ContractionWithBatchNorm(const NodeDef* contraction,
                           const NodeDef* fused_batch_norm, float epsilon = 0.0)
      : contraction(contraction),
        fused_batch_norm(fused_batch_norm),
        epsilon(epsilon) {}

  const NodeDef* contraction = nullptr;
  const NodeDef* fused_batch_norm = nullptr;
  float epsilon = 0.0;
};

// Contraction node followed by a FusedBatchNorm and Activation.
struct ContractionWithBatchNormAndActivation {
  ContractionWithBatchNormAndActivation() = default;
  ContractionWithBatchNormAndActivation(const NodeDef* contraction,
                                        const NodeDef* fused_batch_norm,
                                        const NodeDef* activation,
                                        float epsilon = 0.0)
      : contraction(contraction),
        fused_batch_norm(fused_batch_norm),
        activation(activation),
        epsilon(epsilon) {}

  const NodeDef* contraction = nullptr;
  const NodeDef* fused_batch_norm = nullptr;
  const NodeDef* activation = nullptr;
  float epsilon = 0.0;
};

#ifdef INTEL_MKL
// Contraction node followed by a BiasAdd and Add.
struct ContractionWithBiasAddAndAdd {
  ContractionWithBiasAddAndAdd() = default;
  ContractionWithBiasAddAndAdd(const NodeDef* contraction,
                               const NodeDef* bias_add, const NodeDef* add,
                               int port_id)
      : contraction(contraction),
        bias_add(bias_add),
        add(add),
        port_id(port_id) {}

  const NodeDef* contraction = nullptr;
  const NodeDef* bias_add = nullptr;
  const NodeDef* add = nullptr;
  int port_id = 0;
};

// Contraction node followed by a BiasAdd, Add and Relu.
struct ContractionWithBiasAndAddActivation {
  ContractionWithBiasAndAddActivation() = default;
  ContractionWithBiasAndAddActivation(const NodeDef* contraction,
                                      const NodeDef* bias_add,
                                      const NodeDef* add, int port_id,
                                      const NodeDef* activation)
      : contraction(contraction),
        bias_add(bias_add),
        add(add),
        port_id(port_id),
        activation(activation) {}

  const NodeDef* contraction = nullptr;
  const NodeDef* bias_add = nullptr;
  const NodeDef* add = nullptr;
  int port_id = 0;
  const NodeDef* activation = nullptr;
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
  if (IsConv2D(*contraction)) {
    return dtype == DT_FLOAT || dtype == DT_DOUBLE;
  } else if (IsMatMul(*contraction)) {
    return dtype == DT_FLOAT;
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

bool IsCpuCompatibleMatMul(const NodeDef* matmul) {
  DCHECK(IsMatMul(*matmul)) << "Expected MatMul op";
#ifndef INTEL_MKL
  // Temporarily disable Matmul fusions if MKL is enabled.
  // TODO(Intel) renable Matmul fusions when enabled by MKL DNN.
  return NodeIsOnCpu(matmul) && IsCpuCompatibleDataType(matmul);
#else
  return false;
#endif  // !INTEL_MKL
}

// Checks if we can rewrite a pattern to the `_Fused{Conv2D,MatMul}` on CPU.
template <typename Pattern>
bool IsCpuCompatible(const Pattern& matched) {
  if (IsConv2D(*matched.contraction)) {
    return IsCpuCompatibleConv2D(matched.contraction);
  } else if (IsMatMul(*matched.contraction)) {
    return IsCpuCompatibleMatMul(matched.contraction);
  } else {
    return false;
  }
}

// Checks if we can rewrite a pattern to the `_FusedConv2D` on GPU device.
bool IsGpuCompatible(const RemapperContext& ctx,
                     const ContractionWithBiasAddAndActivation& matched) {
  if (!IsConv2D(*matched.contraction)) return false;

  const std::vector<OpInfo::TensorProperties>& input_props =
      ctx.graph_properties.GetInputProperties(matched.contraction->name());
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
  bool is_relu = IsRelu(*matched.activation);

  return is_relu && is_spatial_conv &&
         IsGpuCompatibleConv2D(matched.contraction);
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
  return IsCpuCompatible(matched) || IsGpuCompatible(ctx, matched);
}

bool IsSupportedActivation(const NodeDef& node) {
  return IsRelu(node) || IsRelu6(node) || IsElu(node);
}

bool FindContractionWithBias(const RemapperContext& ctx,
                             const NodeDef* bias_add,
                             ContractionWithBiasAdd* matched,
                             bool check_device_compatible = true) {
  if (!EigenSupportsContractionOutputKernel()) return false;

  // Root of the pattern must be a BiasAdd.
  if (bias_add == nullptr || !IsBiasAdd(*bias_add) ||
      HasControlFaninOrFanout(ctx.graph_view, bias_add))
    return false;

  // Input to the BiasAdd must be a Conv2D or a MatMul.
  const auto input_port = GraphView::InputPort(bias_add, 0);
  const auto contraction = ctx.graph_view.GetRegularFanin(input_port);

  bool is_conv2d_or_matmul = contraction.node && (IsConv2D(*contraction.node) ||
                                                  IsMatMul(*contraction.node));

  if (!is_conv2d_or_matmul || !HaveSameDataType(bias_add, contraction.node) ||
      HasControlFaninOrFanout(ctx.graph_view, contraction.node) ||
      !HasSingleFanoutNode(ctx.graph_view, contraction.node) ||
      IsInPreserveSet(ctx, contraction.node))
    return false;

  // Check that data type and data format are supported on assigned device.
  const ContractionWithBiasAdd pattern{contraction.node, bias_add};
  if (check_device_compatible && !IsDeviceCompatible(ctx, pattern)) {
    return false;
  }

  // We successfully found a {Conv2D, MatMul}+BiasAdd pattern.
  *matched = pattern;

  return true;
}

bool FindContractionWithBiasAndActivation(
    const RemapperContext& ctx, const NodeDef* activation,
    ContractionWithBiasAddAndActivation* matched) {
  if (!EigenSupportsContractionOutputKernel()) return false;

  // Root of the pattern must be an activation node.
  if (!activation || !IsSupportedActivation(*activation) ||
      HasControlFaninOrFanout(ctx.graph_view, activation))
    return false;

  // And input to the activation node must match ContractionWithBiasAdd pattern.
  const auto input_port = GraphView::InputPort(activation, 0);
  const auto bias_add = ctx.graph_view.GetRegularFanin(input_port);

  ContractionWithBiasAdd base;
  if (!FindContractionWithBias(ctx, bias_add.node, &base,
                               /*check_device_compatible=*/false) ||
      !HasSingleFanoutNode(ctx.graph_view, base.bias_add) ||
      !HaveSameDataType(activation, base.bias_add) ||
      IsInPreserveSet(ctx, base.bias_add))
    return false;

  // Check that data type and data format are supported on assigned device.
  const ContractionWithBiasAddAndActivation pattern{base.contraction,
                                                    base.bias_add, activation};
  if (!IsDeviceCompatible(ctx, pattern)) return false;

  // We successfully found a {Conv2D, MatMul}+BiasAdd+Activation pattern.
  *matched = pattern;

  return true;
}

bool FindConv2DWithSqueezeAndBias(const RemapperContext& ctx,
                                  const NodeDef* bias_add,
                                  ContractionWithSqueezeAndBiasAdd* matched) {
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
  const ContractionWithSqueezeAndBiasAdd pattern{conv2d.node, squeeze.node,
                                                 bias_add};
  if (!IsDeviceCompatible(ctx, pattern)) return false;

  // We successfully found a Conv2D+Squeeze+BiasAdd pattern.
  *matched = pattern;

  return true;
}

bool FindConv2DWithBatchNorm(const RemapperContext& ctx,
                             const NodeDef* batch_norm,
                             ContractionWithBatchNorm* matched) {
  if (!EigenSupportsContractionOutputKernel()) return false;

  // Root of the pattern must be a FusedBatchNorm.
  if (!batch_norm || !IsFusedBatchNorm(*batch_norm)) return false;

  // FusedBatchNormV2 and V3 have an extra type parameter.
  if (batch_norm->op() != "FusedBatchNorm" &&
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
  matched->contraction = conv2d.node;
  matched->fused_batch_norm = batch_norm;
  if (!GetNodeAttr(*batch_norm, "epsilon", &matched->epsilon).ok())
    return false;

  return true;
}

bool FindConv2DWithBatchNormAndActivation(
    const RemapperContext& ctx, const NodeDef* node,
    ContractionWithBatchNormAndActivation* matched) {
  if (!EigenSupportsContractionOutputKernel()) return false;

  // Root of the pattern must be an activation node.
  if (!node || !IsSupportedActivation(*node) ||
      HasControlFaninOrFanout(ctx.graph_view, node))
    return false;

  // And input to the activation node must match Conv2DWithBatchNorm pattern.
  const auto input_port = GraphView::InputPort(node, 0);
  const auto batch_norm = ctx.graph_view.GetRegularFanin(input_port);

  ContractionWithBatchNorm base;
  if (!FindConv2DWithBatchNorm(ctx, batch_norm.node, &base) ||
      !HasSingleFanoutNode(ctx.graph_view, base.fused_batch_norm) ||
      !HaveSameDataType(node, base.fused_batch_norm) ||
      IsInPreserveSet(ctx, base.fused_batch_norm))
    return false;

  // We successfully found a Conv2D+FusedBatchNorm+Activation pattern.
  matched->contraction = base.contraction;
  matched->fused_batch_norm = base.fused_batch_norm;
  matched->activation = node;
  matched->epsilon = base.epsilon;

  return true;
}

#ifdef INTEL_MKL
// As AddN has multiple inputs, this function tries to find Conv2D + Bias
// pattern in specific input port.
bool FindContractionWithBiasInPort(const RemapperContext& ctx,
                                   const NodeDef* add, int port_id,
                                   ContractionWithBiasAdd* base) {
  // Input to AddN must match ContractionWithBiasAdd pattern.
  const auto input_port = GraphView::InputPort(add, port_id);
  const auto bias_add = ctx.graph_view.GetRegularFanin(input_port);

  if (!FindContractionWithBias(ctx, bias_add.node, base,
                               /*check_device_compatible=*/false) ||
      !HasSingleFanoutNode(ctx.graph_view, base->bias_add) ||
      !HaveSameDataType(add, base->bias_add) ||
      IsInPreserveSet(ctx, base->bias_add))
    return false;
  return true;
}

bool FindContractionWithBiasAddAndAdd(const RemapperContext& ctx,
                                      const NodeDef* add,
                                      ContractionWithBiasAddAndAdd* matched) {
  // Root of the pattern must be a AddN
  if (!add || !IsAddN(*add) || HasControlFaninOrFanout(ctx.graph_view, add))
    return false;

  // Fusion with AddN is supported only when it has two inputs.
  if (add->input_size() != 2) {
    return false;
  }

  // MKL AddN ops only support float data type.
  if (!HasDataType(add, DT_FLOAT)) return false;

  ContractionWithBiasAdd base;
  matched->port_id = 0;

  // Find the conv+bias pattern in specific port.
  if (!FindContractionWithBiasInPort(ctx, add, matched->port_id, &base)) {
    matched->port_id = 1;
    if (!FindContractionWithBiasInPort(ctx, add, matched->port_id, &base)) {
      return false;
    }
  }

  // We successfully found a Conv2D+BiasAdd+AddN pattern.
  matched->contraction = base.contraction;
  matched->bias_add = base.bias_add;
  matched->add = add;

  return true;
}

bool FindContractionWithBiasAndAddActivation(
    const RemapperContext& ctx, const NodeDef* activation,
    ContractionWithBiasAndAddActivation* matched) {
  // Root of the pattern must be an activation node.
  if (!activation || !IsSupportedActivation(*activation) ||
      HasControlFaninOrFanout(ctx.graph_view, activation))
    return false;

  // MKL activation op only supports float data type.
  if (!HasDataType(activation, DT_FLOAT)) return false;

  // And input to activation must match ContractionWithBiasAddAndAdd pattern.
  const auto input_port = GraphView::InputPort(activation, 0);
  const auto add = ctx.graph_view.GetRegularFanin(input_port);

  ContractionWithBiasAddAndAdd base;

  if (!FindContractionWithBiasAddAndAdd(ctx, add.node, &base)) {
    return false;
  }

  // We successfully found a Conv2D+BiasAdd+AddN+activation pattern.
  const ContractionWithBiasAndAddActivation pattern{
      base.contraction, base.bias_add, base.add, base.port_id, activation};
  *matched = pattern;

  return true;
}
#endif

bool FindFusedBatchNorm(const RemapperContext& ctx, const NodeDef* node,
                        FusedBatchNorm* matched) {
  if (!IsFusedBatchNorm(*node)) return false;
  if (GetDataTypeFromAttr(*node, "T") != DT_FLOAT) return false;

  // Check that the node is in inference mode.
  bool is_training = true;
  if (!GetNodeAttr(*node, kIsTraining, &is_training).ok()) return false;
  if (is_training) return false;

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

bool FindFusedBatchNormEx(const RemapperContext& ctx, const NodeDef* node,
                          FusedBatchNormEx* matched) {
  // Root of the pattern must be a Relu.
  // TODO(ezhulenev): Forward control dependencies.
  if (!IsRelu(*node) || HasControlFaninOrFanout(ctx.graph_view, node))
    return false;

  const NodeDef* relu = node;

  // Returns true iff the node is a compatible FusedBatchNorm node.
  const auto valid_batch_norm = [&](const NodeDef* fused_batch_norm) -> bool {
    if (fused_batch_norm == nullptr || !IsFusedBatchNorm(*fused_batch_norm))
      return false;

    AttrSlice attr(*fused_batch_norm);

    // We fuse FusedBatchNorm only on GPU, because on CPU we fuse it with
    // contraction (MatMul or Conv2D node).
    if (!NodeIsOnGpu(fused_batch_norm)) return false;

    DataType t_dtype = GetDataTypeFromAttr(*fused_batch_norm, "T");
    if (t_dtype != DT_FLOAT && t_dtype != DT_HALF) return false;

    // Get the FusedBatchNorm training mode.
    bool is_training;
    if (!GetNodeAttr(attr, kIsTraining, &is_training).ok()) return false;
    // TODO(ezhulenev): Add support for is_training=True and custom CUDA kernel.
    if (!is_training) return false;

    // In training mode we rely on cuDNN for computing FusedBatchNorm with side
    // inputs and activation, and it has its own limitations. In inference mode
    // we have a custom CUDA kernel that doesn't not have these constraints.
    if (is_training) {
      // cuDNN only supports NHWC data layout.
      string data_format;
      if (!GetNodeAttr(attr, kDataFormat, &data_format).ok()) return false;
      if (data_format != "NHWC") return false;

      // Data type must be DT_HALF.
      if (t_dtype != DT_HALF) return false;

      // Channel dimension must be a multiple of 4.
      const auto& props =
          ctx.graph_properties.GetInputProperties(fused_batch_norm->name());

      const bool valid_channel_dim = !props.empty() &&
                                     props[0].shape().dim_size() == 4 &&
                                     props[0].shape().dim(3).size() % 4 == 0;
      if (!valid_channel_dim) return false;

      // cuDNN must support CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode.
      if (!BatchnormSpatialPersistentEnabled()) return false;
    }

    // FusedBatchNormV2 and V3 have an extra type parameter.
    if ((fused_batch_norm->op() != "FusedBatchNorm") &&
        !HasDataType(fused_batch_norm, DT_FLOAT, "U"))
      return false;

    // Check that only one node consumes the output of a FusedBatchNorm.
    if (HasControlFaninOrFanout(ctx.graph_view, fused_batch_norm) ||
        !HasSingleFanoutNode(ctx.graph_view, fused_batch_norm) ||
        IsInPreserveSet(ctx, fused_batch_norm))
      return false;

    return true;
  };

  const auto relu_input_port = GraphView::InputPort(relu, 0);
  const auto relu_fanin = ctx.graph_view.GetRegularFanin(relu_input_port);
  if (!relu_fanin.node) return false;

  // Input to a Relu can be a FusedBatchNorm.
  if (valid_batch_norm(relu_fanin.node)) {
    matched->activation = relu;
    matched->side_input = nullptr;
    matched->fused_batch_norm = relu_fanin.node;
    matched->invalidated = nullptr;
    return true;
  }

  // Input to a Relu can be an Add node with FusedBatchNorm as one of the inputs
  if (IsAdd(*relu_fanin.node)) {
    const NodeDef* add = relu_fanin.node;

    // Check that only Relu node consumes the output of an Add node.
    if (HasControlFaninOrFanout(ctx.graph_view, add) ||
        !HasSingleFanoutNode(ctx.graph_view, add) || IsInPreserveSet(ctx, add))
      return false;

    const auto add_input_port_0 = GraphView::InputPort(add, 0);
    const auto add_fanin_0 = ctx.graph_view.GetRegularFanin(add_input_port_0);

    const auto add_input_port_1 = GraphView::InputPort(add, 1);
    const auto add_fanin_1 = ctx.graph_view.GetRegularFanin(add_input_port_1);

    if (valid_batch_norm(add_fanin_0.node)) {
      matched->activation = relu;
      matched->side_input = add_fanin_1.node;
      matched->fused_batch_norm = add_fanin_0.node;
      matched->invalidated = add;
      return true;
    }

    if (valid_batch_norm(add_fanin_1.node)) {
      matched->activation = relu;
      matched->side_input = add_fanin_0.node;
      matched->fused_batch_norm = add_fanin_1.node;
      matched->invalidated = add;
      return true;
    }
  }

  return false;
}

void CopyConv2DAttributes(const NodeDef* conv2d, NodeDef* fused_conv2d) {
  DCHECK(IsConv2D(*conv2d)) << "Input node must be a Conv2D";

  auto* attr = fused_conv2d->mutable_attr();
  auto src_attr = conv2d->attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["strides"] = src_attr.at("strides");
  (*attr)["padding"] = src_attr.at("padding");
  (*attr)["explicit_paddings"] = src_attr.at("explicit_paddings");
  (*attr)["dilations"] = src_attr.at("dilations");
  (*attr)["data_format"] = src_attr.at("data_format");
  (*attr)["use_cudnn_on_gpu"] = src_attr.at("use_cudnn_on_gpu");
}

void CopyFusedBatchNormAttributes(const NodeDef* fused_batch_norm,
                                  NodeDef* fused_batch_norm_ex) {
  DCHECK(IsFusedBatchNorm(*fused_batch_norm))
      << "Input node must be a FusedBatchNorm";

  auto* attr = fused_batch_norm_ex->mutable_attr();
  auto src_attr = fused_batch_norm->attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["is_training"] = src_attr.at("is_training");
  (*attr)["data_format"] = src_attr.at("data_format");
  (*attr)["epsilon"] = src_attr.at("epsilon");

  // FusedBatchNormV2 and V3 have an extra type parameter.
  if (fused_batch_norm->op() != "FusedBatchNorm") {
    (*attr)["U"] = src_attr.at("U");
  } else {
    (*attr)["U"] = src_attr.at("T");
  }
}

void CopyMatMulAttributes(const NodeDef* matmul, NodeDef* fused_matmul) {
  DCHECK(IsMatMul(*matmul)) << "Input node must be a MatMul";

  auto* attr = fused_matmul->mutable_attr();
  auto src_attr = matmul->attr();

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

void AddFusedContractionNode(
    const RemapperContext& ctx, const ContractionWithBiasAdd& matched,
    GraphDef* optimized_graph,
    absl::flat_hash_set<const NodeDef*>* invalidated_nodes) {
  DCHECK(IsDeviceCompatible(ctx, matched)) << "Unsupported fusion pattern";

  VLOG(2) << "Fuse " << matched.contraction->op() << " with BiasAdd: "
          << " bias_add=" << matched.bias_add->name()
          << " contraction=" << matched.contraction->name();

  NodeDef* fused_op = optimized_graph->add_node();
  fused_op->set_name(matched.bias_add->name());
  fused_op->set_device(matched.contraction->device());
  fused_op->add_input(matched.contraction->input(0));  // 0: input
  fused_op->add_input(matched.contraction->input(1));  // 1: filter
  fused_op->add_input(matched.bias_add->input(1));     // 2: bias

  if (IsConv2D(*matched.contraction)) {
    fused_op->set_op(kFusedConv2D);
    CopyConv2DAttributes(matched.contraction, fused_op);
  } else if (IsMatMul(*matched.contraction)) {
    fused_op->set_op(kFusedMatMul);
    CopyMatMulAttributes(matched.contraction, fused_op);
  }

  SetFusedOpAttributes(fused_op, {"BiasAdd"});

  invalidated_nodes->insert(matched.bias_add);
  invalidated_nodes->insert(matched.contraction);
}

void AddFusedContractionNode(
    const RemapperContext& ctx,
    const ContractionWithBiasAddAndActivation& matched,
    GraphDef* optimized_graph,
    absl::flat_hash_set<const NodeDef*>* invalidated_nodes) {
  DCHECK(IsDeviceCompatible(ctx, matched)) << "Unsupported fusion pattern";

  VLOG(2) << "Fuse " << matched.contraction->op() << " with BiasAdd and "
          << matched.activation->op() << ":"
          << " activation=" << matched.activation->name()
          << " bias_add=" << matched.bias_add->name()
          << " contraction=" << matched.contraction->name();

  NodeDef* fused_op = optimized_graph->add_node();
  fused_op->set_name(matched.activation->name());
  fused_op->set_device(matched.contraction->device());
  fused_op->add_input(matched.contraction->input(0));  // 0: input
  fused_op->add_input(matched.contraction->input(1));  // 1: filter
  fused_op->add_input(matched.bias_add->input(1));     // 2: bias

  if (IsConv2D(*matched.contraction)) {
    fused_op->set_op(kFusedConv2D);
    CopyConv2DAttributes(matched.contraction, fused_op);
  } else if (IsMatMul(*matched.contraction)) {
    fused_op->set_op(kFusedMatMul);
    CopyMatMulAttributes(matched.contraction, fused_op);
  }

  SetFusedOpAttributes(fused_op, {"BiasAdd", matched.activation->op()});

  invalidated_nodes->insert(matched.activation);
  invalidated_nodes->insert(matched.bias_add);
  invalidated_nodes->insert(matched.contraction);
}

void AddFusedConv2DNode(
    const RemapperContext& ctx, const ContractionWithSqueezeAndBiasAdd& matched,
    GraphDef* optimized_graph,
    absl::flat_hash_set<const NodeDef*>* invalidated_nodes) {
  DCHECK(IsDeviceCompatible(ctx, matched)) << "Unsupported fusion pattern";
  DCHECK(IsConv2D(*matched.contraction)) << "Only Conv2D supported for now";

  VLOG(2) << "Fuse Conv2D with Squeeze and BiasAdd: "
          << " bias_add=" << matched.bias_add->name()
          << " squeeze=" << matched.squeeze->name()
          << " conv2d=" << matched.contraction->name();

  // Replace Conv2D node with a fused Conv2D. Matched pattern guarantees that it
  // has single consumer (only the squeeze node).
  NodeDef* fused_conv2d = optimized_graph->add_node();
  fused_conv2d->set_name(matched.contraction->name());
  fused_conv2d->set_op(kFusedConv2D);
  fused_conv2d->set_device(matched.contraction->device());
  fused_conv2d->add_input(matched.contraction->input(0));  // 0: input
  fused_conv2d->add_input(matched.contraction->input(1));  // 1: filter
  fused_conv2d->add_input(matched.bias_add->input(1));     // 2: bias

  CopyConv2DAttributes(matched.contraction, fused_conv2d);
  SetFusedOpAttributes(fused_conv2d, {"BiasAdd"});

  // Replace BiasAdd node with a Squeeze.
  NodeDef* remapped_squeeze = optimized_graph->add_node();
  *remapped_squeeze = *matched.squeeze;
  remapped_squeeze->set_name(matched.bias_add->name());
  remapped_squeeze->set_input(0, fused_conv2d->name());

  invalidated_nodes->insert(matched.squeeze);
  invalidated_nodes->insert(matched.bias_add);
  invalidated_nodes->insert(matched.contraction);
}

void AddFusedConv2DNode(
    const ContractionWithBatchNorm& matched, GraphDef* optimized_graph,
    absl::flat_hash_set<const NodeDef*>* invalidated_nodes) {
  DCHECK(IsConv2D(*matched.contraction)) << "Only Conv2D supported for now";

  VLOG(2) << "Fuse Conv2D with BatchNorm: batch_norm="
          << matched.fused_batch_norm->name()
          << " conv2d=" << matched.contraction->name();

  NodeDef* fused_conv2d = optimized_graph->add_node();
  fused_conv2d->set_name(matched.fused_batch_norm->name());
  fused_conv2d->set_op(kFusedConv2D);
  fused_conv2d->set_device(matched.contraction->device());
  fused_conv2d->add_input(matched.contraction->input(0));       // 0: input
  fused_conv2d->add_input(matched.contraction->input(1));       // 1: filter
  fused_conv2d->add_input(matched.fused_batch_norm->input(1));  // 2: scale
  fused_conv2d->add_input(matched.fused_batch_norm->input(2));  // 3: offset
  fused_conv2d->add_input(matched.fused_batch_norm->input(3));  // 4: mean
  fused_conv2d->add_input(matched.fused_batch_norm->input(4));  // 5: variance

  CopyConv2DAttributes(matched.contraction, fused_conv2d);
  SetFusedOpAttributes(fused_conv2d, {"FusedBatchNorm"},
                       /*num_args=*/4, /*epsilon=*/matched.epsilon);

  invalidated_nodes->insert(matched.fused_batch_norm);
  invalidated_nodes->insert(matched.contraction);
}

void AddFusedConv2DNode(
    const ContractionWithBatchNormAndActivation& matched,
    GraphDef* optimized_graph,
    absl::flat_hash_set<const NodeDef*>* invalidated_nodes) {
  DCHECK(IsConv2D(*matched.contraction)) << "Only Conv2D supported for now";

  VLOG(2) << "Fuse Conv2D with BatchNorm and " << matched.activation->op()
          << ": activation=" << matched.activation->name()
          << " batch_norm=" << matched.fused_batch_norm->name()
          << " conv2d=" << matched.contraction->name();

  NodeDef* fused_conv2d = optimized_graph->add_node();
  fused_conv2d->set_name(matched.activation->name());
  fused_conv2d->set_op(kFusedConv2D);
  fused_conv2d->set_device(matched.contraction->device());
  fused_conv2d->add_input(matched.contraction->input(0));       // 0: input
  fused_conv2d->add_input(matched.contraction->input(1));       // 1: filter
  fused_conv2d->add_input(matched.fused_batch_norm->input(1));  // 2: scale
  fused_conv2d->add_input(matched.fused_batch_norm->input(2));  // 3: offset
  fused_conv2d->add_input(matched.fused_batch_norm->input(3));  // 4: mean
  fused_conv2d->add_input(matched.fused_batch_norm->input(4));  // 5: variance

  CopyConv2DAttributes(matched.contraction, fused_conv2d);
  SetFusedOpAttributes(fused_conv2d,
                       {"FusedBatchNorm", matched.activation->op()},
                       /*num_args=*/4, /*epsilon=*/matched.epsilon);

  invalidated_nodes->insert(matched.activation);
  invalidated_nodes->insert(matched.fused_batch_norm);
  invalidated_nodes->insert(matched.contraction);
}

#ifdef INTEL_MKL
void AddFusedContractionNode(
    const ContractionWithBiasAddAndAdd& matched, GraphDef* optimized_graph,
    absl::flat_hash_set<const NodeDef*>* invalidated_nodes) {
  // MKL version only support fusion for Conv2D
  DCHECK(IsConv2D(*matched.contraction));

  NodeDef* fused_conv2d = optimized_graph->add_node();
  fused_conv2d->set_name(matched.add->name());
  fused_conv2d->set_op(kFusedConv2D);
  fused_conv2d->set_device(matched.contraction->device());
  fused_conv2d->add_input(matched.contraction->input(0));  // 0: input
  fused_conv2d->add_input(matched.contraction->input(1));  // 1: filter
  fused_conv2d->add_input(matched.bias_add->input(1));     // 2: bias

  // Add OP has two inputs, one is conv+bias pattern matched previously,
  // the other input to add is fused here.
  fused_conv2d->add_input(matched.add->input(1 - matched.port_id));

  CopyConv2DAttributes(matched.contraction, fused_conv2d);
  SetFusedOpAttributes(fused_conv2d, {"BiasAdd", "Add"}, 2);

  invalidated_nodes->insert(matched.add);
  invalidated_nodes->insert(matched.bias_add);
  invalidated_nodes->insert(matched.contraction);
}

void AddFusedContractionNode(
    const ContractionWithBiasAndAddActivation& matched,
    GraphDef* optimized_graph,
    absl::flat_hash_set<const NodeDef*>* invalidated_nodes) {
  // MKL version only support fusion for Conv2D
  DCHECK(IsConv2D(*matched.contraction));

  NodeDef* fused_conv2d = optimized_graph->add_node();
  fused_conv2d->set_name(matched.activation->name());
  fused_conv2d->set_op(kFusedConv2D);
  fused_conv2d->set_device(matched.contraction->device());
  fused_conv2d->add_input(matched.contraction->input(0));  // 0: input
  fused_conv2d->add_input(matched.contraction->input(1));  // 1: filter
  fused_conv2d->add_input(matched.bias_add->input(1));     // 2: bias

  // Add OP has two inputs, one is conv+bias pattern matched previously,
  // the other input to add is fused here.
  fused_conv2d->add_input(matched.add->input(1 - matched.port_id));

  CopyConv2DAttributes(matched.contraction, fused_conv2d);
  SetFusedOpAttributes(fused_conv2d, {"BiasAdd", "Add", "Relu"}, 2);

  invalidated_nodes->insert(matched.activation);
  invalidated_nodes->insert(matched.add);
  invalidated_nodes->insert(matched.bias_add);
  invalidated_nodes->insert(matched.contraction);
}
#endif

void AddFusedBatchNormExNode(
    const FusedBatchNormEx& matched, GraphDef* optimized_graph,
    absl::flat_hash_set<const NodeDef*>* invalidated_nodes) {
  VLOG(2) << "Fuse " << matched.activation->op() << " with FusedBatchNorm:"
          << " side_input="
          << (matched.side_input ? matched.side_input->name() : "<none>")
          << " activation=" << matched.activation->name()
          << " fused_batch_norm=" << matched.fused_batch_norm->name();

  // Replace FusedBatchNorm with _FusedBatchNormEx + <SideInput> + <Activation>.
  NodeDef* fused_op = optimized_graph->add_node();
  fused_op->set_op(kFusedBatchNormEx);
  fused_op->set_name(matched.fused_batch_norm->name());
  fused_op->set_device(matched.fused_batch_norm->device());

  fused_op->add_input(matched.fused_batch_norm->input(0));  // 0: input
  fused_op->add_input(matched.fused_batch_norm->input(1));  // 1: scale
  fused_op->add_input(matched.fused_batch_norm->input(2));  // 2: offset
  fused_op->add_input(matched.fused_batch_norm->input(3));  // 3: estimated_mean
  fused_op->add_input(matched.fused_batch_norm->input(4));  // 4: estimated_var

  CopyFusedBatchNormAttributes(matched.fused_batch_norm, fused_op);

  auto* attrs = fused_op->mutable_attr();
  SetAttrValue(matched.activation->op(), &(*attrs)["activation_mode"]);

  if (matched.side_input != nullptr) {
    SetAttrValue(1, &(*attrs)["num_side_inputs"]);
    fused_op->add_input(matched.side_input->name());  // 5: side_input
  } else {
    SetAttrValue(0, &(*attrs)["num_side_inputs"]);
  }

  // Turn activation node into Identity node.
  NodeDef* identity_op = optimized_graph->add_node();
  identity_op->set_op("Identity");
  identity_op->set_name(matched.activation->name());
  identity_op->set_device(matched.fused_batch_norm->device());
  identity_op->add_input(matched.fused_batch_norm->name());
  (*identity_op->mutable_attr())["T"] = attrs->at("T");

  // Invalidate all nodes bypassed by this rewrite.
  invalidated_nodes->insert(matched.activation);
  invalidated_nodes->insert(matched.fused_batch_norm);
  if (matched.side_input != nullptr) {
    invalidated_nodes->insert(matched.invalidated);
  }
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
      AddPrefixToNodeName("Const", fused_node.name()), TensorValue(&value),
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

// Check if a node is a candidate to one of the patterns that require inferred
// shapes:
//   (1) Splitting FusedBatchNorm into primitives.
//   (2) Fusing side input and/or activation into FusedBatchNorm.
bool RequiresInferredShapes(const RemapperContext& ctx, const NodeDef& node) {
  // Candidate for a FusedBatchNorm splitting.
  const auto is_batch_norm_candidate = [&]() -> bool {
    if (!IsFusedBatchNorm(node)) return false;
    if (GetDataTypeFromAttr(node, "T") != DT_FLOAT) return false;

    bool is_training = true;
    if (!GetNodeAttr(node, kIsTraining, &is_training).ok()) return false;
    if (is_training) return false;

    return true;
  };

  // Candidate for a FusedBatchNorm fusion.
  const auto is_batch_norm_fusion_candidate = [&]() -> bool {
    if (!IsRelu(node)) return false;

    const auto relu_input_port = GraphView::InputPort(&node, 0);
    const auto relu_fanin = ctx.graph_view.GetRegularFanin(relu_input_port);
    if (!relu_fanin.node) return false;

    if (IsFusedBatchNorm(*relu_fanin.node)) {
      // FusedBatchNorm + Relu.
      return true;

    } else if (IsAdd(*relu_fanin.node)) {
      // FusedBatchNorm + Add + Relu.
      const NodeDef* add = relu_fanin.node;

      const auto add_input_port_0 = GraphView::InputPort(add, 0);
      const auto add_fanin_0 = ctx.graph_view.GetRegularFanin(add_input_port_0);
      if (IsFusedBatchNorm(*add_fanin_0.node)) return true;

      const auto add_input_port_1 = GraphView::InputPort(add, 1);
      const auto add_fanin_1 = ctx.graph_view.GetRegularFanin(add_input_port_1);
      if (IsFusedBatchNorm(*add_fanin_1.node)) return true;
    }

    return false;
  };

  return is_batch_norm_candidate() || is_batch_norm_fusion_candidate();
}

}  // namespace

Status Remapper::Optimize(Cluster* /*cluster*/, const GrapplerItem& item,
                          GraphDef* optimized_graph) {
  // Supported graph patterns.
  // clang-format off
  FusedBatchNorm                        fused_batch_norm;
  FusedBatchNormEx                      fused_batch_norm_ex;
  ContractionWithBiasAdd                contract_with_bias;
  ContractionWithBiasAddAndActivation   contract_with_bias_and_activation;
#ifndef INTEL_MKL
  ContractionWithBatchNorm              contract_with_batch_norm;
  ContractionWithBatchNormAndActivation contract_with_batch_norm_and_activation;
  ContractionWithSqueezeAndBiasAdd      contract_with_squeeze_and_bias;
#endif  // !INTEL_MKL
#ifdef INTEL_MKL
  ContractionWithBiasAddAndAdd          contract_with_bias_and_add;
  ContractionWithBiasAndAddActivation   contract_with_bias_and_add_activation;
#endif  // INTEL_MKL
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
  // and Activation nodes that were fused into a Conv2D node.
  absl::flat_hash_set<const NodeDef*> invalidated_nodes;

  // _Fused{...} kernels do not have registered gradient function, so we must
  // not perform rewrite if the graph will be differentiated later.
  bool allow_non_differentiable_rewrites =
      item.optimization_options().allow_non_differentiable_rewrites;

  optimized_graph->mutable_node()->Reserve(topo_sorted_item.graph.node_size());
  for (const NodeDef& node : topo_sorted_item.graph.node()) {
    // Check if node was invalidated by one of the previous remaps.
    if (invalidated_nodes.count(&node) > 0) continue;

#ifdef INTEL_MKL
    if (!item.optimization_options().is_eager_mode) {
      // Remap Conv2D+BiasAdd+Add+relu into the _FusedConv2D.
      if (FindContractionWithBiasAndAddActivation(
              ctx, &node, &contract_with_bias_and_add_activation)) {
        AddFusedContractionNode(contract_with_bias_and_add_activation,
                                optimized_graph, &invalidated_nodes);
        continue;
      }

      // Remap Conv2D+BiasAdd+Add into the _FusedConv2D.
      if (FindContractionWithBiasAddAndAdd(ctx, &node,
                                           &contract_with_bias_and_add)) {
        AddFusedContractionNode(contract_with_bias_and_add, optimized_graph,
                                &invalidated_nodes);
        continue;
      }
    }
#endif  //! INTEL_MKL

    // Remap {Conv2D,MatMul}+BiasAdd into the _Fused{Conv2D,MatMul}
    if (allow_non_differentiable_rewrites &&
        FindContractionWithBias(ctx, &node, &contract_with_bias)) {
      AddFusedContractionNode(ctx, contract_with_bias, optimized_graph,
                              &invalidated_nodes);
      continue;
    }

    // Remap {Conv2D,MatMul}+BiasAdd+Activation into the _Fused{Conv2D,MatMul}.
    if (allow_non_differentiable_rewrites &&
        FindContractionWithBiasAndActivation(
            ctx, &node, &contract_with_bias_and_activation)) {
      AddFusedContractionNode(ctx, contract_with_bias_and_activation,
                              optimized_graph, &invalidated_nodes);
      continue;
    }

// NOTE: We can only fuse BatchNorm into Conv2D nodes. In theory we can do
// it for MatMul as well, but in practice this pattern does not appear in
// real Tensorflow graphs.

// TODO(penporn):
// Remove this once TF-MKL supports _FusedConv2D with these operations.
#ifndef INTEL_MKL
    // Remap Conv2D+Squeeze+BiasAdd into the _FusedConv2D+Squeeze.
    if (allow_non_differentiable_rewrites &&
        FindConv2DWithSqueezeAndBias(ctx, &node,
                                     &contract_with_squeeze_and_bias)) {
      AddFusedConv2DNode(ctx, contract_with_squeeze_and_bias, optimized_graph,
                         &invalidated_nodes);
      continue;
    }

    // Remap Conv2D+FusedBatchNorm into the _FusedConv2D;
    if (allow_non_differentiable_rewrites &&
        FindConv2DWithBatchNorm(ctx, &node, &contract_with_batch_norm)) {
      AddFusedConv2DNode(contract_with_batch_norm, optimized_graph,
                         &invalidated_nodes);
      continue;
    }

    // Remap Conv2D+FusedBatchNorm+Activation into the _FusedConv2D;
    if (allow_non_differentiable_rewrites &&
        FindConv2DWithBatchNormAndActivation(
            ctx, &node, &contract_with_batch_norm_and_activation)) {
      AddFusedConv2DNode(contract_with_batch_norm_and_activation,
                         optimized_graph, &invalidated_nodes);
      continue;
    }
#endif  // !INTEL_MKL

    // Infer properties lazily in case they are not needed.
    if (!ctx.inferred_graph_properties && RequiresInferredShapes(ctx, node)) {
      const bool assume_valid_feeds = opt_level_ == RewriterConfig::AGGRESSIVE;
      // TODO(rmlarsen): Get rid of tensor value copies.
      TF_RETURN_IF_ERROR(ctx.graph_properties.InferStatically(
          assume_valid_feeds,
          /*aggressive_shape_inference=*/false,
          /*include_input_tensor_values=*/true,
          /*include_output_tensor_values=*/false));
      ctx.inferred_graph_properties = true;
    }

    // Remap FusedBatchNorm+<SideInput>+<Activation> into the _FusedBatchNormEx.
    if (allow_non_differentiable_rewrites &&
        FindFusedBatchNormEx(ctx, &node, &fused_batch_norm_ex)) {
      AddFusedBatchNormExNode(fused_batch_norm_ex, optimized_graph,
                              &invalidated_nodes);
      continue;
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
