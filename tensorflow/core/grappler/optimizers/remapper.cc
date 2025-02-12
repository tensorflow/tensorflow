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

#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/grappler/utils/pattern_utils.h"
#include "tensorflow/core/grappler/utils/symbolic_shapes.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tsl/platform/errors.h"
#ifdef INTEL_MKL
#include "tensorflow/core/util/mkl_heuristics.h"
#endif  // INTEL_MKL
#include "tensorflow/core/util/util.h"

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
// DepthwiseConv2dNative + ... -> _FusedDepthwiseConv2dNative:
//   (1) DepthwiseConv2dNative + BiasAdd + <Activation>
//
// FusedBatchNorm[$is_training] + ... -> _FusedBatchNormEx[$is_training]
//   (1) FusedBatchNorm + <Activation>
//   (2) FusedBatchNorm + SideInput + <Activation>
//
// Sigmoid + Mul -> _MklSwish  // This fusion only works on Intel CPU.
//
//
// In all cases, the supported activation functions are Relu, Relu6, and Elu.
//
// Both Conv2D and MatMul implemented as Tensor contraction (on CPU), so all the
// patterns are "ContractionWith...".
//
// _FusedConv2D/_FusedConv3D + <Activation> -> _FusedConv2D/_FusedConv3D
// Supported Activations: LeakyRelu, Mish

namespace {

constexpr char kFusedConv2D[] = "_FusedConv2D";
constexpr char kFusedConv3D[] = "_FusedConv3D";
constexpr char kFusedMatMul[] = "_FusedMatMul";
constexpr char kFusedDepthwiseConv2dNative[] = "_FusedDepthwiseConv2dNative";
constexpr char kFusedBatchNormEx[] = "_FusedBatchNormEx";
constexpr char kFusedBatchNormGradEx[] = "_FusedBatchNormGradEx";
constexpr char kTensorToHashBucket[] = "_TensorToHashBucketFast";
constexpr char kLeakyRelu[] = "LeakyRelu";
constexpr char kMklFusedMish[] = "_MklFusedMish";
constexpr char kRelu[] = "Relu";
constexpr char kRelu6[] = "Relu6";
constexpr char kElu[] = "Elu";

constexpr char kDataFormat[] = "data_format";
constexpr char kIsTraining[] = "is_training";

constexpr char kWidth[] = "width";
constexpr char kFill[] = "fill";

constexpr int kMissingIndex = -1;

struct RemapperContext {
  explicit RemapperContext(GrapplerItem* item, absl::Status* status,
                           RewriterConfig::CpuLayout cpu_layout_conversion,
                           bool xla_auto_clustering_on,
                           bool xla_cpu_jit_disable_fusion)
      : nodes_to_preserve(item->NodesToPreserve()),
        graph_view(&item->graph, status),
        graph_properties(*item),
        inferred_graph_properties(false),
        cpu_layout_conversion(cpu_layout_conversion),
        xla_auto_clustering_on(xla_auto_clustering_on),
        xla_cpu_jit_disable_fusion(xla_cpu_jit_disable_fusion) {}

  std::unordered_set<string> nodes_to_preserve;
  utils::MutableGraphView graph_view;
  GraphProperties graph_properties;
  bool inferred_graph_properties;
  RewriterConfig::CpuLayout cpu_layout_conversion;
  bool xla_auto_clustering_on;
  bool xla_cpu_jit_disable_fusion;
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

// FusedBatchNormGrad with fused side output and/or activation.
struct FusedBatchNormGradEx {
  int fused_batch_norm_grad = kMissingIndex;
  int activation_grad = kMissingIndex;
  int side_input_grad = kMissingIndex;
  // Add node of the forward pass to access its "offset" input.
  int fwd_fused_batch_norm = kMissingIndex;
};

// TensorToHashBucket that can be replaced with AsString + StringToHashBucket.
// We also include the fanin node of AsString ("pre_as_string") to determine the
// device.
struct TensorToHashBucket {
  TensorToHashBucket() = default;
  explicit TensorToHashBucket(int op1, int op2, int op3)
      : pre_as_string(op1), as_string(op2), string_to_hash_bucket(op3) {}

  int pre_as_string = kMissingIndex;
  int as_string = kMissingIndex;
  int string_to_hash_bucket = kMissingIndex;
};

// Pad followed by Conv3D/FusedConv3D
struct PadWithConv3D {
  PadWithConv3D() = default;
  PadWithConv3D(int contraction_idx, int pad_idx, int padding_const_idx)
      : contraction_idx(contraction_idx),
        pad_idx(pad_idx),
        padding_const_idx(padding_const_idx) {}

  int contraction_idx = kMissingIndex;
  int pad_idx = kMissingIndex;
  int padding_const_idx = kMissingIndex;
};

// Contraction node followed by a BiasAdd.
struct ContractionWithBiasAdd {
  ContractionWithBiasAdd() = default;
  ContractionWithBiasAdd(int contraction, int bias_add, int bias_port)
      : contraction(contraction), bias_add(bias_add), bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int bias_port = 1;
};

// Contraction node followed by Activation
struct ContractionWithActivation {
  ContractionWithActivation() = default;
  ContractionWithActivation(int contraction, int activation)
      : contraction(contraction), activation(activation) {}

  int contraction = kMissingIndex;
  int activation = kMissingIndex;
};

// Contraction node followed by a BiasAdd and Activation.
struct ContractionWithBiasAddAndActivation {
  ContractionWithBiasAddAndActivation() = default;
  ContractionWithBiasAddAndActivation(int contraction, int bias_add,
                                      int activation, int bias_port)
      : contraction(contraction),
        bias_add(bias_add),
        activation(activation),
        bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int activation = kMissingIndex;
  int bias_port = 1;
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

// Contraction node followed by a BiasAdd and Add.
struct ContractionWithBiasAddAndAdd {
  ContractionWithBiasAddAndAdd() = default;
  ContractionWithBiasAddAndAdd(int contraction, int bias_add, int add,
                               int port_id, int bias_port)
      : contraction(contraction),
        bias_add(bias_add),
        add(add),
        port_id(port_id),
        bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int add = kMissingIndex;
  int port_id = 0;
  int bias_port = 1;
};

// Contraction node followed by a BiasAdd, Add and Relu.
// Plus Tanh and Sigmoid for MatMul in MKL
struct ContractionWithBiasAndAddActivation {
  ContractionWithBiasAndAddActivation() = default;
  ContractionWithBiasAndAddActivation(int contraction, int bias_add, int add,
                                      int port_id, int activation,
                                      int bias_port)
      : contraction(contraction),
        bias_add(bias_add),
        add(add),
        port_id(port_id),
        activation(activation),
        bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int add = kMissingIndex;
  int port_id = 0;
  int activation = kMissingIndex;
  int bias_port = 1;
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

bool IsCpuCompatibleDataType(const NodeDef* contraction,
                             const string& type_attr = "T") {
  DataType dtype = GetDataTypeFromAttr(*contraction, type_attr);
  // Stock TF without oneDNN build will always be `false`.
  bool is_one_dnn_enabled = IsMKLEnabled();

  if (is_one_dnn_enabled) {
    // Currently, oneDNN based fused-kernel does not support transpose_a on
    // MatMul. Since bfloat16 precision fused-kernel is only enabled through
    // oneDNN, the fusion is disabled here. Float32 and float16 precisions,
    // however, will use the Eigen library based kernel in case of transpose_a
    // since the mkl_layout_pass will not rewrite for transpose_a. So for
    // float32 and float16 precisions, the fusion is enabled for transpose_a.
    bool is_supported_matmul = false;
    if (IsMatMul(*contraction)) {
      is_supported_matmul = (dtype == DT_BFLOAT16)
                                ? contraction->attr().contains("transpose_a") &&
                                      !contraction->attr().at("transpose_a").b()
                                : true;
    }
    return ((IsConv2D(*contraction) || IsDepthwiseConv2dNative(*contraction) ||
             IsConv3D(*contraction) || IsAnyBatchMatMul(*contraction) ||
             is_supported_matmul) &&
            IsDataTypeSupportedByOneDNNOnThisCPU(dtype));
  }
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
  if (IsConv2D(*contraction) || IsMatMul(*contraction)) {
    return dtype == DT_FLOAT || dtype == DT_HALF;
  } else {
    return false;
  }
}

bool IsCpuCompatibleDataFormat(const RemapperContext& ctx,
                               const NodeDef* conv_node) {
  const string& data_format = conv_node->attr().at(kDataFormat).s();
  if (IsConv2D(*conv_node)) {
    return data_format == "NHWC" || (IsMKLEnabled() && data_format == "NCHW") ||
           (ctx.cpu_layout_conversion == RewriterConfig::NHWC_TO_NCHW &&
            data_format == "NCHW");
  } else if (IsConv3D(*conv_node)) {
    return data_format == "NDHWC" || (IsMKLEnabled() && data_format == "NCDHW");
  } else {
    return false;
  }
}

bool BlasLtMatmulEnabled() {
  static bool is_enabled = [] {
    bool is_enabled = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar(
        "TF_USE_CUBLASLT", /*default_val=*/false, &is_enabled));
    return is_enabled;
  }();
  return is_enabled;
}

bool IsGpuCompatibleDataFormat(const RemapperContext& ctx,
                               const NodeDef* conv2d) {
  DCHECK(IsConv2D(*conv2d)) << "Expected Conv2D op";
  const string& data_format = conv2d->attr().at(kDataFormat).s();
  return data_format == "NHWC" || data_format == "NCHW";
}

bool IsCpuCompatibleConv2D(const RemapperContext& ctx, const NodeDef* conv2d) {
  DCHECK(IsConv2D(*conv2d)) << "Expected Conv2D op";
  return NodeIsOnCpu(conv2d) && IsCpuCompatibleDataType(conv2d) &&
         IsCpuCompatibleDataFormat(ctx, conv2d);
}

bool IsCpuCompatibleConv3D(const RemapperContext& ctx, const NodeDef* conv3d) {
  DCHECK(IsConv3D(*conv3d)) << "Expected Conv3D op";
  return NodeIsOnCpu(conv3d) && IsCpuCompatibleDataType(conv3d) &&
         IsCpuCompatibleDataFormat(ctx, conv3d);
}

bool IsGpuCompatibleConv2D(const RemapperContext& ctx, const NodeDef* conv2d,
                           const NodeDef* activation) {
  DCHECK(IsConv2D(*conv2d)) << "Expected Conv2D op";
  if (IsRelu(*activation)) {
    return NodeIsOnGpu(conv2d) && IsGpuCompatibleDataType(conv2d) &&
           IsGpuCompatibleDataFormat(ctx, conv2d);
  } else if (IsRelu6(*activation) || IsElu(*activation) ||
             IsLeakyRelu(*activation)) {
    DataType dtype = GetDataTypeFromAttr(*conv2d, "T");
    const string& data_format = conv2d->attr().at(kDataFormat).s();
    return NodeIsOnGpu(conv2d) && dtype == DT_HALF && data_format == "NHWC";
  }
  return false;
}

bool IsGpuCompatibleMatMul(const RemapperContext& ctx, const NodeDef* matmul,
                           const NodeDef* activation) {
  DCHECK(IsMatMul(*matmul)) << "Expected MatMul op";
  if (activation == nullptr || IsRelu(*activation)) {
    return BlasLtMatmulEnabled() && NodeIsOnGpu(matmul) &&
           IsGpuCompatibleDataType(matmul);
  } else if (IsTanh(*activation) || IsSigmoid(*activation)) {
    DataType dtype = GetDataTypeFromAttr(*matmul, "T");
    return NodeIsOnGpu(matmul) && dtype == DT_HALF;
  }
  return false;
}

bool IsCpuCompatibleMatMul(const RemapperContext& ctx, const NodeDef* matmul) {
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
  // Disable fusions on CPU when XLA JIT compilation enabled.
  if (ctx.xla_cpu_jit_disable_fusion) return false;

  const NodeDef& node = ctx.graph_view.graph()->node(matched.contraction);
  if (IsConv2D(node)) {
    return IsCpuCompatibleConv2D(ctx, &node);
  } else if (IsDepthwiseConv2dNative(node)) {
    return (IsMKLEnabled() && IsCpuCompatibleDepthwiseConv2dNative(&node));
  } else if (IsMatMul(node)) {
    return IsCpuCompatibleMatMul(ctx, &node);
  } else if (IsConv3D(node)) {
    return (IsMKLEnabled() && IsCpuCompatibleConv3D(ctx, &node));
  } else {
    return false;
  }
}

bool RuntimeFusionEnabled(const Cluster* cluster) {
  static bool is_enabled = [&] {
#if CUDNN_VERSION >= 8400
    // Cudnn runtime fusion feature is recommended for Ampere GPUs or later.
    // For pre-Ampere GPUs, the overhead of runtime compilation would be very
    // large and there are more limitations of supported cases.
    if (!cluster) return false;
    auto devices = cluster->GetDevices();
    int num_gpus = 0;
    int num_ampere = 0;
    for (const auto& d : devices) {
      if (d.second.type() == "GPU") {
        num_gpus++;
        auto cc_it = d.second.environment().find("architecture");
        if (cc_it != d.second.environment().end()) {
          double compute_capability = 0.0;
          if (absl::SimpleAtod(cc_it->second, &compute_capability) &&
              compute_capability >= 8.0) {
            num_ampere++;
          }
        }
      }
    }
    bool runtime_fusion_enabled = CudnnUseRuntimeFusion() &&
                                  CudnnUseFrontend() && num_gpus > 0 &&
                                  num_gpus == num_ampere;

    if (CudnnUseRuntimeFusion() && !runtime_fusion_enabled) {
      VLOG(1) << "Enabling Cudnn with runtime compilation requires the "
              << "Cudnn frontend and Ampere GPUs or later, but we got "
              << "Cudnn frontend is "
              << (CudnnUseFrontend() ? "enabled" : "disabled") << " and "
              << num_ampere << " Ampere GPU(s) out of total " << num_gpus
              << " GPU(s)";
    }

    return runtime_fusion_enabled;
#else
    return false;
#endif
  }();
  return is_enabled;
}

bool IsSupportedActivation(const NodeDef& node, const Cluster* cluster) {
  bool is_default_supported =
      IsRelu(node) || IsRelu6(node) || IsElu(node) || IsLeakyRelu(node);
  bool is_device_specific = (IsMKLEnabled() || RuntimeFusionEnabled(cluster)) &&
                            (IsTanh(node) || IsSigmoid(node));
  return (is_default_supported || is_device_specific);
}

// Checks if we can rewrite a pattern to the `_FusedConv2D` on GPU device.
bool IsGpuCompatible(const RemapperContext& ctx,
                     const ContractionWithBiasAddAndActivation& matched,
                     const Cluster* cluster) {
#if TENSORFLOW_USE_ROCM
  // TODO: add a hipblaslt pathway
  return false;
#endif
  // The TF->XLA bridge does not support `_Fused[Conv2D|MatMul]` so we avoid
  // creating this op. Furthermore, XLA already does this fusion internally so
  // there is no true benefit from doing this optimization if XLA is going to
  // compile the unfused operations anyway.
  if (ctx.xla_auto_clustering_on) return false;

  const GraphDef* graph = ctx.graph_view.graph();

  // We rely on cuDNN for fused convolution and cublasLt for fused matmul.
  const NodeDef& activation_node = graph->node(matched.activation);
  if (!IsSupportedActivation(activation_node, cluster)) return false;

  const NodeDef& contraction_node = graph->node(matched.contraction);
  if (IsConv2D(contraction_node)) {
    const std::vector<OpInfo::TensorProperties>& input_props =
        ctx.graph_properties.GetInputProperties(contraction_node.name());
    const TensorShapeProto& filter_shape =
        input_props.size() >= 2 ? input_props[1].shape() : TensorShapeProto();

    // FusedConv2D on GPU with 1x1 convolution is marginally faster than
    // in-graph computation in micro benchmarks (see kernels/conv_ops_test.cc),
    // and significantly slower in large scale benchmarks.
    bool is_spatial_conv = Rank(filter_shape) == 4 &&          //
                           IsKnown(filter_shape.dim(0)) &&     //
                           IsKnown(filter_shape.dim(1)) &&     //
                           filter_shape.dim(0).size() != 1 &&  //
                           filter_shape.dim(1).size() != 1;

    // FusedConv2D on GPU will use cuDNN static kernels when the activation is
    // Relu. For other activations, it will rely on cuDNN runtime funsion
    // kernels which require 32-bit aligned data access. Here, we check if the
    // in and out channels of filter are even numbers.
    bool valid_channels = Rank(filter_shape) == 4 &&              //
                          IsKnown(filter_shape.dim(2)) &&         //
                          IsKnown(filter_shape.dim(3)) &&         //
                          filter_shape.dim(2).size() % 2 == 0 &&  //
                          filter_shape.dim(3).size() % 2 == 0;
    return is_spatial_conv &&
           (IsRelu(activation_node) ||
            (RuntimeFusionEnabled(cluster) && valid_channels)) &&
           IsGpuCompatibleConv2D(ctx, &contraction_node, &activation_node);
  } else if (IsMatMul(contraction_node)) {
    const std::vector<OpInfo::TensorProperties>& input_props =
        ctx.graph_properties.GetInputProperties(contraction_node.name());
    const TensorShapeProto& a_shape =
        !input_props.empty() ? input_props[0].shape() : TensorShapeProto();
    const TensorShapeProto& b_shape =
        !input_props.empty() ? input_props[1].shape() : TensorShapeProto();
    // FusedMatMul on GPU will use cublasLt when the activation is Relu. For
    // other activations, it will rely on cuDNN runtime funsion kernels which
    // require 32-bit aligned data access. Here, we check if the leading dims of
    // input matrices are even numbers.
    bool valid_dims = Rank(a_shape) == 2 && Rank(b_shape) == 2 &&
                      IsKnown(a_shape.dim(1)) &&         //
                      IsKnown(b_shape.dim(1)) &&         //
                      a_shape.dim(1).size() % 2 == 0 &&  //
                      b_shape.dim(1).size() % 2 == 0;

    return (IsRelu(activation_node) ||
            (RuntimeFusionEnabled(cluster) && valid_dims)) &&
           IsGpuCompatibleMatMul(ctx, &contraction_node, &activation_node);
  }

  return false;
}

// Checks if we can rewrite a pattern to the `_FusedMatMul` on GPU device.
bool IsGpuCompatible(const RemapperContext& ctx,
                     const ContractionWithBiasAdd& matched,
                     const Cluster* cluster) {
#if TENSORFLOW_USE_ROCM && !TF_HIPBLASLT
  // ROCm does not support _FusedMatMul
  return false;
#endif
  // The TF->XLA bridge does not support `_FusedMatMul` so we avoid creating
  // this op. Furthermore, XLA already does this fusion internally so there
  // is no true benefit from doing this optimization if XLA is going to compile
  // the unfused operations anyway.
  if (ctx.xla_auto_clustering_on) return false;

  const GraphDef* graph = ctx.graph_view.graph();
  const NodeDef& contraction_node = graph->node(matched.contraction);
  if (!IsMatMul(contraction_node)) return false;

  return IsGpuCompatibleMatMul(ctx, &contraction_node, nullptr);
}

bool IsGpuCompatible(const RemapperContext& ctx,
                     const ContractionWithSqueezeAndBiasAdd& matched,
                     const Cluster* cluster) {
  return false;
}

// Returns true if the given pattern is supported on the assigned device.
template <typename Pattern>
bool IsDeviceCompatible(const RemapperContext& ctx, Pattern& matched,
                        Cluster* cluster = nullptr) {
  return IsCpuCompatible(ctx, matched) ||
         IsGpuCompatible(ctx, matched, cluster);
}

// Returns the generic op name for an _Mkl activation op
std::string GetActivationName(const std::string& s) {
  if (s == kMklFusedMish) {
    return "Mish";
  } else {
    return s;
  }
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

bool IsConvOrMatMul(const NodeDef& node) {
  return IsConv2D(node) || IsDepthwiseConv2dNative(node) || IsMatMul(node) ||
         IsConv3D(node);
}

// Returns true if one input to Add is Conv2D/3D or DepthwiseConv2dNative or
// MatMul, and the other input is semantically equivalent to BiasAdd.
bool IsBiasSemanticAdd(const RemapperContext& ctx,
                       const utils::MutableNodeView& node_view,
                       int& bias_port) {
  if (!IsMKLEnabled()) return false;

  const auto* node_def = node_view.node();
  if (!NodeIsOnCpu(node_def)) return false;
  if (!IsAdd(*node_def) || node_view.NumRegularFanins() != 2) return false;

  const auto& props = ctx.graph_properties.GetInputProperties(node_def->name());
  if (props.size() < 2) return false;

  const auto& regular_fanin_0 = node_view.GetRegularFanin(0);
  const auto* node_view_0 = regular_fanin_0.node_view();
  const auto* node_def_0 = node_view_0->node();
  const auto& regular_fanin_1 = node_view.GetRegularFanin(1);
  const auto* node_view_1 = regular_fanin_1.node_view();
  const auto* node_def_1 = node_view_1->node();

  if (!IsConvOrMatMul(*node_def_0) && !IsConvOrMatMul(*node_def_1))
    return false;

  auto is_channel_last_format = [](const NodeDef& node) -> bool {
    if (node.attr().contains("data_format")) {
      const string data_format = node.attr().at("data_format").s();
      return (data_format == "NHWC" || data_format == "NDHWC");
    }
    return true;
  };

  // Currently supported data formats are NHWC and NDHWC.
  if (!is_channel_last_format(*node_def_0) ||
      !is_channel_last_format(*node_def_1))
    return false;

  const TensorShapeProto& prot0_shape = props[0].shape();
  const TensorShapeProto& prot1_shape = props[1].shape();

  if (prot0_shape.unknown_rank() || prot1_shape.unknown_rank() ||
      prot0_shape.dim_size() < 1 || prot1_shape.dim_size() < 1 ||
      !IsKnown(prot0_shape.dim(prot0_shape.dim_size() - 1)) ||
      !IsKnown(prot1_shape.dim(prot1_shape.dim_size() - 1)))
    return false;

  // Helper function to check Add/AddV2 could be replaced with BiasAdd.
  const auto is_supported_shape =
      [&](const TensorShapeProto& shape,
          const TensorShapeProto& bcast_shape) -> bool {
    int conv_channel_dim;
    conv_channel_dim = shape.dim(shape.dim_size() - 1).size();

    if (shape.dim_size() == 4 && bcast_shape.dim_size() > 4) return false;
    if (shape.dim_size() == 5 && bcast_shape.dim_size() > 5) return false;

    if (shape.dim_size() < 2) return false;
    // Check that the conv node's channel dim is equal to the 1-dim add node's
    // dim
    if (conv_channel_dim != bcast_shape.dim(bcast_shape.dim_size() - 1).size())
      return false;

    // Check that add nodes dims are all 1's except the channel dim
    for (int i = 0; i < bcast_shape.dim_size() - 1; i++) {
      if (1 != bcast_shape.dim(i).size()) return false;
    }
    return true;
  };

  if (ShapesSymbolicallyEqual(prot0_shape, prot1_shape) ||
      !ShapesBroadcastable(prot0_shape, prot1_shape))
    return false;

  if (IsConvOrMatMul(*node_def_0)) {
    bias_port = 1;
    return (is_supported_shape(prot0_shape, prot1_shape));
  } else if (IsConvOrMatMul(*node_def_1)) {
    bias_port = 0;
    return (is_supported_shape(prot1_shape, prot0_shape));
  }
  return false;
}

void AddInputShapesAttr(const RemapperContext& ctx, int node_index) {
  auto mutable_node = ctx.graph_view.graph()->mutable_node(node_index);

  AttrValue attr_input_shape;
  auto tensor_properties =
      ctx.graph_properties.GetInputProperties(mutable_node->name());
  for (const auto& tensor_property : tensor_properties) {
    TensorShapeProto* proto = attr_input_shape.mutable_list()->add_shape();
    *proto = tensor_property.shape();
  }

  if (IsMKLEnabled() && !tensor_properties.empty()) {
    (*mutable_node->mutable_attr())["_input_shapes"] =
        std::move(attr_input_shape);
  }
}

bool FindContractionWithBias(const RemapperContext& ctx, int node_index,
                             ContractionWithBiasAdd* matched,
                             bool check_device_compatible = true) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be a BiasAdd.
  // TODO(lyandy): Forward controls for patterns with control dependencies.
  if (HasControlFaninOrFanout(*node_view)) return false;

  const auto* node_def = node_view->node();
  int bias_port = 1;
  if (!IsBiasAdd(*node_def) && !IsBiasSemanticAdd(ctx, *node_view, bias_port))
    return false;

  // Input to the BiasAdd must be a Conv2D/3D or a MatMul.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(1 - bias_port);
  const auto* contraction_node_view = regular_fanin_0.node_view();
  const auto* contraction_node_def = contraction_node_view->node();

  // Conv2D/3D, MatMul or DepthwiseConv2D
  bool is_contraction = IsConv2D(*contraction_node_def) ||
                        (IsConv3D(*contraction_node_def) && IsMKLEnabled()) ||
                        IsMatMul(*contraction_node_def) ||
                        IsDepthwiseConv2dNative(*contraction_node_def);

#ifdef DNNL_AARCH64_USE_ACL
  if (IsDepthwiseConv2dNative(*contraction_node_def)) is_contraction = false;
#endif

  if (!is_contraction || !HaveSameDataType(node_def, contraction_node_def) ||
      HasControlFaninOrFanout(*contraction_node_view) ||
      !HasAtMostOneFanoutAtPort0(*contraction_node_view) ||
      IsInPreserveSet(ctx, contraction_node_def))
    return false;

  // Check that data type and data format are supported on assigned device.
  const ContractionWithBiasAdd pattern{contraction_node_view->node_index(),
                                       node_index, bias_port};
  if (check_device_compatible && !IsDeviceCompatible(ctx, pattern))
    return false;

  // We successfully found a {Conv2D, MatMul}+BiasAdd pattern.
  *matched = pattern;

  return true;
}

// Fuse _FusedConv{2,3}D with elementwise ops that
// gets fused in the first iteration of remapper
// Currently supports: LeakyRelu, _MklFusedMish
bool FindFusedConvWithFusedActivation(const RemapperContext& ctx,
                                      int node_index,
                                      ContractionWithActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  if (HasControlFaninOrFanout(*node_view)) return false;

  const auto* node_def = node_view->node();

  // Root of the pattern must be on CPU with MKL enabled
  if (!NodeIsOnCpu(node_def) && !IsMKLEnabled()) return false;

  // Root of the pattern must be a LeakyRelu or _MklFusedMish
  if (!IsLeakyRelu(*node_def) && !IsMklFusedMish(*node_def)) return false;

  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* contraction_node_view = regular_fanin_0.node_view();
  const auto* contraction_node_def = contraction_node_view->node();

  // Input to the activation must be a _FusedConv2D or _FusedConv3D
  if (!(contraction_node_def->op() == kFusedConv2D ||
        contraction_node_def->op() == kFusedConv3D))
    return false;

  // Check if any activation is already fused into _FusedConv2D or _FusedConv3D
  auto contraction_fused_ops_list =
      contraction_node_def->attr().at("fused_ops").list().s();
  for (auto it = contraction_fused_ops_list.begin();
       it != contraction_fused_ops_list.end(); it++) {
    if (*it == kLeakyRelu || *it == kMklFusedMish || *it == kRelu ||
        *it == kRelu6 || *it == kElu) {
      return false;
    }
  }

  // We found the pattern
  const ContractionWithActivation pattern{contraction_node_view->node_index(),
                                          node_view->node_index()};

  *matched = pattern;

  return true;
}

bool FindContractionWithBiasAndActivation(
    const RemapperContext& ctx, Cluster* cluster, int node_index,
    ContractionWithBiasAddAndActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be an activation node.
  // TODO(lyandy): Forward controls for patterns with control dependencies.
  if (HasControlFaninOrFanout(*node_view)) return false;

  const auto* node_def = node_view->node();
  if (!IsSupportedActivation(*node_def, cluster)) return false;

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

  // Get the contraction node
  const auto* contraction_node_view =
      bias_add_node_view->GetRegularFanin(1 - base.bias_port).node_view();
  const auto* contraction_node_def = contraction_node_view->node();

  // Currently, only matmul + bias + (tanh or Sigmoid) is enabled
  if (!IsMatMul(*contraction_node_def) &&
      (IsTanh(*node_def) || IsSigmoid(*node_def)))
    return false;

  // Currently, only (conv | matmul) + bias + leakyrelu is enabled
  if (!(IsConv2D(*contraction_node_def) || IsMatMul(*contraction_node_def) ||
        (IsConv3D(*contraction_node_def) && IsMKLEnabled())) &&
      IsLeakyRelu(*node_def))
    return false;

  // Check that data type and data format are supported on assigned device.
  const ContractionWithBiasAddAndActivation pattern{
      base.contraction, base.bias_add, node_index, base.bias_port};
  if (!IsDeviceCompatible(ctx, pattern, cluster)) return false;

  // We successfully found a {Conv2D, MatMul}+BiasAdd+Activation pattern.
  *matched = pattern;

  return true;
}

bool FindConvWithSqueezeAndBias(const RemapperContext& ctx, int node_index,
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

  // Input to the Squeeze must be a Conv2D/3D.
  if (squeeze_node_view->NumRegularFanins() < 1) return false;
  const auto& squeeze_regular_fanin_0 = squeeze_node_view->GetRegularFanin(0);
  const auto* conv_node_view = squeeze_regular_fanin_0.node_view();
  const auto* conv_node_def = conv_node_view->node();

  if (!(IsConv2D(*conv_node_def) ||
        (IsConv3D(*conv_node_def) && IsMKLEnabled())) ||
      !HaveSameDataType(node_def, conv_node_def, "T") ||
      HasControlFaninOrFanout(*conv_node_view) ||
      !HasAtMostOneFanoutAtPort0(*conv_node_view) ||
      IsInPreserveSet(ctx, conv_node_def))
    return false;

  // Squeeze must not squeeze output channel dimension.
  std::vector<int32> dims;
  if (!TryGetNodeAttr(*squeeze_node_def, "squeeze_dims", &dims)) return false;
  for (auto dim : dims) {
    if ((dim == 3 && IsConv2D(*conv_node_def)) ||
        (dim == 4 && IsConv3D(*conv_node_def)))
      return false;
  }

  // Check that data type and data format are supported on assigned device.
  const ContractionWithSqueezeAndBiasAdd pattern{
      conv_node_view->node_index(), squeeze_node_view->node_index(),
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
  // Conv2D + FusedBatchNormV2/V3 fusion is currently supported only for fp32.
  // TODO(intel-tf): Enable the fusion for bf16 and fp16.
  bool dtypeU_is_float = HasDataType(node_def, DT_FLOAT, "U");
  bool dtypeT_is_bf16 = HasDataType(node_def, DT_BFLOAT16, "T");
  bool dtypeT_is_mkl_fp16 =
      IsMKLEnabled() && HasDataType(node_def, DT_HALF, "T");
  if (node_view->GetOp() != "FusedBatchNorm" &&
      (!dtypeU_is_float || dtypeT_is_bf16 || dtypeT_is_mkl_fp16)) {
    return false;
  }

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

  // Disable fusions on CPU when XLA JIT compilation enabled.
  if (NodeIsOnCpu(conv2d_node_def) && ctx.xla_cpu_jit_disable_fusion) {
    return false;
  }

  if (!IsConv2D(*conv2d_node_def) || !NodeIsOnCpu(conv2d_node_def) ||
      !HaveSameDataType(node_def, conv2d_node_def) ||
      !IsCpuCompatibleDataType(conv2d_node_def) ||
      !IsCpuCompatibleDataFormat(ctx, conv2d_node_def) ||
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
  if (!IsSupportedActivation(*node_def, nullptr)) return false;

  // Need to test and enable in Kernel Op before enabling
  // this activation TODO(intel-tf)
  if (IsSigmoid(*node_def)) return false;

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

// As AddN has multiple inputs, this function tries to find Conv2D + Bias
// pattern in specific input port.
bool FindContractionWithBiasInPort(const RemapperContext& ctx,
                                   const utils::MutableNodeView& add_node_view,
                                   const NodeDef& add_node_def, int port_id,
                                   ContractionWithBiasAdd* base,
                                   const int allowed_fanouts = 1) {
  // Input to AddN must match ContractionWithBiasAdd pattern.
  if (add_node_view.NumRegularFanins() < port_id + 1) return false;
  const auto& bias_add_node_view =
      add_node_view.GetRegularFanin(port_id).node_view();
  if (bias_add_node_view == nullptr) return false;
  const auto* bias_add_node_def = bias_add_node_view->node();

  if (!FindContractionWithBias(ctx, bias_add_node_view->node_index(), base,
                               /*check_device_compatible=*/false))
    return false;
  if (bias_add_node_view->GetRegularFanout(0).size() > allowed_fanouts ||
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

bool FindPadWithConv3D(const RemapperContext& ctx, int node_index,
                       PadWithConv3D* matched) {
  if (!IsMKLEnabled()) return false;
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  // The optimization is only for CPU
  if (!NodeIsOnCpu(node_def)) return false;
  // Root of the pattern must be a Conv3D or _FusedConv3D
  if (!(IsConv3D(*node_def) || node_def->op() == kFusedConv3D)) return false;
  if (!(HasDataType(node_def, DT_FLOAT) || HasDataType(node_def, DT_BFLOAT16) ||
        HasDataType(node_def, DT_HALF)))
    return false;

  // Input to Conv3D/_FusedConv3D must be a Pad
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* pad_node_view = regular_fanin_0.node_view();
  const auto* pad_node_def = pad_node_view->node();
  const auto& padding_const = pad_node_view->GetRegularFanin(1);
  const auto* padding_const_node_view = padding_const.node_view();

  if (!(pad_node_def->op() == "Pad") ||
      !HaveSameDataType(node_def, pad_node_def))
    return false;
  const PadWithConv3D pattern{node_view->node_index(),
                              pad_node_view->node_index(),
                              padding_const_node_view->node_index()};

  // Successfully found a Pad+{Conv3D, _FusedConv3D} pattern.
  *matched = pattern;
  return true;
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

  if (!NodeIsOnCpu(node_def)) return false;

  // MKL AddN ops only support float32, bfloat16 and float16 data types.
  if (!(HasDataType(node_def, DT_FLOAT) || HasDataType(node_def, DT_BFLOAT16) ||
        HasDataType(node_def, DT_HALF)))
    return false;

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

  // We successfully found a {Conv2D,Conv3D}+BiasAdd+{AddN,Add} pattern.
  matched->contraction = base.contraction;
  matched->bias_add = base.bias_add;
  matched->add = node_view.node_index();
  matched->bias_port = base.bias_port;

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
  if (!IsSupportedActivation(*node_def, nullptr)) return false;

  if (!NodeIsOnCpu(node_def)) return false;

  // Currently, Contraction + Bias + Add + Tanh pattern is not supported
  if (IsTanh(*node_def)) return false;

  // Need to test and enable in Kernel Op before enabling
  // this activation. TODO(intel-tf)
  if (IsSigmoid(*node_def)) return false;

  // MKL activation op only supports float32, bfloat16 and float16 data types.
  if (!(HasDataType(node_def, DT_FLOAT) || HasDataType(node_def, DT_BFLOAT16) ||
        HasDataType(node_def, DT_HALF)))
    return false;

  // And input to activation must match ContractionWithBiasAddAndAdd pattern.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* add_node_view = regular_fanin_0.node_view();

  ContractionWithBiasAddAndAdd base;

  if (!FindContractionWithBiasAddAndAdd(ctx, *add_node_view, &base)) {
    return false;
  }

  // Get the contraction node
  const auto* bias_add_node_view =
      add_node_view->GetRegularFanin(base.port_id).node_view();
  const auto* contraction_node_view =
      bias_add_node_view->GetRegularFanin(0).node_view();
  const auto* contraction_node_def = contraction_node_view->node();

  // Currently, only conv + bias + add + leakyrelu is enabled
  if (!(IsConv2D(*contraction_node_def) || IsConv3D(*contraction_node_def)) &&
      IsLeakyRelu(*node_def))
    return false;
  // Conv3D fusion is available with oneDNN enabled
  if (IsConv3D(*contraction_node_def) && !IsMKLEnabled()) return false;

  // We successfully found a Conv2D+BiasAdd+AddN+activation pattern
  // or Conv3D+BiasAdd+AddN+activation pattern
  const ContractionWithBiasAndAddActivation pattern{
      base.contraction, base.bias_add, base.add,
      base.port_id,     node_index,    base.bias_port};
  *matched = pattern;

  return true;
}

bool FindConv2DSwish(RemapperContext* ctx, int node_index,
                     std::map<string, int>* matched_nodes_map,
                     std::set<int>* remove_node_indices) {
  using utils::MatchingDirection;
  using utils::NodeStatus;
  // clang-format off

  //    Fuse Conv2D + BiasAdd/FusedBatchNorm + Sigmoid + Mul to _FuesdConv2D
  //   From Graph                                To Graph
  //   -----------                              ---------
  //    Conv2D
  //      !
  //      V
  //  BiasAdd / FusedBatchNorm/V2/V3
  //      !
  //      V
  //  ---- ----
  //  !       !                    ----->       _FusedConv2D
  //  !       V
  //  !    Sigmoid
  //  !       !
  //  ---   ---
  //     !  !
  //     V  V
  //      Mul
  //      !
  //      V

  utils::OpTypePattern conv2dbiasaddswish_pattern{
    "Mul", "mulToswish", NodeStatus::kReplace,
    {
      { "Sigmoid", "sigmoid", NodeStatus::kRemove,
        {
          { "BiasAdd", "biasadd", NodeStatus::kRemove,
            {
              { "Conv2D", "conv", NodeStatus::kRemove},
              { "*", "bias", NodeStatus::kRemain}
            }
          }
        }
      },
      { "BiasAdd", "biasadd", NodeStatus::kRemove}
    }
  };

  utils::OpTypePattern conv2dbatchnormswish_pattern{
    "Mul", "mulToswish", NodeStatus::kReplace,
    {
      { "Sigmoid", "sigmoid", NodeStatus::kRemove,
        {
          { "FusedBatchNorm", "fusebatchnorm", NodeStatus::kRemove,
            {
              { "Conv2D", "conv", NodeStatus::kRemove},
              { "*", "scale", NodeStatus::kRemain},
              { "*", "offset", NodeStatus::kRemain},
              { "*", "mean", NodeStatus::kRemain},
              { "*", "var", NodeStatus::kRemain}
            }
          }
        }
      },
      { "FusedBatchNorm", "fusebatchnorm", NodeStatus::kRemove}
    }
  };

  utils::OpTypePattern conv2dbatchnormv2swish_pattern{
    "Mul", "mulToswish", NodeStatus::kReplace,
    {
      { "Sigmoid", "sigmoid", NodeStatus::kRemove,
        {
          { "FusedBatchNormV2", "fusebatchnorm", NodeStatus::kRemove,
            {
              { "Conv2D", "conv", NodeStatus::kRemove},
              { "*", "scale", NodeStatus::kRemain},
              { "*", "offset", NodeStatus::kRemain},
              { "*", "mean", NodeStatus::kRemain},
              { "*", "var", NodeStatus::kRemain}
            }
          }
        }
      },
      { "FusedBatchNormV2", "fusebatchnorm", NodeStatus::kRemove}
    }
  };

  utils::OpTypePattern conv2dbatchnormv3swish_pattern{
    "Mul", "mulToswish", NodeStatus::kReplace,
    {
      { "Sigmoid", "sigmoid", NodeStatus::kRemove,
        {
          { "FusedBatchNormV3", "fusebatchnorm", NodeStatus::kRemove,
            {
              { "Conv2D", "conv", NodeStatus::kRemove},
              { "*", "scale", NodeStatus::kRemain},
              { "*", "offset", NodeStatus::kRemain},
              { "*", "mean", NodeStatus::kRemain},
              { "*", "var", NodeStatus::kRemain}
            }
          }
        }
      },
      { "FusedBatchNormV3", "fusebatchnorm", NodeStatus::kRemove}
    }
  };
  // clang-format on
  // check for data types
  auto* mul_node_def = ctx->graph_view.GetNode(node_index)->node();
  if (!(HasDataType(mul_node_def, DT_FLOAT) ||
        HasDataType(mul_node_def, DT_HALF) ||
        HasDataType(mul_node_def, DT_BFLOAT16)))
    return false;

  if (!NodeIsOnCpu(mul_node_def)) return false;
  // Check first if the swish pattern is present
  bool found_op_type_match = false;
  bool is_biasadd_pattern = false;
  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx->graph_view));
  matched_nodes_map->clear();
  remove_node_indices->clear();
  found_op_type_match = graph_matcher.GetMatchedNodes(
      conv2dbiasaddswish_pattern, {}, ctx->graph_view.GetNode(node_index),
      matched_nodes_map, remove_node_indices);
  is_biasadd_pattern = found_op_type_match;

  // If Conv2D + BiasAdd + Sigmoid + Mul Not found , check for FusedBatchNorm
  // pattern
  if (!found_op_type_match) {
    utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
        &(ctx->graph_view));
    matched_nodes_map->clear();
    remove_node_indices->clear();
    found_op_type_match = graph_matcher.GetMatchedNodes(
        conv2dbatchnormswish_pattern, {}, ctx->graph_view.GetNode(node_index),
        matched_nodes_map, remove_node_indices);
  }

  // if above fails check for FusedBatchNormV2 pattern
  if (!found_op_type_match) {
    utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
        &(ctx->graph_view));
    matched_nodes_map->clear();
    remove_node_indices->clear();
    found_op_type_match = graph_matcher.GetMatchedNodes(
        conv2dbatchnormv2swish_pattern, {}, ctx->graph_view.GetNode(node_index),
        matched_nodes_map, remove_node_indices);
  }

  // if above fails check for FusedBatchNormV3 pattern
  if (!found_op_type_match) {
    utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
        &(ctx->graph_view));
    matched_nodes_map->clear();
    remove_node_indices->clear();
    found_op_type_match = graph_matcher.GetMatchedNodes(
        conv2dbatchnormv3swish_pattern, {}, ctx->graph_view.GetNode(node_index),
        matched_nodes_map, remove_node_indices);
  }

  // Check if the Conv2d to be fused is CPU compatible
  if (found_op_type_match) {
    NodeDef* conv2d_node =
        ctx->graph_view.GetNode(matched_nodes_map->at("conv"))->node();
    if (!IsCpuCompatibleConv2D(*ctx, conv2d_node)) return false;
    if (!is_biasadd_pattern) {
      NodeDef* fusedbatchnorm_node =
          ctx->graph_view.GetNode(matched_nodes_map->at("fusebatchnorm"))
              ->node();
      // Check if FusedBatchNorm node is in inference mode
      bool is_training = true;
      if (!TryGetNodeAttr(*fusedbatchnorm_node, kIsTraining, &is_training) ||
          is_training)
        return false;

      if (fusedbatchnorm_node->op() != "FusedBatchNorm" &&
          (!HasDataType(fusedbatchnorm_node, DT_FLOAT, "U") ||
           (HasDataType(fusedbatchnorm_node, DT_FLOAT, "U") &&
            !HasDataType(fusedbatchnorm_node, DT_FLOAT, "T")))) {
        return false;
      }
    }
  }

  return found_op_type_match;
}

inline bool VerifyConstants(RemapperContext* ctx,
                            std::map<string, int>* nodes_map,
                            std::map<string, float>* values_map) {
  using utils::MutableNodeView;

  for (auto it = values_map->begin(); it != values_map->end(); ++it) {
    int node_idx = nodes_map->at(it->first);
    MutableNodeView* node_view = ctx->graph_view.GetNode(node_idx);
    NodeDef* node_def = node_view->node();
    Tensor const_tensor;

    // If node is a Cast, look for Const in fan-ins.
    if (node_def != nullptr && node_def->op() == "Cast") {
      const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
      const auto* regular_node_view = regular_fanin_0.node_view();
      node_def = regular_node_view->node();
    }

    // Verify if the node is a constant.
    if (node_def == nullptr || node_def->op() != "Const" ||
        !const_tensor.FromProto(node_def->attr().at("value").tensor()) ||
        const_tensor.NumElements() != 1) {
      return false;
    }
    DataType dtype = const_tensor.dtype();
    float const_value;
    if (dtype == DT_FLOAT) {
      const_value = const_tensor.flat<float>()(0);
    } else if (dtype == DT_BFLOAT16) {
      const_value = static_cast<float>(const_tensor.flat<bfloat16>()(0));
    } else if (dtype == DT_HALF) {
      const_value = static_cast<float>(const_tensor.flat<Eigen::half>()(0));
    } else {
      return false;
    }
    if (std::abs(const_value - it->second) > 1e-2) return false;
  }
  return true;
}

bool IsMatchedMatMulBiasAddAndGeluExact(
    RemapperContext& ctx, int node_index,
    std::map<string, int>* matched_nodes_map = nullptr,
    std::set<int>* remove_node_indices = nullptr) {
  auto* node_view = ctx.graph_view.GetNode(node_index);
  using utils::MatchingDirection;
  using utils::NodeStatus;
  // clang-format off
  // Pattern 1:
  //    Const: 1/sqrt(2)        Const: 1    Const: 1/2
  //                  \               \         \
  //  * --> BiasAdd --> Mul --> Erf --> AddV2 --> Mul --> Mul
  //        /       \____________________________________/
  //  MatMul
  static utils::OpTypePattern* gelu_exact_pattern = new utils::OpTypePattern
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Mul", "erf_plus_one_times_one_half", NodeStatus::kRemove,
          {
            {"Add|AddV2", "erf_plus_one", NodeStatus::kRemove,
              {
                {"Erf", "erf", NodeStatus::kRemove,
                  {
                    {"Mul", "bias_add_x_sqrt_one_half",
                     NodeStatus::kRemove,
                      {
                        {"BiasAdd", "bias_add", NodeStatus::kRemove},
                        {"Cast|Const", "sqrt_one_half", NodeStatus::kRemain}
                      }
                    }  // Mul: "bias_add_x_sqrt_one_half"
                  }
                },  // Erf: "erf"
                {"Cast|Const", "one", NodeStatus::kRemain}
              }  // Add|AddV2: "erf_plus_one"
            },
            {"Cast|Const", "one_half", NodeStatus::kRemain}
          }
        },  // Mul: "erf_plus_one_times_one_half"
        {"BiasAdd", "bias_add", NodeStatus::kRemove,
          {
            {"MatMul", "matmul", NodeStatus::kRemove},
            {"*", "bias", NodeStatus::kRemain}
          }
        }  // BiasAdd: "bias_add"
      }  // Mul: "output"
    };

  // Pattern 2: Erfc
  //             Const: 1/sqrt(2)       Const: 1/2
  //                            \                \
  //  * --> BiasAdd --> Neg --> Mul --> Erfc --> Mul --> Mul
  //        /       \____________________________________/
  //  MatMul
  static utils::OpTypePattern* gelu_exact_pattern2 = new utils::OpTypePattern
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Mul", "one_half_x_erfc", NodeStatus::kRemove,
          {
            {"Const", "one_half", NodeStatus::kRemain},
            {"Erfc", "erfc", NodeStatus::kRemove,
              {
                {"Mul", "neg_bias_add_x_sqrt_one_half", NodeStatus::kRemove,
                  {
                    {"Const", "sqrt_one_half", NodeStatus::kRemain},
                    {"Neg", "neg", NodeStatus::kRemove,
                      {{"BiasAdd", "bias_add", NodeStatus::kRemove}}
                    },  // Neg: "neg"
                  }
                }  // Mul: "neg_bias_add_x_sqrt_one_half"
              }  // Erfc: "erfc"
            }
          }  // Mul: "one_half_x_erfc"
        },
        {"BiasAdd", "bias_add", NodeStatus::kRemove,
          {
            {"MatMul", "matmul", NodeStatus::kRemove},
            {"*", "bias", NodeStatus::kRemain}
          }
        }  // BiasAdd: "bias_add"
      }
    };  // Mul: "output"


  // Pattern 3:
  //  Cast|Const: 1/sqrt(2)    Cast|Const: 1
  //                  \               \
  //  * --> BiasAdd --> Mul --> Erf --> Add|AddV2 --> Mul
  //      /         \                                 /
  // MatMul           ----------------------------> Mul
  //                                                /
  //                                  Cast|Const: 1/2
  static utils::OpTypePattern* gelu_exact_pattern3 = new utils::OpTypePattern
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Add|AddV2", "erf_plus_one", NodeStatus::kRemove,
          {
            {"Erf", "erf", NodeStatus::kRemove,
              {
                {"Mul", "bias_add_x_sqrt_one_half", NodeStatus::kRemove,
                  {
                    {"BiasAdd", "bias_add", NodeStatus::kRemove},
                    {"Cast|Const", "sqrt_one_half", NodeStatus::kRemain}
                  }
                }  // Mul: "bias_add_x_sqrt_one_half"
              }
            },  // Erf: "erf"
            {"Cast|Const", "one", NodeStatus::kRemain}
          }
        },  // Add|AddV2: "erf_plus_one"
        {"Mul", "erf_plus_one_times_one_half", NodeStatus::kRemove,
          {
            {"BiasAdd", "bias_add", NodeStatus::kRemove,
              {
                {"MatMul", "matmul", NodeStatus::kRemove},
                {"*", "bias", NodeStatus::kRemain}
              }
            },  // BiasAdd: "bias_add"
            {"Cast|Const", "one_half", NodeStatus::kRemain}
          }
        }  // Mul: "erf_plus_one_times_one_half"
      }
    };  // Mul: "output"
  // clang-format on

  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx.graph_view));
  // Find GeluExact
  std::map<string, int> dummy_matched_nodes_map;
  std::set<int> dummy_remove_node_indices;
  if (!matched_nodes_map) matched_nodes_map = &dummy_matched_nodes_map;
  if (!remove_node_indices) remove_node_indices = &dummy_remove_node_indices;
  auto patterns = {gelu_exact_pattern, gelu_exact_pattern2,
                   gelu_exact_pattern3};
  for (auto& pattern : patterns) {
    matched_nodes_map->clear();
    remove_node_indices->clear();
    if (graph_matcher.GetMatchedNodes(*pattern, ctx.nodes_to_preserve,
                                      node_view, matched_nodes_map,
                                      remove_node_indices)) {
      return true;
    }
  }
  return false;
}

// Gelu in python api generates a number of nodes in the graph. Depending on the
// parmeter `approximate={True/False}` different types of ops are generated. We
// distinguish them as `GeluExact` that uses Erf and `GeluApproximate` that
// uses Tanh.
bool FindMatMulBiasAddAndGelu(RemapperContext* ctx, int node_index,
                              const Cluster* cluster,
                              std::map<string, int>* matched_nodes_map,
                              std::set<int>* remove_node_indices,
                              bool* is_gelu_approximate) {
  // Gelu fusion is enabled with oneDNN or cublasLt or cuDNN library.
  if (!IsMKLEnabled() && !BlasLtMatmulEnabled() &&
      !RuntimeFusionEnabled(cluster))
    return false;

  using utils::MatchingDirection;
  using utils::NodeStatus;

  bool found_gelu_exact = false;
  bool found_gelu_approximate = false;

  // Find GeluExact
  matched_nodes_map->clear();
  remove_node_indices->clear();
  found_gelu_exact = IsMatchedMatMulBiasAddAndGeluExact(
      *ctx, node_index, matched_nodes_map, remove_node_indices);
  // Find GeluApproximate
  if (!found_gelu_exact) {
    // Gelu approximate uses Pow(x, 3). On GPU, it is a single Pow() node, but
    // on CPU, it is optimized by arithmetic optimizer as Mul(x, Square(x)) with
    // an arifact of control dependency. So we try to match pattern at second
    // pass of remapper which reccieves _FusedMatMul (MatMul + BiasAdd) with
    // control dependency removed.
    // clang-format off
    utils::OpTypePattern subgraph_gpu =
      {"Mul", "mul", NodeStatus::kRemove,
        {
          {"Pow", "pow", NodeStatus::kRemove,
            {
              {"_FusedMatMul", "matmul", NodeStatus::kRemove},
              {"Const", "three", NodeStatus::kRemain}
            }
          },
          {"Const", "empirical_const", NodeStatus::kRemain}
        }
      };
    utils::OpTypePattern subgraph_cpu =
      {"Mul", "mul", NodeStatus::kRemove,
        {
          {"Mul", "empirical_const_times_matmul", NodeStatus::kRemove,
            {
              {"Const", "empirical_const", NodeStatus::kRemain},
              {"_FusedMatMul", "matmul", NodeStatus::kRemove}
            }
          },
          {"Square", "square", NodeStatus::kRemove,
            {
              {"_FusedMatMul", "matmul", NodeStatus::kRemove}
            }
          }
        }
      };
    // clang-format on

    utils::MutableNodeView* node_view = ctx->graph_view.GetNode(node_index);
    const NodeDef* node_def = node_view->node();
    bool root_on_gpu = NodeIsOnGpu(node_def);
    utils::OpTypePattern* subgraph_pattern =
        root_on_gpu ? &subgraph_gpu : &subgraph_cpu;

    // clang-format off
    utils::OpTypePattern gelu_approximate_pattern =
      {"Mul", "output", NodeStatus::kReplace,
        {
          {"Mul", "tanh_plus_one_times_one_half", NodeStatus::kRemove,
            {
              {"AddV2", "tanh_plus_one", NodeStatus::kRemove,
                {
                  {"Tanh", "tanh", NodeStatus::kRemove,
                    {
                      {"Mul", "matmul_plus_mul_times_square_root_two_over_pi", NodeStatus::kRemove,
                        {
                          {"AddV2", "matmul_plus_mul", NodeStatus::kRemove,
                            {
                              {"_FusedMatMul", "matmul", NodeStatus::kRemove},
                              *subgraph_pattern
                            }
                          },
                          {"Const", "square_root_two_over_pi", NodeStatus::kRemain}
                        }
                      }
                    }
                  },
                  {"Const", "one", NodeStatus::kRemain}
                }
              },
              {"Const", "one_half", NodeStatus::kRemain}
            }
          },
          {"_FusedMatMul", "matmul", NodeStatus::kRemove}
        }
      };
    // clang-format on

    utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
        &(ctx->graph_view));

    // Find GeluApproximate
    matched_nodes_map->clear();
    remove_node_indices->clear();
    found_gelu_approximate = graph_matcher.GetMatchedNodes(
        gelu_approximate_pattern, ctx->nodes_to_preserve, node_view,
        matched_nodes_map, remove_node_indices);
  }

  // Pattern matcher does subgraph matching based on op types only. The matcher
  // also does a sanity check on nodes tagged as `kRemove`, i.e., they do not
  // have any consumer outside the matched nodes. In order to replace the
  // subgraph, we need additional checks, for example, if the key ops have been
  // placed on CPU or GPU, desired data type, const has desired value etc. For
  // the following fusion: MatMul + BiasAdd + Gelu (disintegrated into smaller
  // ops), we check if (i) MatMul op is CpuCompatible or GpuComptible, (ii)
  // const nodes have desired values.
  if (found_gelu_exact) {
    // Check if the MatMul to be fused is device compatible.
    NodeDef* matmul_node =
        ctx->graph_view.GetNode(matched_nodes_map->at("matmul"))->node();

    // Disable fusions on CPU when XLA JIT compilation enabled.
    if (NodeIsOnCpu(matmul_node) && ctx->xla_cpu_jit_disable_fusion) {
      return false;
    }

    DataType matmul_dtype = GetDataTypeFromAttr(*matmul_node, "T");

    bool cpu_ok = IsMKLEnabled() && IsCpuCompatibleMatMul(*ctx, matmul_node);
    // Currently, the fusion is not supported on CPU for transpose_a in the
    // MatMul op.
    cpu_ok = cpu_ok && matmul_node->attr().contains("transpose_a") &&
             !matmul_node->attr().at("transpose_a").b();

    bool gpu_ok = NodeIsOnGpu(matmul_node) && RuntimeFusionEnabled(cluster) &&
                  matmul_dtype == DT_HALF;
    if (!cpu_ok && !gpu_ok) return false;

    // Check if the leading dims of input matrices are even numbers. The CuDNN
    // runtime fusion kernels require 32-bit aligned data loading.
    if (gpu_ok) {
      const std::vector<OpInfo::TensorProperties>& input_props =
          ctx->graph_properties.GetInputProperties(matmul_node->name());
      const TensorShapeProto& a_shape =
          !input_props.empty() ? input_props[0].shape() : TensorShapeProto();
      const TensorShapeProto& b_shape =
          !input_props.empty() ? input_props[1].shape() : TensorShapeProto();
      bool valid_dims = Rank(a_shape) == 2 && Rank(b_shape) == 2 &&
                        IsKnown(a_shape.dim(1)) &&         //
                        IsKnown(b_shape.dim(1)) &&         //
                        a_shape.dim(1).size() % 2 == 0 &&  //
                        b_shape.dim(1).size() % 2 == 0;
      if (!valid_dims) return false;
    }

    // Check if the matched constants have desired values.
    std::map<string, float> values_map = {{"sqrt_one_half", 0.707106},
                                          {"one_half", 0.5}};
    // GeluExact Pattern 2 (Erfc) does not have constant "one".
    if (matched_nodes_map->find("one") != matched_nodes_map->end()) {
      values_map["one"] = 1.0;
    }
    if (!VerifyConstants(ctx, matched_nodes_map, &values_map)) return false;
  } else if (found_gelu_approximate) {
    NodeDef* matmul_node =
        ctx->graph_view.GetNode(matched_nodes_map->at("matmul"))->node();

    // Disable fusions on CPU when XLA JIT compilation enabled.
    if (NodeIsOnCpu(matmul_node) && ctx->xla_cpu_jit_disable_fusion) {
      return false;
    }

    // matmul_node is already the _FusedMatMul and we don't need to check its
    // data type again.
    if (!IsMKLEnabled() && !NodeIsOnGpu(matmul_node)) return false;

    // Currently, the fusion is not supported on CPU for transpose_a in the
    // MatMul op.
    if (NodeIsOnCpu(matmul_node) &&
        matmul_node->attr().contains("transpose_a") &&
        matmul_node->attr().at("transpose_a").b()) {
      return false;
    }

    // Check if _FusedMatMul contains only BiasAdd
    auto fused_ops = matmul_node->attr().at("fused_ops").list().s();
    if (fused_ops.size() == 1) {
      if (fused_ops.at(0) != "BiasAdd") return false;
    } else {
      return false;
    }
    // Check if the matched constants have desired values.
    std::map<string, float> values_map = {{"square_root_two_over_pi", 0.797884},
                                          {"one", 1.0},
                                          {"one_half", 0.5},
                                          {"empirical_const", 0.044715}};
    if (NodeIsOnGpu(matmul_node)) {
      values_map["three"] = 3.0;
    }

    if (!VerifyConstants(ctx, matched_nodes_map, &values_map)) return false;
  } else {
    return false;
  }
  *is_gelu_approximate = found_gelu_approximate ? true : false;
  return (found_gelu_exact || found_gelu_approximate);
}

bool FindMulAndMaximum(RemapperContext* ctx, int node_index,
                       std::map<string, int>* matched_nodes_map,
                       std::set<int>* remove_node_indices, float* alpha) {
  using utils::MatchingDirection;
  using utils::NodeStatus;

  // clang-format off
  // Convert Mul+Maximum to LeakyRelu
  // maximum(x, alpha * x) = LeakyRelu(x)
  utils::OpTypePattern mulmax_pattern{
    "Maximum", "max_to_leakyrelu", NodeStatus::kReplace,
    {
      { "Mul", "mul", NodeStatus::kRemove,
        {
          { "*", "input", NodeStatus::kRemain},
          { "Const|Cast", "alpha", NodeStatus::kRemain}
        }
      },
      { "*", "input", NodeStatus::kRemain}
    }
  };
  // clang-format on
  // Check for allowed datatypes
  auto* max_node_def = ctx->graph_view.GetNode(node_index)->node();
  if (!HasDataType(max_node_def, DT_HALF) &&
      !HasDataType(max_node_def, DT_BFLOAT16) &&
      !HasDataType(max_node_def, DT_FLOAT) &&
      !HasDataType(max_node_def, DT_DOUBLE))
    return false;

  // Current implementation has support only
  // for CPU when oneDNN is enabled.
  // TODO(intel-tf): This will be removed when fully tested with GPU
  if (!NodeIsOnCpu(max_node_def) && !IsMKLEnabled()) return false;

  bool found_op_type_match = false;
  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx->graph_view));
  matched_nodes_map->clear();
  remove_node_indices->clear();

  found_op_type_match = graph_matcher.GetMatchedNodes(
      mulmax_pattern, {}, ctx->graph_view.GetNode(node_index),
      matched_nodes_map, remove_node_indices);

  // Check if the value of alpha >= 0 as required for LeakyRelu
  if (found_op_type_match) {
    const auto* alpha_node_view =
        ctx->graph_view.GetNode(matched_nodes_map->at("alpha"));
    const auto* alpha_node_def = alpha_node_view->node();

    if (alpha_node_def != nullptr && alpha_node_def->op() == "Cast") {
      const auto& regular_fanin_0 = alpha_node_view->GetRegularFanin(0);
      const auto* regular_node_view = regular_fanin_0.node_view();
      alpha_node_def = regular_node_view->node();
    }

    Tensor alpha_tensor;
    // Only fusing if the node is a const scalar
    if (alpha_node_def == nullptr || alpha_node_def->op() != "Const" ||
        !alpha_tensor.FromProto(alpha_node_def->attr().at("value").tensor()) ||
        alpha_tensor.NumElements() != 1) {
      return false;
    }

    DataType dtype = alpha_tensor.dtype();
    float alpha_val;
    if (dtype == DT_FLOAT) {
      alpha_val = alpha_tensor.flat<float>()(0);
    } else if (dtype == DT_BFLOAT16) {
      alpha_val = static_cast<float>(alpha_tensor.flat<bfloat16>()(0));
    } else if (dtype == DT_HALF) {
      alpha_val = static_cast<float>(alpha_tensor.flat<Eigen::half>()(0));
    } else {
      return false;
    }

    if (alpha_val < 0) return false;
    *alpha = alpha_val;
  }
  return found_op_type_match;
}

bool FindSigmoidAndMul(RemapperContext* ctx, int node_index,
                       std::map<string, int>* matched_nodes_map,
                       std::set<int>* remove_node_indices) {
  // Gelu fusion is enabled only with oneDNN library.
  if (!IsMKLEnabled()) return false;

  using utils::MatchingDirection;
  using utils::NodeStatus;
  // clang-format off
  // Convert Sigmoid+Mul to Swish
  // Mul(x, Sigmoid(x)) --> _MklSwish(x)

  utils::OpTypePattern sigmoidmul_pattern{
    "Mul", "mul_to_swish", NodeStatus::kReplace,
    {
      { "Sigmoid", "sigmoid", NodeStatus::kRemove,
        {
          { "*", "input", NodeStatus::kRemain}
        }
      },
      { "*", "input", NodeStatus::kRemain}
    }
  };
  // clang-format on
  // check for data types
  auto* mul_node_def = ctx->graph_view.GetNode(node_index)->node();
  if (!(HasDataType(mul_node_def, DT_FLOAT) ||
        HasDataType(mul_node_def, DT_HALF) ||
        HasDataType(mul_node_def, DT_BFLOAT16)))
    return false;

  if (!NodeIsOnCpu(mul_node_def)) return false;

  bool found_op_type_match = false;
  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx->graph_view));
  matched_nodes_map->clear();
  remove_node_indices->clear();
  found_op_type_match = graph_matcher.GetMatchedNodes(
      sigmoidmul_pattern, {}, ctx->graph_view.GetNode(node_index),
      matched_nodes_map, remove_node_indices);

  if (found_op_type_match) {
    NodeDef* matched_sigmoid_node =
        ctx->graph_view.GetNode(matched_nodes_map->at("sigmoid"))->node();
    auto in_tensor_sigmoid = matched_sigmoid_node->input(0);
    if ((mul_node_def->input(0) != in_tensor_sigmoid) &&
        (mul_node_def->input(1) != in_tensor_sigmoid)) {
      // If the input tensor of Sigmoid doesn't match with either of input
      // tensors of mul return false
      found_op_type_match = false;
    }
  }

  return found_op_type_match;
}

// Find a group of ops that make up an instance/layer normalization pattern
// for fusion
bool IsCommonNormPattern(RemapperContext* ctx, int node_index,
                         std::map<string, int>* matched_nodes_map,
                         std::set<int>* remove_node_indices) {
  using utils::MatchingDirection;
  using utils::NodeStatus;
  // clang-format off
  utils::OpTypePattern subgraph_pattern =
  {"Rsqrt", "rsqrt", NodeStatus::kRemove,
    {
      {"AddV2|Add", "add", NodeStatus::kRemove,
        {
          {"Mean", "mean0", NodeStatus::kRemove,
            {
              {"SquaredDifference", "squareddiff", NodeStatus::kRemove,
                {
                  {"*", "input", NodeStatus::kRemain},
                  {"Mean", "mean1", NodeStatus::kRemove,
                    {
                      {"*", "input", NodeStatus::kRemain},
                      {"Const", "r_indices1", NodeStatus::kRemain}
                    }
                  } // end mean1
                }
              }, // end squareddiff
              {"Const", "r_indices0", NodeStatus::kRemain}
            }
          }, // end mean0
          {"Const", "epsilon", NodeStatus::kRemain}
        }
      } // end add
    }
  };
  // The following pattern will be searched in the graph with additional
  // contraints. Here * means any type of op.
  //              Subgraph for fusion
  //              -------------------
  //   *(input)
  //    |    | \____________
  //    |    |              \
  //    |    |             Mean1                                      FusedOp
  //    |    |            /    \                                      -------
  //    |    |           /      \                           *(input)  Const  Const
  //    |    |          /        \                              \    (gamma) (beta)
  //    |    |         /          \                              \     |     /
  //    |    |        /           |                _MklFusedInstanceNorm/_MklLayerNorm
  //    |    |       /            |
  //    \   SquaredDiff  Const    |
  //     \      \      /          |
  //      \      \    /           |
  //       \     Mean0  Const     |
  //        \      \    /         |
  //         \  AddV2|Add         |
  //          \       \    Const  |
  //           \    Rsqrt (gamma) |
  //            \        \ /      |
  //             \       Mul1     |
  //              \      |   \    |
  //               \     |    \   |
  //                \    |     \  |
  //                 \   | Const\ |
  //                  \  | (beta)Mul2
  //                   \ |    |  /
  //                   Mul0   Sub
  //                      \   /
  //                  AddV2|Add(output)
  utils::OpTypePattern common_norm_pattern =
    {"AddV2|Add", "output", NodeStatus::kReplace,
      {
        {"Mul", "mul0", NodeStatus::kRemove,
          {
            {"*", "input", NodeStatus::kRemain},
            {"Mul", "mul1", NodeStatus::kRemove,
              {
                subgraph_pattern,
                {"Const", "gamma", NodeStatus::kRemain}
              }
            } // end mul1
          }
        }, // end mul0
        {"Sub", "sub0", NodeStatus::kRemove,
          {
            {"Const", "beta", NodeStatus::kRemain},
            {"Mul", "mul2", NodeStatus::kRemove,
              {
                {"Mul", "mul1", NodeStatus::kRemove},
                {"Mean", "mean1", NodeStatus::kRemove}
              }
            }, // end mul2
          }
        } // end sub
      }
    };
  // clang-format on
  // A slight variation of the common layernorm pattern
  // clang-format off
  //        Subgraph for fusion
  //        -------------------
  //             *(input)
  //            /  |      \
  //           /   | Const \
  //          /    | /      \
  //          |   Mean1     |                                    FusedOp
  //          |   / \       |                                    -------
  //          |  /   \      |                            *(input)  Const   Const
  //          Sub     \     |                                \    (gamma)  (beta)
  //          |     SquaredDiff  Const                        \     |     /
  //          |          \      /                              _MklLayerNorm
  //          |           \    /
  //          |            Mean0  Const
  //          |              \   /
  //          |               AddV2
  //          |                 |
  //          \               Rsqrt
  //           \______     _____/
  //                  \   /
  //        Const      Mul1
  //       (gamma)    /
  //             \   /     Const
  //             Mul0      (beta)
  //                \      /
  //               AddV2(output)
  utils::OpTypePattern common_norm_pattern_1 =
    {"AddV2|Add", "output", NodeStatus::kReplace,
      {
        {"Mul", "mul0", NodeStatus::kRemove,
          {
            {"Mul", "mul1", NodeStatus::kRemove,
              {
                {"Sub", "sub0", NodeStatus::kRemove,
                  {
                    {"*", "input", NodeStatus::kRemain},
                    {"Mean", "mean1", NodeStatus::kRemove,
                      {
                        {"*", "input", NodeStatus::kRemain},
                        {"Const", "r_indices1", NodeStatus::kRemain}
                      }
                    }
                  }
                },
                subgraph_pattern
              }
            },
            {"*", "gamma", NodeStatus::kRemain}
          }
        },
        {"*", "beta", NodeStatus::kRemain},
      }
    };
  // clang-format on
  // TODO(intel-tf) : Use * for gamma and beta instead of Const
  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx->graph_view));
  matched_nodes_map->clear();
  remove_node_indices->clear();
  bool found_op_type_match =
      graph_matcher.GetMatchedNodes(common_norm_pattern, {},
                                    ctx->graph_view.GetNode(node_index),
                                    matched_nodes_map, remove_node_indices) ||
      graph_matcher.GetMatchedNodes(common_norm_pattern_1, {},
                                    ctx->graph_view.GetNode(node_index),
                                    matched_nodes_map, remove_node_indices);
  return found_op_type_match;
}

// Keras LayerNormalization api uses multiple TensorFlow ops. Current fusion
// pattern is only for the case, when LayerNormalization uses FusedBatcNormV3.
// We further restrict it to only 2D or 3D tensor inputs to keras
// LayerNormalization api.
bool FindMklLayerNorm(RemapperContext* ctx, int node_index,
                      std::map<string, int>* matched_nodes_map,
                      std::set<int>* remove_node_indices,
                      std::vector<string>* input_node_names, float* epsilon) {
  if (!IsMKLEnabled()) return false;

  // The following pattern will be searched in the graph with additional
  // contraints. Here * means any type of op.
  // clang-format off
  //              Subgraph for fusion
  //              -------------------
  //
  //     *(input)  *  * Const  *  Const                       FusedOp
  //          \    |   \  |    |  /        Const              -------
  //           \   |    \ |    | /  Const   /
  //           Reshape  Fill   Fill  /     /         *(input) *(gamma)  *(beta)
  //              \      /      /   /     /                \     |      /
  //               \    /      /   /     /                  \    |     /
  //          F u s e d B a t c h N o r m V 3              _MklLayerNorm
  //                 \
  //                  \   *
  //                   \ /
  //                 Reshape
  //                    \   *(gamma)
  //                     \ /
  //                     Mul
  //             *(beta) /
  //                \   /
  //                AddV2(output)
  // clang-format on
  using utils::MatchingDirection;
  using utils::NodeStatus;
  // clang-format off
  utils::OpTypePattern layer_norm_pattern =
    {"AddV2", "output", NodeStatus::kReplace,
      {
        {"*", "beta", NodeStatus::kRemain},
        {"Mul", "scale", NodeStatus::kRemove,
          {
            {"Reshape", "post_reshape", NodeStatus::kRemove,
              {
                {"FusedBatchNormV3", "fused_batch_norm", NodeStatus::kRemove,
                  {
                    {"Reshape", "pre_reshape", NodeStatus::kRemove,
                      {
                        {"*", "input", NodeStatus::kRemain},
                        {"*", "pre_shape", NodeStatus::kRemain}
                      }
                    },
                    {"Fill", "fill_scale", NodeStatus::kRemove,
                      {
                        {"*", "dims_fill_scale", NodeStatus::kRemain},
                        {"Const", "unit_gamma", NodeStatus::kRemain}
                      }
                    },
                    {"Fill", "fill_offset", NodeStatus::kRemove,
                      {
                        {"*", "dims_fill_offset", NodeStatus::kRemain},
                        {"Const", "zero_beta", NodeStatus::kRemain}
                      }
                    },
                    {"Const", "empty", NodeStatus::kRemain},
                    {"Const", "empty", NodeStatus::kRemain}
                  }
                },
                {"*", "post_shape", NodeStatus::kRemain}
              }
            },
            {"*", "gamma", NodeStatus::kRemain}
          }
        }
      }
    };  // clang-format on

  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx->graph_view));
  bool found_op_type_match = false;
  matched_nodes_map->clear();
  remove_node_indices->clear();
  found_op_type_match =
      graph_matcher.GetMatchedNodes(layer_norm_pattern, ctx->nodes_to_preserve,
                                    ctx->graph_view.GetNode(node_index),
                                    matched_nodes_map, remove_node_indices);
  // If Keras api based layer-norm is not found, check if custom layer-norm is
  // present in the graph
  if (!found_op_type_match) {
    matched_nodes_map->clear();
    remove_node_indices->clear();
    found_op_type_match = IsCommonNormPattern(
        ctx, node_index, matched_nodes_map, remove_node_indices);
  }

  // Additional check for LayerNorm
  if (found_op_type_match) {
    if (!ctx->inferred_graph_properties) {
      absl::Status s = ctx->graph_properties.InferStatically(
          /*assume_valid_feeds=*/true,
          /*aggressive_shape_inference=*/false,
          /*include_input_tensor_values=*/true,
          /*include_output_tensor_values=*/true);
      if (!s.ok()) return false;
      ctx->inferred_graph_properties = true;
    }
    *epsilon = 0.001;  // default value
    // Keras layer-norm uses FusedBatchNorm in training mode. Check the
    // FusedBatchNorm conforms to layer-norm semantics.
    if (matched_nodes_map->count("fused_batch_norm")) {
      // LayerNorm uses FusedBatchNorm in training mode.
      NodeDef* fused_batch_norm_node =
          ctx->graph_view.GetNode(matched_nodes_map->at("fused_batch_norm"))
              ->node();
      if (fused_batch_norm_node->attr().count("epsilon")) {
        *epsilon = fused_batch_norm_node->attr().at("epsilon").f();
      }
      bool is_training = false;
      if (!TryGetNodeAttr(*fused_batch_norm_node, kIsTraining, &is_training) ||
          !is_training)
        return false;

      // FusedBatchNorm node should have mean/variance as empty constant
      NodeDef* empty_const_node =
          ctx->graph_view.GetNode(matched_nodes_map->at("empty"))->node();
      Tensor const_tensor;
      if (empty_const_node != nullptr && empty_const_node->op() == "Const" &&
          const_tensor.FromProto(
              empty_const_node->attr().at("value").tensor())) {
        if (const_tensor.NumElements() != 0) return false;
      } else {
        return false;
      }
      auto* pre_reshape_node =
          ctx->graph_view.GetNode(matched_nodes_map->at("pre_reshape"))->node();
      auto* scale_node =
          ctx->graph_view.GetNode(matched_nodes_map->at("gamma"))->node();
      auto* beta_node =
          ctx->graph_view.GetNode(matched_nodes_map->at("beta"))->node();
      input_node_names->clear();
      input_node_names->resize(3);
      input_node_names->at(0) = pre_reshape_node->input(0);
      input_node_names->at(1) = scale_node->name();
      input_node_names->at(2) = beta_node->name();

    } else {
      // Make sure the custom pattern conforms to layer-norm semantics by
      // checking the reduction axis
      NodeDef* mean1_node =
          ctx->graph_view.GetNode(matched_nodes_map->at("mean1"))->node();
      bool keep_dims = false;
      if (!mean1_node ||
          !TryGetNodeAttr(*mean1_node, "keep_dims", &keep_dims) || !keep_dims)
        return false;
      // Get the reduction axes for mean node to check if the
      // mean computation complies with layer normalization
      // i.e the axis count should be 1 and the reduction axis
      // should be the last axis
      NodeDef* mean_axis_node =
          ctx->graph_view.GetNode(matched_nodes_map->at("r_indices1"))->node();
      if (!mean_axis_node) {
        VLOG(1) << "Unable to find reduction axis node";
        return false;
      }
      Tensor mean_axis_tensor;
      if (!mean_axis_tensor.FromProto(
              mean_axis_node->attr().at("value").tensor())) {
        return false;
      }
      DataType dtype = mean_axis_tensor.dtype();
      if (dtype != DT_INT32 && dtype != DT_INT64) return false;

      int expected_axis_count = 1;
      if (mean_axis_tensor.NumElements() != expected_axis_count) return false;

      NodeDef* input_node =
          ctx->graph_view.GetNode(matched_nodes_map->at("input"))->node();
      auto input_node_props =
          ctx->graph_properties.GetOutputProperties(input_node->name());
      int rank = Rank(input_node_props[0].shape());
      if (dtype == DT_INT32) {
        if (static_cast<int32>(rank - 1) != mean_axis_tensor.flat<int32>()(0))
          return false;
      } else {
        if (static_cast<int64>(rank - 1) != mean_axis_tensor.flat<int64>()(0))
          return false;
      }
      auto* gamma_node =
          ctx->graph_view.GetNode(matched_nodes_map->at("gamma"))->node();
      auto* beta_node =
          ctx->graph_view.GetNode(matched_nodes_map->at("beta"))->node();
      input_node_names->clear();
      input_node_names->resize(3);
      input_node_names->at(0) = mean1_node->input(0);
      input_node_names->at(1) = gamma_node->name();
      input_node_names->at(2) = beta_node->name();
    }

    // TODO(intel-tf): Relax the restriction of 2D/3D tensor once kernel
    // supports that.
    NodeDef* input_node_def =
        ctx->graph_view.GetNode(matched_nodes_map->at("input"))->node();
    auto input_props =
        ctx->graph_properties.GetOutputProperties(input_node_def->name());
    NodeDef* output_node_def =
        ctx->graph_view.GetNode(matched_nodes_map->at("output"))->node();
    auto output_props =
        ctx->graph_properties.GetOutputProperties(output_node_def->name());
    if (ShapesSymbolicallyEqual(input_props[0].shape(),
                                output_props[0].shape())) {
      int rank = Rank(input_props[0].shape());
      if (rank < 2 || rank > 3) return false;
    } else {
      return false;
    }
  }
  return found_op_type_match;
}

bool FindFusedBatchNorm(const RemapperContext& ctx, int node_index,
                        FusedBatchNorm* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  // Disable fusions on CPU when XLA JIT compilation enabled.
  if (ctx.xla_cpu_jit_disable_fusion && NodeIsOnCpu(node_def)) return false;

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
// `tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc` for details.
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

    // We fuse FusedBatchNorm on GPU or oneDNN CPU.
    if (!IsMKLEnabled() && !NodeIsOnGpu(fused_batch_norm_node_def))
      return false;

    DataType t_dtype = GetDataTypeFromAttr(*fused_batch_norm_node_def, "T");

    if (NodeIsOnGpu(fused_batch_norm_node_def)) {
      // GPU supports float and half.
      // Put this condition before check `IsMKLEnabled()` because this node
      // should be processed when it's on GPU and oneDNN CPU is enabled.
      if (t_dtype != DT_FLOAT && t_dtype != DT_HALF) return false;
    } else {
      // Disable fusions on CPU when XLA JIT compilation enabled.
      if (ctx.xla_cpu_jit_disable_fusion) return false;
      if (IsMKLEnabled() && !IsDataTypeSupportedByOneDNNOnThisCPU(t_dtype))
        return false;
    }

    // Get the FusedBatchNorm training mode.
    bool is_training;
    if (!GetNodeAttr(*fused_batch_norm_node_def, kIsTraining, &is_training)
             .ok())
      return false;
    string data_format;
    if (!GetNodeAttr(*fused_batch_norm_node_def, kDataFormat, &data_format)
             .ok())
      return false;
    if (data_format != "NHWC" && data_format != "NCHW") return false;

    // In training mode we rely on cuDNN for computing FusedBatchNorm with side
    // inputs and activation, and it has its own limitations. In inference mode
    // we have a custom CUDA kernel that doesn't not have these constraints.
    if (is_training && NodeIsOnGpu(fused_batch_norm_node_def)) {
      // cuDNN only supports NHWC data layout.
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
    // Currently no CPU implementation for "FusedBatchNorm + SideInput +
    // <Activation>""
    if (IsMKLEnabled() && !NodeIsOnGpu(node_def)) return false;

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

bool FindFusedBatchNormGradEx(const RemapperContext& ctx, int node_index,
                              FusedBatchNormGradEx* matched) {
  // Root of the pattern must be a FusedBatchNormGrad.
  const utils::MutableNodeView* node_view = ctx.graph_view.GetNode(node_index);

  // Returns true iff the node is a compatible FusedBatchNormGrad node.
  const auto valid_batch_norm_grad =
      [&](const utils::MutableNodeView& fused_batch_norm_grad) -> bool {
    const NodeDef* node_def = fused_batch_norm_grad.node();
    if (!IsFusedBatchNormGrad(*node_def) ||
        HasControlFaninOrFanout(fused_batch_norm_grad))
      return false;

    // We fuse FusedBatchNormGrad on GPU.
    if (!NodeIsOnGpu(node_def)) return false;

    // We fuse FusedBatchNormGrad only for the training mode.
    bool is_training;
    if (!GetNodeAttr(*node_def, kIsTraining, &is_training).ok() || !is_training)
      return false;

    // Data type must be DT_HALF.
    DataType t_dtype = GetDataTypeFromAttr(*node_def, "T");
    if (t_dtype != DT_HALF) return false;

    // We rely on cuDNN for computing FusedBatchNormGrad with side
    // outputs and activation. cuDNN only supports NHWC data layout.
    string data_format;
    if (!GetNodeAttr(*node_def, kDataFormat, &data_format).ok()) return false;
    if (data_format != "NHWC") return false;

    // Channel dimension must be a multiple of 4.
    const auto& props =
        ctx.graph_properties.GetInputProperties(node_def->name());
    const bool valid_channel_dim = !props.empty() &&
                                   props[0].shape().dim_size() == 4 &&
                                   props[0].shape().dim(3).size() % 4 == 0;
    if (!valid_channel_dim) return false;

    // cuDNN must support CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode.
    if (!BatchnormSpatialPersistentEnabled()) return false;

    // FusedBatchNormV2 and V3 have an extra type parameter.
    if (node_def->op() != "FusedBatchNorm" &&
        !HasDataType(node_def, DT_FLOAT, "U"))
      return false;

    return true;
  };

  if (ctx.xla_auto_clustering_on) return false;

  if (!valid_batch_norm_grad(*node_view)) return false;

  if (node_view->NumRegularFanins() < 1) return false;

  const utils::MutableFanoutView& regular_fanin_0 =
      node_view->GetRegularFanin(0);
  const utils::MutableNodeView* relugrad_node_view =
      regular_fanin_0.node_view();
  const NodeDef* relugrad_node_def = relugrad_node_view->node();
  bool is_relugrad = IsReluGrad(*relugrad_node_def);

  if (!is_relugrad || HasControlFaninOrFanout(*relugrad_node_view) ||
      IsInPreserveSet(ctx, relugrad_node_def))
    return false;

  if (relugrad_node_view->NumRegularFanins() < 1) return false;
  // Find its corresponding forward node. We need the node to determine if the
  // type is bn+add+act or bn+act. Also, we need to access its "offset" input.
  const utils::MutableFanoutView& fanin_1 =
      relugrad_node_view->GetRegularFanin(1);
  const utils::MutableNodeView* fwd_node_view = fanin_1.node_view();
  FusedBatchNormEx fwd_matched;
  FindFusedBatchNormEx(ctx, fwd_node_view->node_index(), &fwd_matched);
  bool fwd_bn_act_used = fwd_matched.activation != kMissingIndex &&
                         fwd_matched.side_input == kMissingIndex;
  bool fwd_bn_add_act_used = fwd_matched.activation != kMissingIndex &&
                             fwd_matched.side_input != kMissingIndex;

  // Check that only 1 node consumes the output of the ReluGrad node.
  if (fwd_bn_act_used && relugrad_node_view->GetRegularFanout(0).size() == 1) {
    matched->activation_grad = regular_fanin_0.node_index();
    matched->fused_batch_norm_grad = node_index;
    matched->fwd_fused_batch_norm = fwd_matched.fused_batch_norm;
    return true;
  }

  // Check that only 2 nodes consume the output of the ReluGrad node.
  if (fwd_bn_add_act_used &&
      relugrad_node_view->GetRegularFanout(0).size() == 2) {
    // In a graph with the Add node having two BatchNorm nodes as the inputs, we
    // need to make sure only the one backward BatchNorm that correponds to the
    // to-be-fused forward BatchNorm should be fused. We use the edge for the
    // reserve space to get the directly corresponded forward BatchNorm node.
    const utils::MutableFanoutView& fwd_batch_norm_node =
        node_view->GetRegularFanin(5);
    if (fwd_matched.fused_batch_norm != fwd_batch_norm_node.node_index()) {
      return false;
    }

    const std::vector<utils::MutableFaninView>& fanouts_at_port_0 =
        relugrad_node_view->GetRegularFanouts()[0];
    const utils::MutableNodeView* fanout_0_node_view =
        ctx.graph_view.GetNode(fanouts_at_port_0[0].node_view()->GetName());
    const utils::MutableNodeView* fanout_1_node_view =
        ctx.graph_view.GetNode(fanouts_at_port_0[1].node_view()->GetName());
    const NodeDef* fanout_0_node_def = fanout_0_node_view->node();
    const NodeDef* fanout_1_node_def = fanout_1_node_view->node();
    const NodeDef* node_def = node_view->node();

    matched->activation_grad = regular_fanin_0.node_index();
    matched->fused_batch_norm_grad = node_index;
    matched->fwd_fused_batch_norm = fwd_matched.fused_batch_norm;

    if (fanout_0_node_def == node_def) {
      matched->side_input_grad = fanout_1_node_view->node_index();
      return true;
    }

    if (fanout_1_node_def == node_def) {
      matched->side_input_grad = fanout_0_node_view->node_index();
      return true;
    }
  }

  return false;
}

bool FindTensorToHashBucket(const RemapperContext& ctx, int node_index,
                            TensorToHashBucket* matched) {
  // Root of the pattern must be a StringToHashBucketFast.
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  if (!IsStringToHashBucketFast(*node_def) ||
      HasControlFaninOrFanout(*node_view)) {
    return false;
  }

  // Input to the StringToHashBucketFast must be AsString.
  if (node_view->NumRegularFanins() < 1) return false;

  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* as_string_node_view = regular_fanin_0.node_view();
  const auto* as_string_node_def = as_string_node_view->node();
  bool is_as_string = IsAsString(*as_string_node_def);

  if (!is_as_string || HasControlFaninOrFanout(*as_string_node_view) ||
      !HasAtMostOneFanoutAtPort0(*as_string_node_view) ||
      IsInPreserveSet(ctx, as_string_node_def))
    return false;

  // DataType of AsString must be int8/16/32/64 and width/fill attrs must be
  // default values.
  if (!HasDataType(as_string_node_def, DT_INT8) &&
      !HasDataType(as_string_node_def, DT_INT16) &&
      !HasDataType(as_string_node_def, DT_INT32) &&
      !HasDataType(as_string_node_def, DT_INT64)) {
    return false;
  }

  int width;
  if (!GetNodeAttr(*as_string_node_def, kWidth, &width).ok() || width != -1) {
    return false;
  }

  string fill;
  if (!GetNodeAttr(*as_string_node_def, kFill, &fill).ok() || !fill.empty()) {
    return false;
  }

  // An input to the AsString must exist to determine the device.
  if (as_string_node_view->NumRegularFanins() < 1) return false;

  const auto& fanin_0 = as_string_node_view->GetRegularFanin(0);
  const auto* pre_node_view = fanin_0.node_view();

  // We successfully found a AsString + StringToHashBucketFast pattern.
  const TensorToHashBucket pattern{pre_node_view->node_index(),
                                   as_string_node_view->node_index(),
                                   node_index};

  *matched = pattern;

  return true;
}

// clang-format off
// HardSwish pattern
//                        input     Const (value: 3)
//                          |  \   /
//                          |  Add or AddV2
//                          |        |
//                          |       Relu6
//                          |        /
//                          |       /
// Const (value: 0.1666)    |      /
//                      \   |     /
//                        Mul    /
//                          \   /
//                           Mul
// clang-format on
bool FindHardSwish(RemapperContext& ctx, int node_index,
                   std::map<string, int>* matched_nodes_map,
                   std::set<int>* remove_node_indices) {
  if (!IsMKLEnabled()) return false;

  using utils::MatchingDirection;
  using utils::NodeStatus;
  // clang-format off
  utils::OpTypePattern pattern {"Mul", "output", NodeStatus::kReplace,
    {
      {"Mul", "mul_one_sixth", NodeStatus::kRemove,
        {
          {"Const|Cast", "one_sixth", NodeStatus::kRemain},
          {"*", "input", NodeStatus::kRemain}
        }
      },
      {"Relu6", "relu6", NodeStatus::kRemove,
        {
          {"Add|AddV2", "add", NodeStatus::kRemove,
            {
              {"*", "input", NodeStatus::kRemain},
              {"Const|Cast", "three", NodeStatus::kRemain}
            }
          }
        }
      },
    }
  };
  // clang-format on
  bool found_match = false;
  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx.graph_view));

  matched_nodes_map->clear();
  remove_node_indices->clear();

  found_match = graph_matcher.GetMatchedNodes(
      pattern, ctx.nodes_to_preserve, ctx.graph_view.GetNode(node_index),
      matched_nodes_map, remove_node_indices);

  if (found_match) {
    // Check if the values of Const nodes are as expected
    std::map<string, float> values_map = {{"three", 3.0},
                                          {"one_sixth", 0.16666}};
    if (!VerifyConstants(&ctx, matched_nodes_map, &values_map)) return false;
  }

  return found_match;
}

// clang-format off
// Contraction + BiasAdd + _FusedHardSwish activation
// input     filter
//     \     /
// Contraction  bias
//         |   /
//      BiasAdd
//         |
//   _FusedHardSwish
// clang-format on
bool FindContractionWithBiasAddAndHardSwish(
    RemapperContext& ctx, int node_index,
    std::map<string, int>* matched_nodes_map,
    std::set<int>* remove_node_indices) {
  if (!IsMKLEnabled()) return false;

  const auto* node_view = ctx.graph_view.GetNode(node_index);
  if (HasControlFaninOrFanout(*node_view)) return false;

  // Check if HardSwish pattern is available
  if (!FindHardSwish(ctx, node_index, matched_nodes_map, remove_node_indices))
    return false;
  // Get handle of Add|AddV2 op that is the root of HardSwish pattern.
  const auto* add_node_view =
      ctx.graph_view.GetNode(matched_nodes_map->at("add"));
  const auto* add_node_def = add_node_view->node();

  // Check if ContractionWithBias pattern is feeding HardSwish
  ContractionWithBiasAdd base;
  int port_id = 0;
  // BiasAdd node is expected to have 2 fanouts feeding the HardSwish pattern.
  if (!FindContractionWithBiasInPort(ctx, *add_node_view, *add_node_def,
                                     port_id, &base, /*allowed_fanouts*/ 2)) {
    port_id = 1;
    if (!FindContractionWithBiasInPort(ctx, *add_node_view, *add_node_def,
                                       port_id, &base, /*allowed_fanouts*/ 2)) {
      VLOG(2) << "Contraction + BiasAdd pattern was not found although"
              << " HardSwish pattern was found, so fusion failed.";
      return false;
    }
  }

  // Get the BiasAdd node
  const auto* bias_node_def = ctx.graph_view.GetNode(base.bias_add)->node();
  if (!HaveSameDataType(add_node_def, bias_node_def)) return false;

  // Get the contraction node
  const auto* contraction_node_view = ctx.graph_view.GetNode(base.contraction);
  const auto* contraction_node_def = contraction_node_view->node();

  // Currently only Conv2D and DepthwiseConv2D contraction ops are supported
  if (!IsConv2D(*contraction_node_def) &&
      !IsDepthwiseConv2dNative(*contraction_node_def))
    return false;

  // Check if contraction is compatible with CPU
  if (!IsCpuCompatibleConv2D(ctx, contraction_node_def) &&
      !IsCpuCompatibleDepthwiseConv2dNative(contraction_node_def))
    return false;

  // We found a {Conv2D, DepthwiseConv2D}+BiasAdd+_FusedHardSwish pattern.
  matched_nodes_map->insert({"contraction", base.contraction});
  matched_nodes_map->insert({"bias_add", base.bias_add});

  remove_node_indices->insert(base.contraction);
  remove_node_indices->insert(base.bias_add);
  return true;
}

bool FindFusedBatchMatMul(RemapperContext* ctx, int node_index,
                          std::map<string, int>* matched_nodes_map,
                          std::set<int>* remove_node_indices,
                          std::vector<string>* input_node_names) {
  if (!IsMKLEnabled()) return false;

  using utils::MatchingDirection;
  using utils::NodeStatus;
  int pattern = 0;
  // clang-format off
  utils::OpTypePattern fusion_pattern1 =
    {"Add|AddV2", "output", NodeStatus::kReplace,
      {
        {"Mul", "mul", NodeStatus::kRemove,
          {
            {"BatchMatMulV2", "batch_matmul", NodeStatus::kRemove},
            {"*", "multiplicand", NodeStatus::kRemain}
          }
        },
        {"*", "addend", NodeStatus::kRemain}
      }
    };

  utils::OpTypePattern fusion_pattern2 =
    {"Add|AddV2", "output", NodeStatus::kReplace,
      {
        {"BatchMatMulV2", "batch_matmul", NodeStatus::kRemove,
          {
            {"Mul", "mul", NodeStatus::kRemove,
              {
                {"*", "mul_input0", NodeStatus::kRemain},
                {"Const|Cast", "multiplicand", NodeStatus::kRemain}
              }
            },
            {"*", "bmm_input1", NodeStatus::kRemain}
          }
        },
        {"*", "addend", NodeStatus::kRemain}
      }
    };
  // clang-format on

  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx->graph_view));
  bool found_op_type_match = false;
  matched_nodes_map->clear();
  remove_node_indices->clear();
  found_op_type_match =
      graph_matcher.GetMatchedNodes(fusion_pattern1, ctx->nodes_to_preserve,
                                    ctx->graph_view.GetNode(node_index),
                                    matched_nodes_map, remove_node_indices);
  if (found_op_type_match) pattern = 1;

  if (!found_op_type_match) {
    matched_nodes_map->clear();
    remove_node_indices->clear();
    found_op_type_match =
        graph_matcher.GetMatchedNodes(fusion_pattern2, ctx->nodes_to_preserve,
                                      ctx->graph_view.GetNode(node_index),
                                      matched_nodes_map, remove_node_indices);
    if (found_op_type_match) pattern = 2;
  }

  // OneDNN is not optimized for all shapes with regard to binary-post ops
  // fusion. Allow limited cases only for now that are optimized, (i)
  // multiplicand is scalar, (ii) BatchMatmulV2 output is 4D tensor, and (iii)
  // addend is 4D tensor with second dim_size = 1.
  if (!found_op_type_match) return false;
  if (!ctx->inferred_graph_properties) {
    absl::Status s = ctx->graph_properties.InferStatically(
        /*assume_valid_feeds=*/true,
        /*aggressive_shape_inference=*/false,
        /*include_input_tensor_values=*/false,
        /*include_output_tensor_values=*/true);
    if (!s.ok()) return false;
    ctx->inferred_graph_properties = true;
  }
  NodeDef* multiplicand_node_def =
      ctx->graph_view.GetNode(matched_nodes_map->at("multiplicand"))->node();
  auto multiplicand_props =
      ctx->graph_properties.GetOutputProperties(multiplicand_node_def->name());
  if (NumCoefficients(multiplicand_props[0].shape()) != 1) return false;

  NodeDef* batch_matmul_node_def =
      ctx->graph_view.GetNode(matched_nodes_map->at("batch_matmul"))->node();
  if (!IsCpuCompatibleMatMul(*ctx, batch_matmul_node_def)) return false;

  auto batch_matmul_props =
      ctx->graph_properties.GetOutputProperties(batch_matmul_node_def->name());
  if (Rank(batch_matmul_props[0].shape()) != 4) return false;

  NodeDef* addend_node_def =
      ctx->graph_view.GetNode(matched_nodes_map->at("addend"))->node();
  auto addend_props =
      ctx->graph_properties.GetOutputProperties(addend_node_def->name());
  auto addend_shape = addend_props[0].shape();
  if (!(Rank(addend_shape) == 4 && addend_shape.dim(1).size() == 1)) {
    return false;
  }
  input_node_names->clear();
  input_node_names->resize(4);
  if (pattern == 1) {
    input_node_names->at(0) = batch_matmul_node_def->input(0);
    input_node_names->at(1) = batch_matmul_node_def->input(1);
    input_node_names->at(2) = multiplicand_node_def->name();
    input_node_names->at(3) = addend_node_def->name();
  } else if (pattern == 2) {
    auto* mul_input0_node_def =
        ctx->graph_view.GetNode(matched_nodes_map->at("mul_input0"))->node();
    input_node_names->at(0) = mul_input0_node_def->name();
    input_node_names->at(1) = batch_matmul_node_def->input(1);
    input_node_names->at(2) = multiplicand_node_def->name();
    input_node_names->at(3) = addend_node_def->name();
  }
  return found_op_type_match;
}

// Helper function to check if the reduction axes for a given input
// shape align with instance normalization's mean computation.
// Mean reduction axes for instance norm are expected to be:
// 4D input shape - reduction axes [2,3], data format NCHW;
// 4D input shape - reduction axes [1,2], data format NHWC;
// 5D input shape - reduction axes [2,3,4], data format NCDHW;
// 5D input shape - reduction axes [1,2,3], data format NDHWC;
template <typename T>
bool IsInstanceNormReduction(const TensorShapeProto& input_shape,
                             const Tensor& reduction_axes_data) {
  int input_dims = input_shape.dim_size();
  int reduction_axes = reduction_axes_data.NumElements();

  if ((input_dims != 4 && input_dims != 5) ||
      (reduction_axes + 2) != input_dims) {
    return false;
  }

  if (input_dims == 4) {
    return ((reduction_axes_data.flat<T>()(0) == static_cast<T>(1) &&
             reduction_axes_data.flat<T>()(1) == static_cast<T>(2)) ||
            (reduction_axes_data.flat<T>()(0) == static_cast<T>(2) &&
             reduction_axes_data.flat<T>()(1) == static_cast<T>(3)));
  } else {
    return ((reduction_axes_data.flat<T>()(0) == static_cast<T>(1) &&
             reduction_axes_data.flat<T>()(1) == static_cast<T>(2) &&
             reduction_axes_data.flat<T>()(2) == static_cast<T>(3)) ||
            (reduction_axes_data.flat<T>()(0) == static_cast<T>(2) &&
             reduction_axes_data.flat<T>()(1) == static_cast<T>(3) &&
             reduction_axes_data.flat<T>()(2) == static_cast<T>(4)));
  }
}

// Find a group of ops that make up an instance normalization pattern for fusion
bool FindInstanceNorm(RemapperContext* ctx, int node_index,
                      std::map<string, int>* matched_nodes_map,
                      std::set<int>* remove_node_indices) {
  if (!IsCommonNormPattern(ctx, node_index, matched_nodes_map,
                           remove_node_indices)) {
    return false;
  }

  // Additional checks for InstanceNorm
  if (!ctx->inferred_graph_properties) {
    absl::Status s = ctx->graph_properties.InferStatically(
        /*assume_valid_feeds=*/true,
        /*aggressive_shape_inference=*/false,
        /*include_input_tensor_values=*/false,
        /*include_output_tensor_values=*/true);
    if (!s.ok()) return false;
    ctx->inferred_graph_properties = true;
  }

  NodeDef* mean1_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("mean1"))->node();
  bool keep_dims = false;
  if (!mean1_node || !TryGetNodeAttr(*mean1_node, "keep_dims", &keep_dims) ||
      !keep_dims) {
    return false;
  }

  // Get the input shape
  const auto& input_props =
      ctx->graph_properties.GetInputProperties(mean1_node->name());
  const TensorShapeProto& input_shape = input_props[0].shape();
  if (input_shape.unknown_rank()) return false;

  DataType dtype = GetDataTypeFromAttr(*mean1_node, "T");
  // TODO(intel-tf): enable bfloat16 data type support
  if (dtype != DT_FLOAT && dtype != DT_HALF) return false;

  // Check if gamma and beta constants have the same shape
  NodeDef* gamma_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("gamma"))->node();
  NodeDef* beta_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("beta"))->node();
  if (!gamma_node || !beta_node) {
    VLOG(2) << "Unexpected error to retrieve gamma or beta node";
    return false;
  }
  Tensor gamma_tensor, beta_tensor;
  // TODO(intel-tf): Allow fusions for cases where gamma and beta aren't Consts
  if (gamma_node->op() != "Const" ||
      !gamma_tensor.FromProto(gamma_node->attr().at("value").tensor()) ||
      beta_node->op() != "Const" ||
      !beta_tensor.FromProto(beta_node->attr().at("value").tensor())) {
    return false;
  }
  if (!gamma_tensor.IsSameSize(beta_tensor)) return false;

  // Get the reduction axes for mean node to check if the
  // mean computation complies with instance normalization
  NodeDef* mean_axes_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("r_indices1"))->node();
  if (!mean_axes_node) {
    VLOG(2) << "Unexpected error to retrieve reduction axes node";
    return false;
  }

  Tensor mean_axes_tensor;
  if (!mean_axes_tensor.FromProto(
          mean_axes_node->attr().at("value").tensor())) {
    return false;
  }
  dtype = mean_axes_tensor.dtype();
  if (dtype != DT_INT32 && dtype != DT_INT64) return false;

  return (dtype == DT_INT32)
             ? IsInstanceNormReduction<int32>(input_shape, mean_axes_tensor)
             : IsInstanceNormReduction<int64>(input_shape, mean_axes_tensor);
}

// Find the pattern with activation following instance normalization
bool FindInstanceNormWithActivation(RemapperContext* ctx, int node_index,
                                    std::map<string, int>* matched_nodes_map,
                                    std::set<int>* remove_node_indices) {
  const auto* node_view = ctx->graph_view.GetNode(node_index);
  if (HasControlFaninOrFanout(*node_view)) return false;

  const auto* node_def = node_view->node();
  // Currently only Relu and LeakyRelu are supported by oneDNN
  if (!IsLeakyRelu(*node_def) && !IsRelu(*node_def)) return false;

  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* base_node_view = regular_fanin_0.node_view();
  int base_node_idx = base_node_view->node_index();

  if (!FindInstanceNorm(ctx, base_node_idx, matched_nodes_map,
                        remove_node_indices))
    return false;

  remove_node_indices->insert(matched_nodes_map->at("output"));
  matched_nodes_map->insert(std::pair<string, int>("activation", node_index));
  return true;
}

void CopyConv2DAttributes(const NodeDef& conv2d, NodeDef* fused_conv2d,
                          const NodeDef* activation = nullptr) {
  DCHECK(IsConv2D(conv2d)) << "Input node must be a Conv2D";

  auto* attr = fused_conv2d->mutable_attr();
  auto& src_attr = conv2d.attr();

  (*attr)["T"] = src_attr.at("T");
  int num_args = fused_conv2d->input_size() - 2;
  for (int i = 0; i < num_args; ++i) {
    (*attr)["TArgs"].mutable_list()->add_type(src_attr.at("T").type());
  }
  (*attr)["num_args"].set_i(num_args);
  (*attr)["num_host_args"].set_i(0);
  (*attr)["strides"] = src_attr.at("strides");
  (*attr)["padding"] = src_attr.at("padding");
  (*attr)["explicit_paddings"] = src_attr.at("explicit_paddings");
  (*attr)["dilations"] = src_attr.at("dilations");
  (*attr)["data_format"] = src_attr.at("data_format");
  (*attr)["use_cudnn_on_gpu"] = src_attr.at("use_cudnn_on_gpu");
  // When copying attributes check whether this convolution has
  // attribute that describes the shapes on which it is working.
  if (IsMKLEnabled() && src_attr.find("_input_shapes") != src_attr.end()) {
    (*attr)["_input_shapes"] = src_attr.at("_input_shapes");
  }
  // Copy LeakyRelu's attr alpha to FusedConv2D's attr leakyrelu_alpha
  if (activation != nullptr && IsLeakyRelu(*activation)) {
    auto& activation_attr = activation->attr();
    (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
  }
}

void CopyConv3DAttributes(const NodeDef& conv3d, NodeDef* fused_conv3d,
                          const NodeDef* activation = nullptr) {
  DCHECK(IsConv3D(conv3d)) << "Input node must be a Conv3D";

  auto* attr = fused_conv3d->mutable_attr();
  auto& src_attr = conv3d.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["strides"] = src_attr.at("strides");
  (*attr)["padding"] = src_attr.at("padding");
  (*attr)["dilations"] = src_attr.at("dilations");
  (*attr)["data_format"] = src_attr.at("data_format");
  // Copy LeakyRelu's attr alpha to FusedConv3D's attr leakyrelu_alpha
  if (activation != nullptr && IsLeakyRelu(*activation)) {
    auto& activation_attr = activation->attr();
    (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
  }
}

void CopyDepthwiseConv2dNativeAttributes(const NodeDef& dw_conv2d,
                                         NodeDef* fused_dw_conv2d,
                                         const NodeDef* activation = nullptr) {
  DCHECK(IsDepthwiseConv2dNative(dw_conv2d))
      << "Input node must be a DepthwiseConv2dNative";

  auto* attr = fused_dw_conv2d->mutable_attr();
  auto& src_attr = dw_conv2d.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["strides"] = src_attr.at("strides");
  (*attr)["padding"] = src_attr.at("padding");
  (*attr)["dilations"] = src_attr.at("dilations");
  (*attr)["data_format"] = src_attr.at("data_format");
  // Copy LeakyRelu's attr alpha to FusedDepthwiseConv2d's attr leakyrelu_alpha
  if (activation != nullptr && IsLeakyRelu(*activation)) {
    auto& activation_attr = activation->attr();
    (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
  }
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
  (*attr)["exponential_avg_factor"] = src_attr.at("exponential_avg_factor");

  // FusedBatchNormV2 and V3 have an extra type parameter.
  if (fused_batch_norm.op() != "FusedBatchNorm") {
    SetAttrValue(src_attr.at("U"), &(*attr)["U"]);
  } else {
    if (!IsMKLEnabled())
      SetAttrValue(src_attr.at("T"), &(*attr)["U"]);
    else
      SetAttrValue(DT_FLOAT, &(*attr)["U"]);
  }
}

void CopyFusedBatchNormGradAttributes(const NodeDef& fused_batch_norm_grad,
                                      NodeDef* fused_batch_norm_grad_ex) {
  DCHECK(IsFusedBatchNormGrad(fused_batch_norm_grad))
      << "Input node must be a FusedBatchNormGrad";

  auto* attr = fused_batch_norm_grad_ex->mutable_attr();
  auto src_attr = fused_batch_norm_grad.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["is_training"] = src_attr.at("is_training");
  (*attr)["data_format"] = src_attr.at("data_format");
  (*attr)["epsilon"] = src_attr.at("epsilon");

  // FusedBatchNormV2 and V3 have an extra type parameter.
  if (fused_batch_norm_grad.op() != "FusedBatchNormGrad") {
    SetAttrValue(src_attr.at("U"), &(*attr)["U"]);
  } else {
    SetAttrValue(DT_FLOAT, &(*attr)["U"]);
  }
}

void CopyMatMulAttributes(const NodeDef& matmul, NodeDef* fused_matmul,
                          const NodeDef* activation = nullptr) {
  DCHECK(IsMatMul(matmul)) << "Input node must be a MatMul";

  auto* attr = fused_matmul->mutable_attr();
  auto& src_attr = matmul.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["transpose_a"] = src_attr.at("transpose_a");
  (*attr)["transpose_b"] = src_attr.at("transpose_b");
  // Copy LeakyRelu's attr alpha to _FusedMatMul's attr leakyrelu_alpha
  if (activation != nullptr && IsLeakyRelu(*activation)) {
    auto& activation_attr = activation->attr();
    (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
  }
  if (IsMKLEnabled()) {
    auto input_shapes = src_attr.find("_input_shapes");
    if (input_shapes != src_attr.end()) {
      (*attr)["_input_shapes"] = input_shapes->second;
    }
  }
}

void CopyBatchMatMulAttributes(const NodeDef& batchmatmul,
                               NodeDef* fused_batch_matmul) {
  DCHECK(IsAnyBatchMatMul(batchmatmul)) << "Input node must be a BatchMatMul";

  auto* attr = fused_batch_matmul->mutable_attr();
  auto& src_attr = batchmatmul.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["adj_x"] = src_attr.at("adj_x");
  (*attr)["adj_y"] = src_attr.at("adj_y");
  if (IsMKLEnabled()) {
    auto input_shapes = src_attr.find("_input_shapes");
    if (input_shapes != src_attr.end()) {
      (*attr)["_input_shapes"] = input_shapes->second;
    }
  }
}

void SetFusedOpAttributes(NodeDef* fused,
                          const absl::Span<const absl::string_view> fused_ops,
                          int num_args = 1, float epsilon = 0.0) {
  auto* attr = fused->mutable_attr();
  SetAttrValue(fused_ops, &(*attr)["fused_ops"]);
  SetAttrValue(num_args, &(*attr)["num_args"]);
  SetAttrValue(epsilon, &(*attr)["epsilon"]);  // required only for BatchNorm
}

absl::Status AddFusedContractionNode(RemapperContext* ctx,
                                     const ContractionWithBiasAdd& matched,
                                     std::vector<bool>* invalidated_nodes,
                                     std::vector<bool>* nodes_to_delete) {
  DCHECK(IsDeviceCompatible(*ctx, matched)) << "Unsupported fusion pattern";

  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  VLOG(2) << "Fuse " << contraction.op()
          << " with BiasAdd: " << " bias_add=" << bias_add.name()
          << " contraction=" << contraction.name();

  NodeDef fused_op;
  fused_op.set_name(bias_add.name());
  fused_op.set_device(contraction.device());
  fused_op.add_input(contraction.input(0));               // 0: input
  fused_op.add_input(contraction.input(1));               // 1: filter
  fused_op.add_input(bias_add.input(matched.bias_port));  // 2: bias
  if (IsConv2D(contraction)) {
    fused_op.set_op(kFusedConv2D);
    AddInputShapesAttr(*ctx, matched.contraction);
    CopyConv2DAttributes(contraction, &fused_op);
  } else if (IsDepthwiseConv2dNative(contraction)) {
    fused_op.set_op(kFusedDepthwiseConv2dNative);
    CopyDepthwiseConv2dNativeAttributes(contraction, &fused_op);
  } else if (IsMatMul(contraction)) {
    fused_op.set_op(kFusedMatMul);
    AddInputShapesAttr(*ctx, matched.contraction);
    CopyMatMulAttributes(contraction, &fused_op);
  } else if (IsConv3D(contraction)) {
    fused_op.set_op(kFusedConv3D);
    CopyConv3DAttributes(contraction, &fused_op);
  }

  SetFusedOpAttributes(&fused_op, {"BiasAdd"});
  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return absl::OkStatus();
}

absl::Status AddFusedContractionNode(RemapperContext* ctx,
                                     const ContractionWithActivation& matched,
                                     std::vector<bool>* invalidated_nodes,
                                     std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& activation = graph->node(matched.activation);

  VLOG(2) << "Fuse " << contraction.op() << " and " << activation.op() << ":"
          << " activation=" << activation.name()
          << " contraction=" << contraction.name();

  NodeDef fused_op;

  // In case of _FusedConv2{3}D, only updating the fused_ops
  // attr and the value of alpha in case of LeakyRelu activation

  // creating a copy of the contraction
  fused_op = contraction;

  auto* attr = fused_op.mutable_attr();
  auto contraction_fused_ops_list =
      contraction.attr().at("fused_ops").list().s();

  // updating the fused_ops attr
  std::vector<std::string> fused_items;
  for (auto it = contraction_fused_ops_list.begin();
       it != contraction_fused_ops_list.end(); it++) {
    fused_items.push_back(*it);
  }
  fused_items.push_back(GetActivationName(activation.op()));

  SetAttrValue(fused_items, &(*attr)["fused_ops"]);

  // LeakyRelu has a special attribute
  if (IsLeakyRelu(activation)) {
    auto& activation_attr = activation.attr();
    (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
  }
  fused_op.set_name(activation.name());

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.contraction] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return absl::OkStatus();
}

absl::Status AddFusedContractionNode(
    RemapperContext* ctx, const ContractionWithBiasAddAndActivation& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  DCHECK(IsDeviceCompatible(*ctx, matched)) << "Unsupported fusion pattern";

  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& activation = graph->node(matched.activation);

  VLOG(2) << "Fuse " << contraction.op() << " with BiasAdd and "
          << activation.op() << ":" << " activation=" << activation.name()
          << " bias_add=" << bias_add.name()
          << " contraction=" << contraction.name();

  NodeDef fused_op;
  fused_op.set_name(activation.name());
  fused_op.set_device(contraction.device());
  fused_op.add_input(contraction.input(0));               // 0: input
  fused_op.add_input(contraction.input(1));               // 1: filter
  fused_op.add_input(bias_add.input(matched.bias_port));  // 2: bias

  if (IsConv2D(contraction)) {
    fused_op.set_op(kFusedConv2D);
    AddInputShapesAttr(*ctx, matched.contraction);
    // leaky relu has a special attribute alpha
    CopyConv2DAttributes(contraction, &fused_op, &activation);
  } else if (IsDepthwiseConv2dNative(contraction)) {
    fused_op.set_op(kFusedDepthwiseConv2dNative);
    CopyDepthwiseConv2dNativeAttributes(contraction, &fused_op);
  } else if (IsMatMul(contraction)) {
    fused_op.set_op(kFusedMatMul);
    AddInputShapesAttr(*ctx, matched.contraction);
    CopyMatMulAttributes(contraction, &fused_op, &activation);
  } else if (IsConv3D(contraction)) {
    fused_op.set_op(kFusedConv3D);
    CopyConv3DAttributes(contraction, &fused_op, &activation);
  }

  SetFusedOpAttributes(&fused_op, {"BiasAdd", activation.op()});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return absl::OkStatus();
}

absl::Status AddFusedConvNode(RemapperContext* ctx,
                              const ContractionWithSqueezeAndBiasAdd& matched,
                              std::vector<bool>* invalidated_nodes,
                              std::vector<bool>* nodes_to_delete) {
  DCHECK(IsDeviceCompatible(*ctx, matched)) << "Unsupported fusion pattern";

  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);

  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& squeeze = graph->node(matched.squeeze);
  VLOG(2) << "Fuse Conv2D/3D with Squeeze and BiasAdd: " << " bias_add="
          << bias_add.name() << " squeeze=" << squeeze.name()
          << " conv=" << contraction.name();

  // Replace Conv2D/3D node with a fused Conv2D/3D. Matched pattern guarantees
  // that it has single consumer (only the squeeze node).
  NodeDef fused_conv;
  fused_conv.set_name(contraction.name());
  fused_conv.set_device(contraction.device());
  fused_conv.add_input(contraction.input(0));  // 0: input
  fused_conv.add_input(contraction.input(1));  // 1: filter
  fused_conv.add_input(bias_add.input(1));     // 2: bias

  if (IsConv2D(contraction)) {
    fused_conv.set_op(kFusedConv2D);
    AddInputShapesAttr(*ctx, matched.contraction);
    CopyConv2DAttributes(contraction, &fused_conv);
  } else if (IsConv3D(contraction)) {
    fused_conv.set_op(kFusedConv3D);
    CopyConv3DAttributes(contraction, &fused_conv);
  }

  SetFusedOpAttributes(&fused_conv, {"BiasAdd"});

  // Replace BiasAdd node with a Squeeze.
  NodeDef remapped_squeeze = squeeze;
  remapped_squeeze.set_name(bias_add.name());
  remapped_squeeze.set_input(0, contraction.name());

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_conv), &status);
  TF_RETURN_IF_ERROR(status);
  mutation->AddNode(std::move(remapped_squeeze), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.contraction] = true;
  (*invalidated_nodes)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.squeeze] = true;

  return absl::OkStatus();
}

absl::Status AddFusedConv2DNode(RemapperContext* ctx,
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

  AddInputShapesAttr(*ctx, matched.contraction);
  CopyConv2DAttributes(contraction, &fused_conv2d);
  SetFusedOpAttributes(&fused_conv2d, {"FusedBatchNorm"},
                       /*num_args=*/4, /*epsilon=*/matched.epsilon);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_conv2d), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.fused_batch_norm] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return absl::OkStatus();
}

absl::Status AddFusedConv2DNode(
    RemapperContext* ctx, const ContractionWithBatchNormAndActivation& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
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

  AddInputShapesAttr(*ctx, matched.contraction);
  CopyConv2DAttributes(contraction, &fused_conv2d, &activation);
  SetFusedOpAttributes(&fused_conv2d, {"FusedBatchNorm", activation.op()},
                       /*num_args=*/4, /*epsilon=*/matched.epsilon);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_conv2d), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.activation] = true;
  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.fused_batch_norm] = true;

  return absl::OkStatus();
}

absl::Status AddFusedContractionNode(
    RemapperContext* ctx, const ContractionWithBiasAddAndAdd& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);

  // oneDNN version only supports fusion for Conv2D/3D and MatMul
  DCHECK(IsConv2D(contraction) || IsMatMul(contraction) ||
         IsConv3D(contraction));

  NodeDef contraction_node;
  const NodeDef& add = graph->node(matched.add);
  contraction_node.set_name(add.name());
  contraction_node.set_device(contraction.device());
  contraction_node.add_input(
      contraction.input(0));  // 0: input(conv) / a (matmul)
  contraction_node.add_input(
      contraction.input(1));  // 1: filter(conv) / b (matmul)
  contraction_node.add_input(bias_add.input(matched.bias_port));  // 2: bias

  // Add OP has two inputs, one is conv+bias/matmul+bias pattern matched
  // previously, the other input to add is fused here.
  contraction_node.add_input(add.input(1 - matched.port_id));

  if (IsConv2D(contraction)) {
    contraction_node.set_op(kFusedConv2D);
    AddInputShapesAttr(*ctx, matched.contraction);
    CopyConv2DAttributes(contraction, &contraction_node);
  } else if (IsMatMul(contraction)) {
    AddInputShapesAttr(*ctx, matched.contraction);
    contraction_node.set_op(kFusedMatMul);
    CopyMatMulAttributes(contraction, &contraction_node);
  } else if (IsConv3D(contraction)) {
    contraction_node.set_op(kFusedConv3D);
    CopyConv3DAttributes(contraction, &contraction_node);
  }

  SetFusedOpAttributes(&contraction_node, {"BiasAdd", "Add"}, 2);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(contraction_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.add] = true;
  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.bias_add] = true;

  return absl::OkStatus();
}

absl::Status AddFusedConv3DNode(RemapperContext* ctx,
                                const PadWithConv3D& matched,
                                std::vector<bool>* invalidated_nodes,
                                std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction_idx);
  const NodeDef& pad_node_def = graph->node(matched.pad_idx);
  const NodeDef& padding_const_node_def =
      graph->node(matched.padding_const_idx);
  VLOG(2) << "Fuse " << pad_node_def.op()
          << " with contraction: " << " contraction=" << contraction.name();

  NodeDef fused_node;
  // Note: Currently, the attributes of fused op are superset of contraction
  // node. So copy it from contraction node before mutation.
  fused_node = contraction;
  // 0-th input is the 0-th input of pad node, while the remainder inputs are
  // same as the contraction node.
  fused_node.set_input(0, pad_node_def.input(0));
  fused_node.set_op(kFusedConv3D);
  auto* attr = fused_node.mutable_attr();
  // Set num_args attr explicitly, since there is no default value.
  if (!attr->contains("num_args")) {
    SetAttrValue(0, &(*attr)["num_args"]);
  }

  Tensor const_tensor;
  if (padding_const_node_def.op() == "Const" &&
      const_tensor.FromProto(
          padding_const_node_def.attr().at("value").tensor())) {
    auto const_value = const_tensor.flat<int32>();
    std::vector<int32> paddings;
    for (int i = 0; i < const_value.size(); ++i) {
      paddings.push_back(const_value(i));
      SetAttrValue(paddings, &(*attr)["padding_list"]);
    }
  } else {
    VLOG(2) << "Pad fusion with " << contraction.op() << " is invalidated, "
            << "it requires padding dim sizes to be constant.";
    return absl::OkStatus();
  }

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.contraction_idx] = true;
  (*nodes_to_delete)[matched.pad_idx] = true;
  return absl::OkStatus();
}

absl::Status AddFusedContractionNode(
    RemapperContext* ctx, const ContractionWithBiasAndAddActivation& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  // MKL version only support fusion for Conv2D
  const NodeDef& contraction = graph->node(matched.contraction);
  DCHECK(IsConv2D(contraction) || IsConv3D(contraction));
  const NodeDef& activation = graph->node(matched.activation);

  NodeDef fused_conv;
  fused_conv.set_name(activation.name());
  fused_conv.set_device(contraction.device());
  fused_conv.add_input(contraction.input(0));  // 0: input
  fused_conv.add_input(contraction.input(1));  // 1: filter
  const NodeDef& bias_add = graph->node(matched.bias_add);
  fused_conv.add_input(bias_add.input(matched.bias_port));  // 2: bias

  // Add OP has two inputs, one is conv+bias pattern matched previously,
  // the other input to add is fused here.
  const NodeDef& add = graph->node(matched.add);
  fused_conv.add_input(add.input(1 - matched.port_id));

  if (IsConv2D(contraction)) {
    fused_conv.set_op(kFusedConv2D);
    AddInputShapesAttr(*ctx, matched.contraction);
    CopyConv2DAttributes(contraction, &fused_conv);
  } else if (IsConv3D(contraction)) {
    fused_conv.set_op(kFusedConv3D);
    CopyConv3DAttributes(contraction, &fused_conv);
  }

  SetFusedOpAttributes(&fused_conv, {"BiasAdd", "Add", activation.op()}, 2);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_conv), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.activation] = true;
  (*nodes_to_delete)[matched.add] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return absl::OkStatus();
}

absl::Status FuseContractionWithBiasAddAndHardSwish(
    RemapperContext* ctx, std::map<string, int>* matched_nodes_map,
    std::set<int>* remove_node_indices, std::vector<bool>* invalidated_nodes,
    std::vector<bool>* nodes_to_delete) {
  auto* output_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("output"))->node();
  auto* contraction_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("contraction"))->node();
  auto* bias_add_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("bias_add"))->node();

  bool is_conv2d = IsConv2D(*contraction_node);

  NodeDef fused_node;
  fused_node.set_name(output_node->name());
  fused_node.set_op(is_conv2d ? kFusedConv2D : kFusedDepthwiseConv2dNative);
  fused_node.set_device(contraction_node->device());
  fused_node.add_input(contraction_node->input(0));
  fused_node.add_input(contraction_node->input(1));
  fused_node.add_input(bias_add_node->input(1));

  if (is_conv2d) {
    CopyConv2DAttributes(*contraction_node, &fused_node);
  } else {
    CopyDepthwiseConv2dNativeAttributes(*contraction_node, &fused_node);
  }
  SetFusedOpAttributes(&fused_node, {"BiasAdd", "_FusedHardSwish"});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());
  (*invalidated_nodes)[matched_nodes_map->at("output")] = true;

  for (const auto& node_idx : *remove_node_indices) {
    (*nodes_to_delete)[node_idx] = true;
  }
  return absl::OkStatus();
}

absl::Status FuseConv2DSwish(RemapperContext* ctx,
                             const std::map<string, int>& matched_nodes_map,
                             const std::set<int>& remove_node_indices,
                             std::vector<bool>* invalidated_nodes,
                             std::vector<bool>* nodes_to_delete) {
  const NodeDef* mul =
      ctx->graph_view.GetNode(matched_nodes_map.at("mulToswish"))->node();
  const NodeDef* conv2d =
      ctx->graph_view.GetNode(matched_nodes_map.at("conv"))->node();

  NodeDef fused_op;
  fused_op.set_name(mul->name());
  fused_op.set_op(kFusedConv2D);
  fused_op.set_device(mul->device());
  fused_op.add_input(conv2d->input(0));
  fused_op.add_input(conv2d->input(1));
  // Check if the pattern has Conv2d + BiasAdd
  if (matched_nodes_map.find("biasadd") != matched_nodes_map.end()) {
    auto* bias_add_node =
        ctx->graph_view.GetNode(matched_nodes_map.at("biasadd"))->node();
    fused_op.add_input(bias_add_node->input(1));
    SetFusedOpAttributes(&fused_op, {"BiasAdd", "_MklSwish"});
  } else {
    // pattern is conv2D + FuseBatchNorm/v2/v3
    auto* fusebatchnorm_node =
        ctx->graph_view.GetNode(matched_nodes_map.at("fusebatchnorm"))->node();
    fused_op.add_input(fusebatchnorm_node->input(1));
    fused_op.add_input(fusebatchnorm_node->input(2));
    fused_op.add_input(fusebatchnorm_node->input(3));
    fused_op.add_input(fusebatchnorm_node->input(4));
    float epsilon;
    TF_CHECK_OK(GetNodeAttr(*fusebatchnorm_node, "epsilon", &epsilon));
    SetFusedOpAttributes(&fused_op, {"FusedBatchNorm", "_MklSwish"},
                         /*num_args=*/4, /*epsilon=*/epsilon);
  }
  AddInputShapesAttr(*ctx, matched_nodes_map.at("conv"));
  CopyConv2DAttributes(*conv2d, &fused_op);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched_nodes_map.at("mulToswish")] = true;

  for (const auto& node_index : remove_node_indices) {
    (*nodes_to_delete)[node_index] = true;
  }

  return absl::OkStatus();
}

absl::Status AddFusedMatMulBiasAddAndGelu(
    RemapperContext* ctx, const std::map<string, int>& matched_nodes_map,
    const std::set<int>& remove_node_indices,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete,
    bool is_gelu_approximate) {
  auto* output_node =
      ctx->graph_view.GetNode(matched_nodes_map.at("output"))->node();
  auto* matmul_node =
      ctx->graph_view.GetNode(matched_nodes_map.at("matmul"))->node();

  NodeDef fused_node;
  // Fused node should have the name of terminal node of the fusion.
  fused_node.set_name(output_node->name());
  fused_node.set_op("_FusedMatMul");
  fused_node.set_device(matmul_node->device());
  fused_node.add_input(matmul_node->input(0));
  fused_node.add_input(matmul_node->input(1));
  if (is_gelu_approximate) {
    fused_node.add_input(matmul_node->input(2));
  } else {
    auto* bias_add_node =
        ctx->graph_view.GetNode(matched_nodes_map.at("bias_add"))->node();
    fused_node.add_input(bias_add_node->input(1));
  }
  CopyMatMulAttributes(*matmul_node, &fused_node);
  if (is_gelu_approximate)
    SetFusedOpAttributes(&fused_node, {"BiasAdd", "GeluApproximate"});
  else
    SetFusedOpAttributes(&fused_node, {"BiasAdd", "GeluExact"});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());
  (*invalidated_nodes)[matched_nodes_map.at("output")] = true;

  for (const auto& node_idx : remove_node_indices) {
    (*nodes_to_delete)[node_idx] = true;
  }
  return absl::OkStatus();
}

absl::Status AddMklLayerNorm(RemapperContext* ctx,
                             const std::map<string, int>& matched_nodes_map,
                             const std::set<int>& remove_node_indices,
                             const std::vector<string>& input_node_names,
                             std::vector<bool>* invalidated_nodes,
                             std::vector<bool>* nodes_to_delete,
                             const float epsilon) {
  auto* output_node =
      ctx->graph_view.GetNode(matched_nodes_map.at("output"))->node();

  NodeDef fused_node;
  fused_node.set_name(output_node->name());
  fused_node.set_op("_MklLayerNorm");
  fused_node.set_device(output_node->device());
  for (const auto& name : input_node_names) fused_node.add_input(name);
  auto* attr = fused_node.mutable_attr();
  auto& src_attr = output_node->attr();
  (*attr)["T"] = src_attr.at("T");
  SetAttrValue(epsilon, &(*attr)["epsilon"]);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());
  (*invalidated_nodes)[matched_nodes_map.at("output")] = true;

  for (const auto& node_idx : remove_node_indices) {
    (*nodes_to_delete)[node_idx] = true;
  }
  return absl::OkStatus();
}

absl::Status ReplaceMulMaximumWithLeakyRelu(
    RemapperContext* ctx, const std::map<string, int>& matched_nodes_map,
    const std::set<int>& remove_node_indices,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete,
    float alpha) {
  const NodeDef* maximum =
      ctx->graph_view.GetNode(matched_nodes_map.at("max_to_leakyrelu"))->node();
  const NodeDef* input =
      ctx->graph_view.GetNode(matched_nodes_map.at("input"))->node();
  const auto* alpha_node_view =
      ctx->graph_view.GetNode(matched_nodes_map.at("alpha"));

  NodeDef fused_op;
  fused_op.set_name(maximum->name());
  fused_op.set_op("LeakyRelu");
  fused_op.set_device(maximum->device());
  fused_op.add_input(input->name());

  // Addressing the corner case where the Const (which becomes an attr of
  // LeakyRelu) has input control dependencies. In that case, we transfer the
  // control dependencies to the fused op (LeakyRelu).
  if (alpha_node_view->NumControllingFanins() > 0) {
    const auto& control_fanins = alpha_node_view->GetControllingFanins();
    for (int i = 0; i < alpha_node_view->NumControllingFanins(); i++) {
      const auto* control_node_view = control_fanins[i].node_view();
      *fused_op.add_input() =
          AsControlDependency(control_node_view->node()->name());
    }
  }

  auto* attr = fused_op.mutable_attr();
  (*attr)["T"] = maximum->attr().at("T");

  SetAttrValue(alpha, &(*attr)["alpha"]);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched_nodes_map.at("max_to_leakyrelu")] = true;

  for (const auto& node_index : remove_node_indices) {
    (*nodes_to_delete)[node_index] = true;
  }

  return absl::OkStatus();
}

absl::Status ReplaceSigmoidMulWithSwish(
    RemapperContext* ctx, const std::map<string, int>& matched_nodes_map,
    const std::set<int>& remove_node_indices,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const NodeDef* mul =
      ctx->graph_view.GetNode(matched_nodes_map.at("mul_to_swish"))->node();
  const NodeDef* sigmoid =
      ctx->graph_view.GetNode(matched_nodes_map.at("sigmoid"))->node();

  NodeDef fused_op;
  fused_op.set_name(mul->name());
  fused_op.set_op("_MklSwish");
  fused_op.set_device(mul->device());
  fused_op.add_input(sigmoid->input(0));

  auto* attr = fused_op.mutable_attr();
  (*attr)["T"] = mul->attr().at("T");

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched_nodes_map.at("mul_to_swish")] = true;

  for (const auto& node_index : remove_node_indices) {
    (*nodes_to_delete)[node_index] = true;
  }
  return absl::OkStatus();
}

absl::Status AddFusedBatchNormExNode(RemapperContext* ctx,
                                     const FusedBatchNormEx& matched,
                                     std::vector<bool>* invalidated_nodes,
                                     std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& fused_batch_norm = graph->node(matched.fused_batch_norm);
  const NodeDef& activation = graph->node(matched.activation);

  VLOG(2) << "Fuse " << activation.op()
          << " with FusedBatchNorm:" << " activation=" << activation.name()
          << " side_input="
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
  absl::Status status;
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

  return absl::OkStatus();
}

absl::Status AddFusedBatchNormGradExNode(RemapperContext* ctx,
                                         const FusedBatchNormGradEx& matched,
                                         std::vector<bool>* invalidated_nodes,
                                         std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& fused_batch_norm_grad =
      graph->node(matched.fused_batch_norm_grad);
  const NodeDef& activation_grad = graph->node(matched.activation_grad);
  const NodeDef& fwd_fused_batch_norm =
      graph->node(matched.fwd_fused_batch_norm);

  VLOG(2) << "Fuse FusedBatchNormGrad with " << activation_grad.op() << ": "
          << " fused_batch_norm_grad=" << fused_batch_norm_grad.name()
          << " side_input="
          << (matched.side_input_grad != kMissingIndex
                  ? graph->node(matched.side_input_grad).name()
                  : "<none>")
          << " activation=" << activation_grad.name()
          << " corresponding FusedBatchNorm=" << fwd_fused_batch_norm.name();

  NodeDef fused_op;
  fused_op.set_op(kFusedBatchNormGradEx);
  fused_op.set_name(fused_batch_norm_grad.name());
  fused_op.set_device(fused_batch_norm_grad.device());

  fused_op.add_input(activation_grad.input(0));        // 0: y_backprop
  fused_op.add_input(fused_batch_norm_grad.input(1));  // 1: x
  fused_op.add_input(fused_batch_norm_grad.input(2));  // 2: scale
  fused_op.add_input(fused_batch_norm_grad.input(3));  // 3: reserve_space_1
  fused_op.add_input(fused_batch_norm_grad.input(4));  // 4: reserve_space_2
  fused_op.add_input(fused_batch_norm_grad.input(5));  // 5: reserve_space_3
  fused_op.add_input(fwd_fused_batch_norm.input(2));   // 6: offset
  fused_op.add_input(activation_grad.input(1));        // 7: y

  CopyFusedBatchNormGradAttributes(fused_batch_norm_grad, &fused_op);

  auto* attrs = fused_op.mutable_attr();
  // Only support Relu mode.
  SetAttrValue("Relu", &(*attrs)["activation_mode"]);

  if (matched.side_input_grad != kMissingIndex) {
    SetAttrValue(1, &(*attrs)["num_side_inputs"]);
  } else {
    SetAttrValue(0, &(*attrs)["num_side_inputs"]);
  }

  NodeDef identity_op;
  identity_op.set_op("Identity");
  identity_op.set_name(activation_grad.name());
  identity_op.set_device(fused_batch_norm_grad.device());
  identity_op.add_input(absl::StrCat(fused_batch_norm_grad.name(), ":5"));
  (*identity_op.mutable_attr())["T"] = attrs->at("T");

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  if (matched.side_input_grad != kMissingIndex) {
    mutation->AddNode(std::move(identity_op), &status);
    TF_RETURN_IF_ERROR(status);
  }
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.fused_batch_norm_grad] = true;
  if (matched.side_input_grad != kMissingIndex) {
    (*invalidated_nodes)[matched.activation_grad] = true;
  } else {
    (*nodes_to_delete)[matched.activation_grad] = true;
  }

  return absl::OkStatus();
}

absl::Status AddBatchNormNodes(RemapperContext* ctx,
                               const FusedBatchNorm& matched) {
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
  absl::Status status;

  string x_format = fused_node.attr().at(kDataFormat).s();
  if (x_format == "NCHW" || x_format == "NCDHW") {
    // Need to reshape the last 4 inputs
    NodeDef new_shape;
    const string new_shape_name =
        AddPrefixToNodeName(x_format + "Shape", fused_node.name());
    new_shape.set_name(new_shape_name);
    new_shape.set_op("Const");
    new_shape.set_device(fused_node.device());
    *new_shape.add_input() = AsControlDependency(scale);
    (*new_shape.mutable_attr())["dtype"].set_type(DT_INT32);
    if (x_format == "NCHW") {
      Tensor t(DT_INT32, {4});
      t.flat<int32>()(0) = 1;
      t.flat<int32>()(1) = -1;
      t.flat<int32>()(2) = 1;
      t.flat<int32>()(3) = 1;
      t.AsProtoTensorContent(
          (*new_shape.mutable_attr())["value"].mutable_tensor());
    } else {
      Tensor t(DT_INT32, {5});
      t.flat<int32>()(0) = 1;
      t.flat<int32>()(1) = -1;
      t.flat<int32>()(2) = 1;
      t.flat<int32>()(3) = 1;
      t.flat<int32>()(4) = 1;
      t.AsProtoTensorContent(
          (*new_shape.mutable_attr())["value"].mutable_tensor());
    }
    mutation->AddNode(std::move(new_shape), &status);
    TF_RETURN_IF_ERROR(status);

    NodeDef reshaped_scale;
    reshaped_scale.set_name(
        AddPrefixToNodeName(x_format + "ShapedScale", fused_node.name()));
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
        AddPrefixToNodeName(x_format + "ShapedOffset", fused_node.name()));
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
        AddPrefixToNodeName(x_format + "ShapedMean", fused_node.name()));
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
        AddPrefixToNodeName(x_format + "ShapedVariance", fused_node.name()));
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

absl::Status AddTensorToHashBucketNode(RemapperContext* ctx,
                                       const TensorToHashBucket& matched,
                                       std::vector<bool>* invalidated_nodes,
                                       std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& pre_as_string = graph->node(matched.pre_as_string);
  const NodeDef& as_string = graph->node(matched.as_string);
  const NodeDef& string_to_hash_bucket =
      graph->node(matched.string_to_hash_bucket);
  VLOG(2) << "Fuse AsString with StringToHashBucketFast:" << " as_string="
          << as_string.name()
          << " string_to_hash_bucket=" << string_to_hash_bucket.name()
          << " on device=" << pre_as_string.device();

  NodeDef fused_op;
  fused_op.set_name(string_to_hash_bucket.name());
  fused_op.set_device(pre_as_string.device());
  fused_op.add_input(as_string.input(0));  // 0: input
  fused_op.set_op(kTensorToHashBucket);

  auto* attr = fused_op.mutable_attr();
  auto& src_attr0 = as_string.attr();
  auto& src_attr1 = string_to_hash_bucket.attr();
  (*attr)["T"] = src_attr0.at("T");
  (*attr)["num_buckets"] = src_attr1.at("num_buckets");

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.string_to_hash_bucket] = true;
  (*nodes_to_delete)[matched.as_string] = true;

  return absl::OkStatus();
}

absl::Status AddFusedBatchMatMul(RemapperContext* ctx,
                                 const std::map<string, int>& matched_nodes_map,
                                 const std::set<int>& remove_node_indices,
                                 const std::vector<string>& input_node_names,
                                 std::vector<bool>* invalidated_nodes,
                                 std::vector<bool>* nodes_to_delete) {
  auto* output_node =
      ctx->graph_view.GetNode(matched_nodes_map.at("output"))->node();
  auto* batch_matmul_node =
      ctx->graph_view.GetNode(matched_nodes_map.at("batch_matmul"))->node();

  NodeDef fused_node;
  fused_node.set_name(output_node->name());
  fused_node.set_op("_MklFusedBatchMatMulV2");
  fused_node.set_device(batch_matmul_node->device());
  for (const auto& name : input_node_names) fused_node.add_input(name);

  CopyBatchMatMulAttributes(*batch_matmul_node, &fused_node);
  SetFusedOpAttributes(&fused_node, {"Mul", "Add"}, /*num_args=*/2);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());
  (*invalidated_nodes)[matched_nodes_map.at("output")] = true;

  for (const auto& node_idx : remove_node_indices) {
    (*nodes_to_delete)[node_idx] = true;
  }
  return absl::OkStatus();
}

// Helper function to get data of type T from a given tensor and
// return them in a vector and casted to type U.
// Note - use this function only when type cast is safe from T to U.
template <typename T, typename U>
std::vector<U> GetTensorValues(const Tensor& tensor) {
  std::vector<U> result_vector;
  int item_count = tensor.flat<T>().size();
  result_vector.reserve(item_count);
  for (int i = 0; i < item_count; i++) {
    result_vector.push_back((U)(tensor.flat<T>()(i)));
  }
  return result_vector;
}

absl::Status AddMklFusedInstanceNorm(RemapperContext* ctx,
                                     std::map<string, int>* matched_nodes_map,
                                     std::set<int>* remove_node_indices,
                                     std::vector<bool>* invalidated_nodes,
                                     std::vector<bool>* nodes_to_delete,
                                     bool fuse_activation) {
  auto* output_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("output"))->node();
  auto* input_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("input"))->node();
  auto* gamma_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("gamma"))->node();
  auto* beta_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("beta"))->node();
  auto* epsilon_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("epsilon"))->node();
  auto* mean_axes_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("r_indices1"))->node();

  if (!mean_axes_node || mean_axes_node->op() != "Const") {
    VLOG(2) << "Mean reduction axes node is not valid, abort fusion";
    return absl::OkStatus();
  }
  DataType dtype;
  Tensor mean_axes_tensor;
  if (!mean_axes_tensor.FromProto(
          mean_axes_node->attr().at("value").tensor())) {
    VLOG(2) << "Unable to get mean reduction axes, abort fusion";
    return absl::OkStatus();
  }
  dtype = mean_axes_tensor.dtype();
  if (dtype != DT_INT32 && dtype != DT_INT64) {
    VLOG(2) << "Unexpected mean reduction axes data type, abort fusion";
    return absl::OkStatus();
  }
  std::vector<int> reduction_axes =
      (dtype == DT_INT32) ? GetTensorValues<int32, int>(mean_axes_tensor)
                          : GetTensorValues<int64, int>(mean_axes_tensor);

  NodeDef* activation_node = nullptr;
  if (fuse_activation) {
    activation_node =
        ctx->graph_view.GetNode(matched_nodes_map->at("activation"))->node();
    if (!activation_node) {
      VLOG(2) << "Error to retrieve activation node, abort fusion";
      return absl::OkStatus();
    }
    if (!IsLeakyRelu(*activation_node) && !IsRelu(*activation_node)) {
      VLOG(2) << "Unsupported activation node, abort fusion";
      return absl::OkStatus();
    }
  }

  NodeDef fused_node;
  fused_node.set_op("_MklFusedInstanceNorm");
  fused_node.set_device(output_node->device());
  fused_node.add_input(input_node->name());
  fused_node.add_input(gamma_node->name());
  fused_node.add_input(beta_node->name());
  auto* attr = fused_node.mutable_attr();
  auto& src_attr = output_node->attr();
  (*attr)["T"] = src_attr.at("T");

  Tensor epsilon_tensor;
  float epsilon_value = 0.0001;
  if (epsilon_node != nullptr && epsilon_node->op() == "Const" &&
      epsilon_tensor.FromProto(epsilon_node->attr().at("value").tensor())) {
    dtype = epsilon_tensor.dtype();
    if (dtype == DT_BFLOAT16) {
      epsilon_value = static_cast<float>(epsilon_tensor.flat<bfloat16>()(0));
    } else if (dtype == DT_HALF) {
      epsilon_value = static_cast<float>(epsilon_tensor.flat<Eigen::half>()(0));
    } else if (dtype == DT_FLOAT) {
      epsilon_value = epsilon_tensor.flat<float>()(0);
    }
    SetAttrValue(epsilon_value, &(*attr)["epsilon"]);
  }

  SetAttrValue(reduction_axes, &(*attr)["reduction_axes"]);

  if (fuse_activation) {
    fused_node.set_name(activation_node->name());
    string activation_op = activation_node->op();
    absl::string_view fused_items[] = {activation_op};
    SetAttrValue(absl::Span<absl::string_view>(fused_items),
                 &(*attr)["fused_ops"]);
    if (activation_op == "LeakyRelu") {
      auto& activation_attr = activation_node->attr();
      (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
    }
  } else {
    fused_node.set_name(output_node->name());
  }

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());
  if (fuse_activation) {
    (*invalidated_nodes)[matched_nodes_map->at("activation")] = true;
  } else {
    (*invalidated_nodes)[matched_nodes_map->at("output")] = true;
  }
  for (const auto& node_idx : *remove_node_indices) {
    (*nodes_to_delete)[node_idx] = true;
  }
  return absl::OkStatus();
}

// This function supports below patterns that require inferred
// shapes:
// 1. Contraction + Add.
// 2. Contraction + Add + Activation.
// 3. Contraction + BiasAdd/BiasSemanticAdd + Add.
// 4. Contraction + BiasAdd/BiasSemanticAdd + Add + Activation.
// Contraction candidate: MatMul, Conv2D, Conv3D, DepthwiseConv2dNative.
bool IsContractionWithAdd(const RemapperContext& ctx, int node_index) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);

  auto is_supported_add_input = [](const auto* node_view) -> bool {
    if (IsConvOrMatMul(*node_view->node())) return true;
    // IsAdd will verify BiasSemanticAdd.
    if (IsBiasAdd(*node_view->node()) || IsAdd(*node_view->node())) {
      if (node_view->NumRegularFanins() < 2) return false;
      const auto& bias_add_fanin_0 = node_view->GetRegularFanin(0);
      const auto& bias_add_fanin_1 = node_view->GetRegularFanin(1);
      return IsConvOrMatMul(*bias_add_fanin_0.node_view()->node()) ||
             IsConvOrMatMul(*bias_add_fanin_1.node_view()->node());
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

  // Dealing with the Contraction + Add or Contraction + BiasAdd/BiasSemanticAdd
  // + Add patterns.
  if (is_supported_add(node_view)) {
    return true;
  }
  // Dealing with the Contraction + Add + Activation  or Contraction +
  // BiasAdd/BiasSemanticAdd + Add + Activation pattern.
  if (IsSupportedActivation(*node_view->node(), nullptr)) {
    for (int i = 0; i < node_view->NumRegularFanins(); i++) {
      const auto& fanin_i = node_view->GetRegularFanin(i);
      if (is_supported_add(fanin_i.node_view())) return true;
    }
  }

  return false;
}

bool FindSoftplusAndTanhAndMul(RemapperContext* ctx, int node_index,
                               std::map<string, int>* matched_nodes_map,
                               std::set<int>* remove_node_indices) {
  // Mish fusion is enabled only with oneDNN library.
  if (!IsMKLEnabled()) return false;

  using utils::MatchingDirection;
  using utils::NodeStatus;
  // clang-format off
  //                Convert Softplus+Tanh+Mul to Mish
  //          From Graph                          To Graph
  //          -----------                         ---------
  //    Conv2D  <-  Filter(const)           Conv2D  <-  Filter(const)
  //      !                                   !
  //      V                                   V
  //    BiasAdd <-  bias(const)             BiasAdd <-  bias(const)
  //      !                                   !
  //      V                                   !
  //  ---- ----                               !
  //  !       !                               !
  //  !       V                               !
  //  !    Softplus                           !
  //  !       !                               !
  //  !       V                               !
  //  !     Tanh                              !
  //  !       !                               !
  //  !       V                               !
  //  ---   ---                               !
  //     !  !                                 !
  //     !  !                                 !
  //     V  V                                 V
  //      Mul                           _MklFusedMish
  //      !                                   !
  //      V                                   V

  utils::OpTypePattern softplustanhmul_pattern {
    "Mul", "mul_to_mish", NodeStatus::kReplace,
    {
      {
        "Tanh", "tanh", NodeStatus::kRemove,
        {
          {
            "Softplus", "softplus", NodeStatus::kRemove,
            {
              {"*", "input", NodeStatus::kRemain}
            }
          }
        }
      },
      {"*", "input", NodeStatus::kRemain}
    }
  };
  // clang-format on

  // check for data types
  auto* mul_node_def = ctx->graph_view.GetNode(node_index)->node();
  if (!HasDataType(mul_node_def, DT_FLOAT) &&
      !HasDataType(mul_node_def, DT_HALF) &&
      !HasDataType(mul_node_def, DT_BFLOAT16))
    return false;

  if (!NodeIsOnCpu(mul_node_def)) return false;

  bool found_op_type_match = false;
  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx->graph_view));
  matched_nodes_map->clear();
  remove_node_indices->clear();
  found_op_type_match = graph_matcher.GetMatchedNodes(
      softplustanhmul_pattern, {}, ctx->graph_view.GetNode(node_index),
      matched_nodes_map, remove_node_indices);

  if (found_op_type_match) {
    NodeDef* matched_softplus_node =
        ctx->graph_view.GetNode(matched_nodes_map->at("softplus"))->node();
    auto in_tensor_softplus = matched_softplus_node->input(0);
    if ((mul_node_def->input(0) != in_tensor_softplus) &&
        (mul_node_def->input(1) != in_tensor_softplus)) {
      // If the input tensor of Softplus doesn't match with either of input
      // tensors of mul return false
      found_op_type_match = false;
    }
  }

  return found_op_type_match;
}

absl::Status ReplaceSoftplusTanhAndMulWithMish(
    RemapperContext* ctx, const std::map<string, int>* matched_nodes_map,
    const std::set<int>* remove_node_indices,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  // Fuse Softplus + Tanh + Mul to Mish
  auto* old_mul_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("mul_to_mish"))->node();
  auto* softplus_node =
      ctx->graph_view.GetNode(matched_nodes_map->at("softplus"))->node();

  NodeDef fused_node;
  fused_node.set_name(old_mul_node->name());
  fused_node.set_op("_MklFusedMish");
  fused_node.set_device(old_mul_node->device());
  fused_node.add_input(softplus_node->input(0));

  auto* fused_node_attr = fused_node.mutable_attr();
  (*fused_node_attr)["T"] = old_mul_node->attr().at("T");

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  absl::Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());
  (*invalidated_nodes)[matched_nodes_map->at("mul_to_mish")] = true;

  for (const auto& node_index : *remove_node_indices) {
    (*nodes_to_delete)[node_index] = true;
  }

  return absl::OkStatus();
}

// Check if a node is a candidate to one of the patterns that require inferred
// shapes:
//   (1) Splitting FusedBatchNorm into primitives.
//   (2) Fusing side input and/or activation into FusedBatchNorm.
//   (3) Fusing Conv2D biasadd and relu on GPU
//   (4) INTEL_MKL specific: Conv2D -> Add or Conv2D -> BiasAdd -> Add.
//   (5) Fusing side output and/or activation into FusedBatchNormGrad.
bool RequiresInferredShapes(const RemapperContext& ctx, int node_index,
                            const Cluster* cluster) {
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

  const auto is_act_biasadd_conv_candidate = [&]() -> bool {
    if (!IsSupportedActivation(*node_def, cluster)) return false;

    if (!RuntimeFusionEnabled(cluster) && !IsRelu(*node_def)) return false;

    const auto is_compatible_dtype = [&](const NodeDef& node) -> bool {
      // The cuDNN fusion with relu6, elu, leakyrelu is realized by the runtime
      // compiled kernels which only support fp16.
      bool fp16_only =
          IsRelu6(*node_def) || IsElu(*node_def) || IsLeakyRelu(*node_def);
      DataType dtype = GetDataTypeFromAttr(node, "T");
      return dtype == DT_HALF || (!fp16_only && dtype == DT_FLOAT);
    };
    if (!is_compatible_dtype(*node_def)) return false;

    if (node_view->NumRegularFanins() < 1) return false;
    const auto& relu_fanin_0 = node_view->GetRegularFanin(0);
    const auto* relu_fanin_0_node_view = relu_fanin_0.node_view();
    const auto* relu_fanin_0_node_def = relu_fanin_0_node_view->node();

    if (!IsBiasAdd(*relu_fanin_0_node_def) && !IsAdd(*relu_fanin_0_node_def))
      return false;
    if (!is_compatible_dtype(*relu_fanin_0_node_def)) return false;

    if (relu_fanin_0_node_view->NumRegularFanins() < 1) return false;

    const auto& biasadd_fanin_0 = relu_fanin_0_node_view->GetRegularFanin(0);
    const auto* biasadd_fanin_0_node_def = biasadd_fanin_0.node_view()->node();

    if (!IsConv2D(*biasadd_fanin_0_node_def) &&
        !IsConv3D(*biasadd_fanin_0_node_def))
      return false;
    if (!is_compatible_dtype(*biasadd_fanin_0_node_def)) return false;
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

  // Candidate for a FusedBatchNormGrad fusion.
  const auto is_batch_norm_grad_fusion_candidate = [&]() -> bool {
    if (!IsFusedBatchNormGrad(*node_def)) return false;

    if (node_view->NumRegularFanins() < 1) return false;
    const auto& bn_fanin_0 = node_view->GetRegularFanin(0);
    const auto* bn_fanin_0_node_view = bn_fanin_0.node_view();
    const auto* bn_fanin_0_node_def = bn_fanin_0_node_view->node();

    if (IsReluGrad(*bn_fanin_0_node_def)) {
      // ReluGrad + FusedBatchNormGrad.
      return true;
    }

    return false;
  };

  // Candidate for a FusedMatmul fusion (MatMul + BiasAdd + GeluExact).
  const auto is_matmul_gelu_exact_fusion_candidate = [&]() -> bool {
    if (!RuntimeFusionEnabled(cluster)) return false;
    DataType node_dtype = GetDataTypeFromAttr(*node_def, "T");
    if (node_dtype != DT_HALF) return false;
    return IsMatchedMatMulBiasAddAndGeluExact(const_cast<RemapperContext&>(ctx),
                                              node_index);
  };

  // Candidate for a FusedMatmul fusion (MatMul + BiasAdd + Tanh/Sigmoid).
  const auto is_act_biasadd_matmul_candidate = [&]() -> bool {
    if (!IsTanh(*node_def) && !IsSigmoid(*node_def)) return false;
    if (!RuntimeFusionEnabled(cluster)) return false;
    DataType act_dtype = GetDataTypeFromAttr(*node_def, "T");
    if (act_dtype != DT_HALF) return false;

    if (node_view->NumRegularFanins() < 1) return false;
    const auto& relu_fanin_0 = node_view->GetRegularFanin(0);
    const auto* relu_fanin_0_node_view = relu_fanin_0.node_view();
    const auto* relu_fanin_0_node_def = relu_fanin_0_node_view->node();

    if (!IsBiasAdd(*relu_fanin_0_node_def) && !IsAdd(*relu_fanin_0_node_def)) {
      return false;
    }
    DataType biasadd_dtype = GetDataTypeFromAttr(*relu_fanin_0_node_def, "T");
    if (biasadd_dtype != DT_HALF) return false;

    if (relu_fanin_0_node_view->NumRegularFanins() < 1) return false;

    const auto& biasadd_fanin_0 = relu_fanin_0_node_view->GetRegularFanin(0);
    const auto* biasadd_fanin_0_node_def = biasadd_fanin_0.node_view()->node();

    if (!IsMatMul(*biasadd_fanin_0_node_def)) return false;
    DataType matmul_dtype = GetDataTypeFromAttr(*biasadd_fanin_0_node_def, "T");
    if (matmul_dtype != DT_HALF) return false;
    return true;
  };

  if (IsMKLEnabled())
    return is_batch_norm_candidate() || is_batch_norm_fusion_candidate() ||
           IsContractionWithAdd(ctx, node_index) ||
           is_act_biasadd_conv_candidate() || IsBiasAdd(*node_def) ||
           IsTranspose(*node_def);

  return is_act_biasadd_conv_candidate() || is_batch_norm_candidate() ||
         is_batch_norm_fusion_candidate() ||
         is_batch_norm_grad_fusion_candidate() ||
         is_matmul_gelu_exact_fusion_candidate() ||
         is_act_biasadd_matmul_candidate();
}

inline bool IsXlaCpuGlobalJitOn() {
  std::vector<string> tf_xla_flags;
  const std::string tf_xla_cpu_global_jit = "--tf_xla_cpu_global_jit";
  TF_CHECK_OK(ReadStringsFromEnvVar("TF_XLA_FLAGS", "", &tf_xla_flags));
  return std::find(tf_xla_flags.begin(), tf_xla_flags.end(),
                   tf_xla_cpu_global_jit) != tf_xla_flags.end();
}
}  // namespace

absl::Status Remapper::Optimize(Cluster* cluster, const GrapplerItem& item,
                                GraphDef* optimized_graph) {
  GrapplerItem mutable_item = item;
  absl::Status status;
  bool xla_cpu_jit_disable_fusion =
      xla_auto_clustering_on_ && IsXlaCpuGlobalJitOn();
#ifdef DNNL_AARCH64_USE_ACL
  xla_cpu_jit_disable_fusion = false;
#endif  // DNNL_AARCH64_USE_ACL
  RemapperContext ctx(&mutable_item, &status, cpu_layout_conversion_,
                      xla_auto_clustering_on_, xla_cpu_jit_disable_fusion);
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
    if (!ctx.inferred_graph_properties &&
        RequiresInferredShapes(ctx, i, cluster)) {
      const bool assume_valid_feeds = opt_level_ == RewriterConfig::AGGRESSIVE;
      TF_RETURN_IF_ERROR(ctx.graph_properties.InferStatically(
          assume_valid_feeds,
          /*aggressive_shape_inference=*/false,
          /*include_input_tensor_values=*/true,
          /*include_output_tensor_values=*/false));
      ctx.inferred_graph_properties = true;
    }

    ContractionWithBiasAddAndAdd contract_with_bias_and_add;
    ContractionWithActivation contract_with_activation;
    ContractionWithBiasAndAddActivation contract_with_bias_and_add_activation;

    // Store dimensions so that they can be retrieved later in
    // mkl_layout_rewrite_pass when deciding whether to rewrite node.
    if (IsConv2D(ctx.graph_view.graph()->node(i)) ||
        IsFusedBatchNorm(ctx.graph_view.graph()->node(i)) ||
        IsDepthwiseConv2dNative(ctx.graph_view.graph()->node(i)) ||
        IsBiasAdd(ctx.graph_view.graph()->node(i)) ||
        IsTranspose(ctx.graph_view.graph()->node(i)) ||
        IsSigmoid(ctx.graph_view.graph()->node(i)) ||
        IsMatMul(ctx.graph_view.graph()->node(i))) {
      AddInputShapesAttr(ctx, i);
    }

    if (IsMKLEnabled() && !ctx.xla_cpu_jit_disable_fusion) {
      const auto* node_view = ctx.graph_view.GetNode(i);
      const auto* node_def = node_view->node();
      const string& type_attr = "T";
      DataType dtype = GetDataTypeFromAttr(*node_def, type_attr);
      // Check if BF16 and FP16 data types are supported by oneDNN,
      // skipping fusions if it is not supported.
      // Since the input type could be DT_INVALID, run type check only for
      // BF16 and FP16.
      if ((dtype == DT_BFLOAT16 || dtype == DT_HALF) &&
          !IsDataTypeSupportedByOneDNNOnThisCPU(dtype))
        continue;

      // Remap Conv2D+BiasAdd+Add+relu into the _FusedConv2D.
      // or Remap Conv3D+BiasAdd+Add+relu into _FusedConv3D
      if (FindContractionWithBiasAndAddActivation(
              ctx, i, &contract_with_bias_and_add_activation)) {
        TF_RETURN_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_bias_and_add_activation,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap {_FusedConv2D, _FusedConv3D} + {LeakyRelu, _MklFusedMish}
      // into {_FusedConv2D, _FusedConv3D}
      if (FindFusedConvWithFusedActivation(ctx, i, &contract_with_activation)) {
        TF_RETURN_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_activation,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }

#ifndef DNNL_AARCH64_USE_ACL
      // Remap {Conv2D,Conv3D}+BiasAdd+Add into the _FusedConv2D/3D.
      if (FindContractionWithBiasAddAndAdd(ctx, i,
                                           &contract_with_bias_and_add)) {
        TF_RETURN_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_bias_and_add,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }
#endif

      PadWithConv3D pad_with_conv3d;
      // Remap Pad+{Conv3D,_FusedConv3D} into the _FusedConv3D.
      if (FindPadWithConv3D(ctx, i, &pad_with_conv3d)) {
        TF_RETURN_IF_ERROR(AddFusedConv3DNode(
            &ctx, pad_with_conv3d, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      std::map<string, int> matched_nodes_map;
      std::set<int> remove_node_indices;
      std::vector<string> input_node_names;

      // Remap {Conv2D|DepthwiseConv2D} + BiasAdd + HardSwish subgraph
      if (FindContractionWithBiasAddAndHardSwish(ctx, i, &matched_nodes_map,
                                                 &remove_node_indices)) {
        TF_RETURN_IF_ERROR(FuseContractionWithBiasAddAndHardSwish(
            &ctx, &matched_nodes_map, &remove_node_indices, &invalidated_nodes,
            &nodes_to_delete));
        continue;
      }

      // Softplus + Tanh + Mul to Mish conversion
      matched_nodes_map.clear();
      remove_node_indices.clear();
      if (FindSoftplusAndTanhAndMul(&ctx, i, &matched_nodes_map,
                                    &remove_node_indices)) {
        TF_RETURN_IF_ERROR(ReplaceSoftplusTanhAndMulWithMish(
            &ctx, &matched_nodes_map, &remove_node_indices, &invalidated_nodes,
            &nodes_to_delete));
        continue;
      }

      // Remap BatchMatMul+Mul+AddV2 into the _FusedBatchMatMul.
      matched_nodes_map.clear();
      remove_node_indices.clear();
      input_node_names.clear();
      if (FindFusedBatchMatMul(&ctx, i, &matched_nodes_map,
                               &remove_node_indices, &input_node_names)) {
        TF_RETURN_IF_ERROR(AddFusedBatchMatMul(
            &ctx, matched_nodes_map, remove_node_indices, input_node_names,
            &invalidated_nodes, &nodes_to_delete));
        continue;
      }

#ifndef DNNL_AARCH64_USE_ACL
      // Fuse Conv2d + BiasAdd/FusedBatchNorm + Swish.
      std::map<string, int> fusedconv2dSwish_matched_nodes_map;
      std::set<int> fusedconv2dSwish_remove_node_indices;
      if (FindConv2DSwish(&ctx, i, &fusedconv2dSwish_matched_nodes_map,
                          &fusedconv2dSwish_remove_node_indices)) {
        TF_RETURN_IF_ERROR(
            FuseConv2DSwish(&ctx, fusedconv2dSwish_matched_nodes_map,
                            fusedconv2dSwish_remove_node_indices,
                            &invalidated_nodes, &nodes_to_delete));
        continue;
      }
#endif

      // Remap Maximum(x, alpha * x) pattern, fuse them into the LeakyRelu(x).
      std::map<string, int> mulmax_matched_nodes_map;
      std::set<int> mulmax_remove_node_indices;
      float alpha;
      if (FindMulAndMaximum(&ctx, i, &mulmax_matched_nodes_map,
                            &mulmax_remove_node_indices, &alpha)) {
        TF_RETURN_IF_ERROR(ReplaceMulMaximumWithLeakyRelu(
            &ctx, mulmax_matched_nodes_map, mulmax_remove_node_indices,
            &invalidated_nodes, &nodes_to_delete, alpha));
        continue;
      }

      // Remap Mul(x, Sigmoid(x)) pattern, fuse them into the Swish(x).
      std::map<string, int> sigmoidmul_matched_nodes_map;
      std::set<int> sigmoidmul_remove_node_indices;
      if (FindSigmoidAndMul(&ctx, i, &sigmoidmul_matched_nodes_map,
                            &sigmoidmul_remove_node_indices)) {
        bool replace = true;
#ifdef DNNL_AARCH64_USE_ACL
        // Need to check whether the cost of rewriting node
        // to execute using oneDNN kernel will be amortised
        // based on the size of the input
        const int sigmoid_idx = sigmoidmul_matched_nodes_map.at("sigmoid");
        // We need to infer what is the shape of sigmoid
        AddInputShapesAttr(ctx, sigmoid_idx);
        const NodeDef* sigmoid = ctx.graph_view.GetNode(sigmoid_idx)->node();
        const int intra_op_parallelism_threads =
            item.optimization_options().intra_op_parallelism_threads;
        double total_mflops =
            CalculateNodeMFlops(AttrSlice(*sigmoid), "Sigmoid");
        double thr =
            FindRewriteThreshold("Sigmoid", intra_op_parallelism_threads);

        if (total_mflops != -1 && total_mflops < thr) {
          // The overhead of using oneDNN kernel is not amortized
          // so we are not going to rewrite node
          replace = false;
        }
#endif  // DNNL_AARCH64_USE_ACL
        if (replace) {
          TF_RETURN_IF_ERROR(
              ReplaceSigmoidMulWithSwish(&ctx, sigmoidmul_matched_nodes_map,
                                         sigmoidmul_remove_node_indices,
                                         &invalidated_nodes, &nodes_to_delete));
          continue;
        }
      }

      // Remap smaller ops from layernorm python api into _MklLayerNorm
      matched_nodes_map.clear();
      remove_node_indices.clear();
      input_node_names.clear();
      float epsilon = 0.001;
      if (FindMklLayerNorm(&ctx, i, &matched_nodes_map, &remove_node_indices,
                           &input_node_names, &epsilon)) {
        TF_RETURN_IF_ERROR(AddMklLayerNorm(
            &ctx, matched_nodes_map, remove_node_indices, input_node_names,
            &invalidated_nodes, &nodes_to_delete, epsilon));
        continue;
      }

      // Remap ops that make up instancenorm followed by Relu or LeakyRelu
      // into _MklFusedInstanceNorm
      matched_nodes_map.clear();
      remove_node_indices.clear();
      if (FindInstanceNormWithActivation(&ctx, i, &matched_nodes_map,
                                         &remove_node_indices)) {
        TF_RETURN_IF_ERROR(AddMklFusedInstanceNorm(
            &ctx, &matched_nodes_map, &remove_node_indices, &invalidated_nodes,
            &nodes_to_delete, true));
        continue;
      }

      // Remap ops that make up instancenorm into _MklFusedInstanceNorm
      matched_nodes_map.clear();
      remove_node_indices.clear();
      if (FindInstanceNorm(&ctx, i, &matched_nodes_map, &remove_node_indices)) {
        TF_RETURN_IF_ERROR(AddMklFusedInstanceNorm(
            &ctx, &matched_nodes_map, &remove_node_indices, &invalidated_nodes,
            &nodes_to_delete, false));
        continue;
      }
    }

    // Remap MatMul + BiasAdd + gelu-subgraph
    std::map<string, int> matched_nodes_map;
    std::set<int> remove_node_indices;
    bool is_gelu_approximate = false;
    if (FindMatMulBiasAddAndGelu(&ctx, i, cluster, &matched_nodes_map,
                                 &remove_node_indices, &is_gelu_approximate)) {
      TF_RETURN_IF_ERROR(AddFusedMatMulBiasAddAndGelu(
          &ctx, matched_nodes_map, remove_node_indices, &invalidated_nodes,
          &nodes_to_delete, is_gelu_approximate));
      continue;
    }

    // Fusions are disabled on XLA CPU in IsCpuCompatible(...) invoked by the
    // following fusions.
    //
    // Remap {Conv2D,DepthwiseConv2D,MatMul}+BiasAdd into the
    // _Fused{Conv2D,DepthwiseConv2dNative,MatMul}
    ContractionWithBiasAdd contract_with_bias;
    if (allow_non_differentiable_rewrites &&
        FindContractionWithBias(ctx, i, &contract_with_bias)) {
      TF_RETURN_IF_ERROR(AddFusedContractionNode(
          &ctx, contract_with_bias, &invalidated_nodes, &nodes_to_delete));
      continue;
    }

    // Remap {Conv2D,DepthwiseConv2D,MatMul,Conv3D}+BiasAdd+Activation into the
    // _Fused{Conv2D,DepthwiseConv2dNative,MatMul,Conv3D}.
    ContractionWithBiasAddAndActivation contract_with_bias_and_activation;
    if (allow_non_differentiable_rewrites &&
        FindContractionWithBiasAndActivation(
            ctx, cluster, i, &contract_with_bias_and_activation)) {
      TF_RETURN_IF_ERROR(
          AddFusedContractionNode(&ctx, contract_with_bias_and_activation,
                                  &invalidated_nodes, &nodes_to_delete));
      continue;
    }

    // NOTE: We can only fuse BatchNorm into Conv2D nodes. In theory we can do
    // it for MatMul as well, but in practice this pattern does not appear in
    // real Tensorflow graphs.

    // Remap {Conv2D, Conv3D}+Squeeze+BiasAdd into the {_FusedConv2D,
    // _FusedConv3D}+Squeeze.
    ContractionWithSqueezeAndBiasAdd contract_with_squeeze_and_bias;
    if (allow_non_differentiable_rewrites &&
        FindConvWithSqueezeAndBias(ctx, i, &contract_with_squeeze_and_bias)) {
      TF_RETURN_IF_ERROR(AddFusedConvNode(&ctx, contract_with_squeeze_and_bias,
                                          &invalidated_nodes,
                                          &nodes_to_delete));
      continue;
    }

#ifndef DNNL_AARCH64_USE_ACL
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
#endif  // !DNNL_AARCH64_USE_ACL

    // Remap FusedBatchNorm+<SideInput>+<Activation> into the _FusedBatchNormEx.
    FusedBatchNormEx fused_batch_norm_ex;
    if (allow_non_differentiable_rewrites &&
        FindFusedBatchNormEx(ctx, i, &fused_batch_norm_ex)) {
      TF_RETURN_IF_ERROR(AddFusedBatchNormExNode(
          &ctx, fused_batch_norm_ex, &invalidated_nodes, &nodes_to_delete));
      continue;
    }

    // This fusion is enabled on GPU only.
    FusedBatchNormGradEx fused_batch_norm_grad_ex;
    if (allow_non_differentiable_rewrites &&
        FindFusedBatchNormGradEx(ctx, i, &fused_batch_norm_grad_ex)) {
      TF_RETURN_IF_ERROR(
          AddFusedBatchNormGradExNode(&ctx, fused_batch_norm_grad_ex,
                                      &invalidated_nodes, &nodes_to_delete));
      continue;
    }

    TensorToHashBucket tensor_to_hash_bucket;
    if (allow_non_differentiable_rewrites &&
        FindTensorToHashBucket(ctx, i, &tensor_to_hash_bucket)) {
      TF_RETURN_IF_ERROR(AddTensorToHashBucketNode(
          &ctx, tensor_to_hash_bucket, &invalidated_nodes, &nodes_to_delete));
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

  return absl::OkStatus();
}

}  // namespace grappler
}  // namespace tensorflow
