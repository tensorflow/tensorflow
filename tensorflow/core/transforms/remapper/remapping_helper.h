/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TRANSFORMS_REMAPPER_REMAPPING_HELPER_H_
#define TENSORFLOW_CORE_TRANSFORMS_REMAPPER_REMAPPING_HELPER_H_

#include <string>

#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/transforms/utils/op_cat_helper.h"
#include "tensorflow/core/transforms/utils/utils.h"

namespace mlir {
namespace tfg {

// The following structures store info of the operations to be fused. These
// are mainly used for combining operands info and attributes for a fused
// operation. They are also used for some predicate functions like
// `IsCpuCompatible` and `IsGpuCompatible` to check if the relevant fusion is
// supported on CPU and GPU, respectively. Another reason to keep these
// structures is to follow similar logics in current grappler-remapper.
// TODO(intel-tf): Remove redundancies once the similar functionality is
// achieved by tfg-remapper.
struct ContractionBiasAdd {
  Operation* contraction;
  Operation* bias_add;
};

struct ContractionBiasAddActivation {
  Operation* contraction;
  Operation* bias_add;
  Operation* activation;
};

struct ContractionBiasAddAdd {
  Operation* contraction;
  Operation* bias_add;
  Operation* add;
};

struct ContractionBiasAddAddActivation {
  Operation* contraction;
  Operation* bias_add;
  Operation* add;
  Operation* activation;
};

struct FusedBatchNormEx {
  Operation* fused_batch_norm;
  Value side_input;
  Operation* activation;
};

class OpPropertyHelper : public OpCatHelper {
 public:
  OpPropertyHelper() = default;
  explicit OpPropertyHelper(TFGraphDialect* dialect,
                            bool onednn_enabled = false,
                            bool xla_auto_clustering = false)
      : OpCatHelper(dialect),
        is_onednn_enabled_(onednn_enabled),
        is_xla_auto_clustering_enabled_(xla_auto_clustering) {}

  bool HasControlOperandsOrResultUsers(Operation* op) const {
    TFOp wrapper_op(op);
    bool has_ctl_operands = !(wrapper_op.getControlOperands().empty());
    bool has_ctl_ret_users = !(wrapper_op.controlRet().getUsers().empty());
    if (has_ctl_operands || has_ctl_ret_users)
      return true;
    else
      return false;
  }

  // This function is to be used for an operation that has at least 1
  // non-control result.
  bool HasAtMostOneUserOfResult0(Operation* op) const {
    // All tfg operation has 1 control result. When the operation has at least 1
    // non-control result, the number of results should be at least 2.
    return op->getNumResults() > 1 &&
           (op->getResult(0).hasOneUse() || op->getResult(0).use_empty());
  }

  bool IsContraction(Operation* op) const {
    return dialect_->IsConv2D(op) || dialect_->IsConv3D(op) ||
           dialect_->IsDepthwiseConv2dNative(op) || dialect_->IsMatMul(op);
  }

  bool HaveSameDataType(Operation* lhs_op, Operation* rhs_op,
                        StringRef attr_name = "T") const {
    auto lhs_attr = lhs_op->getAttrOfType<TypeAttr>(attr_name);
    auto rhs_attr = rhs_op->getAttrOfType<TypeAttr>(attr_name);
    if (!lhs_attr || !rhs_attr) return false;
    return lhs_attr == rhs_attr;
  }

  // This function is currently used by contraction ops.
  bool IsGpuCompatibleDataType(Operation* contraction_op,
                               StringRef attr_name = "T") const {
    auto attr = contraction_op->getAttrOfType<TypeAttr>(attr_name);
    if (!attr) return false;
    Type dtype = attr.getValue();
    if (dialect_->IsConv2D(contraction_op)) {
      return mlir::isa<Float32Type>(dtype);
    } else if (dialect_->IsMatMul(contraction_op)) {
      return mlir::isa<Float32Type, Float64Type>(dtype);
    } else {
      return false;
    }
  }

  // This function is currently used by contraction ops.
  bool IsCpuCompatibleDataType(Operation* contraction_op,
                               StringRef attr_name = "T") const {
    auto attr = contraction_op->getAttrOfType<TypeAttr>(attr_name);
    if (!attr) return false;
    Type dtype = attr.getValue();
    if (is_onednn_enabled_) {
      // Only contraction ops (MatMul, Conv2D, Conv3D, and
      // DepthwiseConv2dNative) and BatchMatMul are supported. BatchMatMul
      // fusions are handled differently than contraction ops.
      bool is_supported = IsContraction(contraction_op) ||
                          dialect_->IsAnyBatchMatMul(contraction_op);
      return is_supported && mlir::isa<Float32Type, BFloat16Type>(dtype);
    }

    if (dialect_->IsConv2D(contraction_op)) {
      return mlir::isa<Float32Type, Float64Type>(dtype);
    } else if (dialect_->IsMatMul(contraction_op)) {
      return mlir::isa<Float32Type>(dtype);
    } else {
      return false;
    }
  }

  // This function is currently used by convolution type op
  bool IsGpuCompatibleDataFormat(Operation* conv_op,
                                 StringRef attr_name = "data_format") const {
    StringRef data_format;
    if (auto attr = conv_op->getAttrOfType<StringAttr>(attr_name)) {
      data_format = attr.getValue();
    } else {
      return false;
    }
    if (dialect_->IsConv2D(conv_op)) {
      return data_format == "NHWC" || data_format == "NCHW";
    } else {
      return false;
    }
  }

  // This function is currently used by convolution type op
  bool IsCpuCompatibleDataFormat(Operation* conv_op,
                                 StringRef attr_name = "data_format") const {
    StringRef data_format;
    if (auto attr = conv_op->getAttrOfType<StringAttr>(attr_name)) {
      data_format = attr.getValue();
    } else {
      return false;
    }
    if (dialect_->IsConv2D(conv_op)) {
      return data_format == "NHWC" ||
             (is_onednn_enabled_ && data_format == "NCHW");
    } else if (dialect_->IsConv3D(conv_op)) {
      return data_format == "NDHWC" ||
             (is_onednn_enabled_ && data_format == "NCDHW");
    } else {
      return false;
    }
  }

  bool IsGpuCompatible(const ContractionBiasAddActivation& pattern) const {
#if TENSORFLOW_USE_ROCM
    // ROCm does not support _FusedConv2D. Does it suppport _FusedMatMul?
    return false;
#endif
    // The TF->XLA bridge does not support `_FusedMatMul` so we avoid creating
    // this op. Furthermore, XLA already does this fusion internally so there
    // is no true benefit from doing this optimization if XLA is going to
    // compile the unfused operations anyway.
    if (is_xla_auto_clustering_enabled_) return false;
    if (!util::OpHasDevice(pattern.contraction, tensorflow::DEVICE_GPU))
      return false;
    if (!dialect_->IsRelu(pattern.activation)) return false;
    if (dialect_->IsMatMul(pattern.contraction)) {
      return IsGpuCompatibleDataType(pattern.contraction);
    } else {
      // TODO(intel-tf): Add spatial convolution support on GPU
      return false;
    }
  }

  // Currently GPU does not supprt contraction + bias_add
  bool IsGpuCompatible(const ContractionBiasAdd&) const { return false; }

  bool IsCpuCompatible(Operation* contraction_op) const {
    if (!util::OpHasDevice(contraction_op, tensorflow::DEVICE_CPU))
      return false;
    if (dialect_->IsConv2D(contraction_op) ||
        dialect_->IsConv3D(contraction_op)) {
      return IsCpuCompatibleDataType(contraction_op) &&
             IsCpuCompatibleDataFormat(contraction_op);
    } else if (dialect_->IsMatMul(contraction_op) ||
               dialect_->IsAnyBatchMatMul(contraction_op) ||
               dialect_->IsDepthwiseConv2dNative(contraction_op)) {
      return IsCpuCompatibleDataType(contraction_op);
    } else {
      return false;
    }
  }

  template <typename Pattern>
  bool IsDeviceCompatible(const Pattern& pattern) const {
    // Currently, this function is used by contraction based fussion.
    if constexpr (!std::is_same<Pattern, ContractionBiasAdd>::value &&
                  !std::is_same<Pattern, ContractionBiasAddActivation>::value &&
                  !std::is_same<Pattern, ContractionBiasAddAdd>::value &&
                  !std::is_same<Pattern, ContractionBiasAddActivation>::value) {
      return false;
    }
    return IsGpuCompatible(pattern) || IsCpuCompatible(pattern.contraction);
  }

  bool isOneDNNEnabled() const { return is_onednn_enabled_; }

 private:
  bool is_onednn_enabled_;
  bool is_xla_auto_clustering_enabled_;
};

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_TRANSFORMS_REMAPPER_REMAPPING_HELPER_H_
