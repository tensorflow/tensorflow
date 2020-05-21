/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_ESTIMATORS_GPU_ESTIMATORS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_ESTIMATORS_GPU_ESTIMATORS_H_

// GPU
constexpr float kGPUArithmeticUnitCost = 0.2;

// The copy can be non-consectutive copy. This is just fake data.
constexpr float kGPUCopyUnitCost = 0.2;
constexpr float kGPUDefaultCost = 1.0f;

// Default values.
constexpr float kGPUDefaultFixedValuedCost = 10000.0;

// tfl.abs
template <>
class TFLiteCostEstimator<AbsOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.add
template <>
class TFLiteCostEstimator<AddOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    int64_t count;
    if (ArithmeticCountUtilHelper::GetArithmeticCountForBroadcastableOp(op,
                                                                        &count))
      return kGPUArithmeticUnitCost * count;
    return kGPUDefaultFixedValuedCost;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.average_pool_2d
template <>
class TFLiteCostEstimator<AveragePool2DOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.concatenation
template <>
class TFLiteCostEstimator<ConcatenationOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    int64_t count;
    if (ArithmeticCountUtilHelper::GetInputTensorTotalSize(op, &count))
      return kGPUCopyUnitCost * count;
    return kGPUDefaultFixedValuedCost;
  }

  // TODO(renjieliu): We probably need to check for dynamic weights.
  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.conv_2d
template <>
class TFLiteCostEstimator<Conv2DOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    int64_t arithmetic_count;
    if (ArithmeticCountUtilHelper::GetArithmeticCountForConvAndFullyconnectedOp(
            op, &arithmetic_count)) {
      return arithmetic_count * kGPUArithmeticUnitCost;
    }
    return kGPUDefaultFixedValuedCost;
  }

  // TODO(renjieliu): We probably need to check for dynamic weights.
  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.cos
template <>
class TFLiteCostEstimator<CosOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.depthwise_conv_2d
template <>
class TFLiteCostEstimator<DepthwiseConv2DOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    int64_t arithmetic_count;
    if (ArithmeticCountUtilHelper::GetArithmeticCountForConvAndFullyconnectedOp(
            op, &arithmetic_count)) {
      return arithmetic_count * kGPUArithmeticUnitCost;
    }
    return kGPUDefaultFixedValuedCost;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.div
template <>
class TFLiteCostEstimator<DivOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.exp
template <>
class TFLiteCostEstimator<ExpOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.fully_connected
template <>
class TFLiteCostEstimator<FullyConnectedOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    int64_t arithmetic_count;
    if (ArithmeticCountUtilHelper::GetArithmeticCountForConvAndFullyconnectedOp(
            op, &arithmetic_count)) {
      return arithmetic_count * kGPUArithmeticUnitCost;
    }
    return kGPUDefaultFixedValuedCost;
  }

  // TODO(renjieliu): we need to check for dynamic weights.
  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.hard_swish
template <>
class TFLiteCostEstimator<HardSwishOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.log
template <>
class TFLiteCostEstimator<LogOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.logistic
template <>
class TFLiteCostEstimator<LogisticOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.max_pool_2d
template <>
class TFLiteCostEstimator<MaxPool2DOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.mirror_pad
template <>
class TFLiteCostEstimator<MirrorPadOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.maximum
template <>
class TFLiteCostEstimator<MaximumOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.custom
template <>
class TFLiteCostEstimator<CustomOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.mean
template <>
class TFLiteCostEstimator<MeanOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  // TODO(renjieiu): check for constraints.
  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.minimum
template <>
class TFLiteCostEstimator<MinimumOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.mul
template <>
class TFLiteCostEstimator<MulOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    int64_t count;
    if (ArithmeticCountUtilHelper::GetArithmeticCountForBroadcastableOp(op,
                                                                        &count))
      return kGPUArithmeticUnitCost * count;
    return kGPUDefaultFixedValuedCost;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.pad
template <>
class TFLiteCostEstimator<PadOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.pow
template <>
class TFLiteCostEstimator<PowOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.prelu
template <>
class TFLiteCostEstimator<PReluOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.relu
template <>
class TFLiteCostEstimator<ReluOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.relu6
template <>
class TFLiteCostEstimator<Relu6Op, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.reshape
template <>
class TFLiteCostEstimator<ReshapeOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    int64_t count;
    if (ArithmeticCountUtilHelper::GetInputTensorTotalSize(op, &count))
      return kGPUCopyUnitCost * count;
    return kGPUDefaultFixedValuedCost;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.rsqrt
template <>
class TFLiteCostEstimator<RsqrtOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.sin
template <>
class TFLiteCostEstimator<SinOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.slice
template <>
class TFLiteCostEstimator<SliceOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.softmax
template <>
class TFLiteCostEstimator<SoftmaxOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.space_to_depth
template <>
class TFLiteCostEstimator<SpaceToDepthOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.sqrt
template <>
class TFLiteCostEstimator<SqrtOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.square
template <>
class TFLiteCostEstimator<SquareOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.squared_difference
template <>
class TFLiteCostEstimator<SquaredDifferenceOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.strided_slice
template <>
class TFLiteCostEstimator<StridedSliceOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.tanh
template <>
class TFLiteCostEstimator<TanhOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.transpose
template <>
class TFLiteCostEstimator<TransposeOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.transpose_conv
template <>
class TFLiteCostEstimator<TransposeConvOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  static bool IsSupported(mlir::Operation* op) { return true; }
};

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_ESTIMATORS_GPU_ESTIMATORS_H_

