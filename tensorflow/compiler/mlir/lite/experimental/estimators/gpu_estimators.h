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

// tfl.add
template <>
class TFLiteCostEstimator<AddOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
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

// tfl.conv_2d
template <>
class TFLiteCostEstimator<Conv2DOp, hardware::GPU> {
 public:
  static double GetCost(mlir::Operation* op) {
    llvm::errs() << "No defined cost function for op: "
                 << op->getName().getStringRef().str();
    return 0.0;
  }

  // TODO(renjieliu): We probably need to check for dynamic weights.
  static bool IsSupported(mlir::Operation* op) { return true; }
};

// tfl.depthwise_conv_2d
template <>
class TFLiteCostEstimator<DepthwiseConv2DOp, hardware::GPU> {
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

// tfl.mul
template <>
class TFLiteCostEstimator<MulOp, hardware::GPU> {
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

// tfl.reshape
template <>
class TFLiteCostEstimator<ReshapeOp, hardware::GPU> {
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

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_ESTIMATORS_GPU_ESTIMATORS_H_

