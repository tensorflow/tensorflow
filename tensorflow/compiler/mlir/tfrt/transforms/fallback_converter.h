/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_FALLBACK_CONVERTER_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_FALLBACK_CONVERTER_H_

#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace tensorflow {
namespace tfrt_compiler {

inline llvm::StringRef GetDefaultCpuDeviceName() {
  static constexpr char kCpuDeviceName[] =
      "/job:localhost/replica:0/task:0/device:CPU:0";
  return kCpuDeviceName;
}

class FallbackConverter : public mlir::TypeConverter {
 public:
  explicit FallbackConverter(mlir::MLIRContext *context);

  // Return the next dense key for fallback ops. The key is simply an array
  // index so that in runtime, the fallback ops can be efficiently retrieved.
  int64_t GetNextFallbackKey() const { return fallback_ops_.size(); }

  void RegisterFallbackOp(mlir::Operation *op) { fallback_ops_.push_back(op); }

  void ReplaceFallbackOp(int64_t key, mlir::Operation *op) {
    fallback_ops_[key] = op;
  }

  llvm::ArrayRef<mlir::Operation *> GetFallbackOps() const {
    return fallback_ops_;
  }

 private:
  mlir::Builder builder_;
  // Using a vector to keep fallback ops in order, and the key for a fallback op
  // is its corresponding index here.
  llvm::SmallVector<mlir::Operation *, 8> fallback_ops_;
};

// Convert the `value` that is a !corert.tensorhandle to
// !tfrt_fallback.tf_tensor. If needed, tensor conversion kernels will be added.
// On error it returns nullptr.
mlir::Value ConvertCoreRTTensorHandleToFallbackTensor(
    mlir::Location loc, llvm::StringRef device, mlir::Value value,
    mlir::ConversionPatternRewriter &rewriter);

// Convert the `value` that is a !tfrt_fallback.tf_tensor to
// !corert.tensorhandle. If needed, tensor conversion kernels will be added. On
// error it returns nullptr.
mlir::Value ConvertFallbackTensorToCoreRTTensorHandle(
    mlir::Location loc, mlir::Value value,
    mlir::ConversionPatternRewriter &rewriter);

// Convert operands that might be !tfrt_fallback.tf_tensor for corert operations
// that take only !corert.tensorhandle.
mlir::LogicalResult ConvertCoreRTOperands(
    mlir::Operation *op, mlir::ValueRange operands,
    llvm::SmallVectorImpl<mlir::Value> *new_operands,
    mlir::ConversionPatternRewriter &rewriter);

// Convert operands that might be !corert.tensorhandle for fallback operations
// that take only !tfrt_fallback.tf_tensor.
mlir::LogicalResult ConvertFallbackOperands(
    mlir::Operation *op, llvm::StringRef device, mlir::ValueRange operands,
    llvm::SmallVectorImpl<mlir::Value> *new_operands,
    mlir::ConversionPatternRewriter &rewriter);

}  // namespace tfrt_compiler
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_FALLBACK_CONVERTER_H_
