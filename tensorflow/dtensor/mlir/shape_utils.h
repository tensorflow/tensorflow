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

#ifndef TENSORFLOW_DTENSOR_MLIR_SHAPE_UTILS_H_
#define TENSORFLOW_DTENSOR_MLIR_SHAPE_UTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/core/platform/status.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"

namespace tensorflow {
namespace dtensor {

StatusOr<llvm::ArrayRef<int64_t>> ExtractGlobalInputShape(
    mlir::OpOperand& input_value);

StatusOr<llvm::ArrayRef<int64_t>> ExtractGlobalOutputShape(
    mlir::OpResult result_value);

// If result is a resource, the shape of the result should be adjusted to
// local value of the resource, based on the layout for output.
absl::Status InferSPMDExpandedLocalShapeForResourceOutput(
    mlir::OpResult* op_result, const Layout& output_layout,
    mlir::MLIRContext* context);

// Returns op with recalculated local shape of `op` given all it's operands.
mlir::Operation* InferSPMDExpandedLocalShape(mlir::Operation* op);

// Gets the shape of a Value if the type is a RankedTensorType, otherwise
// returns an error.
StatusOr<llvm::ArrayRef<int64_t>> GetShapeOfValue(const mlir::Value& value,
                                                  bool fail_on_dynamic = false);

// If the producer or consumer of this value is a DTensorLayout, retrieves
// the global shape from that layout, otherwise returns an error.
StatusOr<llvm::ArrayRef<int64_t>> GetGlobalShapeOfValueFromDTensorLayout(
    const mlir::Value& value);

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_SHAPE_UTILS_H_
