/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_SHAPE_AND_SIZE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_SHAPE_AND_SIZE_UTILS_H_

#include <cstdint>

#include "mlir/Support/LLVM.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project

namespace mlir {
namespace TFL {

// Value used for a dynamic dimension in TFLite (and TF) shapes.
inline constexpr int64_t kTFLiteDynamicSize = -1;

// Converts a shape that uses `kTFLiteDynamicSize` for dynamic dimensions to an
// MLIR shape that uses `ShapedType::kDynamic`.
llvm::SmallVector<int64_t> ConvertTFLiteShapeToMlir(
    llvm::ArrayRef<int64_t> shape);

// Converts an MLIR shape that uses `ShapedType::kDynamic` for dynamic
// dimensions to a shape that uses `kTFLiteDynamicSize`.
llvm::SmallVector<int64_t> ConvertMlirShapeToTFLite(
    llvm::ArrayRef<int64_t> shape);

// Returns a ranked tensor type for a shape that uses `kTFLiteDynamicSize` for
// dynamic dimensions.
mlir::RankedTensorType GetTypeFromTFLiteShape(llvm::ArrayRef<int64_t> shape,
                                              mlir::Type element_type,
                                              mlir::Attribute encoding = {});

// Converts a TF size (64-bit) to TFLite (32-bit) and properly converts TF's
// value for dynamic size (`std::numeric_limits<int64_t>::min()`) to the
// TFLite-specific value.
int32_t ConvertToTfliteSize(int64_t size);

// Returns the quantization dimension after a reshape operation.
//
// TFL Reshape Op can fold multiple dimensions into one, or split one
// dimension into multiple dimensions. This function will return an error if the
// input quantization dimension is part of the folded/split dimensions.
absl::StatusOr<int32_t> GetQuantDimensionAfterReshape(
    mlir::ArrayRef<int64_t> input_shape, mlir::ArrayRef<int64_t> output_shape,
    int32_t input_quant_dim);

// Padding schemes supported by TFLite windowed ops (e.g. convolutions and
// pooling).
enum class WindowPadding { kValid, kSame };

// Parses a TFLite padding attribute value ("SAME" or "VALID").
absl::StatusOr<WindowPadding> ParseWindowPadding(llvm::StringRef padding);

// Computes the output size of a windowed op along one spatial dimension, as
// well as the padding added before and after the input.
absl::Status GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                                   int64_t dilation_rate, int64_t stride,
                                   WindowPadding padding, int64_t* output_size,
                                   int64_t* padding_before,
                                   int64_t* padding_after);

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_SHAPE_AND_SIZE_UTILS_H_
