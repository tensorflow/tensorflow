/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/stablehlo/uniform_quantized_types.h"

#include <cstdint>

#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace quant {

UniformQuantizedType CreateI8F32UniformQuantizedType(const Location loc,
                                                     MLIRContext& context,
                                                     const float scale,
                                                     const int8_t zero_point) {
  return UniformQuantizedType::getChecked(
      loc, /*flags=*/QuantizationFlags::Signed,
      /*storageType=*/IntegerType::get(&context, /*width=*/8),
      /*expressedType=*/FloatType::getF32(&context), scale, zero_point,
      /*storageTypeMin=*/llvm::minIntN(8), /*storageTypeMax=*/llvm::maxIntN(8));
}

UniformQuantizedPerAxisType CreateI8F32UniformQuantizedPerAxisType(
    const Location loc, MLIRContext& context, const ArrayRef<float> scales,
    const ArrayRef<int8_t> zero_points, const int quantization_dimension) {
  return UniformQuantizedPerAxisType::getChecked(
      loc, /*flags=*/QuantizationFlags::Signed,
      /*storageType=*/IntegerType::get(&context, /*width=*/8),
      /*expressedType=*/FloatType::getF32(&context),
      SmallVector<double>(scales), SmallVector<int64_t>(zero_points),
      quantization_dimension, /*storageTypeMin=*/llvm::minIntN(8),
      /*storageTypeMax=*/llvm::maxIntN(8));
}

}  // namespace quant
}  // namespace mlir
