/* Copyright 2023 The StableHLO Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_UTILS_FILL_QUANTIZATION_OPTIONS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_UTILS_FILL_QUANTIZATION_OPTIONS_H_

#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_options.pb.h"

namespace mlir::quant::stablehlo {

using ::stablehlo::quantization::QuantizationOptions;

// Returns QuantizationOptions filled with detailed specs when user specifies
// an optional preset method name. The preset methods are defined in
// quantization_options.proto. This function will only be executed if a user
// gives a preset method, not a custom method.
QuantizationOptions FillPresetQuantizationOptions(
    QuantizationOptions quantization_options);

// Returns LogicalResult depending on the look up of activation bit width in the
// custom quantization method. If such information exists, returns success,
// otherwise, returns false.
LogicalResult GetActivationBitWidth(QuantizationOptions quantization_options,
                                    int* bit_width);

}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_UTILS_FILL_QUANTIZATION_OPTIONS_H_
