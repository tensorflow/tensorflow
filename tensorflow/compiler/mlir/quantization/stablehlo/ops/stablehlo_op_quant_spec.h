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

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_OPS_STABLEHLO_OP_QUANT_SPEC_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_OPS_STABLEHLO_OP_QUANT_SPEC_H_

#include <memory>

#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace mlir::quant::stablehlo {

// Returns StableHLO quantization specs for an op.
std::unique_ptr<OpQuantSpec> GetStableHloOpQuantSpec(Operation* op);

// Returns quantization constraints (ex: fixed output, same scale) given
// a StableHLO op.
std::unique_ptr<OpQuantScaleSpec> GetStableHloQuantConstraints(Operation* op);

// Checks if an op is quantizable in StableHLO quantizer. Argument op is not
// necessarily a StableHLO op.
bool IsOpQuantizableStableHlo(Operation* op);

}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_OPS_STABLEHLO_OP_QUANT_SPEC_H_
