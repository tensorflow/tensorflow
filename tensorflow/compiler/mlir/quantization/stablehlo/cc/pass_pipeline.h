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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PASS_PIPELINE_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PASS_PIPELINE_H_

#include "mlir/Pass/PassManager.h"  // from @llvm-project

namespace mlir::quant::stablehlo {

// Deserializes StableHLO functions serialized and embedded in XlaCallModuleOps.
void AddXlaCallModuleOpDeserializationPasses(OpPassManager& pm);

// Legalizes shape/tensor/arith dialect ops to StableHLO for handling dynamic
// shapes, by going through a round-trip to MHLO.
void AddShapeLegalizationPasses(OpPassManager& pm);

// Serializes the StableHLO module into a tf.XlaCallModuleOp for compatibility
// with passes that expect TF format. This also allows the StableHLO ops to be
// exported as a TF SavedModel.
void AddCallModuleSerializationPasses(OpPassManager& pm);

// Passes for unpacking quantized ops to int valued StableHLO ops. This is
// useful when uniform quantized types are suboptimal for the hardware. It goes
// through a StableHLO <-> MHLO roundtrip to utilize the MHLOQuantToInt pass.
void AddStablehloQuantToIntPasses(OpPassManager& pm);

}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PASS_PIPELINE_H_
