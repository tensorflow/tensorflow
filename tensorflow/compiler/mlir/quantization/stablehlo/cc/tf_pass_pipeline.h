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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_TF_PASS_PIPELINE_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_TF_PASS_PIPELINE_H_

#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace mlir::tf_quant::stablehlo {

// Adds passes for static-range quantization pre-calibration. Inserts ops
// required to collect tensor statistics.
void AddPreCalibrationPasses(
    OpPassManager& pm,
    const ::stablehlo::quantization::CalibrationOptions& calibration_options,
    const ::stablehlo::quantization::QuantizationSpecs& specs,
    const ::stablehlo::quantization::DebuggerConfig& debugger_config);

// Adds passes for static-range quantization post-calibration. Utilizes tensor
// statistics collected from the calibration step and performs quantization.
void AddPostCalibrationPasses(
    OpPassManager& pm,
    const ::stablehlo::quantization::PipelineConfig& pipeline_config,
    const ::stablehlo::quantization::QuantizationSpecs& specs);

// Adds passes for weight-only quantization.
void AddWeightOnlyQuantizationPasses(
    OpPassManager& pm,
    const ::stablehlo::quantization::QuantizationSpecs& quantization_specs,
    const ::stablehlo::quantization::PipelineConfig& pipeline_config,
    const ::stablehlo::quantization::DebuggerConfig& debugger_config);

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

// Processes tensors with NCHW format (== (batch, channel, height, weight)) by
// converting them to NHWC formats along with extra optimizations such as
// constant folding the transpose->convolution pattern. This is useful when
// downstream pipeline (e.g. XLA) is more optimized when accepting NHWC formats.
void AddProcessNchwTensorPasses(OpPassManager& pm);

// Registers quantization pass pipelines. This is only required when running
// MLIR opt binaries and not required when adding passes programmatically.
void RegisterPassPipelines();

}  // namespace mlir::tf_quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_TF_PASS_PIPELINE_H_
