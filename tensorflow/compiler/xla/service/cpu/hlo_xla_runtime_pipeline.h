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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_HLO_XLA_RUNTIME_PIPELINE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_HLO_XLA_RUNTIME_PIPELINE_H_

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
// Include tf_jitrt_pipeline.h to get TfJitRtPipelineOptions.
#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_pipeline.h"

namespace xla {
namespace cpu {

using HloXlaRuntimePipelineOptions = tensorflow::TfJitRtPipelineOptions;

// Creates a pipeline that lowers modules from HLO to Linalg on buffers.
// `HloXlaRuntimePipelineOptions` contains flags to enable/disable
// experimental features.
void CreateHloXlaRuntimePipeline(mlir::OpPassManager& pm,
                                 const HloXlaRuntimePipelineOptions& options);

// Calls CreateHloXlaRuntimePipeline with the default
// HloXlaRuntimePipelineOptions.
void CreateDefaultHloXlaRuntimePipeline(mlir::OpPassManager& pm);

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_HLO_XLA_RUNTIME_PIPELINE_H_
