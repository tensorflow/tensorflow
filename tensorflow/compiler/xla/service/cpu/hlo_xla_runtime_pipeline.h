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

#include "tensorflow/compiler/xla/runtime/compiler.h"
#include "tensorflow/compiler/xla/status.h"

namespace mlir {
class DialectRegistry;
}  // namespace mlir

namespace xla {
namespace cpu {

struct HloXlaRuntimePipelineOptions {
  bool sparse_bufferization = true;
  bool outline_with_xla_framework = false;
};

// Creates a pipeline that lowers modules from HLO to Linalg on buffers.
Status CreateHloXlaRuntimePipeline(xla::runtime::PassManager& passes,
                                   const HloXlaRuntimePipelineOptions& options);
Status CreateDefaultHloXlaRuntimePipeline(xla::runtime::PassManager& passes);

void RegisterHloXlaRuntimePipelineDialects(mlir::DialectRegistry& dialects);
}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_HLO_XLA_RUNTIME_PIPELINE_H_
