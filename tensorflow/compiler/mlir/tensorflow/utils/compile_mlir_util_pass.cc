/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"

namespace {
void CreateConvertMlirToXlaHloPipelineWithDefaults(mlir::OpPassManager& pm) {
  tensorflow::CreateConvertMlirToXlaHloPipeline(
      pm, /*device_type=*/"XLA_CPU_JIT",
      /*custom_legalization_passes=*/{});
}

mlir::PassPipelineRegistration<> pipeline(
    "tf-to-hlo-pipeline",
    "Convert TF dialect to HLO dialect (used for compilation in bridge).",
    CreateConvertMlirToXlaHloPipelineWithDefaults);
}  // anonymous namespace
