/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback.h"
#include "tensorflow/compiler/mlir/tfrt/tfrt_fallback_registration.h"
#include "tfrt/bef_converter/mlir_to_bef_translate.h"  // from @tf_runtime
#include "tfrt/init_tfrt_dialects.h"  // from @tf_runtime

static mlir::TranslateFromMLIRRegistration registration(
    "mlir-to-bef", "translate MLIR to BEF", tfrt::MLIRToBEFTranslate,
    [](mlir::DialectRegistry& registry) {
      mlir::RegisterAllTensorFlowDialects(registry);
      tfrt::RegisterTFRTDialects(registry);
      tfrt::RegisterTFRTCompiledDialects(registry);
      tensorflow::tfd::RegisterTfrtFallbackDialect(registry);
    });
