/* Copyright 2025 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_REGISTER_LITE_DIALECTS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_REGISTER_LITE_DIALECTS_H_

#include "mlir/IR/DialectRegistry.h"  // from @llvm-project

namespace tflite {

// Inserts LiteRT dialects used for offline tools.
void RegisterLiteToolingDialects(mlir::DialectRegistry& registry);

};  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_REGISTER_LITE_DIALECTS_H_
