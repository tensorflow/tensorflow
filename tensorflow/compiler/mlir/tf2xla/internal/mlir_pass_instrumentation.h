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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_MLIR_PASS_INSTRUMENTATION_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_MLIR_PASS_INSTRUMENTATION_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mlir/Pass/PassInstrumentation.h"  // from @llvm-project

namespace mlir {

void RegisterPassInstrumentor(
    const std::string& name,
    std::function<std::unique_ptr<PassInstrumentation>()> creator);
std::vector<std::function<std::unique_ptr<PassInstrumentation>()>>
GetPassInstrumentors();

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_MLIR_PASS_INSTRUMENTATION_H_
