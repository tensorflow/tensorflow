/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_EMITTERS_TRANSFORMS_PASS_PIPELINES_H_
#define XLA_CODEGEN_EMITTERS_TRANSFORMS_PASS_PIPELINES_H_

#include "mlir/Pass/PassManager.h"

namespace xla::emitters {

// Adds passes that simplify arithmetic operations and remove dead code.
void RegisterOptimizationPasses(mlir::OpPassManager& pm);

}  // namespace xla::emitters

#endif  // XLA_CODEGEN_EMITTERS_TRANSFORMS_PASS_PIPELINES_H_
