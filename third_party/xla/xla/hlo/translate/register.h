/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSLATE_REGISTER_H_
#define XLA_HLO_TRANSLATE_REGISTER_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

namespace xla {

// Register all dialects used by XLA plugins
// This is needed to ensure that XLA PJRT plugins work in async environment
// where all registrations must take place before any async compilation.
void RegisterMlirToHloDependentDialects(mlir::DialectRegistry& registry);

}  // namespace xla

#endif  // XLA_HLO_TRANSLATE_REGISTER_H_
