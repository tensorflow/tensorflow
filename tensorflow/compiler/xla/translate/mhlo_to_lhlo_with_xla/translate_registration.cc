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

#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/xla/translate/mhlo_to_lhlo_with_xla/mhlo_to_lhlo_with_xla.h"

namespace {
// NOLINTNEXTLINE
llvm::cl::opt<bool> optimize_xla_hlo(
    "optimize-xla-hlo",
    llvm::cl::desc("Enable optimizations when translating XLA HLO -> LHLO"),
    llvm::cl::init(true));
}  // namespace

//----------------------------------------------------------------------------//
// Hooks for tf-mlir-translate
//----------------------------------------------------------------------------/

// MHLO doesn't support explicit layouts, while XLA service does.
// TODO(timshen): remove it once MHLO supports explicit layouts.
static mlir::TranslateToMLIRRegistration HloTextToLhloMlirTranslate(
    "hlo-text-to-lhlo", "hlo-text-to-lhlo",
    [](llvm::StringRef input, mlir::MLIRContext* context) {
      return mlir::HloTextToLhloTranslateFunction(input, context,
                                                  optimize_xla_hlo);
    });
