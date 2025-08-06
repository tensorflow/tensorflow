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

#include "xla/hlo/translate/register.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"

namespace xla {

void RegisterMlirToHloDependentDialects(mlir::DialectRegistry& registry) {
  mlir::stablehlo::registerAllDialects(registry);
  mlir::func::registerAllExtensions(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::sdy::registerAllDialects(registry);

  // TODO(b/435720503): These dialects are mostly unsupported and should be
  // removed, keeping for parity for now.
  registry.insert<mlir::tensor::TensorDialect, mlir::arith::ArithDialect,
                  mlir::shape::ShapeDialect>();

  // MLIR Canonicalization relies on the UB dialect so include it
  registry.insert<mlir::ub::UBDialect>();
}

}  // namespace xla
