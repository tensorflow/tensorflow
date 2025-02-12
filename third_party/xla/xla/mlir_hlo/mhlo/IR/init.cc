/* Copyright 2020 The OpenXLA Authors.

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

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/IR/register.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "stablehlo/dialect/ChloOps.h"

void mlir::mhlo::registerAllMhloDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::sparse_tensor::SparseTensorDialect>();
  // Backward compatibility with the old way of registering CHLO dialect
  registry.insert<mlir::chlo::ChloDialect>();
}
