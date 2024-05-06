/* Copyright 2021 The OpenXLA Authors.
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

#include "bindings/c/Types.h"

#include "mhlo/IR/hlo_ops.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Support/LLVM.h"

MlirType mlirMhloTokenTypeGet(MlirContext ctx) {
  return wrap(mlir::mhlo::TokenType::get(unwrap(ctx)));
}

bool mlirMhloTypeIsAToken(MlirType type) {
  return mlir::isa<mlir::mhlo::TokenType>(unwrap(type));
}
