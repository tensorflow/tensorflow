/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.
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

#include "integrations/c/ChloAttributes.h"

#include "dialect/ChloOps.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr
//===----------------------------------------------------------------------===//

MlirAttribute chloComparisonDirectionAttrGet(MlirContext ctx,
                                             MlirStringRef direction) {
  llvm::Optional<mlir::chlo::ComparisonDirection> compareDirection =
      mlir::chlo::symbolizeComparisonDirection(unwrap(direction));
  if (!compareDirection)
    llvm_unreachable("Invalid comparison-direction specified.");
  return wrap(mlir::chlo::ComparisonDirectionAttr::get(
      unwrap(ctx), compareDirection.value()));
}

bool chloAttributeIsAComparisonDirectionAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::chlo::ComparisonDirectionAttr>();
}

MlirStringRef chloComparisonDirectionAttrGetDirection(MlirAttribute attr) {
  return wrap(mlir::chlo::stringifyComparisonDirection(
      unwrap(attr).cast<mlir::chlo::ComparisonDirectionAttr>().getValue()));
}

//===----------------------------------------------------------------------===//
// ComparisonTypeAttr
//===----------------------------------------------------------------------===//

MlirAttribute chloComparisonTypeAttrGet(MlirContext ctx, MlirStringRef type) {
  llvm::Optional<mlir::chlo::ComparisonType> compareType =
      mlir::chlo::symbolizeComparisonType(unwrap(type));
  if (!compareType) llvm_unreachable("Invalid comparison-type specified.");
  return wrap(
      mlir::chlo::ComparisonTypeAttr::get(unwrap(ctx), compareType.value()));
}

bool chloAttributeIsAComparisonTypeAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::chlo::ComparisonTypeAttr>();
}

MlirStringRef chloComparisonTypeAttrGetType(MlirAttribute attr) {
  return wrap(mlir::chlo::stringifyComparisonType(
      unwrap(attr).cast<mlir::chlo::ComparisonTypeAttr>().getValue()));
}
