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
#ifndef STABLEHLO_INTEGRATIONS_C_CHLO_ATTRIBUTES_H
#define STABLEHLO_INTEGRATIONS_C_CHLO_ATTRIBUTES_H

#include <sys/types.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
chloComparisonDirectionAttrGet(MlirContext ctx, MlirStringRef direction);

MLIR_CAPI_EXPORTED bool chloAttributeIsAComparisonDirectionAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
chloComparisonDirectionAttrGetDirection(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// ComparisonTypeAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute chloComparisonTypeAttrGet(MlirContext ctx,
                                                           MlirStringRef type);

MLIR_CAPI_EXPORTED bool chloAttributeIsAComparisonTypeAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
chloComparisonTypeAttrGetType(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif  // STABLEHLO_INTEGRATIONS_C_CHLO_ATTRIBUTES_H
