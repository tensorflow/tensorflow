/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_MOSAIC_DIALECT_TPU_INTEGRATIONS_C_TPU_DIALECT_H_
#define XLA_MOSAIC_DIALECT_TPU_INTEGRATIONS_C_TPU_DIALECT_H_

#include "xla/mosaic/dialect/tpu/integrations/c/tpu_dialect.h"
#ifndef __cplusplus
#include <stdbool.h>
#endif
#include <stddef.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "xla/mosaic/dialect/tpu/integrations/c/tpu_passes.capi.h.inc"

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TPU, tpu);

MLIR_CAPI_EXPORTED void mlirTPUAnalyzePotentialCommunication(
    MlirOperation op, bool* has_communication, bool* has_custom_barrier);

MLIR_CAPI_EXPORTED void mlirTpuRegisterMosaicSerdePass();

MLIR_CAPI_EXPORTED MlirType
mlirTpuFloat8EXMYTypeGetUnderlyingType(MlirType exmy_type);

MLIR_CAPI_EXPORTED bool mlirTpuIsAFloat8EXMYType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirTpuFloat8EXMYTypeGet(MlirContext ctx,
                                                     MlirType exmy_type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XLA_MOSAIC_DIALECT_TPU_INTEGRATIONS_C_TPU_DIALECT_H_
