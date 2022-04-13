/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#ifndef MLIR_HLO_C_TYPES_H
#define MLIR_HLO_C_TYPES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates a token type in the given context.
MLIR_CAPI_EXPORTED MlirType mlirMhloTokenTypeGet(MlirContext ctx);

// Returns true if the type is an MHLO Token type.
MLIR_CAPI_EXPORTED bool mlirMhloTypeIsAToken(MlirType type);

#ifdef __cplusplus
}
#endif

#endif  // MLIR_HLO_C_TYPES_H
