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

#ifndef MLIR_HLO_BINDINGS_C_DIALECTS_H
#define MLIR_HLO_BINDINGS_C_DIALECTS_H

#include "mlir-c/RegisterEverything.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Chlo, chlo);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Mhlo, mhlo);

#ifdef __cplusplus
}
#endif

#endif  // MLIR_HLO_BINDINGS_C_DIALECTS_H
