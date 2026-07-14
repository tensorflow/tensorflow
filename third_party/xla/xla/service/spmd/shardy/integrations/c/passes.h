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

#ifndef XLA_SERVICE_SPMD_SHARDY_INTEGRATIONS_C_PASSES_H_
#define XLA_SERVICE_SPMD_SHARDY_INTEGRATIONS_C_PASSES_H_

#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Register all compiler passes and pipelines of XLA Shardy.
MLIR_CAPI_EXPORTED void mlirRegisterAllXlaSdyPassesAndPipelines();

#ifdef __cplusplus
}
#endif

#endif  // XLA_SERVICE_SPMD_SHARDY_INTEGRATIONS_C_PASSES_H_
