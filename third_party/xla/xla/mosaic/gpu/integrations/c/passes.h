/* Copyright 2024 The JAX Authors.

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

#ifndef JAXLIB_MOSAIC_GPU_INTEGRATIONS_C_PASSES_H_
#define JAXLIB_MOSAIC_GPU_INTEGRATIONS_C_PASSES_H_

#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void mlirMosaicGpuRegisterPasses();

#ifdef __cplusplus
}
#endif

#endif  // JAXLIB_MOSAIC_GPU_INTEGRATIONS_C_PASSES_H_
