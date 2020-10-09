/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_C_API_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_C_API_H_

#include <stddef.h>

#include "tensorflow/core/tpu/kernels/tpu_mesh_state_c_api.h"
#include "tensorflow/core/tpu/kernels/tpu_program_c_api.h"
#include "tensorflow/core/tpu/libtftpu.h"
#include "tensorflow/stream_executor/tpu/c_api_decl.h"

extern "C" {

// Compiles Mlir or TF function computation by lowering into HLO IR and returns
// `count` number of TPU programs ready for execution.
// The API allocates the `XLA_TpuProgram*[]` array `tpu_programs` and creates
// `XLA_TpuProgram` object(s) using the `TpuProgram_New` API. The caller is
// responsible to deallocate both the `XLA_TpuProgram*[]` array and the
// `XLA_TpuProgram` object(s) using `TpuProgram_FreeArray` and `TpuProgram_Free`
// API respectively.
TFTPU_CAPI_EXPORT void TpuCompile_CompileAndBuild(
    TpuSerializedProto compilation_request, const XLA_TpuMeshState* mesh_state,
    XLA_TpuProgram** tpu_programs[], size_t* count, SE_Status* status);

struct TfTpu_CompileApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_CompileAndBuild);
};

}  // extern "C"

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_C_API_H_
