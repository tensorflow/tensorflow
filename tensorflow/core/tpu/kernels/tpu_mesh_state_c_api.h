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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_MESH_STATE_C_API_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_MESH_STATE_C_API_H_

#include "tensorflow/core/tpu/libtftpu.h"

typedef struct XLA_TpuMeshState XLA_TpuMeshState;

extern "C" {

// Creates a new TPU mesh state object.
TFTPU_CAPI_EXPORT XLA_TpuMeshState* TpuMeshState_Create();

// Deletes the given TPU `mesh_state` object. Once deleted the object is
// unusable.
TFTPU_CAPI_EXPORT void TpuMeshState_Free(XLA_TpuMeshState* mesh_state);

// Returns a pointer to an opaque mesh data structure used internally.
TFTPU_CAPI_EXPORT void* TpuMeshState_MeshCommonState(
    XLA_TpuMeshState* mesh_state);

}  // extern "C"

struct TfTpu_MeshStateApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuMeshState_Create);
  TFTPU_ADD_FN_IN_STRUCT(TpuMeshState_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuMeshState_MeshCommonState);
};

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_MESH_STATE_C_API_H_
