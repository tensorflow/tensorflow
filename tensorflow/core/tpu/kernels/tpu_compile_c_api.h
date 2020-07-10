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

#include "tensorflow/core/tpu/kernels/tpu_mesh_state_c_api.h"
#include "tensorflow/core/tpu/kernels/tpu_program_c_api.h"
#include "tensorflow/core/tpu/kernels/tpu_util_c_api.h"
#include "tensorflow/core/tpu/libtftpu.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"

extern "C" {

// Executes the computations using XLA TPU compiler and returns TPU programs
// ready for execution.
TFTPU_CAPI_EXPORT void TpuCompile_CompileAheadOfTime(
    TpuSerializedProto aot_compilation_request, XLA_TpuProgram** tpu_programs[],
    size_t* count, SE_Status* status);

// Builds `DeviceAssignment` from `TpuCompileMetadata` serialized proto.
TFTPU_CAPI_EXPORT void TpuCompile_BuildXLADeviceAssignment(
    TpuSerializedProto serialized_tpu_compile_metadata,
    const XLA_TpuMeshState* mesh_state,
    TpuSerializedProto* serialized_device_assignment, SE_Status* status);

struct TfTpu_CompileApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_CompileAheadOfTime);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_BuildXLADeviceAssignment);
};

}  // extern "C"

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_C_API_H_
