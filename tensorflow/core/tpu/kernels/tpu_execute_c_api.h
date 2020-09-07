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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_EXECUTE_C_API_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_EXECUTE_C_API_H_

#include "tensorflow/core/tpu/kernels/tpu_program_c_api.h"
#include "tensorflow/core/tpu/kernels/tpu_util_c_api.h"
#include "tensorflow/core/tpu/libtftpu.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"

extern "C" {

typedef struct XLA_DeviceAssignment {
  const char* bytes;
  size_t size;
} XLA_DeviceAssignment;

TFTPU_CAPI_EXPORT void TpuExecutable_LoadProgramAndEnqueueToStream(
    const XLA_TpuProgram* program, SE_DeviceMemoryBase* arguments,
    size_t arguments_len, SE_DeviceMemoryBase* result,
    SE_DeviceMemoryBase* cross_program_prefetch_addr, int32_t rng_seed,
    XLA_DeviceAssignment* device_assignment, SE_Stream* stream,
    SE_Status* status);

TFTPU_CAPI_EXPORT void HardwareLayout_HostShapeToDeviceShape(
    XLA_Shape* host_shape, XLA_Shape* device_shape);
TFTPU_CAPI_EXPORT int64_t HardwareLayout_ShapeSize(XLA_Shape* shape);
TFTPU_CAPI_EXPORT int64_t HardwareLayout_ShapeSizeCompact(XLA_Shape* shape);
TFTPU_CAPI_EXPORT int64_t HardwareLayout_ShapeSizeCompactRaw(XLA_Shape* shape);

TFTPU_CAPI_EXPORT void TpuExecute_RuntimeInputToPaddedData(
    uint32_t* runtime_input_ptr, size_t runtime_input_size,
    int8_t* padded_data_ptr, size_t padded_data_size, XLA_Shape* runtime_shape,
    XLA_Shape* compile_time_shape, SE_Status* status);

struct TfTpu_ExecuteApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuExecutable_LoadProgramAndEnqueueToStream);
  TFTPU_ADD_FN_IN_STRUCT(HardwareLayout_HostShapeToDeviceShape);
  TFTPU_ADD_FN_IN_STRUCT(HardwareLayout_ShapeSize);
  TFTPU_ADD_FN_IN_STRUCT(HardwareLayout_ShapeSizeCompact);
  TFTPU_ADD_FN_IN_STRUCT(HardwareLayout_ShapeSizeCompactRaw);
  TFTPU_ADD_FN_IN_STRUCT(TpuExecute_RuntimeInputToPaddedData);
};

}  // extern "C"

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_EXECUTE_C_API_H_
