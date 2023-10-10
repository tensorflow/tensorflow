/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_TF_DEVICE_CONTEXT_C_API_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_TF_DEVICE_CONTEXT_C_API_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TF_Tensor TF_Tensor;
typedef struct TSL_Status TF_Status;

// Structs for TF_StatusCallback.

typedef void (*TF_StatusCallback_Function)(void*, TF_Status*);
typedef struct TF_StatusCallback {
  void* context;
  TF_StatusCallback_Function callback;
} TF_StatusCallback;

// Structs for CopyCPUTensorToDevice API.
typedef struct TF_DeviceContext_CopyCPUTensorToDevice_Params {
  TF_Tensor* cpu_tensor;
  // API for `Device` is not available.
  // Device* device;
  TF_Tensor* device_tensor;  // out
  TF_StatusCallback* done;
  bool sync_dst_compute;
} TF_DeviceContext_CopyCPUTensorToDevice_Params;

typedef void (*TF_DeviceContext_CopyCPUTensorToDevice_Function)(
    void*, TF_DeviceContext_CopyCPUTensorToDevice_Params*);

// Structs for CopyDeviceTensorToCPU API.
typedef struct TF_DeviceContext_CopyDeviceTensorToCPU_Params {
  TF_Tensor* device_tensor;
  char* tensor_name;
  // API for `Device` is not available.
  // Device* device;
  uint32_t tensor_name_len;
  TF_Tensor* cpu_tensor;  // out
  TF_StatusCallback* done;
} TF_DeviceContext_CopyDeviceTensorToCPU_Params;

typedef void (*TF_DeviceContext_CopyDeviceTensorToCPU_Function)(
    void*, TF_DeviceContext_CopyDeviceTensorToCPU_Params*);

// Structs for CopyTensorInSameDevice API.
typedef struct TF_DeviceContext_CopyTensorInSameDevice_Params {
  TF_Tensor* input_tensor;
  // API for `Device` is not available.
  // Device* device;
  TF_Tensor* output_tensor;  // out
  TF_StatusCallback* done;
} TF_DeviceContext_CopyTensorInSameDevice_Params;

typedef void (*TF_DeviceContext_CopyTensorInSameDevice_Function)(
    void*, TF_DeviceContext_CopyTensorInSameDevice_Params*);

/* DeviceContext */
typedef struct TF_DeviceContext {
  void* device_context;
  TF_DeviceContext_CopyCPUTensorToDevice_Function cpu_to_device_func;
  TF_DeviceContext_CopyDeviceTensorToCPU_Function device_to_cpu_func;
  TF_DeviceContext_CopyTensorInSameDevice_Function same_device_func;
} TF_DeviceContext;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_TF_DEVICE_CONTEXT_C_API_H_
