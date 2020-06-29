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
#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_NODE_CONTEXT_C_API_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_NODE_CONTEXT_C_API_H_

#include "tensorflow/core/tpu/kernels/tpu_util_c_api.h"
#include "tensorflow/core/tpu/libtftpu.h"

typedef struct XLA_TpuNodeContext XLA_TpuNodeContext;

extern "C" {

XLA_TpuNodeContext* TpuNodeContext_Create(int device_ordinal,
                                          SE_Status* status);
void TpuNodeContext_Free(XLA_TpuNodeContext* node_context);

void TpuNodeContext_Initialize(int device_ordinal, SE_Status* status);

void TpuNodeContext_StopChipHeartbeats(SE_Status* status);
void TpuNodeContext_CloseTpuHost(SE_Status* status);

}  // extern "C"

struct TfTpu_NodeContextApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuNodeContext_Create);
  TFTPU_ADD_FN_IN_STRUCT(TpuNodeContext_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuNodeContext_Initialize);
  TFTPU_ADD_FN_IN_STRUCT(TpuNodeContext_StopChipHeartbeats);
  TFTPU_ADD_FN_IN_STRUCT(TpuNodeContext_CloseTpuHost);
};

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_NODE_CONTEXT_C_API_H_
