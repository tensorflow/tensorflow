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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_PROGRAM_C_API_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_PROGRAM_C_API_H_

#include "tensorflow/core/tpu/kernels/tpu_ops_common_c_api.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"

typedef struct XLA_TpuProgram XLA_TpuProgram;

extern "C" {

// Creates a new TPU program.
XLA_TpuProgram* TpuProgram_New();

// Destroys the `tpu_program`.
void TpuProgram_Free(XLA_TpuProgram* tpu_program);

// Unloads and destroys the `tpu_program`. Once the TPU program is unloaded and
// destroyed, it is in an unusable state.
void TpuProgram_UnloadAndDestroy(XLA_TpuProgram* tpu_program,
                                 SE_Status* status);

// Gets TPU program size in bytes from the `tpu_program`.
int64_t TpuProgram_GetProgramSize(const XLA_TpuProgram* tpu_program);

// Logs the summary of current memory state snapshot of the `tpu_program`.
bool TpuProgram_LogProgramMemorySummary(const XLA_TpuProgram* tpu_program);

// Gets TPU program executable info from the `tpu_program`.
void TpuProgram_GetExecutableInfo(const XLA_TpuProgram* tpu_program,
                                  TpuSerializedProto* executable_info);

// Gets host transfer info proto.
void TpuProgram_GetHostTransferInfo(const XLA_TpuProgram* tpu_program,
                                    TpuSerializedProto* host_transfer_info);

// Gets HLO metadata proto.
void TpuProgram_GetHloMetadata(const XLA_TpuProgram* tpu_program,
                               TpuSerializedProto* hlo_metadata);

}  // extern "C"

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_PROGRAM_C_API_H_
