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

#include "tensorflow/core/tpu/kernels/tpu_util_c_api.h"
#include "tensorflow/core/tpu/libtftpu.h"
#include "tensorflow/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"

typedef struct XLA_TpuProgram XLA_TpuProgram;

// Enum for choosing sharding/unsharding program from a `XLA_TpuProgram` obj.
enum TpuProgramShardingType { kInvalid = 0, kMain, kSharding, kUnsharding };

struct TpuExecutableSerializedProto {
  const char* bytes;
  size_t size;
};

struct CompilerMetadataSerializedProto {
  const char* bytes;
  size_t size;
};

struct HostComputeMetadataSerializedProto {
  const char* bytes;
  size_t size;
};

extern "C" {

// Creates a new TPU program.
TFTPU_CAPI_EXPORT XLA_TpuProgram* TpuProgram_New();

// Destroys the `tpu_program`.
TFTPU_CAPI_EXPORT void TpuProgram_Free(XLA_TpuProgram* tpu_program);

// Creates an array of `XLA_TpuProgram*`.
TFTPU_CAPI_EXPORT XLA_TpuProgram** TpuProgram_NewArray(size_t count);

// Destroys an array of `XLA_TpuProgram*`.
TFTPU_CAPI_EXPORT void TpuProgram_FreeArray(XLA_TpuProgram* tpu_program[]);

// Unloads and destroys the `tpu_program`. Once the TPU program is unloaded and
// destroyed, it is in an unusable state.
TFTPU_CAPI_EXPORT void TpuProgram_UnloadAndDestroy(XLA_TpuProgram* tpu_program,
                                                   SE_Status* status);

// Gets TPU program size in bytes from the `tpu_program`.
TFTPU_CAPI_EXPORT int64_t
TpuProgram_GetProgramSize(const XLA_TpuProgram* tpu_program);

// Logs the summary of current memory state snapshot of the `tpu_program`.
TFTPU_CAPI_EXPORT bool TpuProgram_LogProgramMemorySummary(
    const XLA_TpuProgram* tpu_program);

// Gets TPU program executable info from the `tpu_program`.
TFTPU_CAPI_EXPORT void TpuProgram_GetExecutableInfo(
    const XLA_TpuProgram* tpu_program, TpuSerializedProto* executable_info,
    SE_Status* status);

// Gets host transfer info proto.
TFTPU_CAPI_EXPORT void TpuProgram_GetHostTransferInfo(
    const XLA_TpuProgram* tpu_program, TpuSerializedProto* host_transfer_info,
    SE_Status* status);

// Gets HLO metadata proto.
TFTPU_CAPI_EXPORT void TpuProgram_GetHloMetadata(
    const XLA_TpuProgram* tpu_program, TpuSerializedProto* hlo_metadata,
    SE_Status* status);

// Gets may modify variables boolean value.
TFTPU_CAPI_EXPORT void TpuProgram_GetMayModifyVariables(
    const XLA_TpuProgram* tpu_program, bool* may_modify_variables);

// Checks if TPU program has sharding.
TFTPU_CAPI_EXPORT bool TpuProgram_HasSharding(
    const XLA_TpuProgram* tpu_program);

// Gets TPU program by sharding type. Return value is valid only when the
// `status.status()` returns `OK`.
TFTPU_CAPI_EXPORT XLA_TpuProgram* TpuProgram_GetTpuProgram(
    XLA_TpuProgram* tpu_program, TpuProgramShardingType type);

// Gets TPU executable proto from a `tpu_program`.
TFTPU_CAPI_EXPORT void TpuProgram_SerializeTpuExecutable(
    const XLA_TpuProgram* tpu_program, TpuExecutableSerializedProto* executable,
    SE_Status* status);

// Gets compilation metadata proto from a `tpu_program`.
TFTPU_CAPI_EXPORT void TpuProgram_SerializeCompilerMetadata(
    const XLA_TpuProgram* tpu_program,
    CompilerMetadataSerializedProto* compiler_metadata, SE_Status* status);


// Deserializes the `GetTpuProgramResponse` proto into an `XLA_TpuProgram`.
TFTPU_CAPI_EXPORT void TpuProgram_DeserializeFromGetTpuProgramResponseProto(
    TpuSerializedProto get_tpu_program_response, XLA_TpuProgram* tpu_program,
    SE_Status* status);

struct TfTpu_TpuProgramApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_NewArray);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_FreeArray);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_UnloadAndDestroy);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_GetProgramSize);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_LogProgramMemorySummary);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_GetExecutableInfo);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_GetHostTransferInfo);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_GetHloMetadata);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_GetMayModifyVariables);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_HasSharding);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_GetTpuProgram);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_SerializeTpuExecutable);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_SerializeCompilerMetadata);
  TFTPU_ADD_FN_IN_STRUCT(TpuProgram_DeserializeFromGetTpuProgramResponseProto);
};

}  // extern "C"

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_PROGRAM_C_API_H_
