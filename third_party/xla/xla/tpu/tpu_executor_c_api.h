/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_TPU_TPU_EXECUTOR_C_API_H_
#define XLA_TPU_TPU_EXECUTOR_C_API_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/tpu/c_api_decl.h"
#include "xla/tpu/libtftpu.h"

extern "C" {

TF_Status* TpuStatus_New();
TF_Status* TpuStatus_Create(int32_t code, const char* msg);
void TpuStatus_Set(TF_Status* status, int32_t code, const char* msg,
                   int32_t len);
void TpuStatus_Free(TF_Status* status);
const char* TpuStatus_Message(TF_Status* status);
int TpuStatus_Code(TF_Status* status);
bool TpuStatus_Ok(TF_Status* status);
void TpuTransferManager_GetInfeedLayout(XLA_Shape* shape,
                                        XLA_Shape* infeed_shape);

int TpuTopology_LogicalDevicesPerHost(const SE_TpuTopology* tpu_topology,
                                      TpuCoreTypeEnum tpu_core_type);
int TpuTopology_LogicalDevicesPerChip(const SE_TpuTopology* tpu_topology,
                                      TpuCoreTypeEnum tpu_core_type);
int TpuTopology_HostCount(const SE_TpuTopology* tpu_topology);
int TpuTopology_ChipsPerHost(const SE_TpuTopology* tpu_topology);

int TpuTopology_ChipBounds_X(const SE_TpuTopology* tpu_topology);
int TpuTopology_ChipBounds_Y(const SE_TpuTopology* tpu_topology);
int TpuTopology_ChipBounds_Z(const SE_TpuTopology* tpu_topology);
bool TpuTopology_HasChip(const SE_TpuTopology* tpu_topology, int x, int y,
                         int z);
SE_TpuTopology_Core* TpuTopology_CoreForId(const SE_TpuTopology* tpu_topology,
                                           TpuCoreTypeEnum tpu_core_type,
                                           int id);
SE_TpuTopology_Core* TpuTopology_Core(const SE_TpuTopology* tpu_topology,
                                      TpuCoreTypeEnum tpu_core_type, int x,
                                      int y, int z, int index);
int TpuTopology_NumCores(const SE_TpuTopology* tpu_topology,
                         TpuCoreTypeEnum tpu_core_type);
// 'cores' should be a preallocated array of size TpuTopology_NumCores.
void TpuTopology_Cores(const SE_TpuTopology* tpu_topology,
                       TpuCoreTypeEnum tpu_core_type,
                       SE_TpuTopology_Core** cores);
int TpuTopology_IdForHost(const SE_TpuTopology* tpu_topology, int x, int y,
                          int z);
TpuVersionEnum TpuTopology_Version(const SE_TpuTopology* tpu_topology);
void TpuCoreLocation_ChipCoordinates(SE_TpuTopology_Core* tpu_core_location,
                                     int* x, int* y, int* z);
void TpuCoreLocation_HostCoordinates(SE_TpuTopology_Core* tpu_core_location,
                                     int* x, int* y, int* z);
int TpuCoreLocation_Index(SE_TpuTopology_Core* tpu_core_location);
int TpuCoreLocation_Id(SE_TpuTopology_Core* tpu_core_location);

int TpuHostLocation_Id(SE_TpuTopology_Host* tpu_host_location);
int TpuHostLocation_NumCores(SE_TpuTopology_Host* tpu_host_location,
                             TpuCoreTypeEnum tpu_core_type);
// 'cores' should be a preallocated array of size TpuHostLocation_NumCores.
void TpuHostLocation_Cores(SE_TpuTopology_Host* tpu_host_location,
                           TpuCoreTypeEnum tpu_core_type,
                           SE_TpuTopology_Core** cores);

// C API for XLA::Compiler interface

struct TfTpu_ExecutorApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Create);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Set);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Message);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Code);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Ok);

  TFTPU_ADD_FN_IN_STRUCT(TpuTransferManager_GetInfeedLayout);

  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_LogicalDevicesPerHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_LogicalDevicesPerChip);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_HostCount);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_ChipsPerHost);

  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_ChipBounds_X);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_ChipBounds_Y);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_ChipBounds_Z);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_HasChip);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_CoreForId);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_Core);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_NumCores);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_Cores);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_IdForHost);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_Version);

  TFTPU_ADD_FN_IN_STRUCT(TpuCoreLocation_ChipCoordinates);
  TFTPU_ADD_FN_IN_STRUCT(TpuCoreLocation_HostCoordinates);
  TFTPU_ADD_FN_IN_STRUCT(TpuCoreLocation_Index);
  TFTPU_ADD_FN_IN_STRUCT(TpuCoreLocation_Id);

  TFTPU_ADD_FN_IN_STRUCT(TpuHostLocation_Id);
  TFTPU_ADD_FN_IN_STRUCT(TpuHostLocation_NumCores);
  TFTPU_ADD_FN_IN_STRUCT(TpuHostLocation_Cores);
};
}

// extern "C"

#endif  // XLA_TPU_TPU_EXECUTOR_C_API_H_
