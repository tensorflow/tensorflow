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

#ifndef TENSORFLOW_CORE_TPU_TPU_CONFIG_C_API_H_
#define TENSORFLOW_CORE_TPU_TPU_CONFIG_C_API_H_

#include <cstddef>
#include <cstdint>

#include "tensorflow/c/tf_status.h"

typedef struct TpuSerializedProto TpuSerializedProto;

extern "C" {

bool TPUHostInitialized();

void ConfigureDistributedTpuOp_DoWork(const size_t num_cores_per_host_size,
                                      const int32_t* num_cores_per_host,
                                      size_t* host_config_output_size,
                                      char** host_config_output,
                                      TF_Status* status);

void WaitForDistributedTpuOp_DoWork(
    const size_t num_hosts, const size_t num_cores_per_host,
    const int32_t** host_ordinal_to_global_core_id_map,
    size_t* tpu_topology_output_size, char** tpu_topology_output,
    TF_Status* status);

void ShutdownDistributedTpuOp_DoWork(TF_Status* status);

void InitializeHostForDistributedTpuOp_DoWork(
    const size_t tpu_host_config_size, const char* tpu_host_config,
    const bool enable_whole_mesh_compilations, size_t* core_id_output_size,
    int32_t** core_id_output, TF_Status* status);

void SetGlobalTPUArrayOp_DoWork(const size_t tpu_topology_size,
                                const char* tpu_topology, TF_Status* status);

void DisconnectDistributedTpuChipsOp_DoWork(int32_t* number_of_chips_output,
                                            TF_Status* status);

void TpuConfigurationApi_FreeCharArray(char* output);
void TpuConfigurationApi_FreeInt32Array(int32_t* output);
}

#endif  // TENSORFLOW_CORE_TPU_TPU_CONFIG_C_API_H_
