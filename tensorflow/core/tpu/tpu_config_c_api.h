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

#include "tensorflow/c/tf_status.h"

typedef struct TpuSerializedProto TpuSerializedProto;

extern "C" {

bool TPUHostInitialized();

// TODO(frankchn): Modify API to take in raw values instead of Tensors.
void ConfigureDistributedTpuOp_DoWork(size_t input_size,
                                      TpuSerializedProto** inputs,
                                      TpuSerializedProto* output,
                                      TF_Status* status);

void WaitForDistributedTpuOp_DoWork(size_t input_size,
                                    TpuSerializedProto** inputs,
                                    TpuSerializedProto* output,
                                    TF_Status* status);

void ShutdownDistributedTpuOp_DoWork(TF_Status* status);

void InitializeHostForDistributedTpuOp_DoWork(
    size_t input_size, TpuSerializedProto** inputs,
    bool enable_whole_mesh_compilations, TpuSerializedProto* output,
    TF_Status* status);

void SetGlobalTPUArrayOp_DoWork(size_t input_size, TpuSerializedProto** inputs,
                                TF_Status* status);

void DisconnectDistributedTpuChipsOp_DoWork(TpuSerializedProto* output,
                                            TF_Status* status);
}

#endif  // TENSORFLOW_CORE_TPU_TPU_CONFIG_C_API_H_
