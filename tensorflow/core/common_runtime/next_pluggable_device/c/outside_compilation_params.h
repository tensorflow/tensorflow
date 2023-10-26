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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_OUTSIDE_COMPILATION_PARAMS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_OUTSIDE_COMPILATION_PARAMS_H_

#include "xla/stream_executor/tpu/c_api_decl.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SE_OutsideCompilationParams {
  char* device_name;
  char* rendezvous_key;
  TF_RendezvousThunk* rendezvous;
  TpuSerializedProto host_transfers;
};

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_OUTSIDE_COMPILATION_PARAMS_H_
