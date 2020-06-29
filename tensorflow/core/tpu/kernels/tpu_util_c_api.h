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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_UTIL_C_API_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_UTIL_C_API_H_

#include "tensorflow/core/tpu/libtftpu.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"

typedef struct SE_Status SE_Status;

extern "C" {

// Checks if whether a TPU compilation is enabled.
bool TpuCompile_IsTpuCompilationEnabled();

// Converts an XLA `Shape` into its equivalent TPU `Shape` representation.
void TpuCompile_ToTpuShapeRepresentation(
    TpuSerializedProto serialized_xla_shape, int data_type,
    bool use_fast_memory, TpuSerializedProto* serialized_tensor_shape,
    SE_Status* status);

// XLA compilation cannot be cancelled. To avoid hanging the TF worker will exit
// when cancellation is requested for an XLA compile op. Some tests require this
// behavior to be disabled, and we test for this condition with the following
// flag function.
bool TpuCompile_ShouldTpuCompileOpIgnoreCancellation();

}  // extern "C"

struct TfTpu_UtilApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_IsTpuCompilationEnabled);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_ShouldTpuCompileOpIgnoreCancellation);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_ToTpuShapeRepresentation);
};

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_UTIL_C_API_H_
