/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "scratch_buffer.h"

// todo: remove this function once context->AllocateTemporaryTensor() is
// implemented.

// This buffer is used by CMSIS-NN optimized operator implementations.
// SCRATCH_BUFFER_BYTES bytes is chosen empirically. It needs to be large
// enough to hold the biggest buffer needed by all CMSIS-NN operators in the
// network.
// note: buffer must be 32-bit aligned for SIMD
#define SCRATCH_BUFFER_BYTES 13000

TfLiteStatus get_cmsis_scratch_buffer(TfLiteContext* context, int16_t** buf,
                                      int32_t buf_size_bytes) {
  __attribute__((aligned(
      4))) static int16_t cmsis_scratch_buffer[SCRATCH_BUFFER_BYTES / 2] = {0};

  TF_LITE_ENSURE(context, buf_size_bytes <= SCRATCH_BUFFER_BYTES);
  *buf = cmsis_scratch_buffer;
  return kTfLiteOk;
}
