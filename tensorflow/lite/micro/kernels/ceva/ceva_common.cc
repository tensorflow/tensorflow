/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/ceva/ceva_tflm_lib.h"
#define CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL_DEF 32768
int32_t CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL =
    CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL_DEF;
#ifndef WIN32
__attribute__((section(".MODEL_DATA")))
#endif
int32_t CEVA_TFLM_KERNELS_SCRATCH[CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL_DEF];
