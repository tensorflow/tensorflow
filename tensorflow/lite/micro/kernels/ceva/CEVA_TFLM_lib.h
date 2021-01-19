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

#ifndef CEVA_TFLM_LIB_H_
#define CEVA_TFLM_LIB_H_

#include "types.h"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */



void CEVA_TFLM_AffineQuantize_Int8(const float_32* input_data,
                                   int8_t* output_data, int flat_size,
                                   float_32 scale, int zero_point);


#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif  // CEVA_TFLM_LIB_H_
