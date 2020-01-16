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

#ifndef TENSORFLOW_LITE_MICRO_ARC_SCRATCH_BUFFERS_H_
#define TENSORFLOW_LITE_MICRO_ARC_SCRATCH_BUFFERS_H_

#include "tensorflow/lite/c/common.h"
#include "mli_api.h"

/**
 * @brief Function to allocate scratch buffers for the convolution tensors
 *
 * @detail This function will update the data pointers in the 4 tensors with pointers
 * to scratch buffers in fast local memory.
 *
 * @param context  [I] pointer to TfLite context (needed for error handling)
 * @param in [IO] pointer to the input tensor
 * @param weights [IO] pointer to the weights tensor
 * @param bias [IO] pointer to the bias tensor
 * @param output [IO] pointer to the output tensor
 *
 * @return Tf Lite status code
 */
TfLiteStatus get_arc_scratch_buffer_for_conv_tensors(TfLiteContext* context,
    mli_tensor* in, 
    mli_tensor* weights, 
    mli_tensor* bias, 
    mli_tensor* out);

/**
 * @brief Function to allocate scratch buffers for kernels with only input and output buffers
 *
 * @detail This function will update the data pointers in the 2 tensors with pointers
 * to scratch buffers in fast local memory.
 *
 * @param context  [I] pointer to TfLite context (needed for error handling)
 * @param in [IO] pointer to the input tensor
 * @param output [IO] pointer to the output tensor
 *
 * @return Tf Lite status code
 */
TfLiteStatus get_arc_scratch_buffer_for_io_tensors(TfLiteContext* context,
    mli_tensor* in, 
    mli_tensor* out);

#endif  // TENSORFLOW_LITE_MICRO_ARC_SCRATCH_BUFFERS_H_
