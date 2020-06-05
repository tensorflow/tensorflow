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

#ifndef TENSORFLOW_LITE_MICRO_ARC_SCRATCH_BUF_MGR_H_
#define TENSORFLOW_LITE_MICRO_ARC_SCRATCH_BUF_MGR_H_

#include "mli_api.h"  // NOLINT
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace ops {
namespace micro {

/**
 * @brief Function to allocate scratch buffers for the convolution tensors
 *
 * @detail This function will update the data pointers in the 4 tensors with
 * pointers to scratch buffers in fast local memory.
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
 * @brief Function to allocate scratch buffers for pooling kernels with only
 * input and output buffers
 *
 * @detail This function will update the data pointers in the 2 tensors with
 * pointers to scratch buffers in fast local memory.
 *
 * @param context  [I] pointer to TfLite context (needed for error handling)
 * @param in [IO] pointer to the input tensor
 * @param output [IO] pointer to the output tensor
 *
 * @return Tf Lite status code
 */
TfLiteStatus get_arc_scratch_buffer_for_pooling_tensors(TfLiteContext* context,
                                                        mli_tensor* in,
                                                        mli_tensor* out);

/**
 * @brief Function to allocate scratch buffers for the fully connect tensors
 *
 * @detail This function will update the data pointers in the 4 tensors with
 * pointers to scratch buffers in fast local memory.
 *
 * @param context  [I] pointer to TfLite context (needed for error handling)
 * @param in [IO] pointer to the input tensor
 * @param weights [IO] pointer to the weights tensor
 * @param bias [IO] pointer to the bias tensor
 * @param output [IO] pointer to the output tensor
 *
 * @return Tf Lite status code
 */
TfLiteStatus get_arc_scratch_buffer_for_fully_connect_tensors(
    TfLiteContext* context, mli_tensor* in, mli_tensor* weights,
    mli_tensor* bias, mli_tensor* out);

/**
 * @brief Function to calculate slice size for io tensors
 *
 * @detail This function will calculate the slice size in the height dimension
 * for input and output tensors. it takes into account the kernel size and the
 * padding. the function will look at the capacity filed in the in and out
 * tensor to determine the available buffersize.
 *
 * @param in [I] pointer to the input tensor
 * @param out [I] pointer to the output tensor
 * @param kernelHeight [I] size of the kernel in height dimension
 * @param strideHeight [I] input stride in height dimension
 * @param padding_top [I] number of lines with zeros at the top
 * @param padding_bot [I] number of lines with zeros at the bottom
 * @param inSliceHeight [O] slice size in height dimension for the input tensor
 * @param outSliceHeight [O] slice size in height dimension for the output
 * tensor
 *
 * @return Tf Lite status code
 */
TfLiteStatus arc_scratch_buffer_calc_slice_size_io(
    const mli_tensor* in, const mli_tensor* out, const int kernelHeight,
    const int strideHeight, const int padding_top, const int padding_bot,
    int* in_slice_height, int* out_slice_height);

/**
 * @brief Function to calculate slice size for weight slicing
 *
 * @detail This function will calculate the slice size in the output channel
 * dimension for weight and bias tensors. the function will look at the capacity
 * filed in the weights and bias tensor to determine the available buffersize.
 *
 * @param weights [I] pointer to the input tensor
 * @param bias [I] pointer to the output tensor
 * @param weightOutChDimension [I] dimension of the output channels in the
 * weights tensor
 * @param sliceChannels [O] slice size in output channel dimension
 *
 * @return Tf Lite status code
 */
TfLiteStatus arc_scratch_buffer_calc_slice_size_weights(
    const mli_tensor* weights, const mli_tensor* bias,
    const int weight_out_ch_dimension, int* slice_channels);

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_ARC_SCRATCH_BUF_MGR_H_
