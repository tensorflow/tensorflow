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

#include "tensorflow/lite/micro/kernels/arc/scratch_buf_mgr.h"
#include "tensorflow/lite/micro/kernels/arc/scratch_buffers.h"
#include <limits.h>
#define MAX(A,B) (((A) > (B))? (A): (B))
#define MIN(A,B) (((A) > (B))? (B): (A)) 

namespace tflite {
namespace ops {
namespace micro {



void get_arc_two_buffer_sizes(int requestsize1, int requestsize2, int *grantsize1, int *grantsize2) {
  int maxrequest = 0;
  int secondrequest = 0;
  int maxavailable = 0;
  int secondavail = 0;

  // determine the largest requested buffer.
  if (requestsize1 > requestsize2) {
    maxrequest = requestsize1;
    secondrequest = requestsize2;
  } else {
    maxrequest = requestsize2;
    secondrequest = requestsize1;
  }

  // find the two largest available buffers.
  get_arc_scratch_buffer_two_max_sizes(&maxavailable, &secondavail);

  // in case two buffers are available, the largest buffer can go to the largest request.
  if (secondavail > 0) { // this condition can be enhanced to prevent cases where the second buffer is so small that it is better to use one buffer and split it.
    if (requestsize1 > requestsize2) {
      *grantsize1 = maxavailable;
      *grantsize2 = secondavail;
    } else {
      *grantsize1 = secondavail;
      *grantsize2 = maxavailable;
    }
  } else {
    // In case only one buffer is available,
    // use only the max buffer, and split it.
    // TODO compute optimal split ratio based on request ratio.
    *grantsize1 = maxavailable / 2;
    *grantsize2 = maxavailable / 2;
  }
}

TfLiteStatus get_arc_scratch_buffer_for_conv_tensors(TfLiteContext* context,
    mli_tensor* in, 
    mli_tensor* weights, 
    mli_tensor* bias, 
    mli_tensor* out) {
TfLiteStatus ret_val = kTfLiteOk;
#ifdef __Xxy

  if (!inside_arc_ccm(weights->data)) {
    int weights_size = mli_hlp_count_elem_num(weights, 0) * mli_hlp_tensor_element_size(weights);
    int maxWeightsSize = 0;
    weights->data = get_arc_scratch_buffer(weights_size);
    weights->capacity = weights_size;
    if (weights->data == NULL) {
      get_arc_scratch_buffer_max_size(&maxWeightsSize);
      weights->data = get_arc_scratch_buffer(maxWeightsSize);
      weights->capacity = maxWeightsSize;
      if (maxWeightsSize == 0) ret_val = kTfLiteError;
    }
    if (weights->data == NULL) ret_val = kTfLiteError;
  }

  if (!inside_arc_ccm(bias->data)) {
    uint32_t bias_mem_requirements = mli_hlp_count_elem_num(bias, 0) * mli_hlp_tensor_element_size(bias);
    bias->data = get_arc_scratch_buffer(bias_mem_requirements);
    bias->capacity = bias_mem_requirements;
  }
  if (ret_val == kTfLiteOk) {
    ret_val = get_arc_scratch_buffer_for_io_tensors(context, in, out);
  }
  if (bias->data == NULL) {
    int maxBiasSize = 0;
    get_arc_scratch_buffer_max_size(&maxBiasSize);
    bias->data = get_arc_scratch_buffer(maxBiasSize);
    bias->capacity = maxBiasSize;
    if (maxBiasSize == 0) ret_val = kTfLiteError;
  }
  if (bias->data == NULL) ret_val = kTfLiteError;

#endif
  return ret_val;
}

TfLiteStatus arc_scratch_buffer_calc_slice_size_io(
    const mli_tensor *in,
    const mli_tensor *out,
    const int kernelHeight,
    const int strideHeight,
    const int padding_top,
    const int padding_bot,
    int *inSliceHeight,
    int *outSliceHeight) {
  const int heightDimension = 1; // todo: compute from rank
  const int inHeight = in->shape[heightDimension];
  const int outHeight = out->shape[heightDimension];
  const int lineSizeIn = mli_hlp_count_elem_num(in, heightDimension + 1) * mli_hlp_tensor_element_size(in);
  const int lineSizeOut = mli_hlp_count_elem_num(out, heightDimension + 1) * mli_hlp_tensor_element_size(out);
  int maxLinesIn = 0;
  int maxLinesOut = 0;
  int maxOutLinesForInput = 0;
  bool fit = (in->capacity >= inHeight * lineSizeIn) && (out->capacity >= outHeight * lineSizeOut);
  if (fit) {
    // in case both tensors completely fit in the capacity, there is no need for slicing
    *inSliceHeight = inHeight;
    *outSliceHeight = outHeight;
  } else {
    // First compute how many lines fit into the input tensor, and compute how many output lines can be computed with that.
    maxLinesIn = MIN(inHeight, in->capacity / lineSizeIn);
    if (maxLinesIn >= inHeight) {
      maxOutLinesForInput = outHeight;
    } else if (2 * maxLinesIn >= inHeight) {
      // in this case only two slices are needed, so both could benefit from padding. take the MIN to get the worst case.
      maxOutLinesForInput = (maxLinesIn + MIN(padding_top, padding_bot) - kernelHeight + 1) / strideHeight;
    } else {
      maxOutLinesForInput = (maxLinesIn - kernelHeight + 1) / strideHeight; // TODO add padding exceptions and test by makin fit=false;
    }
    // Ten compute how many ouput lines fit into the output tensor.
    maxLinesOut = MIN(outHeight, out->capacity / lineSizeOut);
    // the smallest of the two determines the slice height for the output, and the derived sliceheight for the input.
    *outSliceHeight = MIN(maxOutLinesForInput, maxLinesOut);
    *inSliceHeight = *outSliceHeight * strideHeight;
  }

  if ((*inSliceHeight > 0) && (*outSliceHeight > 0)) {
    return kTfLiteOk;
  } else {
    return kTfLiteError;
  }
}

TfLiteStatus arc_scratch_buffer_calc_slice_size_weights(
    const mli_tensor *weights,
    const mli_tensor *bias,
    int *sliceChannels) {
  const int weightOutChDimension = 0; // NHWC layout for weigths, output channel dimension is the first dimension.
  const int channels = weights->shape[weightOutChDimension];


  const int chSizeW = mli_hlp_count_elem_num(weights, weightOutChDimension + 1) * mli_hlp_tensor_element_size(weights);
  const int chSizeB = mli_hlp_count_elem_num(bias, weightOutChDimension + 1) * mli_hlp_tensor_element_size(bias);
  int maxChWeights = 0;
  int maxChBias = 0;

  bool fit = (weights->capacity >= channels * chSizeW) && (bias->capacity >= channels * chSizeB);
  if (fit) {
    // in case both tensors completely fit in the capacity, there is no need for slicing
    *sliceChannels = channels;
  } else {
    // First compute how many channels fit into the weights tensor
    maxChWeights = MIN(channels, weights->capacity / chSizeW);
    // Ten compute how many channels fit into the bias tensor.
    maxChBias = MIN(channels, bias->capacity / chSizeB);
    // the smallest of the two determines the slice size
    *sliceChannels = MIN(maxChWeights, maxChBias);
  }

  if (*sliceChannels > 0) {
    return kTfLiteOk;
  } else {
    return kTfLiteError;
  }
}

TfLiteStatus get_arc_scratch_buffer_for_io_tensors(TfLiteContext* context,
    mli_tensor* in, 
    mli_tensor* out) {
#ifdef __Xxy
  int requestSizeIn = 0;
  int requestSizeOut = 0;
  int grantsizeIn = 0;
  int grantsizeOut = 0;
  if (!inside_arc_ccm(in->data)) {
    // In case the input tensor contains multiple batches, it has rank 4
    // because the mli kernel cannot operate on batches, we need to have the size
    // of a single HWC tensor. that is why the startRank is 1 in case of input rank 4
    int startRank = in->rank - 3;
    requestSizeIn = mli_hlp_count_elem_num(in, startRank) * mli_hlp_tensor_element_size(in);
  }
  if (!inside_arc_ccm(out->data)) {
    // In case the input tensor contains multiple batches, it has rank 4
    // because the mli kernel cannot operate on batches, we need to have the size
    // of a single batch. that is why the startRank is 1 in case of input rank 4
    int startRank = out->rank - 3;
    requestSizeOut = mli_hlp_count_elem_num(out, startRank) * mli_hlp_tensor_element_size(out);
  }

  get_arc_two_buffer_sizes(requestSizeIn, requestSizeOut, &grantsizeIn, &grantsizeOut);

  if (!inside_arc_ccm(in->data)) {
    in->data = get_arc_scratch_buffer(grantsizeIn);
    in->capacity = grantsizeIn;
    if (in->data == NULL) return kTfLiteError;
  }
  if (!inside_arc_ccm(out->data)) {
    out->data = get_arc_scratch_buffer(grantsizeOut);
    out->capacity = grantsizeOut;
    if (out->data == NULL) return kTfLiteError;
  }
#endif
  return kTfLiteOk;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite