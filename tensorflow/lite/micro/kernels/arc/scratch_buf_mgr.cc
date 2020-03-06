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
#ifdef __Xxy

  if (!inside_arc_ccm(weights->data)) {
    int weights_size = mli_hlp_count_elem_num(weights, 0) * mli_hlp_tensor_element_size(weights);
    weights->data = get_arc_scratch_buffer(weights_size);
    weights->capacity = weights_size;
    if (weights->data == NULL) return kTfLiteError;
  }

  if (!inside_arc_ccm(bias->data)) {
    uint32_t bias_mem_requirements = mli_hlp_count_elem_num(bias, 0) * mli_hlp_tensor_element_size(bias);
    bias->data = get_arc_scratch_buffer(bias_mem_requirements);
    bias->capacity = bias_mem_requirements;
    if (bias->data == NULL) return kTfLiteError;
  }

  int requestSizeIn = 0;
  int requestSizeOut = 0;
  int grantsizeIn = 0;
  int grantsizeOut = 0;
  if (!inside_arc_ccm(in->data)) {
    // In case the input tensor contains multiple batches, it has rank 4
    // because the mli kernel cannot operate on batches, we need to have the size
    // of a single batch. that is why the startRank is 1 in case of input rank 4
    int startRank = in->rank - 3; // tOdo explain
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

  return kTfLiteOk;
#else
  return kTfLiteOk;
#endif
}

TfLiteStatus arc_scratch_buffer_calc_slice_size_io(
    const mli_tensor *in,
    const mli_tensor *out,
    const int kernelHeight,
    const int strideHeight,
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

TfLiteStatus get_arc_scratch_buffer_for_io_tensors(TfLiteContext* context,
    mli_tensor* in, 
    mli_tensor* out) {
#ifdef __Xxy
  // Function to assign fast memory from one of 3 scratch buffers.
  // Best Fit strategy - memory is asigned to those tensor which leave less memory of bank unused
  mli_tensor* tensors[2] = { in, out };
  uint32_t tensor_sizes[2] = {
    mli_hlp_count_elem_num(tensors[0], 0), mli_hlp_count_elem_num(tensors[1], 0)};
  int num_tensors = 2;
  

  for (int i = 0; i < num_tensors; ++i) {
    // only for tensors that are not already located in one of the ccm memories, find a local memory that fits the data size.
    if (inside_arc_ccm(tensors[i]->data)) continue;
    tensors[i]->data = get_arc_scratch_buffer(tensor_sizes[i]);
    tensors[i]->capacity = tensor_sizes[i];

    if (tensors[i]->data == NULL) {
      return kTfLiteError;
    }
  }
#endif
  return kTfLiteOk;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite