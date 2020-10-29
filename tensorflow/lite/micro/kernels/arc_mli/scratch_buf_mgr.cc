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

#include "tensorflow/lite/micro/kernels/arc_mli/scratch_buf_mgr.h"

#include <limits.h>

#include <algorithm>

#include "tensorflow/lite/micro/kernels/arc_mli/scratch_buffers.h"

namespace tflite {
namespace ops {
namespace micro {

#ifdef __Xxy
static void get_arc_two_buffer_sizes(int request_size_1, int request_size_2,
                                     int* grant_size_1, int* grant_size_2) {
  int maxrequest = 0;
  int secondrequest = 0;
  int maxavailable = 0;
  int secondavail = 0;

  // determine the largest requested buffer.
  if (request_size_1 > request_size_2) {
    maxrequest = request_size_1;
    secondrequest = request_size_2;
  } else {
    maxrequest = request_size_2;
    secondrequest = request_size_1;
  }

  // find the two largest available buffers.
  get_arc_scratch_buffer_two_max_sizes(&maxavailable, &secondavail);

  // in case two buffers are available, the largest buffer can go to the largest
  // request.
  if (secondavail > 0) {  // this condition can be enhanced to prevent cases
                          // where the second buffer is so small that it is
                          // better to use one buffer and split it.
    if (request_size_1 > request_size_2) {
      *grant_size_1 = maxavailable;
      *grant_size_2 = secondavail;
    } else {
      *grant_size_1 = secondavail;
      *grant_size_2 = maxavailable;
    }
  } else {
    // In case only one buffer is available,
    // use only the max buffer, and split it.
    *grant_size_1 = maxavailable / 2;
    *grant_size_2 = maxavailable / 2;
  }
}

static TfLiteStatus get_arc_scratch_buffer_for_io_tensors(
    TfLiteContext* context, mli_tensor* in, mli_tensor* out) {
  int request_size_in = 0;
  int request_size_out = 0;
  int grant_size_in = 0;
  int grant_size_out = 0;
  if (!inside_arc_ccm(in->data)) {
    // In case the input tensor contains multiple batches, it has rank 4
    // because the mli kernel cannot operate on batches, we need to have the
    // size of a single HWC tensor. that is why the start_rank is 1 in case of
    // input rank 4
    int start_rank = in->rank - 3;
    request_size_in = mli_hlp_count_elem_num(in, start_rank) *
                      mli_hlp_tensor_element_size(in);
  }
  if (!inside_arc_ccm(out->data)) {
    // In case the input tensor contains multiple batches, it has rank 4
    // because the mli kernel cannot operate on batches, we need to have the
    // size of a single batch. that is why the start_rank is 1 in case of input
    // rank 4
    int start_rank = out->rank - 3;
    request_size_out = mli_hlp_count_elem_num(out, start_rank) *
                       mli_hlp_tensor_element_size(out);
  }

  get_arc_two_buffer_sizes(request_size_in, request_size_out, &grant_size_in,
                           &grant_size_out);

  if (!inside_arc_ccm(in->data)) {
    in->data = get_arc_scratch_buffer(grant_size_in);
    in->capacity = grant_size_in;
    if (in->data == NULL) return kTfLiteError;
  }
  if (!inside_arc_ccm(out->data)) {
    out->data = get_arc_scratch_buffer(grant_size_out);
    out->capacity = grant_size_out;
    if (out->data == NULL) return kTfLiteError;
  }

  return kTfLiteOk;
}
#endif

TfLiteStatus get_arc_scratch_buffer_for_conv_tensors(TfLiteContext* context,
                                                     mli_tensor* in,
                                                     mli_tensor* weights,
                                                     mli_tensor* bias,
                                                     mli_tensor* out) {
  TfLiteStatus ret_val = kTfLiteOk;
#ifdef __Xxy
  init_arc_scratch_buffers();
  if (!inside_arc_ccm(weights->data)) {
    int weights_size = mli_hlp_count_elem_num(weights, 0) *
                       mli_hlp_tensor_element_size(weights);
    int max_weights_size = 0;
    weights->data = get_arc_scratch_buffer(weights_size);
    weights->capacity = weights_size;
    if (weights->data == NULL) {
      get_arc_scratch_buffer_max_size(&max_weights_size);
      weights->data = get_arc_scratch_buffer(max_weights_size);
      weights->capacity = max_weights_size;
      if (max_weights_size == 0) ret_val = kTfLiteError;
    }
    if (weights->data == NULL) ret_val = kTfLiteError;
  }

  if (!inside_arc_ccm(bias->data)) {
    uint32_t bias_mem_requirements =
        mli_hlp_count_elem_num(bias, 0) * mli_hlp_tensor_element_size(bias);
    bias->data = get_arc_scratch_buffer(bias_mem_requirements);
    bias->capacity = bias_mem_requirements;
  }

  if (ret_val == kTfLiteOk) {
    ret_val = get_arc_scratch_buffer_for_io_tensors(context, in, out);
  }

  if (bias->data == NULL) {
    int max_bias_size = 0;
    get_arc_scratch_buffer_max_size(&max_bias_size);
    bias->data = get_arc_scratch_buffer(max_bias_size);
    bias->capacity = max_bias_size;
    if (max_bias_size == 0) ret_val = kTfLiteError;
  }
  if (bias->data == NULL) ret_val = kTfLiteError;

#endif
  return ret_val;
}

TfLiteStatus get_arc_scratch_buffer_for_fully_connect_tensors(
    TfLiteContext* context, mli_tensor* in, mli_tensor* weights,
    mli_tensor* bias, mli_tensor* out) {
  TfLiteStatus ret_val = kTfLiteOk;
#ifdef __Xxy
  init_arc_scratch_buffers();
  /* strategy for FC kernels:
     first allocate input, because this cannot be sliced. (in case of batch
     processing, only a single input needs to be allocated) then weights & bias
     because if fully loaded, they can be reused over batches. then output.
     The number of output channels (for weights slicing) depends on size of
     output and size of weights&bias */

  if (!inside_arc_ccm(in->data)) {
    /* In case the input tensor contains multiple batches,
       only count the size if the inner most dimension */
    int size_in = mli_hlp_count_elem_num(in, in->rank - 1) *
                  mli_hlp_tensor_element_size(in);
    in->data = get_arc_scratch_buffer(size_in);
    in->capacity = size_in;
    if (in->data == NULL) {
      in->capacity = 0;
      ret_val = kTfLiteError;
    }
  }

  if (!inside_arc_ccm(weights->data)) {
    int weights_size = mli_hlp_count_elem_num(weights, 0) *
                       mli_hlp_tensor_element_size(weights);
    int max_weights_size = 0;
    weights->data = get_arc_scratch_buffer(weights_size);
    weights->capacity = weights_size;
    if (weights->data == NULL) {
      get_arc_scratch_buffer_max_size(&max_weights_size);
      weights->data = get_arc_scratch_buffer(max_weights_size);
      weights->capacity = max_weights_size;
      if (max_weights_size == 0) ret_val = kTfLiteError;
    }
    if (weights->data == NULL) ret_val = kTfLiteError;
  }

  if (!inside_arc_ccm(bias->data)) {
    int bias_mem_requirements =
        mli_hlp_count_elem_num(bias, 0) * mli_hlp_tensor_element_size(bias);
    bias->data = get_arc_scratch_buffer(bias_mem_requirements);
    bias->capacity = bias_mem_requirements;
  }

  if (!inside_arc_ccm(out->data)) {
    /* In case the input tensor contains multiple batches,
       only count the size if the inner most dimension */
    int out_size = mli_hlp_count_elem_num(out, out->rank - 1) *
                   mli_hlp_tensor_element_size(out);
    int max_out_size = 0;
    out->data = get_arc_scratch_buffer(out_size);
    out->capacity = out_size;
    if (out->data == NULL) {
      get_arc_scratch_buffer_max_size(&max_out_size);
      out->data = get_arc_scratch_buffer(max_out_size);
      out->capacity = max_out_size;
      if (max_out_size == 0) ret_val = kTfLiteError;
    }
    if (out->data == NULL) ret_val = kTfLiteError;
  }

  if (bias->data == NULL) {
    int max_bias_size = 0;
    get_arc_scratch_buffer_max_size(&max_bias_size);
    bias->data = get_arc_scratch_buffer(max_bias_size);
    bias->capacity = max_bias_size;
    if (max_bias_size == 0) ret_val = kTfLiteError;
  }
  if (bias->data == NULL) ret_val = kTfLiteError;

#endif
  return ret_val;
}

TfLiteStatus arc_scratch_buffer_calc_slice_size_io(
    const mli_tensor* in, const mli_tensor* out, const int kernel_height,
    const int stride_height, const int padding_top, const int padding_bot,
    int* in_slice_height, int* out_slice_height) {
  const int height_dimension = 1;
  const int in_height = in->shape[height_dimension];
  const int out_height = out->shape[height_dimension];
  const int line_size_in = mli_hlp_count_elem_num(in, height_dimension + 1) *
                           mli_hlp_tensor_element_size(in);
  const int line_size_out = mli_hlp_count_elem_num(out, height_dimension + 1) *
                            mli_hlp_tensor_element_size(out);
  int max_lines_in = 0;
  int max_lines_out = 0;
  int max_out_lines_for_input = 0;
  bool fit = (static_cast<int>(in->capacity) >= in_height * line_size_in) &&
             (static_cast<int>(out->capacity) >= out_height * line_size_out);
  if (fit) {
    // in case both tensors completely fit in the capacity, there is no need for
    // slicing. As padding can affect effective input region, we also derive it
    // from output height, and rely on a clipping logic which intend to reduce
    // last smaller slice. I.e the only slice is a kind of
    // "smaller last slice that need to be corrected"
    *in_slice_height = std::max(in_height, out_height * stride_height);
    *out_slice_height = out_height;
  } else {
    // First compute how many lines fit into the input tensor, and compute how
    // many output lines can be computed with that.
    max_lines_in =
        std::min(in_height, static_cast<int>(in->capacity) / line_size_in);
    if (max_lines_in >= in_height) {
      max_out_lines_for_input = out_height;
    } else if (2 * max_lines_in >= in_height) {
      // in this case only two slices are needed, so both could benefit from
      // padding. take the MIN to get the worst case.
      max_out_lines_for_input =
          (max_lines_in + std::min(padding_top, padding_bot) - kernel_height +
           1) /
          stride_height;
    } else {
      max_out_lines_for_input =
          (max_lines_in - kernel_height + 1) / stride_height;
    }
    // Then compute how many output lines fit into the output tensor.
    max_lines_out =
        std::min(out_height, static_cast<int>(out->capacity) / line_size_out);
    // the smallest of the two determines the slice height for the output, and
    // the derived sliceheight for the input.
    *out_slice_height = std::min(max_out_lines_for_input, max_lines_out);
    *in_slice_height = *out_slice_height * stride_height;
  }

  if ((*in_slice_height > 0) && (*out_slice_height > 0)) {
    return kTfLiteOk;
  } else {
    return kTfLiteError;
  }
}

TfLiteStatus arc_scratch_buffer_calc_slice_size_weights(
    const mli_tensor* weights, const mli_tensor* bias,
    const int weight_out_ch_dimension, int* slice_channels) {
  const int channels = weights->shape[weight_out_ch_dimension];
  const int ch_size_w = (mli_hlp_count_elem_num(weights, 0) / channels) *
                        mli_hlp_tensor_element_size(weights);
  const int ch_size_b = (mli_hlp_count_elem_num(bias, 0) / channels) *
                        mli_hlp_tensor_element_size(bias);
  int max_ch_weigths = 0;
  int max_ch_bias = 0;

  bool fit = (static_cast<int>(weights->capacity) >= channels * ch_size_w) &&
             (static_cast<int>(bias->capacity) >= channels * ch_size_b);
  if (fit) {
    // in case both tensors completely fit in the capacity, there is no need for
    // slicing
    *slice_channels = channels;
  } else {
    // First compute how many channels fit into the weights tensor
    max_ch_weigths =
        std::min(channels, static_cast<int>(weights->capacity) / ch_size_w);
    // Ten compute how many channels fit into the bias tensor.
    max_ch_bias =
        std::min(channels, static_cast<int>(bias->capacity) / ch_size_b);
    // the smallest of the two determines the slice size
    *slice_channels = std::min(max_ch_weigths, max_ch_bias);
  }

  if (*slice_channels > 0) {
    return kTfLiteOk;
  } else {
    return kTfLiteError;
  }
}

TfLiteStatus get_arc_scratch_buffer_for_pooling_tensors(TfLiteContext* context,
                                                        mli_tensor* in,
                                                        mli_tensor* out) {
#ifdef __Xxy
  init_arc_scratch_buffers();
  return get_arc_scratch_buffer_for_io_tensors(context, in, out);
#else
  return kTfLiteOk;
#endif
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
