/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_KERNEL_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_KERNEL_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace kernel_utils {

// Element-count range required by the caller. Use kInt when the kernel passes
// the count to an implementation that indexes with int; otherwise kSizeT only
// requires the count to fit size_t.
enum class ElementCountLimit {
  kSizeT,
  kInt,
};

// Reads an int32 or int64 axis scalar and normalizes negative values against
// `rank`. Validates that `rank` is non-negative, the axis data buffer is
// present, the axis tensor type is supported, and the normalized axis is in
// [0, rank). This does not validate that `axis_tensor` has exactly one element;
// callers should check that when required by the op contract.
inline TfLiteStatus ReadAndNormalizeAxis(TfLiteContext* context,
                                         const TfLiteTensor& axis_tensor,
                                         int rank, int& normalized_axis) {
  TF_LITE_ENSURE(context, rank >= 0);
  TF_LITE_ENSURE_MSG(context, axis_tensor.data.raw != nullptr,
                     "Axis data is null.");

  int64_t axis_value = 0;
  switch (axis_tensor.type) {
    case kTfLiteInt32:
      axis_value = *GetTensorData<int32_t>(&axis_tensor);
      break;
    case kTfLiteInt64:
      // Retrieve all 8 bytes when axis type is kTfLiteInt64 to avoid data loss.
      axis_value = *GetTensorData<int64_t>(&axis_tensor);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Unsupported axis type: %s",
                         TfLiteTypeGetName(axis_tensor.type));
      return kTfLiteError;
  }

  if (axis_value < 0) {
    axis_value += rank;
  }
  TF_LITE_ENSURE_MSG(context, axis_value >= 0 && axis_value < rank,
                     "Invalid axis value.");
  normalized_axis = static_cast<int>(axis_value);
  return kTfLiteOk;
}

// Computes the product of tensor dimensions in the half-open range
// [begin, end). Validates that the range is within the tensor rank, every
// dimension in the range is non-negative, and every intermediate product fits
// int. The intermediate overflow check is intentional: kernels that loop over
// this product can otherwise overflow even when a later zero dimension would
// make the final tensor element count zero.
inline TfLiteStatus CheckedDimensionProduct(TfLiteContext* context,
                                            const TfLiteTensor& tensor,
                                            int begin, int end, int& product) {
  TF_LITE_ENSURE(context, begin >= 0);
  TF_LITE_ENSURE(context, end >= begin);
  TF_LITE_ENSURE(context, end <= NumDimensions(&tensor));

  CheckedInt<int> checked_product = 1;
  for (int i = begin; i < end; ++i) {
    const int dim = SizeOfDimension(&tensor, i);
    TF_LITE_ENSURE(context, dim >= 0);
    checked_product *= dim;
    TF_LITE_ENSURE_MSG(context, !checked_product.Overflow(),
                       "Dimension product overflows int.");
  }
  product = checked_product.Value();
  return kTfLiteOk;
}

// Gets the checked element count for `tensor` and validates that its data
// storage is usable. The element count must fit size_t, and when `limit` is
// kInt it must also fit int. If the checked element count is non-zero, the
// tensor data buffer must be non-null. Stores the checked element count in
// `element_count`.
inline TfLiteStatus GetTensorElementCountAndValidateData(
    TfLiteContext* context, const TfLiteTensor& tensor, const char* tensor_name,
    ElementCountLimit limit, size_t& element_count) {
  TF_LITE_ENSURE_MSG(
      context, CheckedNumElements(&tensor, element_count) == kTfLiteOk,
      "%s tensor shape is invalid or size overflowed.", tensor_name);
  if (limit == ElementCountLimit::kInt) {
    TF_LITE_ENSURE_MSG(
        context,
        element_count <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "%s tensor shape is invalid or size overflowed.", tensor_name);
  }
  TF_LITE_ENSURE_MSG(context, element_count == 0 || tensor.data.raw != nullptr,
                     "%s data is null.", tensor_name);
  return kTfLiteOk;
}

// Validates that `tensor` has a checked element count and usable data storage,
// but discards the checked element count. Use this when the caller only needs
// to reject invalid shapes, oversized element counts, or missing data buffers.
inline TfLiteStatus ValidateTensorElementsAndData(
    TfLiteContext* context, const TfLiteTensor& tensor, const char* tensor_name,
    ElementCountLimit limit = ElementCountLimit::kSizeT) {
  size_t element_count = 0;
  return GetTensorElementCountAndValidateData(context, tensor, tensor_name,
                                              limit, element_count);
}

// Validates that `tensor` has a checked element count that fits int and, when
// non-empty, has a non-null data buffer. Stores the checked int element count
// in `element_count`.
inline TfLiteStatus CheckedTensorElementsAndData(TfLiteContext* context,
                                                 const TfLiteTensor& tensor,
                                                 const char* tensor_name,
                                                 int& element_count) {
  size_t count = 0;
  TF_LITE_ENSURE_OK(context, GetTensorElementCountAndValidateData(
                                 context, tensor, tensor_name,
                                 ElementCountLimit::kInt, count));
  element_count = static_cast<int>(count);
  return kTfLiteOk;
}

// Performs an RNN batch inference step for inputs specified by input_ptr_batch.
// The RNN cell is specified by the pointers to its input and recurrent weights,
// and biases, along with the input size, number of units, activation.
//
// The pointers to the hidden state and the output are updated as a result.
//
// The pointers with the suffix "_batch" point to data aligned in batch_major
// order, and each step processes batch_size many inputs from input_ptr_batch,
// and updates batch_size many outputs and hidden states.
//
// The output_batch_dim is output.shape[-1], i.e. the outermost dimension of the
// output tensor, and in most cases will be equal to num_units. It is usually
// not when we want to store the RNN output into a slice of the output tensor,
// e.g. for bidirectional RNNs with merge_outputs. In this case, the batched
// operations cannot be used since they assume that the batched outputs are
// contiguous, and we manually loop over the batched outputs.
void RnnBatchStep(const float* input_ptr_batch, const float* input_weights_ptr,
                  const float* recurrent_weights_ptr, const float* bias_ptr,
                  int input_size, int num_units, int batch_size,
                  int output_batch_leading_dim,
                  TfLiteFusedActivation activation,
                  float* hidden_state_ptr_batch, float* output_ptr_batch);

// Same as above but includes an auxiliary input with the corresponding weights.
void RnnBatchStep(const float* input_ptr_batch, const float* input_weights_ptr,
                  const float* aux_input_ptr_batch,
                  const float* aux_input_weights_ptr,
                  const float* recurrent_weights_ptr, const float* bias_ptr,
                  int input_size, int aux_input_size, int num_units,
                  int batch_size, int output_batch_leading_dim,
                  TfLiteFusedActivation activation,
                  float* hidden_state_ptr_batch, float* output_ptr_batch);

// Performs a quantized RNN batch inference step. Same as above, but for
// quantization purposes, we also pass in quantized_hidden_state_ptr_batch and
// quantized_input_ptr_batch pointers for temporary storage of the quantized
// values of hidden_state_ptr_batch and input_ptr_batch, respectively.
// These temporary storages are expected to be preallocated to the same size as
// the respective pointers.
// An additional preallocated temporary storage 'scaling_factors' (of size
// batch_size) is used to store the scaling factors of the quantization (used
// for recovery).
// {input,recurrent}_weights_scale params are used for dequantization/recovery.
void RnnBatchStep(
    const float* input_ptr_batch, const int8_t* input_weights_ptr,
    float input_weights_scale, const int8_t* recurrent_weights_ptr,
    float recurrent_weights_scale, const float* bias_ptr, int input_size,
    int num_units, int batch_size, int output_batch_leading_dim,
    TfLiteFusedActivation activation, int8_t* quantized_input_ptr_batch,
    int8_t* quantized_hidden_state_ptr_batch, float* scaling_factors,
    float* hidden_state_ptr_batch, float* output_ptr_batch,
    bool asymmetric_quantize_inputs, int32_t* zero_points,
    int32_t* accum_scratch, int32_t* row_sums, bool* compute_row_sums);

void RnnBatchStep(
    const float* input_ptr_batch, const int8_t* input_weights_ptr,
    float input_weights_scale, const float* aux_input_ptr_batch,
    const int8_t* aux_input_weights_ptr, float aux_input_weights_scale,
    const int8_t* recurrent_weights_ptr, float recurrent_weights_scale,
    const float* bias_ptr, int input_size, int aux_input_size, int num_units,
    int batch_size, int output_batch_leading_dim,
    TfLiteFusedActivation activation, int8_t* quantized_input_ptr_batch,
    int8_t* aux_quantized_input_ptr_batch,
    int8_t* quantized_hidden_state_ptr_batch, float* scaling_factors,
    float* hidden_state_ptr_batch, float* output_ptr_batch,
    bool asymmetric_quantize_inputs, int32_t* zero_points,
    int32_t* accum_scratch, int32_t* row_sums, bool* compute_row_sums);

}  // namespace kernel_utils
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_KERNEL_UTILS_H_
