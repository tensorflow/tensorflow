/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <stdint.h>

#include <climits>
#include <cstring>
#include <utility>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/variants/list_ops_lib.h"
#include "tensorflow/lite/kernels/variants/list_ops_util.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {
namespace ops {
namespace {

constexpr int kTensorInput = 0;
constexpr int kElementShapeInput = 1;
constexpr int kListOut = 0;

void CopyPackedTensor(const TfLiteTensor* src, TfLiteTensor* dst,
                      int src_row_offset, int num_elements) {
  int bit_width = TfLiteTypeGetSizeBits(src->type);
  const uint8_t* src_raw = reinterpret_cast<const uint8_t*>(src->data.raw);
  uint8_t* dst_raw = reinterpret_cast<uint8_t*>(dst->data.raw);

  // Number of elements that can be packed into one byte.
  int elements_per_byte = CHAR_BIT / bit_width;

  // If the source row is byte-aligned, copy the bytes directly, otherwise it
  // is needed to do some bit-fiddling to extract the elements.
  if (src_row_offset % elements_per_byte == 0) {
    int num_bytes_to_copy = num_elements / elements_per_byte;
    // Copy the whole bytes that can be copied.
    if (num_bytes_to_copy > 0) {
      memcpy(dst_raw, src_raw + (src_row_offset / elements_per_byte),
             num_bytes_to_copy);
    }
    // Copy the remaining bits, if any.
    int tail_elements = num_elements % elements_per_byte;
    if (tail_elements > 0) {
      int src_byte_idx =
          (src_row_offset / elements_per_byte) + num_bytes_to_copy;
      // Create and apply a mask to the source byte to avoid reading out of
      // bounds.
      uint8_t tail_mask = (1 << (tail_elements * bit_width)) - 1;
      dst_raw[num_bytes_to_copy] = src_raw[src_byte_idx] & tail_mask;
    }
  } else {
    // If the source row is not byte-aligned, it is needed to do some
    // bit-fiddling to extract the elements.
    int src_byte_idx = src_row_offset / elements_per_byte;
    int shift_bits = (src_row_offset % elements_per_byte) * bit_width;
    int num_bytes_to_copy = num_elements / elements_per_byte;

    for (int b = 0; b < num_bytes_to_copy; ++b) {
      dst_raw[b] = (src_raw[src_byte_idx + b] >> shift_bits) |
                   (src_raw[src_byte_idx + b + 1] << (8 - shift_bits));
    }

    int tail_elements = num_elements % elements_per_byte;
    if (tail_elements > 0) {
      uint8_t tail_mask = (1 << (tail_elements * bit_width)) - 1;
      uint8_t val = src_raw[src_byte_idx + num_bytes_to_copy] >> shift_bits;
      if (shift_bits + tail_elements * bit_width > 8) {
        val |= src_raw[src_byte_idx + num_bytes_to_copy + 1]
               << (8 - shift_bits);
      }
      dst_raw[num_bytes_to_copy] = val & tail_mask;
    }
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);

  const TfLiteTensor* element_shape;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kElementShapeInput, &element_shape));

  TF_LITE_ENSURE_TYPES_EQ(context, element_shape->type, kTfLiteInt32);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kListOut, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteVariant);
  output->allocation_type = kTfLiteVariantObject;

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* tensor_input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kTensorInput, &tensor_input));
  const int rank = tensor_input->dims->size;

  // As in Tensorflow, input is not permitted be a scalar.
  TF_LITE_ENSURE(context, rank > 0);

  // Output list has `num_elements` equal to the first dim of `tensor_input`,
  // and element tensors with shape equal to `Shape(tensor_input)[1:]`.
  const int list_len = tensor_input->dims->data[0];
  IntArrayUniquePtr element_shape_for_tensors =
      BuildTfLiteArray(rank - 1, tensor_input->dims->data + 1);
  TF_LITE_ENSURE(context, element_shape_for_tensors != nullptr);

  // `element_shape_tensor` is an auxiliary input shape signature which
  // is to be used as the `ElementShape()` attribute of the resulting
  // `TensorArray`.
  const TfLiteTensor* element_shape_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kElementShapeInput,
                                          &element_shape_tensor));
  TF_LITE_ENSURE(context, (element_shape_tensor->dims->size == 1 &&
                           element_shape_tensor->dims->data[0] == rank - 1) ||
                              element_shape_tensor->dims->size == 0);

  IntArrayUniquePtr element_shape_for_list;
  TF_LITE_ENSURE_OK(context, TensorAsShape(context, *element_shape_tensor,
                                           element_shape_for_list));
  TF_LITE_ENSURE(context, element_shape_for_list != nullptr);
  // Check given element shape is compatible with the suffix of input tensor's
  // shape. TODO(b/257472333) consider wrapping this in `#ifndef NDEBUG`.
  if (element_shape_for_list->size > 0) {
    TF_LITE_ENSURE_EQ(context, element_shape_for_list->size,
                      element_shape_for_tensors->size);
    for (int i = 0; i < element_shape_for_tensors->size; ++i) {
      const int lhs = element_shape_for_list->data[i];
      const int rhs = element_shape_for_tensors->data[i];
      TF_LITE_ENSURE(context, lhs == -1 || rhs == -1 || lhs == rhs);
    }
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kListOut, &output));

  // Build and retrieve output list.
  IntArrayUniquePtr copied_element_shape =
      BuildTfLiteArray(*element_shape_for_list);
  TF_LITE_ENSURE(context, copied_element_shape != nullptr);
  TF_LITE_ENSURE_OK(context, TfLiteTensorVariantRealloc<TensorArray>(
                                 output, tensor_input->type,
                                 std::move(copied_element_shape)));
  TensorArray* arr =
      static_cast<TensorArray*>(static_cast<VariantData*>(output->data.data));

  TF_LITE_ENSURE(context, arr->Resize(list_len));

  int num_elements_per_row = 1;
  for (int j = 0; j < element_shape_for_tensors->size; ++j) {
    num_elements_per_row *= element_shape_for_tensors->data[j];
  }
  const bool is_packed_type =
      (tensor_input->type == kTfLiteInt4 ||
       tensor_input->type == kTfLiteUInt4 || tensor_input->type == kTfLiteInt2);

  // Copy each row of input into the elements of the new list.
  size_t data_offset = 0;
  for (int i = 0; i < list_len; ++i) {
    IntArrayUniquePtr cur_shape = BuildTfLiteArray(*element_shape_for_tensors);
    TF_LITE_ENSURE(context, cur_shape != nullptr);
    TensorUniquePtr tensor_to_set = BuildTfLiteTensor(
        tensor_input->type, std::move(cur_shape), kTfLiteDynamic);
    TF_LITE_ENSURE(context, tensor_to_set != nullptr);

    if (tensor_to_set->bytes > 0) {
      TF_LITE_ENSURE(context, tensor_to_set->data.raw != nullptr);
      TF_LITE_ENSURE(context, tensor_input->data.raw != nullptr);
      if (is_packed_type) {
        CopyPackedTensor(tensor_input, tensor_to_set.get(),
                         i * num_elements_per_row, num_elements_per_row);
      } else {
        TF_LITE_ENSURE(
            context, tensor_to_set->bytes <= tensor_input->bytes - data_offset);
        memcpy(tensor_to_set->data.raw, tensor_input->data.raw + data_offset,
               tensor_to_set->bytes);
        data_offset += tensor_to_set->bytes;
      }
    }

    TF_LITE_ENSURE(context, arr->Set(i, std::move(tensor_to_set)));
  }

  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration* Register_LIST_FROM_TENSOR() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare, Eval};
  return &r;
}

}  // namespace ops
}  // namespace variants
}  // namespace tflite
