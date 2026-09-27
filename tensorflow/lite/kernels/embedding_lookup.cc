/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Ops that looks up items from matrix.
//
// Input:
//     Tensor[0]: Row number to lookup, dim.size == 1, int32
//     Tensor[1]: 2-dimensional matrix of multi-dimensional items
//                dim.size >= 2, any data type.
//                first dimension is row, second dimension is column.
//
// Output:
//   Output.dim[0] == Tensor[0].dim[0], num of lookups
//   Output.dim[1] == Tensor[1].dim[1],  num of items per row
//   Each item in output is a raw bytes copy of the corresponding item in input,
//   or a dequantized value in the case of a uint8 input.
//   When indices are out of bound, the ops will not succeed.
//

#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/types/half.h"
#include "tensorflow/lite/util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace embedding_lookup {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* lookup;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &lookup));
  TF_LITE_ENSURE_EQ(context, NumDimensions(lookup), 1);
  TF_LITE_ENSURE_EQ(context, lookup->type, kTfLiteInt32);

  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &value));
  TF_LITE_ENSURE(context, NumDimensions(value) >= 2);

  if (value->quantization.type == kTfLiteAffineQuantization) {
    const auto qparams = static_cast<const TfLiteAffineQuantization*>(
        value->quantization.params);
    TF_LITE_ENSURE(context, qparams->scale != nullptr);
    TF_LITE_ENSURE(context, qparams->zero_point != nullptr);
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
    if ((value->type == kTfLiteUInt8 || value->type == kTfLiteInt8 ||
         value->type == kTfLiteInt4 || value->type == kTfLiteInt2) &&
        (output->type == kTfLiteFloat32)) {
      // EvalHybrid supports only symmetric quantization for now.
      TF_LITE_ENSURE(context, qparams->zero_point->data[0] == 0);
    }
    if (qparams->scale->size > 1) {
      // Per-axis quantization is supported by EvalHybrid only.
      TF_LITE_ENSURE(context, value->type == kTfLiteUInt8 ||
                                  value->type == kTfLiteInt8 ||
                                  value->type == kTfLiteInt4 ||
                                  value->type == kTfLiteInt2);
      TF_LITE_ENSURE(context, output->type == kTfLiteFloat32 ||
                                  output->type == kTfLiteFloat16);
      // Per-axis quantization must have quantized_dimension == 0 and correct
      // sizes for scale and zero_point.
      TF_LITE_ENSURE(context, qparams->quantized_dimension == 0);
      const int row_size = SizeOfDimension(value, 0);
      TF_LITE_ENSURE(context, qparams->scale->size == row_size);
      TF_LITE_ENSURE(context, qparams->zero_point->size == row_size ||
                                  qparams->zero_point->size == 1);
    }
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(NumDimensions(value));

  output_size->data[0] = SizeOfDimension(lookup, 0);
  output_size->data[1] = SizeOfDimension(value, 1);
  for (int i = 2; i < NumDimensions(value); i++) {
    output_size->data[i] = SizeOfDimension(value, i);
  }
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus EvalSimple(TfLiteContext* context, TfLiteNode* node,
                        const TfLiteTensor* lookup, const TfLiteTensor* value,
                        TfLiteTensor* output) {
  const int row_size = SizeOfDimension(value, 0);
  if (row_size == 0) {
    // Propagate empty tensor if input is empty
    return kTfLiteOk;
  }
  const size_t row_bytes = value->bytes / row_size;

  char* output_raw = GetTensorData<char>(output);
  const char* value_raw = GetTensorData<char>(value);
  const int32_t* lookup_data = GetTensorData<int32_t>(lookup);
  for (int i = 0; i < SizeOfDimension(lookup, 0); i++) {
    const int32_t idx = lookup_data[i];
    if (idx >= row_size || idx < 0) {
      TF_LITE_KERNEL_LOG(context,
                         "Embedding Lookup: index out of bounds. "
                         "Got %" PRId32 ", and bounds are [0, %d]",
                         idx, row_size - 1);
      return kTfLiteError;
    } else {
      CheckedInt<size_t> offset_output = CheckedInt<size_t>(i) * row_bytes;
      CheckedInt<size_t> offset_value = CheckedInt<size_t>(idx) * row_bytes;
      if (offset_output.Overflow() || offset_value.Overflow()) {
        TF_LITE_KERNEL_LOG(context, "Embedding Lookup: offset overflow.");
        return kTfLiteError;
      }
      std::memcpy(output_raw + offset_output.Value(),
                  value_raw + offset_value.Value(), row_bytes);
    }
  }
  return kTfLiteOk;
}

template <typename T>
void Unpack4Bit(float scaling_factor, size_t col_size, const int8_t* value_ptr,
                T* output_ptr) {
  float scaling_factor0 = scaling_factor / 16;
  size_t j = 0;
  size_t i4_idx = 0;
  for (; j + 1 < col_size; j += 2, ++i4_idx) {
    int8_t i4_val = value_ptr[i4_idx];
    int8_t i8_val0 = i4_val << 4;
    int8_t i8_val1 = i4_val & 0xF0;

    output_ptr[j] = i8_val0 * scaling_factor0;
    output_ptr[j + 1] = i8_val1 * scaling_factor0;
  }
  if (col_size & 1) {
    int8_t i4_val = value_ptr[i4_idx];
    int8_t i8_val0 = i4_val << 4;
    output_ptr[j] = i8_val0 * scaling_factor0;
  }
}

template <typename T>
void Unpack2Bit(float scaling_factor, size_t col_size, const int8_t* value_ptr,
                T* output_ptr) {
  float scaling_factor0 = scaling_factor / 64;  // 2**6
  size_t j = 0;
  size_t i2_idx = 0;
  for (; j + 3 < col_size; j += 4, ++i2_idx) {
    int8_t i2_val = value_ptr[i2_idx];
    int8_t i8_val0 = static_cast<int8_t>(i2_val << 6);
    int8_t i8_val1 = static_cast<int8_t>(i2_val << 4) & 0xC0;
    int8_t i8_val2 = static_cast<int8_t>(i2_val << 2) & 0xC0;
    int8_t i8_val3 = i2_val & 0xC0;

    output_ptr[j] = i8_val0 * scaling_factor0;
    output_ptr[j + 1] = i8_val1 * scaling_factor0;
    output_ptr[j + 2] = i8_val2 * scaling_factor0;
    output_ptr[j + 3] = i8_val3 * scaling_factor0;
  }
  size_t rem = col_size - j;
  if (rem) {
    int8_t i2_val = value_ptr[i2_idx];
    int8_t i8_val0 = static_cast<int8_t>(i2_val << 6);
    output_ptr[j] = i8_val0 * scaling_factor0;
    if (rem & 2) {
      int8_t i8_val1 = static_cast<int8_t>(i2_val << 4) & 0xC0;
      output_ptr[j + 1] = i8_val1 * scaling_factor0;
      if (rem & 1) {
        int8_t i8_val2 = static_cast<int8_t>(i2_val << 2) & 0xC0;
        output_ptr[j + 2] = i8_val2 * scaling_factor0;
      }
    }
  }
}

TfLiteStatus EvalBlockwise(TfLiteContext* context, TfLiteNode* node,
                           const TfLiteTensor* lookup,
                           const TfLiteTensor* value, TfLiteTensor* output) {
  if (value->type != kTfLiteInt4 && value->type != kTfLiteInt2) {
    TF_LITE_KERNEL_LOG(context,
                       "Embedding Lookup: Blockwise embedding lookup only "
                       "supports Int4 and Int2 data");
    return kTfLiteError;
  }
  if (output->type != kTfLiteFloat32 && output->type != kTfLiteFloat16) {
    TF_LITE_KERNEL_LOG(context,
                       "Embedding Lookup: Blockwise embedding lookup only "
                       "supports Float32 and Float16 outputs");
    return kTfLiteError;
  }
  if (value->dims->size != 2) {
    TF_LITE_KERNEL_LOG(
        context,
        "Embedding Lookup: Blockwise embedding lookup only supports 2D data");
    return kTfLiteError;
  }
  const int row_size = SizeOfDimension(value, 0);

  // col_size after we flatten tensor into 2D.
  CheckedInt<size_t> col_size = 1;
  for (int i = 1; i < NumDimensions(value); i++) {
    col_size = col_size * CheckedInt<size_t>(SizeOfDimension(value, i));
    if (col_size.Overflow()) {
      TF_LITE_KERNEL_LOG(context, "Embedding Lookup: col_size overflow.");
      return kTfLiteError;
    }
  }
  const auto quantization_params =
      reinterpret_cast<const TfLiteBlockwiseQuantization*>(
          value->quantization.params);
  const TfLiteTensor& scale = context->tensors[quantization_params->scale];
  const int blocksize = quantization_params->blocksize;
  const int dimension_size = SizeOfDimension(lookup, 0);

  float* output_fp32_ptr = GetTensorData<float>(output);
  half* output_fp16_ptr = GetTensorData<half>(output);

  const int8_t* value_ptr = GetTensorData<int8_t>(value);
  const int32_t* lookup_data = GetTensorData<int32_t>(lookup);

  // Wrap the correct 2/4-bit float32/float16 unpacking function.
  auto [unpack_to_fp32, unpack_to_fp16] =
      value->type == kTfLiteInt2
          ? std::make_pair(Unpack2Bit<float>, Unpack2Bit<half>)
          : std::make_pair(Unpack4Bit<float>, Unpack4Bit<half>);
  const int values_per_byte = value->type == kTfLiteInt2 ? 4 : 2;
  std::function<void(float, size_t, size_t)> unpack;
  if (output->type == kTfLiteFloat32) {
    unpack = [&, unpack = unpack_to_fp32](float scaling_factor,
                                          size_t value_offset,
                                          size_t output_offset) {
      unpack(scaling_factor, blocksize,
             &value_ptr[value_offset / values_per_byte],
             &output_fp32_ptr[output_offset]);
    };
  } else {
    unpack = [&, unpack = unpack_to_fp16](float scaling_factor,
                                          size_t value_offset,
                                          size_t output_offset) {
      unpack(scaling_factor, blocksize,
             &value_ptr[value_offset / values_per_byte],
             &output_fp16_ptr[output_offset]);
    };
  }

  if (col_size.Value() % blocksize != 0) {
    TF_LITE_KERNEL_LOG(context,
                       "Embedding Lookup: lookup dimension %zu must be "
                       "divisible by blocksize %d",
                       col_size.Value(), blocksize);
    return kTfLiteError;
  }
  size_t num_blocks = col_size.Value() / blocksize;
  for (int i = 0; i < dimension_size; i++) {
    int idx = lookup_data[i];
    if (idx >= row_size || idx < 0) {
      TF_LITE_KERNEL_LOG(context,
                         "Embedding Lookup: index out of bounds. "
                         "Got %d, and bounds are [0, %d]",
                         idx, row_size - 1);
      return kTfLiteError;
    }
    CheckedInt<size_t> output_row_offset = CheckedInt<size_t>(i) * col_size;
    CheckedInt<size_t> scale_offset = CheckedInt<size_t>(idx) * num_blocks;
    CheckedInt<size_t> value_offset = CheckedInt<size_t>(idx) * col_size;
    if (output_row_offset.Overflow() || scale_offset.Overflow() ||
        value_offset.Overflow()) {
      TF_LITE_KERNEL_LOG(context, "Embedding Lookup: offset overflow.");
      return kTfLiteError;
    }
    for (size_t j = 0; j < num_blocks; ++j) {
      float scaling_factor =
          GetTensorData<half>(&scale)[scale_offset.Value() + j];
      CheckedInt<size_t> val_off =
          value_offset + CheckedInt<size_t>(j) * blocksize;
      CheckedInt<size_t> out_off =
          output_row_offset + CheckedInt<size_t>(j) * blocksize;
      if (val_off.Overflow() || out_off.Overflow()) {
        TF_LITE_KERNEL_LOG(context, "Embedding Lookup: offset overflow.");
        return kTfLiteError;
      }
      unpack(scaling_factor, val_off.Value(), out_off.Value());
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalHybrid(TfLiteContext* context, TfLiteNode* node,
                        const TfLiteTensor* lookup, const TfLiteTensor* value,
                        TfLiteTensor* output) {
  const int row_size = SizeOfDimension(value, 0);

  // col_size after we flatten tensor into 2D.
  CheckedInt<size_t> col_size = 1;
  for (int i = 1; i < NumDimensions(value); i++) {
    col_size = col_size * CheckedInt<size_t>(SizeOfDimension(value, i));
    if (col_size.Overflow()) {
      TF_LITE_KERNEL_LOG(context, "Embedding Lookup: col_size overflow.");
      return kTfLiteError;
    }
  }
  auto copy_row = [&](float scaling_factor, auto output_ptr, auto value_ptr,
                      int idx, size_t out_off) -> TfLiteStatus {
    CheckedInt<size_t> offset = col_size * CheckedInt<size_t>(idx);
    if (offset.Overflow()) {
      TF_LITE_KERNEL_LOG(context, "Embedding Lookup: offset overflow.");
      return kTfLiteError;
    }
    if (value->type == kTfLiteInt4) {
      Unpack4Bit(scaling_factor, col_size.Value(),
                 &value_ptr[offset.Value() >> 1], &output_ptr[out_off]);
    } else if (value->type == kTfLiteInt2) {
      Unpack2Bit(scaling_factor, col_size.Value(),
                 &value_ptr[offset.Value() >> 2], &output_ptr[out_off]);
    } else {
      for (size_t j = 0; j < col_size.Value(); j++) {
        CheckedInt<size_t> output_idx = CheckedInt<size_t>(j) + out_off;
        CheckedInt<size_t> value_idx = offset + CheckedInt<size_t>(j);
        if (output_idx.Overflow() || value_idx.Overflow()) {
          TF_LITE_KERNEL_LOG(context, "Embedding Lookup: index overflow.");
          return kTfLiteError;
        }
        output_ptr[output_idx.Value()] =
            value_ptr[value_idx.Value()] * scaling_factor;
      }
    }
    return kTfLiteOk;
  };

  float* output_fp32_ptr =
      output->type == kTfLiteFloat32 ? GetTensorData<float>(output) : nullptr;
  half* output_fp16_ptr =
      output->type == kTfLiteFloat16 ? GetTensorData<half>(output) : nullptr;
  const int8_t* value_ptr = GetTensorData<int8_t>(value);
  const int32_t* lookup_data = GetTensorData<int32_t>(lookup);

  for (int i = 0; i < SizeOfDimension(lookup, 0); i++) {
    const int32_t idx = lookup_data[i];
    if (idx >= row_size || idx < 0) {
      TF_LITE_KERNEL_LOG(context,
                         "Embedding Lookup: index out of bounds. "
                         "Got %" PRId32 ", and bounds are [0, %d]",
                         idx, row_size - 1);
      return kTfLiteError;
    } else {
      CheckedInt<size_t> out_off = CheckedInt<size_t>(i) * col_size.Value();
      if (out_off.Overflow()) {
        TF_LITE_KERNEL_LOG(context, "Embedding Lookup: offset overflow.");
        return kTfLiteError;
      }
      // Dequantize embedding values.
      // TODO(alanchiao): refactor scalar multiply into separate function
      // for ease of adding a neon equivalent if ever necessary.
      float scaling_factor = value->params.scale;
      if (value->quantization.type == kTfLiteAffineQuantization) {
        const auto qparams = static_cast<const TfLiteAffineQuantization*>(
            value->quantization.params);
        if (qparams->scale->size > 1) {
          // get this row's scale for per-axis quantization
          scaling_factor = qparams->scale->data[idx];
        }
      }

      if (output_fp32_ptr) {
        TF_LITE_ENSURE_OK(context, copy_row(scaling_factor, output_fp32_ptr,
                                            value_ptr, idx, out_off.Value()));
      } else {
        TF_LITE_ENSURE_OK(context, copy_row(scaling_factor, output_fp16_ptr,
                                            value_ptr, idx, out_off.Value()));
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* lookup;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &lookup));
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &value));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  if (value->quantization.type == kTfLiteBlockwiseQuantization) {
    return EvalBlockwise(context, node, lookup, value, output);
  } else if (value->type != output->type && (output->type == kTfLiteFloat32 ||
                                             output->type == kTfLiteFloat16)) {
    return EvalHybrid(context, node, lookup, value, output);
  } else {
    return EvalSimple(context, node, lookup, value, output);
  }
}

}  // namespace embedding_lookup

TfLiteRegistration* Register_EMBEDDING_LOOKUP() {
  static TfLiteRegistration r = {nullptr, nullptr, embedding_lookup::Prepare,
                                 embedding_lookup::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
