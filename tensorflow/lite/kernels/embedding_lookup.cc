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

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

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
         value->type == kTfLiteInt4) &&
        (output->type == kTfLiteFloat32)) {
      // EvalHybrid supports only symmetric quantization for now.
      TF_LITE_ENSURE(context, qparams->zero_point->data[0] == 0);
    }
    if (qparams->scale->size > 1) {
      // Per-axis quantization is supported by EvalHybrid only.
      TF_LITE_ENSURE(context, value->type == kTfLiteUInt8 ||
                                  value->type == kTfLiteInt8 ||
                                  value->type == kTfLiteInt4);
      TF_LITE_ENSURE(context, output->type == kTfLiteFloat32);
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
      std::memcpy(output_raw + i * row_bytes, value_raw + idx * row_bytes,
                  row_bytes);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalHybrid(TfLiteContext* context, TfLiteNode* node,
                        const TfLiteTensor* lookup, const TfLiteTensor* value,
                        TfLiteTensor* output) {
  const int row_size = SizeOfDimension(value, 0);

  // col_size after we flatten tensor into 2D.
  int col_size = 1;
  for (int i = 1; i < NumDimensions(value); i++) {
    col_size *= SizeOfDimension(value, i);
  }

  float* output_ptr = GetTensorData<float>(output);
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
      // Dequantize embedding values.
      // TODO(alanchiao): refactor scalar multiply into separate function
      // for ease of adding a neon equivalent if ever necessary.
      double scaling_factor = value->params.scale;
      if (value->quantization.type == kTfLiteAffineQuantization) {
        const auto qparams = static_cast<const TfLiteAffineQuantization*>(
            value->quantization.params);
        if (qparams->scale->size > 1) {
          // get this row's scale for per-axis quantization
          scaling_factor = qparams->scale->data[idx];
        }
      }

      if (value->type == kTfLiteInt4) {
        for (int j = 0; j < col_size; j++) {
          int i8_idx = j + idx * col_size;
          int i4_idx = i8_idx / 2;
          bool even = i8_idx % 2 == 0;
          int8_t i4_val = value_ptr[i4_idx];
          int8_t i8_val =
              even ? static_cast<int8_t>(i4_val << 4) >> 4 : i4_val >> 4;
          output_ptr[j + i * col_size] = i8_val * scaling_factor;
        }
      } else {
        for (int j = 0; j < col_size; j++) {
          output_ptr[j + i * col_size] =
              value_ptr[j + idx * col_size] * scaling_factor;
        }
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
  switch (value->type) {
    case kTfLiteFloat32:
      return EvalSimple(context, node, lookup, value, output);
    case kTfLiteInt4:
      return EvalHybrid(context, node, lookup, value, output);
    case kTfLiteUInt8:
    case kTfLiteInt8:
      if (output->type == kTfLiteFloat32) {
        return EvalHybrid(context, node, lookup, value, output);
      } else {
        return EvalSimple(context, node, lookup, value, output);
      }
    default:
      TF_LITE_KERNEL_LOG(context, "Type not currently supported.");
      return kTfLiteError;
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
