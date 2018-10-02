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

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include "tensorflow/contrib/lite/c/builtin_op_data.h"
#include "tensorflow/contrib/lite/c/c_api_internal.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace embedding_lookup {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* lookup = GetInput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(lookup), 1);
  TF_LITE_ENSURE_EQ(context, lookup->type, kTfLiteInt32);

  const TfLiteTensor* value = GetInput(context, node, 1);
  TF_LITE_ENSURE(context, NumDimensions(value) >= 2);

  TfLiteTensor* output = GetOutput(context, node, 0);
  TfLiteIntArray* outputSize = TfLiteIntArrayCreate(NumDimensions(value));

  outputSize->data[0] = SizeOfDimension(lookup, 0);
  outputSize->data[1] = SizeOfDimension(value, 1);
  for (int i = 2; i < NumDimensions(value); i++) {
    outputSize->data[i] = SizeOfDimension(value, i);
  }
  return context->ResizeTensor(context, output, outputSize);
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       const TfLiteTensor* lookup, const TfLiteTensor* value,
                       TfLiteTensor* output) {
  const int row_size = SizeOfDimension(value, 0);
  const int row_bytes = value->bytes / row_size;

  for (int i = 0; i < SizeOfDimension(lookup, 0); i++) {
    int idx = lookup->data.i32[i];
    if (idx >= row_size || idx < 0) {
      context->ReportError(context, "Embedding Lookup: index out of bounds.");
      return kTfLiteError;
    } else {
      memcpy(output->data.raw + i * row_bytes,
             value->data.raw + idx * row_bytes, row_bytes);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalHybrid(TfLiteContext* context, TfLiteNode* node,
                        const TfLiteTensor* lookup, const TfLiteTensor* value,
                        TfLiteTensor* output) {
  const int row_size = SizeOfDimension(value, 0);
  const double scaling_factor = value->params.scale;

  // col_size after we flatten tensor into 2D.
  int col_size = 1;
  for (int i = 1; i < NumDimensions(value); i++) {
    col_size *= SizeOfDimension(value, i);
  }

  for (int i = 0; i < SizeOfDimension(lookup, 0); i++) {
    int idx = lookup->data.i32[i];
    if (idx >= row_size || idx < 0) {
      context->ReportError(context, "Embedding Lookup: index out of bounds.");
      return kTfLiteError;
    } else {
      // Dequantize embedding values.
      // TODO(alanchiao): refactor scalar multiply into separate function
      // for ease of adding a neon equivalent if ever necessary.
      for (int j = 0; j < col_size; j++) {
        const int8_t* value_ptr = reinterpret_cast<int8_t*>(value->data.uint8);
        output->data.f[j + i * col_size] =
            value_ptr[j + idx * col_size] * scaling_factor;
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* lookup = GetInput(context, node, 0);
  const TfLiteTensor* value = GetInput(context, node, 1);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (value->type) {
    case kTfLiteFloat32:
      return EvalFloat(context, node, lookup, value, output);
    case kTfLiteUInt8:
      return EvalHybrid(context, node, lookup, value, output);
    default:
      context->ReportError(context, "Type not currently supported.");
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
