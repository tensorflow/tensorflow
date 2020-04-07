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

// Op that looks up items from hashtable.
//
// Input:
//     Tensor[0]: Hash key to lookup, dim.size == 1, int32
//     Tensor[1]: Key of hashtable, dim.size == 1, int32
//                *MUST* be sorted in ascending order.
//     Tensor[2]: Value of hashtable, dim.size >= 1
//                Tensor[1].Dim[0] == Tensor[2].Dim[0]
//
// Output:
//   Output[0].dim[0] == Tensor[0].dim[0], num of lookups
//   Each item in output is a raw bytes copy of corresponding item in input.
//   When key does not exist in hashtable, the returned bytes are all 0s.
//
//   Output[1].dim = { Tensor[0].dim[0] }, num of lookups
//   Each item indicates whether the corresponding lookup has a returned value.
//   0 for missing key, 1 for found key.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {

namespace {

int greater(const void* a, const void* b) {
  return *static_cast<const int*>(a) - *static_cast<const int*>(b);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  const TfLiteTensor* lookup = GetInput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(lookup), 1);
  TF_LITE_ENSURE_EQ(context, lookup->type, kTfLiteInt32);

  const TfLiteTensor* key = GetInput(context, node, 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(key), 1);
  TF_LITE_ENSURE_EQ(context, key->type, kTfLiteInt32);

  const TfLiteTensor* value = GetInput(context, node, 2);
  TF_LITE_ENSURE(context, NumDimensions(value) >= 1);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(key, 0),
                    SizeOfDimension(value, 0));
  if (value->type == kTfLiteString) {
    TF_LITE_ENSURE_EQ(context, NumDimensions(value), 1);
  }

  TfLiteTensor* hits = GetOutput(context, node, 1);
  TF_LITE_ENSURE_EQ(context, hits->type, kTfLiteUInt8);
  TfLiteIntArray* hitSize = TfLiteIntArrayCreate(1);
  hitSize->data[0] = SizeOfDimension(lookup, 0);

  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, value->type, output->type);

  TfLiteStatus status = kTfLiteOk;
  if (output->type != kTfLiteString) {
    TfLiteIntArray* outputSize = TfLiteIntArrayCreate(NumDimensions(value));
    outputSize->data[0] = SizeOfDimension(lookup, 0);
    for (int i = 1; i < NumDimensions(value); i++) {
      outputSize->data[i] = SizeOfDimension(value, i);
    }
    status = context->ResizeTensor(context, output, outputSize);
  }
  if (context->ResizeTensor(context, hits, hitSize) == kTfLiteError) {
    status = kTfLiteError;
  }
  return status;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = GetOutput(context, node, 0);
  TfLiteTensor* hits = GetOutput(context, node, 1);
  const TfLiteTensor* lookup = GetInput(context, node, 0);
  const TfLiteTensor* key = GetInput(context, node, 1);
  const TfLiteTensor* value = GetInput(context, node, 2);

  const int num_rows = SizeOfDimension(value, 0);
  const int row_bytes = value->bytes / num_rows;
  void* pointer = nullptr;
  DynamicBuffer buf;

  for (int i = 0; i < SizeOfDimension(lookup, 0); i++) {
    int idx = -1;
    pointer = bsearch(&(lookup->data.i32[i]), key->data.i32, num_rows,
                      sizeof(int32_t), greater);
    if (pointer != nullptr) {
      idx = (reinterpret_cast<char*>(pointer) - (key->data.raw)) /
            sizeof(int32_t);
    }

    if (idx >= num_rows || idx < 0) {
      if (output->type == kTfLiteString) {
        buf.AddString(nullptr, 0);
      } else {
        memset(output->data.raw + i * row_bytes, 0, row_bytes);
      }
      hits->data.uint8[i] = 0;
    } else {
      if (output->type == kTfLiteString) {
        buf.AddString(GetString(value, idx));
      } else {
        memcpy(output->data.raw + i * row_bytes,
               value->data.raw + idx * row_bytes, row_bytes);
      }
      hits->data.uint8[i] = 1;
    }
  }
  if (output->type == kTfLiteString) {
    buf.WriteToTensorAsVector(output);
  }

  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration* Register_HASHTABLE_LOOKUP() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare, Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
