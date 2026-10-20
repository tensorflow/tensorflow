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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace hashtable_lookup {

namespace {

TfLiteStatus AppendValidatedString(TfLiteContext* context,
                                   const TfLiteTensor& value, int idx,
                                   int num_rows, DynamicBuffer& buf) {
  TF_LITE_ENSURE(context, idx >= 0 && idx < num_rows);
  TF_LITE_ENSURE(context, value.data.raw != nullptr);
  TF_LITE_ENSURE(context, value.bytes >= sizeof(int32_t));
  const int num_strings = GetStringCount(&value);
  TF_LITE_ENSURE(context, num_strings >= num_rows);
  TF_LITE_ENSURE(context, value.bytes / sizeof(int32_t) >=
                              static_cast<size_t>(num_strings) + 2);
  const int32_t* offsets = reinterpret_cast<const int32_t*>(value.data.raw);
  const int32_t start_offset = offsets[static_cast<size_t>(idx) + 1];
  const int32_t end_offset = offsets[static_cast<size_t>(idx) + 2];
  TF_LITE_ENSURE(context, start_offset >= 0);
  TF_LITE_ENSURE(context, end_offset >= start_offset);
  TF_LITE_ENSURE(context, static_cast<size_t>(end_offset) <= value.bytes);
  return buf.AddString(value.data.raw + start_offset,
                       end_offset - start_offset);
}

}  // namespace

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  const TfLiteTensor* lookup;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &lookup));
  TF_LITE_ENSURE_EQ(context, NumDimensions(lookup), 1);
  TF_LITE_ENSURE_EQ(context, lookup->type, kTfLiteInt32);

  const TfLiteTensor* key;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &key));
  TF_LITE_ENSURE_EQ(context, NumDimensions(key), 1);
  TF_LITE_ENSURE_EQ(context, key->type, kTfLiteInt32);

  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &value));
  TF_LITE_ENSURE(context, NumDimensions(value) >= 1);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(key, 0),
                    SizeOfDimension(value, 0));
  if (value->type == kTfLiteString) {
    TF_LITE_ENSURE_EQ(context, NumDimensions(value), 1);
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_EQ(context, value->type, output->type);

  TfLiteTensor* hits;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &hits));
  TF_LITE_ENSURE_EQ(context, hits->type, kTfLiteUInt8);

  if (output->type != kTfLiteString) {
    TF_LITE_ENSURE_OK(context, kernel_utils::ResizeLookupOutputTensor(
                                   context, *lookup, *value, output));
  }

  TfLiteIntArray* hit_size = TfLiteIntArrayCreate(1);
  hit_size->data[0] = SizeOfDimension(lookup, 0);
  return context->ResizeTensor(context, hits, hit_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TfLiteTensor* hits;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &hits));
  const TfLiteTensor* lookup;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &lookup));
  const TfLiteTensor* key;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &key));
  const TfLiteTensor* value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &value));

  const int num_rows = SizeOfDimension(value, 0);
  const int lookup_size = SizeOfDimension(lookup, 0);

  DynamicBuffer buf;
  if (lookup_size == 0) {
    if (output->type == kTfLiteString) {
      buf.WriteToTensorAsVector(output);
    }
    return kTfLiteOk;
  }

  TF_LITE_ENSURE(context, num_rows > 0);
  const size_t row_bytes = value->bytes / num_rows;

  const int32_t* key_begin = key->data.i32;
  const int32_t* key_end = key_begin + num_rows;
  for (int i = 0; i < lookup_size; i++) {
    const int32_t target = lookup->data.i32[i];
    const int32_t* it =
        std::lower_bound(key_begin, key_end, target,
                         [](int32_t lhs, int32_t rhs) { return lhs < rhs; });
    const int idx = (it != key_end && *it == target)
                        ? static_cast<int>(it - key_begin)
                        : -1;

    if (idx >= num_rows || idx < 0) {
      if (output->type == kTfLiteString) {
        TF_LITE_ENSURE_OK(context, buf.AddString(nullptr, 0));
      } else if (row_bytes > 0) {
        std::memset(output->data.raw + static_cast<size_t>(i) * row_bytes, 0,
                    row_bytes);
      }
      hits->data.uint8[i] = 0;
    } else {
      if (output->type == kTfLiteString) {
        TF_LITE_ENSURE_OK(context, AppendValidatedString(context, *value, idx,
                                                         num_rows, buf));
      } else if (row_bytes > 0) {
        std::memcpy(output->data.raw + static_cast<size_t>(i) * row_bytes,
                    value->data.raw + static_cast<size_t>(idx) * row_bytes,
                    row_bytes);
      }
      hits->data.uint8[i] = 1;
    }
  }
  if (output->type == kTfLiteString) {
    buf.WriteToTensorAsVector(output);
  }

  return kTfLiteOk;
}

}  // namespace hashtable_lookup

TfLiteRegistration* Register_HASHTABLE_LOOKUP() {
  static TfLiteRegistration r = {nullptr, nullptr, hashtable_lookup::Prepare,
                                 hashtable_lookup::Eval};
  return &r;
}
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
