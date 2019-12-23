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

// Convert a list of strings to integers via hashing.
// Input:
//     Input[0]: A list of ngrams. string[num of input]
//
// Output:
//     Output[0]: Hashed features. int32[num of input]
//     Output[1]: Weights. float[num of input]

#include <algorithm>
#include <map>

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"
#include <farmhash.h>

namespace tflite {
namespace ops {
namespace custom {

namespace extract {

static const int kMaxDimension = 1000000;
static const std::vector<string> kBlacklistNgram = {"<S>", "<E>", "<S> <E>"};

bool Equals(const string& x, const tflite::StringRef& strref) {
  if (strref.len != x.length()) {
    return false;
  }
  if (strref.len > 0) {
    int r = memcmp(strref.str, x.data(), strref.len);
    return r == 0;
  }
  return true;
}

bool IsValidNgram(const tflite::StringRef& strref) {
  for (const auto& s : kBlacklistNgram) {
    if (Equals(s, strref)) {
      return false;
    }
  }
  return true;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  int dim = input->dims->data[0];
  if (dim == 0) {
    // TFLite non-string output should have size greater than 0.
    dim = 1;
  }
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteString);
  TfLiteIntArray* outputSize1 = TfLiteIntArrayCreate(1);
  TfLiteIntArray* outputSize2 = TfLiteIntArrayCreate(1);
  outputSize1->data[0] = dim;
  outputSize2->data[0] = dim;
  context->ResizeTensor(context, GetOutput(context, node, 0), outputSize1);
  context->ResizeTensor(context, GetOutput(context, node, 1), outputSize2);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  int num_strings = tflite::GetStringCount(input);
  TfLiteTensor* label = GetOutput(context, node, 0);
  TfLiteTensor* weight = GetOutput(context, node, 1);

  int32_t* label_data = GetTensorData<int32_t>(label);
  float* weight_data = GetTensorData<float>(weight);

  std::map<int64_t, int> feature_id_counts;
  for (int i = 0; i < num_strings; i++) {
    // Use fingerprint of feature name as id.
    auto strref = tflite::GetString(input, i);
    if (!IsValidNgram(strref)) {
      label_data[i] = 0;
      weight_data[i] = 0;
      continue;
    }

    int64_t feature_id =
        ::util::Fingerprint64(strref.str, strref.len) % kMaxDimension;
    label_data[i] = static_cast<int32_t>(feature_id);
    weight_data[i] = std::count(strref.str, strref.str + strref.len, ' ') + 1;
  }
  // Explicitly set an empty result to make preceding ops run.
  if (num_strings == 0) {
    label_data[0] = 0;
    weight_data[0] = 0;
  }
  return kTfLiteOk;
}

}  // namespace extract

TfLiteRegistration* Register_EXTRACT_FEATURES() {
  static TfLiteRegistration r = {nullptr, nullptr, extract::Prepare,
                                 extract::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
