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

// Lookup projected hash signatures in Predictor model,
// output predicted labels and weights in decreasing order.
//
// Input:
//     Input[0]: A list of hash signatures. int32[num of input]
//     Input[1]: Hash signature keys in the model. int32[keys of model]
//     Input[2]: Labels in the model. int32[keys of model, item per entry]
//     Input[3]: Weights in the model. float[keys of model, item per entry]
//
// Output:
//     Output[0]: Predicted labels. int32[num of output]
//     Output[1]: Predicted weights. float[num of output]
//

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/context.h"

namespace tflite {
namespace ops {
namespace custom {

namespace predict {

struct PredictOption {
  int32_t num_output;
  float weight_threshold;

  static PredictOption* Cast(void* ptr) {
    return reinterpret_cast<PredictOption*>(ptr);
  }
};

bool WeightGreater(const std::pair<int32_t, float>& a,
                   const std::pair<int32_t, float>& b) {
  return a.second > b.second;
}

void* Init(TfLiteContext* context, const char* custom_option, size_t length) {
  if (custom_option == nullptr || length != sizeof(PredictOption)) {
    fprintf(stderr, "No Custom option set\n");
    exit(1);
  }
  PredictOption* option = new PredictOption;
  int offset = 0;
  option->num_output =
      *reinterpret_cast<const int32_t*>(custom_option + offset);
  offset += sizeof(int32_t);
  option->weight_threshold =
      *reinterpret_cast<const float*>(custom_option + offset);
  return reinterpret_cast<void*>(option);
}

void Free(TfLiteContext* context, void* buffer) {
  delete PredictOption::Cast(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 4);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 2);

  TfLiteTensor* lookup = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* model_key = &context->tensors[node->inputs->data[1]];
  TfLiteTensor* model_label = &context->tensors[node->inputs->data[2]];
  TfLiteTensor* model_weight = &context->tensors[node->inputs->data[3]];
  TF_LITE_ENSURE_EQ(context, lookup->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, model_key->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, model_label->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, model_weight->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, lookup->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, model_key->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, model_label->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, model_weight->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, model_key->dims->data[0],
                    model_label->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, model_key->dims->data[0],
                    model_weight->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, model_label->dims->data[1],
                    model_weight->dims->data[1]);

  PredictOption* option = PredictOption::Cast(node->user_data);
  TfLiteTensor* output_label = &context->tensors[node->outputs->data[0]];
  TfLiteTensor* output_weight = &context->tensors[node->outputs->data[1]];
  TF_LITE_ENSURE_EQ(context, output_label->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, output_weight->type, kTfLiteFloat32);

  TfLiteIntArray* label_size = TfLiteIntArrayCreate(1);
  label_size->data[0] = option->num_output;
  TfLiteIntArray* weight_size = TfLiteIntArrayCreate(1);
  weight_size->data[0] = option->num_output;
  TfLiteStatus status =
      context->ResizeTensor(context, output_label, label_size);
  if (status != kTfLiteOk) {
    return status;
  }
  return context->ResizeTensor(context, output_weight, weight_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* lookup = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* model_key = &context->tensors[node->inputs->data[1]];
  TfLiteTensor* model_label = &context->tensors[node->inputs->data[2]];
  TfLiteTensor* model_weight = &context->tensors[node->inputs->data[3]];

  // Aggregate by key
  std::unordered_map<int32_t, float> aggregation;
  const int num_input = lookup->dims->data[0];
  const int num_rows = model_key->dims->data[0];
  const int items = model_label->dims->data[1];
  int* model_key_end = model_key->data.i32 + num_rows;

  for (int i = 0; i < num_input; i++) {
    int* ptr = std::lower_bound(model_key->data.i32, model_key_end,
                                lookup->data.i32[i]);
    if (ptr != nullptr && ptr != model_key_end && *ptr == lookup->data.i32[i]) {
      int idx = ptr - model_key->data.i32;
      for (int j = 0; j < items; j++) {
        aggregation[model_label->data.i32[idx * items + j]] +=
            model_weight->data.f[idx * items + j] / num_input;
      }
    }
  }

  // Sort by value
  std::vector<std::pair<int32_t, float>> sorted_labels(aggregation.begin(),
                                                       aggregation.end());
  std::sort(sorted_labels.begin(), sorted_labels.end(), WeightGreater);

  PredictOption* option = PredictOption::Cast(node->user_data);
  TfLiteTensor* output_label = &context->tensors[node->outputs->data[0]];
  TfLiteTensor* output_weight = &context->tensors[node->outputs->data[1]];
  for (int i = 0; i < output_label->dims->data[0]; i++) {
    if (i >= sorted_labels.size() ||
        sorted_labels[i].second < option->weight_threshold) {
      // Set -1 to avoid lookup message with id 0, which is set for backoff.
      output_label->data.i32[i] = -1;
      output_weight->data.f[i] = 0.0f;
    } else {
      output_label->data.i32[i] = sorted_labels[i].first;
      output_weight->data.f[i] = sorted_labels[i].second;
    }
  }

  return kTfLiteOk;
}

}  // namespace predict

TfLiteRegistration* Register_PREDICT() {
  static TfLiteRegistration r = {predict::Init, predict::Free, predict::Prepare,
                                 predict::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
