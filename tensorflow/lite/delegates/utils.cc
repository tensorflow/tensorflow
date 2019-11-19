/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/utils.h"

#include <algorithm>

#include "tensorflow/lite/c/c_api_internal.h"

namespace tflite {
namespace delegates {

TfLiteStatus PruneContinuousSubsets(TfLiteContext* context,
                                    const int max_subsets,
                                    std::vector<int>* indices) {
  if (!indices) {
    context->ReportError(context, "indices cannot be nullptr");
    return kTfLiteError;
  }
  if (indices->empty() || indices->size() < max_subsets) return kTfLiteOk;

  // Sort indices just in case.
  std::sort(indices->begin(), indices->end());

  // Build a vector of subsets.
  std::vector<std::vector<int>> continuous_subsets;
  int last_index = indices->at(0) - 2;
  for (const auto idx : *indices) {
    if (idx > last_index + 1) {
      continuous_subsets.emplace_back();
    }
    continuous_subsets.back().push_back(idx);
    last_index = idx;
  }

  // Nothing to be done if number of subsets is already less than max_subsets.
  if (continuous_subsets.size() <= max_subsets) return kTfLiteOk;

  // Sort the vector of subsets in descending order of length.
  std::sort(continuous_subsets.begin(), continuous_subsets.end(),
            [](const std::vector<int>& a, const std::vector<int>& b) {
              return a.size() > b.size();
            });

  // Re-build indices vector from top subsets.
  indices->clear();
  for (int i = 0; i < max_subsets; ++i) {
    indices->reserve(indices->size() + continuous_subsets[i].size());
    indices->insert(indices->end(), continuous_subsets[i].begin(),
                    continuous_subsets[i].end());
  }
  std::sort(indices->begin(), indices->end());

  return kTfLiteOk;
}

}  // namespace delegates
}  // namespace tflite
