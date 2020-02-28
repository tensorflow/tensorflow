/* Copyright 2020 Google LLC. All Rights Reserved.

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

#include <vector>

#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"

namespace {

void MergeSortRecurse(int level, int size, int* data, int* workspace) {
  ruy::profiler::ScopeLabel function_label(
      "MergeSortRecurse (level=%d, size=%d)", level, size);
  if (size <= 1) {
    return;
  }
  int half_size = size / 2;
  MergeSortRecurse(level + 1, half_size, data, workspace);
  MergeSortRecurse(level + 1, size - half_size, data + half_size,
                   workspace + half_size);

  ruy::profiler::ScopeLabel merging_sorted_halves_label(
      "Merging sorted halves");
  int dst_index = 0;
  int left_index = 0;
  int right_index = half_size;
  while (dst_index < size) {
    int val;
    if (left_index < half_size &&
        ((right_index >= size) || data[left_index] < data[right_index])) {
      val = data[left_index++];
    } else {
      val = data[right_index++];
    }
    workspace[dst_index++] = val;
  }
  for (int i = 0; i < size; i++) {
    data[i] = workspace[i];
  }
}

}  // namespace

void MergeSort(int size, int* data) {
  ruy::profiler::ScopeLabel function_label("MergeSort (size=%d)", size);
  std::vector<int> workspace(size);
  MergeSortRecurse(0, size, data, workspace.data());
}
