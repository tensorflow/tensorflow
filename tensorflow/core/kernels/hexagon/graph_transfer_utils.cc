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

#include "tensorflow/core/kernels/hexagon/graph_transfer_utils.h"

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

/* static */ std::priority_queue<std::tuple<float, int, string>>
GraphTransferUtils::GetTopNFloatResults(const float *const data,
                                        const string *const labels,
                                        const int element_count) {
  CHECK(data != nullptr);
  CHECK(labels != nullptr);
  std::priority_queue<std::tuple<float, int, string>> queue;
  for (int i = 0; i < element_count; ++i) {
    queue.emplace(data[i], i, labels[i]);
  }
  return queue;
}

/* static */ void GraphTransferUtils::DumpTopNFloatResults(
    const float *const data, const string *const labels,
    const int element_count, const int top_n) {
  std::priority_queue<std::tuple<float, int, string>> queue =
      GetTopNFloatResults(data, labels, element_count);
  LOG(INFO) << "=== Dump ranking ===";
  for (int i = 0; i < top_n; ++i) {
    const std::tuple<float, int, string> &entry = queue.top();
    LOG(INFO) << i << ": " << std::get<1>(entry) << ", " << std::get<2>(entry)
              << ", " << std::get<0>(entry);
    queue.pop();
  }
}

}  // namespace tensorflow
