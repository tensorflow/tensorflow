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
#include "tensorflow/lite/testing/split.h"

#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace testing {

std::vector<std::pair<size_t, size_t>> SplitToPos(const string& s,
                                                  const string& delimiter) {
  std::vector<std::pair<size_t, size_t>> fields;
  if (delimiter.length() == 0) {
    fields.emplace_back(0, s.length());
    return fields;
  }
  size_t pos = 0;
  size_t start = 0;
  while ((pos = s.find(delimiter, start)) != string::npos) {
    if (pos != start) {
      fields.emplace_back(start, pos);
    }
    start = pos + delimiter.length();
  }
  if (start != s.length()) {
    fields.emplace_back(start, s.length());
  }
  return fields;
}

}  // namespace testing
}  // namespace tflite
