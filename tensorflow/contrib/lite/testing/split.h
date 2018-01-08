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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TESTING_SPLIT_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TESTING_SPLIT_H_

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>
#include "tensorflow/contrib/lite/string.h"

namespace tflite {
namespace testing {

// Splits a string based on the given delimiter string. Each pair in the
// returned vector has the start and past-the-end positions for each of the
// parts of the original string. Empty fields are not represented in the
// output.
std::vector<std::pair<size_t, size_t>> SplitToPos(const string& s,
                                                  const string& delimiter);

// Splits the given string and converts each part to the given T.
template <typename T>
std::vector<T> Split(const string& s, const string& delimiter);

template <>
inline std::vector<string> Split(const string& s, const string& delimiter) {
  std::vector<string> fields;
  for (const auto& p : SplitToPos(s, delimiter)) {
    fields.push_back(s.substr(p.first, p.second - p.first));
  }
  return fields;
}

template <>
inline std::vector<int> Split(const string& s, const string& delimiter) {
  std::vector<int> fields;
  for (const auto& p : SplitToPos(s, delimiter)) {
    fields.push_back(strtol(s.data() + p.first, nullptr, 10));
  }
  return fields;
}

template <>
inline std::vector<float> Split(const string& s, const string& delimiter) {
  std::vector<float> fields;
  for (const auto& p : SplitToPos(s, delimiter)) {
    fields.push_back(strtod(s.data() + p.first, nullptr));
  }
  return fields;
}

template <>
inline std::vector<uint8_t> Split(const string& s, const string& delimiter) {
  std::vector<uint8_t> fields;
  for (const auto& p : SplitToPos(s, delimiter)) {
    fields.push_back(strtol(s.data() + p.first, nullptr, 10));
  }
  return fields;
}

}  // namespace testing
}  // namespace tflite

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TESTING_SPLIT_H_
