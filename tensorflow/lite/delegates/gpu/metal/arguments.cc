/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/metal/arguments.h"

#include "absl/strings/ascii.h"

namespace tflite {
namespace gpu {
namespace metal {
namespace {
bool IsWordSymbol(char symbol) {
  return absl::ascii_isalnum(symbol) || symbol == '_';
}

bool HasWord(const std::string& word, const std::string& text) {
  size_t pos = text.find(word);
  while (pos != std::string::npos) {
    char prev = pos == 0 ? '.' : text[pos - 1];
    char next = pos + word.size() < text.size() ? text[pos + word.size()] : '.';
    if (!IsWordSymbol(prev) & !IsWordSymbol(next)) {
      return true;
    }
    pos = text.find(word, pos + 1);
  }
  return false;
}
}  // namespace

// Static
constexpr char Arguments::kArgsPrefix[];

void Arguments::AddFloat(const std::string& name, float value) {
  float_values_[name].value = value;
}

void Arguments::AddInt(const std::string& name, int value) {
  int_values_[name].value = value;
}

void Arguments::GetActiveArguments(const std::string& code) {
  for (auto& float_val : float_values_) {
    float_val.second.active = HasWord(kArgsPrefix + float_val.first, code);
  }
  for (auto& int_val : int_values_) {
    int_val.second.active = HasWord(kArgsPrefix + int_val.first, code);
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
