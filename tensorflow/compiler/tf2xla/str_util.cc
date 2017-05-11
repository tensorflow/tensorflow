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

#include "tensorflow/compiler/tf2xla/str_util.h"

#include <string>
#include <utility>
#include <vector>

namespace tensorflow {
namespace str_util {

static void ReplaceAll(string* text, StringPiece from, StringPiece to) {
  size_t pos = 0;
  while ((pos = text->find(from.data(), pos, from.size())) != string::npos) {
    text->replace(pos, from.size(), to.data(), to.size());
    pos += to.size();
    if (from.empty()) {
      pos++;  // Match at the beginning of the text and after every byte
    }
  }
}

void ReplaceAllPairs(string* text,
                     const std::vector<std::pair<string, string>>& replace) {
  for (const std::pair<string, string>& from_to : replace) {
    ReplaceAll(text, from_to.first, from_to.second);
  }
}

}  // namespace str_util
}  // namespace tensorflow
