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

#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

// Given input string and a delimiter returns back a substring including
// delimiters. If there was only starting delimiter found, returns single char.
absl::string_view FindInlineBlock(absl::string_view s, char delimiter) {
  size_t start = s.find(delimiter);
  if (start != absl::string_view::npos) {
    size_t end = s.find(delimiter, start + 1);
    if (end != std::string::npos) {
      return s.substr(start, end - start + 1);
    }
    // Special case to indicate that we didn't find the end.
    return s.substr(start, 1);
  }
  return s.substr(s.size(), 0);
}

// For the given 's' and its substring 'subs' returns new substring of 's' that
// begins past 'subs'.
absl::string_view PastSubstr(absl::string_view s, absl::string_view subs) {
  return s.substr(subs.data() + subs.size() - s.data());
}

}  // namespace

Status TextPreprocessor::Rewrite(const std::string& input,
                                 std::string* output) {
  absl::string_view s = input;
  std::string result;
  while (true) {
    absl::string_view inline_block = FindInlineBlock(s, inline_delimiter_);
    result.append(s.data(), inline_block.data() - s.data());
    if (inline_block.empty()) {
      break;
    }
    if (inline_block.size() == 1) {
      return NotFoundError("Unable to find end of inline block");
    }
    s = PastSubstr(s, inline_block);
    bool processed = false;
    for (auto& rewrite : inline_rewrites_) {
      if (processed) {
        break;
      }
      switch (rewrite->Rewrite(inline_block.substr(1, inline_block.size() - 2),
                               &result)) {
        case RewriteStatus::NOT_RECOGNIZED:
          // try another rewrite.
          break;
        case RewriteStatus::SUCCESS:
          processed = true;
          break;
        case RewriteStatus::ERROR:
          return InternalError(absl::StrCat("Error while rewriting '",
                                            inline_block, "': ", result));
      }
    }
    if (!processed) {
      if (!keep_unknown_rewrites_) {
        return NotFoundError(absl::StrCat("Didn't find inline rewrite for '",
                                          inline_block, "'"));
      }
      absl::StrAppend(&result, inline_block);
    }
  }
  *output = std::move(result);
  return OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
