/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/scanner.h"

namespace tsl {
namespace strings {

void Scanner::ScanUntilImpl(char end_ch, bool escaped) {
  for (;;) {
    if (cur_.empty()) {
      Error();
      return;
    }
    const char ch = cur_[0];
    if (ch == end_ch) {
      return;
    }

    cur_.remove_prefix(1);
    if (escaped && ch == '\\') {
      // Escape character, skip next character.
      if (cur_.empty()) {
        Error();
        return;
      }
      cur_.remove_prefix(1);
    }
  }
}

bool Scanner::GetResult(absl::string_view* remaining,
                        absl::string_view* capture) {
  if (error_) {
    return false;
  }
  if (remaining != nullptr) {
    *remaining = cur_;
  }
  if (capture != nullptr) {
    const char* end = capture_end_ == nullptr ? cur_.data() : capture_end_;
    *capture = absl::string_view(capture_start_, end - capture_start_);
  }
  return true;
}

}  // namespace strings
}  // namespace tsl
