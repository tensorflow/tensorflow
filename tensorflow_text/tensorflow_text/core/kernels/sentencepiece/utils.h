// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

#ifndef TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_UTILS_H_
#define TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_UTILS_H_

#include <ostream>
#include <string>

namespace tensorflow {
namespace text {
namespace sentencepiece {

// AOSP and WASM doesn't support string_view,
// we put here a minimal re-implementation.
namespace utils {

class string_view {
 public:
  explicit string_view(const std::string& s)
      : str_(s.data()), len_(s.length()) {}
  string_view(const char* str, int len) : str_(str), len_(len) {}
  // A constructor from c string.
  explicit string_view(const char* s) : str_(s), len_(strlen(s)) {}

  int length() const { return len_; }
  const char* data() const { return str_; }
  bool empty() const { return len_ == 0; }
  unsigned char at(int i) const { return str_[i]; }

 private:
  const char* str_ = nullptr;
  const int len_ = 0;
};

inline std::ostream& operator<<(std::ostream& os, const string_view& sv) {
  os << std::string(sv.data(), sv.length());
  return os;
}
inline bool operator==(const string_view& view1, const string_view& view2) {
  if (view1.length() != view2.length()) {
    return false;
  }
  return memcmp(view1.data(), view2.data(), view1.length()) == 0;
}

}  // namespace utils
}  // namespace sentencepiece
}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_UTILS_H_
