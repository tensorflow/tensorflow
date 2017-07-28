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

#ifndef TENSORFLOW_JAVA_SRC_GEN_CC_GEN_UTIL_H_
#define TENSORFLOW_JAVA_SRC_GEN_CC_GEN_UTIL_H_

#include <string>

namespace tensorflow {
namespace gen_util {

inline std::string StripChar(const std::string& str, const char old_char) {
  std::string ret;
  for (const char& c : str) {
    if (c != old_char) {
      ret.push_back(c);
    }
  }
  return ret;
}

inline std::string ReplaceChar(const std::string& str, const char old_char,
    const char new_char) {
  std::string ret;
  for (const char& c : str) {
    ret.push_back(c == old_char ? new_char : c);
  }
  return ret;
}

}  // namespace gen_util
}  // namespace tensorflow

#endif  // TENSORFLOW_JAVA_SRC_GEN_CC_GEN_UTIL_H_
