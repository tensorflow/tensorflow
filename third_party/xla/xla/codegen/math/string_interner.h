/* Copyright 2025 The OpenXLA Authors.

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
#ifndef XLA_CODEGEN_MATH_STRING_INTERNER_H_
#define XLA_CODEGEN_MATH_STRING_INTERNER_H_

#include <string>
#include <unordered_set>

#include "absl/strings/string_view.h"

namespace xla::codegen::math {

// Interns strings in a thread-local string pool.
// Strings exist for the lifetime of the program.
// Many LLVM APIs require StringRef, which does not own the string data.
class StringInterner {
 public:
  static StringInterner& Get() {
    static thread_local StringInterner instance;
    return instance;
  }
  absl::string_view Intern(absl::string_view str) {
    auto [it, inserted] = pool_.insert(std::string(str));
    return *it;
  }

 private:
  StringInterner() = default;
  std::unordered_set<std::string> pool_;
};

}  // namespace xla::codegen::math

#endif  // XLA_CODEGEN_MATH_STRING_INTERNER_H_
