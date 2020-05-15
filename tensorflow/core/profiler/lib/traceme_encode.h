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
#ifndef TENSORFLOW_CORE_PROFILER_LIB_TRACEME_ENCODE_H_
#define TENSORFLOW_CORE_PROFILER_LIB_TRACEME_ENCODE_H_

#include <string.h>

#include <initializer_list>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace profiler {
namespace internal {

// Copies the contents of str to the address pointed by out.
// Returns the address after the copy.
// REQUIRED: The address range [out, out + str.size()] must have been allocated.
TF_ATTRIBUTE_ALWAYS_INLINE inline char* Append(char* out,
                                               absl::string_view str) {
  const size_t str_size = str.size();
  if (str_size > 0) {
    memcpy(out, str.data(), str_size);
    out += str_size;
  }
  return out;
}

}  // namespace internal

// Encodes an event name and arguments into a string stored by TraceMe.
// Use within a lambda to avoid expensive operations when tracing is inactive.
// Example Usage:
//   TraceMe trace_me([&name, value1]() {
//     return TraceMeEncode(name, {{"key1", value1}, {"key2", 42}});
//   });
inline std::string TraceMeEncode(
    std::string name,
    std::initializer_list<std::pair<absl::string_view, absl::AlphaNum>> args) {
  if (TF_PREDICT_TRUE(args.size() > 0)) {
    const auto old_size = name.size();
    auto new_size = old_size + args.size() * 2 + 1;
    for (const auto& arg : args) {
      new_size += arg.first.size() + arg.second.size();
    }
    name.resize(new_size);
    char* const begin = &name[0];
    char* out = begin + old_size;
    *out++ = '#';
    for (const auto& arg : args) {
      out = internal::Append(out, arg.first);
      *out++ = '=';
      out = internal::Append(out, arg.second.Piece());
      *out++ = ',';
    }
    *(out - 1) = '#';
    DCHECK_EQ(out, begin + new_size);
  }
  return name;
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_TRACEME_ENCODE_H_
