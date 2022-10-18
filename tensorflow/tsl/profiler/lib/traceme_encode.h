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
#ifndef TENSORFLOW_TSL_PROFILER_LIB_TRACEME_ENCODE_H_
#define TENSORFLOW_TSL_PROFILER_LIB_TRACEME_ENCODE_H_

#include <string.h>

#include <initializer_list>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/macros.h"

namespace tsl {
namespace profiler {

// An argument passed to TraceMeEncode.
struct TraceMeArg {
  // This constructor is required because absl::AlphaNum is non-copyable.
  template <typename Value>
  TraceMeArg(absl::string_view k, Value v) : key(k), value(v) {}

  TF_DISALLOW_COPY_AND_ASSIGN(TraceMeArg);

  absl::string_view key;
  absl::AlphaNum value;
};

namespace traceme_internal {

// Copies the contents of str to the address pointed by out.
// Returns the address after the copy.
// REQUIRED: The address range [out, out + str.size()] must have been allocated.
TF_ATTRIBUTE_ALWAYS_INLINE inline char* Append(char* out,
                                               absl::string_view str) {
  DCHECK(!absl::StrContains(str, '#'))
      << "'#' is not a valid character in TraceMeEncode";
  const size_t str_size = str.size();
  if (TF_PREDICT_TRUE(str_size > 0)) {
    memcpy(out, str.data(), str_size);
    out += str_size;
  }
  return out;
}

// Appends args encoded as TraceMe metadata to name.
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string AppendArgs(
    std::string name, std::initializer_list<TraceMeArg> args) {
  if (TF_PREDICT_TRUE(args.size() > 0)) {
    const auto old_size = name.size();
    auto new_size = old_size + args.size() * 2 + 1;
    for (const auto& arg : args) {
      new_size += arg.key.size() + arg.value.size();
    }
    name.resize(new_size);
    char* const begin = &name[0];
    char* out = begin + old_size;
    *out++ = '#';
    for (const auto& arg : args) {
      out = Append(out, arg.key);
      *out++ = '=';
      out = Append(out, arg.value.Piece());
      *out++ = ',';
    }
    *(out - 1) = '#';
    DCHECK_EQ(out, begin + new_size);
  }
  return name;
}

// Appends new_metadata to the metadata part of name.
TF_ATTRIBUTE_ALWAYS_INLINE inline void AppendMetadata(
    std::string* name, absl::string_view new_metadata) {
  if (!TF_PREDICT_FALSE(new_metadata.empty())) {
    if (!name->empty() && name->back() == '#') {  // name already has metadata
      name->back() = ',';
      if (TF_PREDICT_TRUE(new_metadata.front() == '#')) {
        new_metadata.remove_prefix(1);
      }
    }
    name->append(new_metadata.data(), new_metadata.size());
  }
}

}  // namespace traceme_internal

// Encodes an event name and arguments into TraceMe metadata.
// Use within a lambda to avoid expensive operations when tracing is disabled.
// Example Usage:
//   TraceMe trace_me([value1]() {
//     return TraceMeEncode("my_trace", {{"key1", value1}, {"key2", 42}});
//   });
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    std::string name, std::initializer_list<TraceMeArg> args) {
  return traceme_internal::AppendArgs(std::move(name), args);
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    absl::string_view name, std::initializer_list<TraceMeArg> args) {
  return traceme_internal::AppendArgs(std::string(name), args);
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    const char* name, std::initializer_list<TraceMeArg> args) {
  return traceme_internal::AppendArgs(std::string(name), args);
}

// Encodes arguments into TraceMe metadata.
// Use within a lambda to avoid expensive operations when tracing is disabled.
// Example Usage:
//   TraceMe trace_me("my_trace");
//   ...
//   trace_me.AppendMetadata([value1]() {
//     return TraceMeEncode({{"key1", value1}, {"key2", 42}});
//   });
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    std::initializer_list<TraceMeArg> args) {
  return traceme_internal::AppendArgs(std::string(), args);
}

// Concatenates op_name and op_type.
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOp(
    absl::string_view op_name, absl::string_view op_type) {
  return absl::StrCat(op_name, ":", op_type);
}

TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOp(const char* op_name,
                                                        const char* op_type) {
  return absl::StrCat(op_name, ":", op_type);
}

TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOp(
    std::string&& op_name, absl::string_view op_type) {
  absl::StrAppend(&op_name, ":", op_type);
  return op_name;
}

// Concatenates op_name and op_type.
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOpOverride(
    absl::string_view op_name, absl::string_view op_type) {
  return absl::StrCat("#tf_op=", op_name, ":", op_type, "#");
}

TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeOpOverride(
    const char* op_name, const char* op_type) {
  return absl::StrCat("#tf_op=", op_name, ":", op_type, "#");
}

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_TRACEME_ENCODE_H_
