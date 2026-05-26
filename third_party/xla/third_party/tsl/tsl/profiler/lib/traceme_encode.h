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
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/config.h"  // IWYU pragma: keep
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/macros.h"
#include "tsl/platform/platform.h"  // IWYU pragma: keep
#if !defined(IS_MOBILE_PLATFORM)
#include "xla/tsl/profiler/utils/traceme_global_flags.h"
#endif

#define TRACEME_ENCODE_STRINGIFY(x) #x
#define TRACEME_ENCODE_TOSTRING(x) TRACEME_ENCODE_STRINGIFY(x)
#define TRACEME_FILE_AND_LINE __FILE__ ":" TRACEME_ENCODE_TOSTRING(__LINE__)

#if !defined(LIBTPU_ON_GCE) && ABSL_HAVE_BUILTIN(__builtin_FILE)
// TODO(b/507077868): Switch to absl::SourceLocation after XLA upgrades to the
// next absl version. For more details, see
// https://gist.github.com/youchunni/24ee88f9daa9566312f055d71513dbea
#define TRACEME_DEFAULT_FILE __builtin_FILE()
#else
#define TRACEME_DEFAULT_FILE ""
#endif

namespace tsl {
namespace profiler {

// An argument passed to TraceMeEncode.
struct TraceMeArg {
  // String conversions of value types are supported via AlphaNum. We keep a
  // reference to the AlphaNum's internal buffer here, so it must remain valid
  // for the lifetime of this object. We cannot store it by value because it is
  // not safe to construct an AlphaNum as a member of a class, particularly when
  // AbslStringify is being used (it may reference default arguments that are on
  // the caller's stack, if we constructed it here those default arguments would
  // be destroyed before they are used).
  TraceMeArg(absl::string_view k,
             const absl::AlphaNum& v ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : key(k), value(v.Piece()) {}

  TraceMeArg(const TraceMeArg&) = delete;
  void operator=(const TraceMeArg&) = delete;

  absl::string_view key;
  absl::string_view value;
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

// Appends arguments encoded as TraceMe metadata to `name`.
//
// The resulting string format is:
//   name#key1=value1,key2=value2,_src=file.cc#
//
// Performance implementation details:
// To minimize overhead on the critical path, this function pre-calculates the
// exact string length for a single resize, then appends the arguments using
// raw buffer manipulation pointed by `out`.
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string AppendArgs(
    std::string name, std::initializer_list<TraceMeArg> args,
    const char* source_loc_arg) {
  absl::string_view source_loc(source_loc_arg);
#if !defined(IS_MOBILE_PLATFORM)
  if (TF_PREDICT_FALSE(!TraceMeGlobalFlags::IsSourceLocationEnabled())) {
    source_loc = "";
  }
#else
  source_loc = "";
#endif
  if (args.size() == 0 && source_loc.empty()) {
    return name;
  }
  const auto old_size = name.size();
  // `args.size() * 2`: Accounts for '=' and ',' added for each arg.
  // `+ 2`: Accounts for the initial '#' and the final '#'.
  auto new_size = old_size + args.size() * 2 + 2;
  for (const auto& arg : args) {
    new_size += arg.key.size() + arg.value.size();
  }

  if (!source_loc.empty()) {
    new_size += source_loc.size() + 5;  // `+ 5`: Accounts for '_src='
  } else {
    new_size -= 1;  // Minus one ',' between the last arg and '_src='.
  }
  name.resize(new_size);
  char* const begin = &name[0];
  char* out = begin + old_size;
  *out++ = '#';
  for (const auto& arg : args) {
    out = Append(out, arg.key);
    *out++ = '=';
    out = Append(out, arg.value);
    *out++ = ',';
  }
  if (!source_loc.empty()) {
    out = Append(out, "_src=");
    out = Append(out, source_loc);
    *out++ = '#';
  } else {
    *(out - 1) = '#';  // Replace the last ',' with '#'.
  }
  DCHECK_EQ(out, begin + new_size);
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
    std::string name, std::initializer_list<TraceMeArg> args,
    const char* source_loc = TRACEME_DEFAULT_FILE) {
  return traceme_internal::AppendArgs(std::move(name), args, source_loc);
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    absl::string_view name, std::initializer_list<TraceMeArg> args,
    const char* source_loc = TRACEME_DEFAULT_FILE) {
  return traceme_internal::AppendArgs(std::string(name), args, source_loc);
}
TF_ATTRIBUTE_ALWAYS_INLINE inline std::string TraceMeEncode(
    const char* name, std::initializer_list<TraceMeArg> args,
    const char* source_loc = TRACEME_DEFAULT_FILE) {
  return traceme_internal::AppendArgs(name, args, source_loc);
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
    std::initializer_list<TraceMeArg> args,
    const char* source_loc = TRACEME_DEFAULT_FILE) {
  return traceme_internal::AppendArgs(std::string(), args, source_loc);
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
