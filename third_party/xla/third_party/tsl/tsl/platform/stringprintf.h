/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Printf variants that place their output in a C++ string.
//
// Usage:
//      string result = strings::Printf("%d %s\n", 10, "hello");
//      strings::Appendf(&result, "%d %s\n", 20, "there");

#ifndef TENSORFLOW_TSL_PLATFORM_STRINGPRINTF_H_
#define TENSORFLOW_TSL_PLATFORM_STRINGPRINTF_H_

#include <stdarg.h>

#include <string>

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/strings/str_format.h"

namespace tsl {
namespace strings {

// Return a C++ string
ABSL_DEPRECATE_AND_INLINE()
inline std::string Printf(const absl::FormatSpec<>& format) {
  return absl::StrFormat(format);
}

template <typename Arg1>
ABSL_DEPRECATE_AND_INLINE()
std::string Printf(const absl::FormatSpec<Arg1>& format, Arg1 arg1) {
  return absl::StrFormat(format, arg1);
}

template <typename Arg1, typename Arg2>
ABSL_DEPRECATE_AND_INLINE()
std::string
    Printf(const absl::FormatSpec<Arg1, Arg2>& format, Arg1 arg1, Arg2 arg2) {
  return absl::StrFormat(format, arg1, arg2);
}

template <typename Arg1, typename Arg2, typename Arg3>
ABSL_DEPRECATE_AND_INLINE()
std::string Printf(const absl::FormatSpec<Arg1, Arg2, Arg3>& format, Arg1 arg1,
                   Arg2 arg2, Arg3 arg3) {
  return absl::StrFormat(format, arg1, arg2, arg3);
}

template <typename Arg1, typename Arg2, typename Arg3, typename Arg4>
ABSL_DEPRECATE_AND_INLINE()
std::string Printf(const absl::FormatSpec<Arg1, Arg2, Arg3, Arg4>& format,
                   Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4) {
  return absl::StrFormat(format, arg1, arg2, arg3, arg4);
}

template <typename Arg1, typename Arg2, typename Arg3, typename Arg4,
          typename Arg5>
ABSL_DEPRECATE_AND_INLINE()
std::string Printf(const absl::FormatSpec<Arg1, Arg2, Arg3, Arg4, Arg5>& format,
                   Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5) {
  return absl::StrFormat(format, arg1, arg2, arg3, arg4, arg5);
}

template <typename Arg1, typename Arg2, typename Arg3, typename Arg4,
          typename Arg5, typename... AV>
ABSL_DEPRECATED("Use absl::StrFormat instead.")
std::string
    Printf(const absl::FormatSpec<Arg1, Arg2, Arg3, Arg4, Arg5, AV...>& format,
           Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, AV... args) {
  return absl::StrFormat(format, arg1, arg2, arg3, arg4, arg5,
                         std::forward<AV>(args)...);
}

// Append result to a supplied string
ABSL_DEPRECATE_AND_INLINE()
inline void Appendf(std::string* dst, const absl::FormatSpec<>& format) {
  absl::StrAppendFormat(dst, format);
}

template <typename Arg1>
ABSL_DEPRECATE_AND_INLINE()
void Appendf(std::string* dst, const absl::FormatSpec<Arg1>& format,
             Arg1 arg1) {
  absl::StrAppendFormat(dst, format, arg1);
}

template <typename Arg1, typename Arg2>
ABSL_DEPRECATE_AND_INLINE()
void Appendf(std::string* dst, const absl::FormatSpec<Arg1, Arg2>& format,
             Arg1 arg1, Arg2 arg2) {
  absl::StrAppendFormat(dst, format, arg1, arg2);
}

template <typename Arg1, typename Arg2, typename Arg3>
ABSL_DEPRECATE_AND_INLINE()
void Appendf(std::string* dst, const absl::FormatSpec<Arg1, Arg2, Arg3>& format,
             Arg1 arg1, Arg2 arg2, Arg3 arg3) {
  absl::StrAppendFormat(dst, format, arg1, arg2, arg3);
}

template <typename Arg1, typename Arg2, typename Arg3, typename Arg4>
ABSL_DEPRECATE_AND_INLINE()
void Appendf(std::string* dst,
             const absl::FormatSpec<Arg1, Arg2, Arg3, Arg4>& format, Arg1 arg1,
             Arg2 arg2, Arg3 arg3, Arg4 arg4) {
  absl::StrAppendFormat(dst, format, arg1, arg2, arg3, arg4);
}

template <typename Arg1, typename Arg2, typename Arg3, typename Arg4,
          typename Arg5>
ABSL_DEPRECATE_AND_INLINE()
void Appendf(std::string* dst,
             const absl::FormatSpec<Arg1, Arg2, Arg3, Arg4, Arg5>& format,
             Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5) {
  absl::StrAppendFormat(dst, format, arg1, arg2, arg3, arg4, arg5);
}

template <typename Arg1, typename Arg2, typename Arg3, typename Arg4,
          typename Arg5, typename... AV>
ABSL_DEPRECATED("Use absl::StrAppendFormat instead.")
void Appendf(
    std::string* dst,
    const absl::FormatSpec<Arg1, Arg2, Arg3, Arg4, Arg5, AV...>& format,
    Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, AV... args) {
  absl::StrAppendFormat(dst, format, arg1, arg2, arg3, arg4, arg5,
                        std::forward<AV>(args)...);
}

}  // namespace strings
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_STRINGPRINTF_H_
