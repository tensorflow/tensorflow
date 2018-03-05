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
// This file is used to provide equivalents of internal util::format::FormatF
// and util::format::AppendF. Unfortunately, type safety is not as good as a
// a full C++ example.
// TODO(aselle): When absl adds support for StrFormat, use that instead.
#ifndef TENSORFLOW_CONTRIB_LITE_TOCO_FORMAT_PORT_H_
#define TENSORFLOW_CONTRIB_LITE_TOCO_FORMAT_PORT_H_

#include "tensorflow/contrib/lite/toco/toco_types.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace toco {
namespace port {

/// Identity (default case)
template <class T>
T IdentityOrConvertStringToRaw(T foo) {
  return foo;
}

// Overloaded case where we return std::string.
inline const char* IdentityOrConvertStringToRaw(const std::string& foo) {
  return foo.c_str();
}

#if defined(PLATFORM_GOOGLE) && defined(HAS_GLOBAL_STRING)
// Overloaded case where we return string.
inline const char* IdentityOrConvertStringToRaw(const string& foo) {
  return foo.c_str();
}
#endif  // PLATFORM_GOOGLE
// Delegate to TensorFlow Appendf function until absl has an equivalent.
template <typename... Args>
inline void AppendFHelper(string* destination, const char* fmt,
                          Args&&... args) {
  tensorflow::strings::Appendf(destination, fmt, args...);
}

// Specialization for no argument format string (avoid security bug).
inline void AppendFHelper(string* destination, const char* fmt) {
  tensorflow::strings::Appendf(destination, "%s", fmt);
}

// Append formatted string (with format fmt and args args) to the string
// pointed to by destination. fmt follows C printf semantics.
// One departure is that %s can be driven by a std::string or string.
template <typename... Args>
inline void AppendF(string* destination, const char* fmt, Args&&... args) {
  AppendFHelper(destination, fmt, IdentityOrConvertStringToRaw(args)...);
}

// Return formatted string (with format fmt and args args). fmt follows C printf
// semantics. One departure is that %s can be driven by a std::string or string.
template <typename... Args>
inline string StringF(const char* fmt, Args&&... args) {
  string result;
  AppendFHelper(&result, fmt, IdentityOrConvertStringToRaw(args)...);
  return result;
}

}  // namespace port
}  // namespace toco

#endif  // TENSORFLOW_CONTRIB_LITE_TOCO_FORMAT_PORT_H_
