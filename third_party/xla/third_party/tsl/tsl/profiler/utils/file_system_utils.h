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

#ifndef TENSORFLOW_TSL_PROFILER_UTILS_FILE_SYSTEM_UTILS_H_
#define TENSORFLOW_TSL_PROFILER_UTILS_FILE_SYSTEM_UTILS_H_

#include <initializer_list>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tsl/platform/platform.h"

#ifdef PLATFORM_WINDOWS
const absl::string_view kPathSep = "\\";
#else
const absl::string_view kPathSep = "/";
#endif

namespace tsl {
namespace profiler {

inline std::string ProfilerJoinPathImpl(
    std::initializer_list<absl::string_view> paths) {
  std::string result;
  for (absl::string_view path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = std::string(path);
      continue;
    }

    path = absl::StripPrefix(path, kPathSep);
    if (absl::EndsWith(result, kPathSep)) {
      absl::StrAppend(&result, path);
    } else {
      absl::StrAppend(&result, kPathSep, path);
    }
  }

  return result;
}

// A local duplication of ::tensorflow::io::JoinPath that supports windows.
// TODO(b/150699701): revert to use ::tensorflow::io::JoinPath when fixed.
template <typename... T>
std::string ProfilerJoinPath(const T&... args) {
  return ProfilerJoinPathImpl({args...});
}

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_UTILS_FILE_SYSTEM_UTILS_H_
