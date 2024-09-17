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

#include "absl/base/macros.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tsl/profiler/lib/traceme_encode.h"

// TODO: b/323943471 - This macro should eventually be provided by Abseil.
#ifndef ABSL_DEPRECATE_AND_INLINE
#define ABSL_DEPRECATE_AND_INLINE()
#endif

namespace tensorflow {
namespace profiler {

using TraceMeArg ABSL_DEPRECATE_AND_INLINE() =
    tsl::profiler::TraceMeArg;  // NOLINT

ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeEncode(
    std::string name, std::initializer_list<tsl::profiler::TraceMeArg> args) {
  return tsl::profiler::TraceMeEncode(std::move(name), args);
}

ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeEncode(
    absl::string_view name,
    std::initializer_list<tsl::profiler::TraceMeArg> args) {
  return tsl::profiler::TraceMeEncode(name, args);
}

ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeEncode(
    const char* name, std::initializer_list<tsl::profiler::TraceMeArg> args) {
  return tsl::profiler::TraceMeEncode(name, args);
}

ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeEncode(
    std::initializer_list<tsl::profiler::TraceMeArg> args) {
  return tsl::profiler::TraceMeEncode(args);
}

ABSL_DEPRECATE_AND_INLINE()
// Concatenates op_name and op_type.
inline std::string TraceMeOp(absl::string_view op_name,
                             absl::string_view op_type) {
  return tsl::profiler::TraceMeOp(op_name, op_type);
}

ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeOp(const char* op_name, const char* op_type) {
  return tsl::profiler::TraceMeOp(op_name, op_type);
}

ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeOp(std::string&& op_name, absl::string_view op_type) {
  return tsl::profiler::TraceMeOp(op_name, op_type);
}

ABSL_DEPRECATE_AND_INLINE()
// Concatenates op_name and op_type.
inline std::string TraceMeOpOverride(absl::string_view op_name,
                                     absl::string_view op_type) {
  return tsl::profiler::TraceMeOpOverride(op_name, op_type);
}

ABSL_DEPRECATE_AND_INLINE()
inline std::string TraceMeOpOverride(const char* op_name, const char* op_type) {
  return tsl::profiler::TraceMeOpOverride(op_name, op_type);
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_TRACEME_ENCODE_H_
