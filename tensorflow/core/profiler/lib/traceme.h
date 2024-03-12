/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_LIB_TRACEME_H_
#define TENSORFLOW_CORE_PROFILER_LIB_TRACEME_H_

#include <new>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"  // IWYU pragma: export
#include "tsl/profiler/lib/traceme.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tsl/profiler/utils/time_utils.h"
#endif

namespace tensorflow {
namespace profiler {

using tsl::profiler::GetTFTraceMeLevel;   // NOLINT
using tsl::profiler::kCritical;           // NOLINT
using tsl::profiler::kInfo;               // NOLINT
using tsl::profiler::kVerbose;            // NOLINT
using tsl::profiler::TfOpDetailsEnabled;  // NOLINT
using tsl::profiler::TraceMe;             // NOLINT
using tsl::profiler::TraceMeLevel;        // NOLINT

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_TRACEME_H_
