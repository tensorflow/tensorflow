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

#ifndef TENSORFLOW_CORE_PLATFORM_LOGGING_H_
#define TENSORFLOW_CORE_PLATFORM_LOGGING_H_

#include "tensorflow/core/platform/types.h"
#include "tensorflow/tsl/platform/logging.h"

// NOLINTBEGIN(misc-unused-using-decls)
namespace tensorflow {
namespace internal {
using tsl::internal::LogString;
}  // namespace internal
using tsl::ERROR;
using tsl::FATAL;
using tsl::INFO;
using tsl::NUM_SEVERITIES;
using tsl::TFAddLogSink;
using tsl::TFGetLogSinks;
using tsl::TFLogEntry;
using tsl::TFLogSink;
using tsl::TFRemoveLogSink;
using tsl::UpdateLogVerbosityIfDefined;
using tsl::WARNING;
}  // namespace tensorflow
// NOLINTEND(misc-unused-using-decls)

#endif  // TENSORFLOW_CORE_PLATFORM_LOGGING_H_
