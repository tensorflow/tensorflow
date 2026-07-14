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

#ifndef TENSORFLOW_CORE_PLATFORM_STACKTRACE_HANDLER_H_
#define TENSORFLOW_CORE_PLATFORM_STACKTRACE_HANDLER_H_

#include "tsl/platform/stacktrace_handler.h"

namespace tensorflow {
namespace testing {

// Installs signal handlers to print out stack trace.
// Although GoogleTest has support for generating stacktraces with abseil via
// https://github.com/google/googletest/pull/1653, this doesn't cover our use
// case of getting C++ stacktraces in our python tests.
using tsl::testing::InstallStacktraceHandler;

}  // namespace testing
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STACKTRACE_HANDLER_H_
