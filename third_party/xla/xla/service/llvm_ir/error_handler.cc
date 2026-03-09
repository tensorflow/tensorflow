/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/llvm_ir/error_handler.h"

#include <utility>

#include "absl/base/call_once.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/ErrorHandling.h"
#include "tsl/platform/stacktrace.h"

namespace xla {

namespace {
using Handler = XlaScopedFatalErrorHandler::Handler;

thread_local Handler* thread_local_handler = nullptr;

void ErrorHandler(void* /*user_data*/, const char* reason, bool /*diag*/) {
  if (thread_local_handler && *thread_local_handler) {
    (*thread_local_handler)(reason);
  } else {
    LOG(ERROR) << "LLVM ERROR: " << reason;
  }
  // We crash here unconditionally.
  // If the handler was to return LLVM will crash anyway.
  // tsl::CurrentStackTrace() generates better stack traces than
  // LOG(FATAL) so we use QFATAL to suppress the redundant stack trace.
  LOG(QFATAL) << tsl::CurrentStackTrace();
}

// Registers the master handler with LLVM exactly once.
void EnsureInstalled() {
  static absl::once_flag once;
  absl::call_once(
      once, []() { llvm::install_fatal_error_handler(ErrorHandler, nullptr); });
}

// Sets the handler for the CURRENT thread only.
// Returns the previous handler for this thread.
Handler* SetThreadHandler(Handler* handler) {
  EnsureInstalled();
  std::swap(thread_local_handler, handler);
  return handler;
}
}  // namespace

XlaScopedFatalErrorHandler::XlaScopedFatalErrorHandler(Handler handler)
    : handler_(std::move(handler)), prev_(SetThreadHandler(&handler_)) {}

XlaScopedFatalErrorHandler::~XlaScopedFatalErrorHandler() {
  SetThreadHandler(prev_);
}

}  // namespace xla
