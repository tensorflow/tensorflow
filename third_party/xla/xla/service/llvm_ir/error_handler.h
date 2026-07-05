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

#ifndef XLA_SERVICE_LLVM_IR_ERROR_HANDLER_H_
#define XLA_SERVICE_LLVM_IR_ERROR_HANDLER_H_

#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"

namespace xla {

// RAII Guard to install an LLVM handler that should be used in place of
// llvm::ScopedFatalErrorHandler.
//
// llvm::ScopedFatalErrorHandler uses a static thread handler and consequently
// trying to register a handler on multiple threads causes a race condition and
// a crash. This class uses thread local storage to avoid this issue.
//
// The class provides per-thread isolation for the handler. Multiple threads
// can safely install handlers without interfering with each other.
//
// Usage:
//   XlaScopedFatalErrorHandler handler([](absl::string_view reason) {
//     LOG(ERROR) << "Something went wrong:" << reason;
//   });
//
// If no handler is provided, a default handler is installed that logs the
// reason and stack trace. If a handler is provided, it is called with the
// reason and the reason is *not* logged anymore.
//
// The registered lambda itself should be thread-safe as in not accessing any
// shared state between threads without proper synchronization.
class [[nodiscard]] XlaScopedFatalErrorHandler final {
 public:
  using Handler = absl::AnyInvocable<void(absl::string_view reason)>;

  // Installs the given handler for the current thread.
  // The previous handler for this thread is kept and restored on destruction.
  explicit XlaScopedFatalErrorHandler(Handler handler = nullptr);
  // Restores the previous handler for this thread.
  ~XlaScopedFatalErrorHandler();

  XlaScopedFatalErrorHandler(const XlaScopedFatalErrorHandler&) = delete;
  XlaScopedFatalErrorHandler(XlaScopedFatalErrorHandler&&) = delete;
  XlaScopedFatalErrorHandler& operator=(const XlaScopedFatalErrorHandler&) =
      delete;
  XlaScopedFatalErrorHandler& operator=(XlaScopedFatalErrorHandler&&) = delete;

 private:
  // Handler the calling thread.
  Handler handler_;
  // Previous handler for this thread.
  Handler* prev_;
};

}  // namespace xla

#endif  // XLA_SERVICE_LLVM_IR_ERROR_HANDLER_H_
