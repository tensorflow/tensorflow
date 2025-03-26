/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LLVM_IR_LLVM_COMMAND_LINE_OPTIONS_H_
#define XLA_SERVICE_LLVM_IR_LLVM_COMMAND_LINE_OPTIONS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/strings/match.h"
#include "absl/synchronization/mutex.h"

namespace xla {
namespace llvm_ir {

// Given a map with options (e.g. originating from xla_backend_extra_options())
// pass those that don't start with xla_ to LLVM.
template <typename T>
std::vector<std::string> ExtractXlaBackendExtraOptions(const T& options) {
  if (!options.empty()) {
    std::vector<std::string> backend_extra_opts;
    for (const auto& it : options) {
      // Skip options the XLA backend itself consumes.
      if (!absl::StartsWith(it.first, "xla_")) {
        if (it.second.empty()) {
          backend_extra_opts.push_back(it.first);
        } else {
          backend_extra_opts.push_back(it.first + "=" + it.second);
        }
      }
    }
    // We non-deterministically iterated over the proto.
    // Sort the options to make their ordering deterministic.
    absl::c_sort(backend_extra_opts);
    return backend_extra_opts;
  }
  return {};
}

// Different XLA clients have different LLVM usage. This is not safe, as each
// client is responsible for setting LLVM's own global state.
//
// Each client before starting compilation, is required to *acquire* the state
// of LLVM, given a list of options. While the lock is acquired, it is
// guaranteed that LLVM will be initialized with this set of options.
//
// Multiple clients are allowed concurrent compilation, as long as their set
// of LLVM options is identical.
class ABSL_SCOPED_LOCKABLE LLVMCommandLineOptionsLock {
 public:
  explicit LLVMCommandLineOptionsLock(
      const std::vector<std::string>& client_options)
      ABSL_EXCLUSIVE_LOCK_FUNCTION(lock_);

  LLVMCommandLineOptionsLock(const LLVMCommandLineOptionsLock&) = delete;
  LLVMCommandLineOptionsLock(LLVMCommandLineOptionsLock&&) = delete;
  LLVMCommandLineOptionsLock& operator=(const LLVMCommandLineOptionsLock&) =
      delete;
  LLVMCommandLineOptionsLock& operator=(LLVMCommandLineOptionsLock&&) = delete;

  ~LLVMCommandLineOptionsLock() ABSL_UNLOCK_FUNCTION();

  static std::vector<std::string>& GetGlobalOptions() {
    // absl::NoDestructor is not available in OSS XLA.
    static std::vector<std::string>* global_options =
        new std::vector<std::string>();
    return *global_options;
  }

 private:
  // Global XLA LLVM options lock.
  static inline absl::Mutex lock_{absl::kConstInit};

  // Number of clients currently using LLVM.
  static inline int32_t num_active_clients_{0};

  // Signature of client options LLVM is currently initialized with.
  static inline uint64_t active_client_signature_{0};

  static std::vector<std::string>& GetActiveClientOptions() {
    // absl::NoDestructor is not available in OSS XLA.
    static std::vector<std::string>* active_client_options =
        new std::vector<std::string>();
    return *active_client_options;
  }

  // Signature of the current clients LLVM options.
  uint64_t client_signature_;
};

}  // namespace llvm_ir
}  // namespace xla

#endif  // XLA_SERVICE_LLVM_IR_LLVM_COMMAND_LINE_OPTIONS_H_
