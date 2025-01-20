/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/llvm_ir/llvm_command_line_options.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/Support/CommandLine.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace llvm_ir {

LLVMCommandLineOptionsLock::LLVMCommandLineOptionsLock(
    const std::vector<std::string>& client_options)
    : client_signature_(absl::HashOf(client_options)) {
  // Wait until other clients are done using LLVM.
  auto no_competing_clients =
      [this, &client_options]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
        return num_active_clients_ == 0 ||
               (client_signature_ == active_client_signature_ &&
                absl::c_equal(client_options, GetActiveClientOptions()));
      };
  lock_.LockWhen(absl::Condition(&no_competing_clients));

  // Check if previous client used a different set of LLVM options,
  // (re)initialize if that's the case.
  if (client_signature_ != active_client_signature_) {
    LOG(INFO) << "XLA (re)initializing LLVM with options fingerprint: "
              << client_signature_;
    VLOG(1) << "XLA LLVM options:";
    CHECK_EQ(num_active_clients_, 0);

    int32_t idx = 1;
    std::vector<const char*> fake_argv(client_options.size() +
                                       GetGlobalOptions().size() + 1);
    fake_argv[0] = "xla";
    for (absl::string_view client_option : client_options) {
      VLOG(1) << absl::StrFormat("XLA LLVM arg[%d]: %s", idx, client_option);
      fake_argv[idx] = client_option.data();
      ++idx;
    }
    for (absl::string_view global_option : GetGlobalOptions()) {
      VLOG(1) << absl::StrFormat("XLA LLVM arg[%d]: %s", idx, global_option);
      fake_argv[idx] = global_option.data();
      ++idx;
    }

    llvm::cl::ResetAllOptionOccurrences();
    llvm::cl::ParseCommandLineOptions(fake_argv.size(), fake_argv.data());

    active_client_signature_ = client_signature_;
    GetActiveClientOptions() = client_options;
  } else {
    VLOG(1) << "XLA skipping reinitializing LLVM with options signature: "
            << client_signature_;
  }

  // We're good to start compilation.
  num_active_clients_ += 1;
  lock_.Unlock();
}

LLVMCommandLineOptionsLock::~LLVMCommandLineOptionsLock() {
  absl::MutexLock lock(&lock_);
  CHECK_GT(num_active_clients_, 0);
  num_active_clients_ -= 1;
}

}  // namespace llvm_ir
}  // namespace xla
