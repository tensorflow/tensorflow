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
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/CodeGen/MachinePassRegistry.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "xla/tsl/platform/logging.h"

namespace xla {
namespace llvm_ir {

namespace {

// Sentinel option registered in LLVM's global command-line registry so XLA can
// detect when an external library in the same process (e.g. Triton's
// ScopedLLVMOptions or LLD) resets LLVM's global options via
// llvm::cl::ResetAllOptionOccurrences(). Must only be accessed while holding
// LLVMCommandLineOptionsLock::lock_.
llvm::cl::opt<bool>& GetXlaOptionsInitializedSentinel() {
  // absl::NoDestructor is not available in OSS XLA.
  static llvm::cl::opt<bool>* sentinel =
      new llvm::cl::opt<bool>("xla-llvm-options-initialized",
                              llvm::cl::init(false), llvm::cl::ReallyHidden);
  return *sentinel;
}

// LLVM caches the value of the `-regalloc` option in RegisterRegAlloc's
// registry the first time any codegen pipeline is built in the process (see
// TargetPassConfig::getOptimizeRegAlloc), and never refreshes it when the
// command line is re-parsed. Other parts of TargetPassConfig read the live
// option, so a client that passes a different `-regalloc` than the first one
// sees inconsistent state (e.g. "Must use fast (default) register allocator
// for unoptimized regalloc."). Re-sync the cached default with the option that
// was just parsed.
void SyncRegisterAllocatorDefault() {
  using RegAllocOption =
      llvm::cl::opt<llvm::RegisterRegAlloc::FunctionPassCtor, false,
                    llvm::RegisterPassParser<llvm::RegisterRegAlloc>>;
  auto& options = llvm::cl::getRegisteredOptions();
  auto it = options.find("regalloc");
  if (it == options.end()) {
    return;  // TargetPassConfig isn't linked in; nothing to sync.
  }
  llvm::RegisterRegAlloc::setDefault(
      static_cast<RegAllocOption*>(it->second)->getValue());
}

}  // namespace

LLVMCommandLineOptionsLock::LLVMCommandLineOptionsLock(
    const std::vector<std::string>& client_options)
    : client_signature_(absl::HashOf(client_options)) {
  // Wait until other clients are done using LLVM.
  auto no_competing_clients =
      [this, &client_options]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_) {
        return num_active_clients_ == 0 ||
               (GetXlaOptionsInitializedSentinel() &&
                client_signature_ == active_client_signature_ &&
                absl::c_equal(client_options, GetActiveClientOptions()));
      };
  lock_.LockWhen(absl::Condition(&no_competing_clients));

  // Check if previous client used a different set of LLVM options, or if an
  // external caller reset LLVM's global options since XLA last initialized
  // them.
  if (!GetXlaOptionsInitializedSentinel() ||
      client_signature_ != active_client_signature_ ||
      !absl::c_equal(client_options, GetActiveClientOptions())) {
    VLOG(1) << "XLA (re)initializing LLVM with options fingerprint: "
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

    // Parse the per-context library options into a context that is copied
    // into each LLVMContext by ApplyLibraryOptions, then parse the options
    // globally.
    delete GetActiveLibraryOptions();
    GetActiveLibraryOptions() = new llvm::cl::LibraryOptionsContext();
    llvm::cl::ResetAllOptionOccurrences();
    llvm::cl::parseCommandLineOptions(*GetActiveLibraryOptions(), fake_argv);
    llvm::cl::ResetAllOptionOccurrences();
    llvm::cl::ParseCommandLineOptions(fake_argv.size(), fake_argv.data());
    GetXlaOptionsInitializedSentinel().setValue(true);
    SyncRegisterAllocatorDefault();

    active_client_signature_ = client_signature_;
    GetActiveClientOptions() = client_options;
  } else {
    VLOG(1) << "XLA skipping reinitializing LLVM with options signature: "
            << client_signature_;
  }

  // We're good to start compilation.
  num_active_clients_ += 1;
  lock_.unlock();
}

void LLVMCommandLineOptionsLock::ApplyLibraryOptions(
    llvm::LLVMContext& context) {
  absl::MutexLock lock(lock_);
  context.getLibraryOptions().assign(*GetActiveLibraryOptions());
}

LLVMCommandLineOptionsLock::~LLVMCommandLineOptionsLock() {
  absl::MutexLock lock(lock_);
  CHECK_GT(num_active_clients_, 0);
  num_active_clients_ -= 1;
}

}  // namespace llvm_ir
}  // namespace xla
