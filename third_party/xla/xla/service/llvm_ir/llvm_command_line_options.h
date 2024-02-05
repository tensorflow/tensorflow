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

#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "llvm/Support/CommandLine.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace llvm_ir {

// Given a map with options (e.g. originating from xla_backend_extra_options())
// pass those that don't start with xla_ to LLVM.
template <typename T>
void InitializeLLVMCommandLineOptions(const T& options) {
  if (!options.empty()) {
    std::vector<std::string> fake_argv_storage;
    fake_argv_storage.push_back("");
    for (const auto& it : options) {
      // Skip options the XLA backend itself consumes.
      if (!absl::StartsWith(it.first, "xla_")) {
        if (it.second.empty()) {
          fake_argv_storage.push_back(it.first);
        } else {
          fake_argv_storage.push_back(it.first + "=" + it.second);
        }
      }
    }

    VLOG(2) << "Passing argv to LLVM:";
    std::vector<const char*> fake_argv;
    for (const auto& s : fake_argv_storage) {
      fake_argv.push_back(s.c_str());
      VLOG(2) << s;
    }
    llvm::cl::ParseCommandLineOptions(static_cast<int>(fake_argv.size()),
                                      &fake_argv[0]);
  }
}

}  // namespace llvm_ir
}  // namespace xla

#endif  // XLA_SERVICE_LLVM_IR_LLVM_COMMAND_LINE_OPTIONS_H_
