/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_RUNNER_RUNNER_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_RUNNER_RUNNER_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/runtime/jit_executable.h"
#include "tensorflow/tsl/util/command_line_flags.h"

namespace xla {
namespace runtime {

struct RunnerFlags {
  std::string function;
  std::string module_path;
  std::string arguments_path;
  std::string results_path;
};

void AppendRunnerFlags(std::vector<tsl::Flag>* flag_list, RunnerFlags* flags);

// Fake AsyncTaskRunner for programs that do not plan to execute any async work.
AsyncTaskRunner* NoAsyncTaskRunner();

// Compiles and executes the MLIR input program defined by `flags` using
// user-provided compilation and execution options.
absl::Status Execute(RunnerFlags flags,
                     const JitExecutable::Options& compile_opts,
                     const Executable::ExecuteOpts& execute_opts);

// A wrapper around `Execute` that does argument parsing and binary
// initialization. Can be used as a main function in user-defined tools.
int Main(int argc, char** argv, const JitExecutable::Options& compile_opts,
         const Executable::ExecuteOpts& execute_opts);

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_RUNNER_RUNNER_H_
