/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file uses absl libraries to parses command-line options and runs a
// given mlir file using test driver library.
//
// This is allowed to link against Tensorflow libraries.

#include "absl/strings/str_split.h"
#include "llvm/Support/Error.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/runtime_fallback/bef_executor_flags.h"
#include "tensorflow/core/runtime_fallback/util/fallback_test_util.h"
#include "tfrt/bef_executor_driver/bef_executor_driver.h"  // from @tf_runtime

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  std::string input_filename = absl::GetFlag(FLAGS_input_filename);

  // Ignore the positional argument if --input_filename is provided.
  if (argc > 1 && input_filename == tfrt::kDefaultInputFilename) {
    input_filename = argv[1];
  }

  // By default, absl::StrSplit includes empty string in its output.
  std::vector<std::string> shared_libs =
      absl::StrSplit(absl::GetFlag(FLAGS_shared_libs), ',', absl::SkipEmpty());
  std::vector<std::string> functions =
      absl::StrSplit(absl::GetFlag(FLAGS_functions), ',', absl::SkipEmpty());
  std::string test_init_function = absl::GetFlag(FLAGS_test_init_function);

  tfrt::RunBefConfig run_config;
  run_config.program_name = argv[0];
  run_config.input_filename = input_filename;
  run_config.shared_libs = shared_libs;
  run_config.functions = functions;
  run_config.test_init_function = test_init_function;
  run_config.work_queue_type = absl::GetFlag(FLAGS_work_queue_type);
  run_config.host_allocator_type = absl::GetFlag(FLAGS_host_allocator_type);

  return RunBefExecutor(
      run_config,
      [](tfrt::HostContext* host, tfrt::ResourceContext* resource_context)
          -> llvm::Expected<tfrt::ExecutionContext> {
        return tensorflow::tfd::CreateFallbackTestExecutionContext(
            host, resource_context);
      });
}
