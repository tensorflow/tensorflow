/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_RUNNER_FLAGS_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_RUNNER_FLAGS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "tensorflow/compiler/xla/tools/multihost_hlo_runner/functional_hlo_runner.h"

namespace xla {

class MultiHostHloRunnerFlags {
 public:
  MultiHostHloRunnerFlags() = default;
  virtual ~MultiHostHloRunnerFlags() = default;

  MultiHostHloRunnerFlags(const MultiHostHloRunnerFlags&) = delete;
  MultiHostHloRunnerFlags& operator=(const MultiHostHloRunnerFlags&) = delete;

  void AppendFlags(std::vector<tsl::Flag>* flags);

  bool CreateOptionsFromFlags(
      FunctionalHloRunner::PreprocessingOptions* preproc_options,
      FunctionalHloRunner::RawCompileOptions* raw_compile_options,
      FunctionalHloRunner::RunningOptions* running_options, std::string* error);

 private:
  struct FlagValues {
    int32_t num_replicas = -1;
    int32_t num_partitions = 1;
    bool log_output = false;
    bool run_xla_backend_only = false;
    bool disable_all_hlo_passes = false;
    bool use_spmd_partitioning = false;
    bool is_spmd_partitioned_module = false;
    std::string xla_dump_to = "";
    bool xla_dump_as_text = false;
    bool xla_dump_as_proto = false;
    std::string hlo_argument_mode = "use_random_inputs";
    int32_t while_execution_count = -1;
    int32_t num_repeats = 1;
    std::string execution_options_path = "";
  };

  void PreprocessFlags();

  FlagValues flag_values_;
  bool added_flags_ = false;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_RUNNER_FLAGS_H_
