/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_TOOLS_RUN_HLO_MODULE_H_
#define XLA_TOOLS_RUN_HLO_MODULE_H_

#include <functional>
#include <memory>
#include <random>
#include <string>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_runner.h"
#include "xla/status.h"
#include "xla/tools/run_hlo_module.pb.h"
#include "tsl/platform/status.h"

namespace xla {

class BufferAssignmentProto;

// Command-line options to this tool.  See main() in run_hlo_module_main.cc for
// descriptions of these fields.
struct RunHloModuleOptions {
  std::string platform;
  std::string reference_platform{"default"};
  bool print_literals{false};
  bool flatten_control_flow{false};
  bool run_test_hlo_passes{true};
  bool run_reference_hlo_passes{true};
  // Using small float range by default, as otherwise all reductions
  // miscompare vs. the interpreter with inf/nan.
  bool use_large_float_range{false};
  bool treat_gte_as_data_formatting{false};
  float abs_error_bound{1e-3};
  float rel_error_bound{1e-3};
  std::string input_format;
  std::string input_module;
  bool use_buffer_assignment_from_proto{false};
  // The format and the usage of the option is platform-dependent.
  std::string input_compilation_environments;
  int iterations{1};
  std::string output_literals_file;
  std::string input_literals_file;
  bool random_init_input_literals{true};
  bool force_fake_data{false};
  bool isolate_instructions{false};
};

// Runs test_module on the platform with the name
// 'test_platform_name', and if 'reference_platform_name' is non-empty, it also
// runs it on the platform with the name 'reference_platform_name' and compares
// the results. 'reference_module_modifier_hook' can be used to transform the
// HloModule before it is run on the reference platform. This may be necessary
// to match the numerics of the test platform.
Status RunAndCompare(
    std::unique_ptr<HloModule> test_module,
    const BufferAssignmentProto* buffer_assignment_proto,
    HloRunnerInterface* test_runner, HloRunnerInterface* reference_runner,
    std::minstd_rand0* engine, const RunHloModuleOptions& options,
    xla::RunHloModuleIterationLiterals* iteration_literals_proto = nullptr,
    std::function<Status(const HloModule&, HloRunnerInterface*, HloModule*)>
        reference_module_modifier_hook = {},
    std::function<void(HloModuleConfig*)> config_modifier_hook = {});

// Same as above but reads an HloModule from 'hlo_filename'. It also takes as
// an argument, a function 'compilation_env_modifier_hook' that potentially sets
// various fields in compilation environments, for an HLO module being loaded
// from the file.
Status RunAndCompare(
    const std::string& hlo_filename, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, std::minstd_rand0* engine,
    const RunHloModuleOptions& options,
    xla::RunHloModuleIterationLiterals* iteration_literals_proto = nullptr,
    std::function<Status(const HloModule&, HloRunnerInterface*, HloModule*)>
        reference_module_modifier_hook = {},
    std::function<void(HloModuleConfig*)> config_modifier_hook = {},
    std::function<Status(const RunHloModuleOptions& options, HloModule& module)>
        compilation_env_modifier_hook = {});
}  // namespace xla

#endif  // XLA_TOOLS_RUN_HLO_MODULE_H_
