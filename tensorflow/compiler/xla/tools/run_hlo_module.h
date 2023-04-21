/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_RUN_HLO_MODULE_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_RUN_HLO_MODULE_H_

#include <functional>
#include <random>
#include <string>

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/tools/run_hlo_module.pb.h"
#include "tensorflow/tsl/platform/status.h"

namespace xla {

// Command-line options to this tool.  See main() in run_hlo_module_main.cc for
// descriptions of these fields.
struct RunHloModuleOptions {
  RunHloModuleOptions()
      : platform(""),
        reference_platform("default"),
        print_literals(false),
        flatten_control_flow(false),
        run_test_hlo_passes(true),
        run_reference_hlo_passes(true),
        // Using small float range by default, as otherwise all reductions
        // miscompare vs. the interpreter with inf/nan.
        use_large_float_range(false),
        treat_gte_as_data_formatting(false),
        abs_error_bound(1e-3),
        rel_error_bound(1e-3),
        input_format(""),
        input_module(""),
        iterations(1),
        output_literals_file(""),
        input_literals_file(""),
        random_init_input_literals(true) {}
  std::string platform;
  std::string reference_platform;
  bool print_literals;
  bool flatten_control_flow;
  bool run_test_hlo_passes;
  bool run_reference_hlo_passes;
  bool use_large_float_range;
  bool treat_gte_as_data_formatting;
  float abs_error_bound;
  float rel_error_bound;
  std::string input_format;
  std::string input_module;
  int iterations;
  std::string output_literals_file;
  std::string input_literals_file;
  bool random_init_input_literals;
};

// Runs test_module on the platform with the name
// 'test_platform_name', and if 'reference_platform_name' is non-empty, it also
// runs it on the platform with the name 'reference_platform_name' and compares
// the results. 'reference_module_modifier_hook' can be used to transform the
// HloModule before it is run on the reference platform. This may be necessary
// to match the numerics of the test platform.
Status RunAndCompare(
    std::unique_ptr<HloModule> test_module, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, std::minstd_rand0* engine,
    const RunHloModuleOptions& options,
    xla::RunHloModuleIterationLiterals* iteration_literals_proto = nullptr,
    std::function<Status(const HloModule&, HloRunnerInterface*, HloModule*)>
        reference_module_modifier_hook = {},
    std::function<void(HloModuleConfig*)> config_modifier_hook = {});

// Same as above but reads a HloModule from 'hlo_filename'.
Status RunAndCompare(
    const std::string& hlo_filename, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, std::minstd_rand0* engine,
    const RunHloModuleOptions& options,
    xla::RunHloModuleIterationLiterals* iteration_literals_proto = nullptr,
    std::function<Status(const HloModule&, HloRunnerInterface*, HloModule*)>
        reference_module_modifier_hook = {},
    std::function<void(HloModuleConfig*)> config_modifier_hook = {});
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_RUN_HLO_MODULE_H_
