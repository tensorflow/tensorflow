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

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/stream_executor/platform.h"

namespace xla {

// Command-line options to this tool.  See main() in run_hlo_module_main.cc for
// descriptions of these fields.
struct RunHloModuleOptions {
  RunHloModuleOptions()
      : platform(""),
        reference_platform("default"),
        print_literals(false),
        run_test_hlo_passes(true),
        run_reference_hlo_passes(true),
        use_large_float_range(true),
        // TODO(b/68721786): These tolerances are set to match the values in the
        // isolation test. The goal is to lower these to 0.001.
        abs_error_bound(0.1),
        rel_error_bound(0.1),
        input_format("hlo"),
        input_module(""),
        iterations(1) {}
  std::string platform;
  std::string reference_platform;
  bool print_literals;
  bool run_test_hlo_passes;
  bool run_reference_hlo_passes;
  bool use_large_float_range;
  float abs_error_bound;
  float rel_error_bound;
  std::string input_format;
  std::string input_module;
  int iterations;
};

// Reads a HloModule from 'hlo_filename', runs it on the platform with the name
// 'test_platform_name', and if 'reference_platform_name' is non-empty, it also
// runs it on the platform with the name 'reference_platform_name' and compares
// the results. 'reference_module_modifier_hook' can be used to transform the
// HloModule before it is run on the reference platform. This may be necessary
// to match the numerics of the test platform.
Status RunAndCompare(
    const std::string& hlo_filename, const std::string& test_platform_name,
    const std::string& reference_platform_name, std::minstd_rand0* engine,
    const RunHloModuleOptions& options,
    std::function<Status(const HloModule&,
                         const ::stream_executor::Platform::Id&, HloModule*)>
        reference_module_modifier_hook = {},
    std::function<void(HloModuleConfig*)> config_modifier_hook = {});

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_RUN_HLO_MODULE_H_
