/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"

#include <mutex>  // NOLINT(build/c++11): only using std::call_once, not mutex.
#include <vector>
#include "tensorflow/compiler/xla/legacy_flags/parse_flags_from_env.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace legacy_flags {

struct DebugOptionsFlags {
  string xla_generate_hlo_graph;
  string xla_disable_hlo_passes;
  bool xla_enable_fast_math;
  int32 xla_backend_optimization_level;
  string xla_backend_extra_options;
};

namespace {

DebugOptionsFlags* flag_values;
std::vector<tensorflow::Flag>* flag_objects;
std::once_flag flags_init;

// Allocates flag_values and flag_objects; this function must not be called more
// than once - its call done via call_once.
void AllocateFlags() {
  flag_values = new DebugOptionsFlags;
  flag_values->xla_generate_hlo_graph = "";
  flag_values->xla_disable_hlo_passes = "";
  flag_values->xla_enable_fast_math = true;
  flag_values->xla_backend_optimization_level = 2;
  flag_values->xla_backend_extra_options = "";

  flag_objects = new std::vector<tensorflow::Flag>(
      {tensorflow::Flag(
           "xla_generate_hlo_graph", &flag_values->xla_generate_hlo_graph,
           "HLO modules matching this regex will be dumped to a .dot file "
           "throughout various stages in compilation."),

       tensorflow::Flag(
           "xla_enable_fast_math", &flag_values->xla_enable_fast_math,
           "Enable unsafe fast-math optimizations in the compiler; "
           "this may produce faster code at the expense of some accuracy."),
       tensorflow::Flag(
           "xla_backend_optimization_level",
           &flag_values->xla_backend_optimization_level,
           "Numerical optimization level for the XLA compiler backend."),

       tensorflow::Flag("xla_backend_extra_options",
                        &flag_values->xla_backend_extra_options,
                        "Extra options to pass to a backend; "
                        "comma-separated list of 'key=val' strings (=val "
                        "may be omitted); no whitespace around commas."),

       tensorflow::Flag(
           "xla_disable_hlo_passes", &flag_values->xla_disable_hlo_passes,
           "Comma-separated list of HLO passes to be disabled. These names "
           "must exactly match the passes' names; "
           "no whitespace around commas.")});
  ParseFlagsFromEnv(*flag_objects);
}

}  // namespace

void AppendDebugOptionsFlags(std::vector<tensorflow::Flag>* flag_list) {
  std::call_once(flags_init, &AllocateFlags);
  flag_list->insert(flag_list->end(), flag_objects->begin(),
                    flag_objects->end());
}

xla::DebugOptions GetDebugOptionsFromFlags() {
  std::call_once(flags_init, &AllocateFlags);

  DebugOptions options;
  options.set_xla_generate_hlo_graph(flag_values->xla_generate_hlo_graph);

  std::vector<string> disabled_passes =
      tensorflow::str_util::Split(flag_values->xla_disable_hlo_passes, ',');
  for (const auto& passname : disabled_passes) {
    options.add_xla_disable_hlo_passes(passname);
  }

  options.set_xla_enable_fast_math(flag_values->xla_enable_fast_math);
  options.set_xla_backend_optimization_level(
      flag_values->xla_backend_optimization_level);

  std::vector<string> extra_options_parts =
      tensorflow::str_util::Split(flag_values->xla_backend_extra_options, ',');
  auto* extra_options_map = options.mutable_xla_backend_extra_options();

  // The flag contains a comma-separated list of options; some options have
  // arguments following "=", some don't.
  for (const auto& part : extra_options_parts) {
    size_t eq_pos = part.find_first_of('=');
    if (eq_pos == string::npos) {
      (*extra_options_map)[part] = "";
    } else {
      string value = "";
      if (eq_pos + 1 < part.size()) {
        value = part.substr(eq_pos + 1);
      }
      (*extra_options_map)[part.substr(0, eq_pos)] = value;
    }
  }

  return options;
}

}  // namespace legacy_flags
}  // namespace xla
