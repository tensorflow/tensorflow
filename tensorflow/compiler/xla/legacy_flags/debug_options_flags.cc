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

  flag_objects = new std::vector<tensorflow::Flag>(
      {tensorflow::Flag(
           "xla_generate_hlo_graph", &flag_values->xla_generate_hlo_graph,
           "HLO modules matching this regex will be dumped to a .dot file "
           "throughout various stages in compilation."),

       tensorflow::Flag(
           "xla_disable_hlo_passes", &flag_values->xla_disable_hlo_passes,
           "Comma-separated list of HLO passes to be disabled. These names "
           "must "
           "exactly match the passes' names; no whitespace around commas.")});
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

  return options;
}

}  // namespace legacy_flags
}  // namespace xla
