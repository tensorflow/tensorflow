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

// Legacy flags for XLA's service module.

#include <mutex>  // NOLINT(build/c++11): only using std::call_once, not mutex.
#include <vector>

#include "tensorflow/compiler/xla/legacy_flags/parse_flags_from_env.h"
#include "tensorflow/compiler/xla/legacy_flags/service_flags.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace legacy_flags {

// Pointers to the parsed value of the flags and flag descriptors, initialized
// via flags_init.
static ServiceFlags* flags;
static std::vector<tensorflow::Flag>* flag_list;
static std::once_flag flags_init;

// Allocate *flags.  Called via call_once(&flags_init,...).
static void AllocateFlags() {
  flags = new ServiceFlags;
  flags->xla_hlo_profile = false;
  flags->xla_log_hlo_text = "";
  flags->xla_generate_hlo_graph = "";
  flags->xla_hlo_graph_addresses = false;
  flags->xla_hlo_graph_layout = false;
  flags->xla_hlo_graph_for_compute_constant = false;
  flags->xla_dump_computations_to = "";
  flags->xla_dump_hlo_text_to = "";
  flags->xla_dump_executions_to = "";
  flag_list = new std::vector<tensorflow::Flag>({
      tensorflow::Flag(
          "xla_hlo_profile", &flags->xla_hlo_profile,
          "Instrument the computation to collect per-HLO cycle counts"),
      tensorflow::Flag(
          "xla_log_hlo_text", &flags->xla_log_hlo_text,
          "If non-empty, print the text format of "
          "HLO modules whose name partially matches this regex. E.g. "
          "xla_log_hlo_text=.* will dump the text for every module."),
      tensorflow::Flag(
          "xla_generate_hlo_graph", &flags->xla_generate_hlo_graph,
          "If non-empty, dump graph of HLO modules whose name partially "
          "matches this regex. E.g. --xla_generate_hlo_graph=.* will dump "
          "the graph of every module."),
      tensorflow::Flag("xla_hlo_graph_addresses",
                       &flags->xla_hlo_graph_addresses,
                       "Show addresses of HLO ops in graph"),
      tensorflow::Flag("xla_hlo_graph_layout", &flags->xla_hlo_graph_layout,
                       "Show layout of HLO ops in graph"),
      tensorflow::Flag(
          "xla_hlo_graph_for_compute_constant",
          &flags->xla_hlo_graph_for_compute_constant,
          "If true, include hlo dumps of graphs from ComputeConstant."
          "Such graphs still need to be matched via xla_generate_hlo_graph."),
      tensorflow::Flag("xla_dump_computations_to",
                       &flags->xla_dump_computations_to,
                       "Dumps computations that XLA executes into the provided "
                       "directory path"),
      tensorflow::Flag("xla_dump_hlo_text_to", &flags->xla_dump_hlo_text_to,
                       "Dumps HLO modules that XLA executes into the provided "
                       "directory path"),
      tensorflow::Flag("xla_dump_executions_to", &flags->xla_dump_executions_to,
                       "Dumps parameters and results of computations that XLA "
                       "executes into the provided directory path"),
  });
  ParseFlagsFromEnv(*flag_list);
}

// Append to *append_to flag definitions associated with XLA's service module.
void AppendServiceFlags(std::vector<tensorflow::Flag>* append_to) {
  std::call_once(flags_init, &AllocateFlags);
  append_to->insert(append_to->end(), flag_list->begin(), flag_list->end());
}

// Return a pointer to the ServiceFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
ServiceFlags* GetServiceFlags() {
  std::call_once(flags_init, &AllocateFlags);
  return flags;
}

}  // namespace legacy_flags
}  // namespace xla
