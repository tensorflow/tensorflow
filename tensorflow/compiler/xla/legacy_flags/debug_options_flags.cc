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
  bool xla_hlo_graph_addresses;
  bool xla_hlo_graph_layout;
  string xla_hlo_graph_path;
  bool xla_hlo_dump_as_graphdef;
  string xla_log_hlo_text;
  string xla_generate_hlo_text_to;

  string xla_disable_hlo_passes;
  bool xla_enable_fast_math;
  bool xla_llvm_enable_alias_scope_metadata;
  bool xla_llvm_enable_noalias_metadata;
  bool xla_llvm_enable_invariant_load_metadata;
  int32 xla_backend_optimization_level;
  bool xla_embed_ir_in_executable;
  string xla_dump_ir_to;
  string xla_dump_debug_json_to;
  bool xla_eliminate_hlo_implicit_broadcast;

  bool xla_cpu_multi_thread_eigen;

  string xla_gpu_cuda_data_dir;
  bool xla_gpu_ftz;

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
  flag_values->xla_hlo_graph_addresses = false;
  flag_values->xla_hlo_graph_layout = false;
  flag_values->xla_hlo_graph_path = "/tmp/";
  flag_values->xla_hlo_dump_as_graphdef = false;
  flag_values->xla_log_hlo_text = "";
  flag_values->xla_generate_hlo_text_to = "";
  flag_values->xla_disable_hlo_passes = "";
  flag_values->xla_enable_fast_math = true;
  flag_values->xla_llvm_enable_alias_scope_metadata = true;
  flag_values->xla_llvm_enable_noalias_metadata = true;
  flag_values->xla_llvm_enable_invariant_load_metadata = true;
  flag_values->xla_backend_optimization_level = 3;
  flag_values->xla_embed_ir_in_executable = false;
  flag_values->xla_dump_ir_to = "";
  flag_values->xla_dump_debug_json_to = "";
  flag_values->xla_eliminate_hlo_implicit_broadcast = false;
  flag_values->xla_cpu_multi_thread_eigen = true;
  flag_values->xla_gpu_cuda_data_dir = "./cuda_sdk_lib";
  flag_values->xla_gpu_ftz = false;
  flag_values->xla_backend_extra_options = "";

  flag_objects = new std::vector<tensorflow::Flag>(
      {tensorflow::Flag(
           "xla_generate_hlo_graph", &flag_values->xla_generate_hlo_graph,
           "HLO modules matching this regex will be dumped to a .dot file "
           "throughout various stages in compilation."),
       tensorflow::Flag(
           "xla_hlo_graph_addresses", &flag_values->xla_hlo_graph_addresses,
           "With xla_generate_hlo_graph, show addresses of HLO ops in "
           "graph dump."),
       tensorflow::Flag(
           "xla_hlo_graph_layout", &flag_values->xla_hlo_graph_layout,
           "With xla_generate_hlo_graph, show layout of HLO ops in "
           "graph dump."),
       tensorflow::Flag(
           "xla_hlo_graph_path", &flag_values->xla_hlo_graph_path,
           "With xla_generate_hlo_graph, dump the graphs into this path."),
       tensorflow::Flag("xla_hlo_dump_as_graphdef",
                        &flag_values->xla_hlo_dump_as_graphdef,
                        "Dump HLO graphs as TensorFlow GraphDefs."),
       tensorflow::Flag(
           "xla_log_hlo_text", &flag_values->xla_log_hlo_text,
           "HLO modules matching this regex will be dumped to LOG(INFO). "),
       tensorflow::Flag(
           "xla_generate_hlo_text_to", &flag_values->xla_generate_hlo_text_to,
           "Dump all HLO modules as text into the provided directory path."),
       tensorflow::Flag(
           "xla_enable_fast_math", &flag_values->xla_enable_fast_math,
           "Enable unsafe fast-math optimizations in the compiler; "
           "this may produce faster code at the expense of some accuracy."),
       tensorflow::Flag("xla_llvm_enable_alias_scope_metadata",
                        &flag_values->xla_llvm_enable_alias_scope_metadata,
                        "In LLVM-based backends, enable the emission of "
                        "!alias.scope metadata in the generated IR."),
       tensorflow::Flag("xla_llvm_enable_noalias_metadata",
                        &flag_values->xla_llvm_enable_noalias_metadata,
                        "In LLVM-based backends, enable the emission of "
                        "!noalias metadata in the generated IR."),
       tensorflow::Flag("xla_llvm_enable_invariant_load_metadata",
                        &flag_values->xla_llvm_enable_invariant_load_metadata,
                        "In LLVM-based backends, enable the emission of "
                        "!invariant.load metadata in "
                        "the generated IR."),
       tensorflow::Flag(
           "xla_backend_optimization_level",
           &flag_values->xla_backend_optimization_level,
           "Numerical optimization level for the XLA compiler backend."),
       tensorflow::Flag(
           "xla_disable_hlo_passes", &flag_values->xla_disable_hlo_passes,
           "Comma-separated list of hlo passes to be disabled. These names "
           "must exactly match the passes' names; no whitespace around "
           "commas."),
       tensorflow::Flag("xla_embed_ir_in_executable",
                        &flag_values->xla_embed_ir_in_executable,
                        "Embed the compiler IR as a string in the executable."),
       tensorflow::Flag("xla_dump_ir_to", &flag_values->xla_dump_ir_to,
                        "Dump the compiler IR into this file/path."),
       tensorflow::Flag("xla_eliminate_hlo_implicit_broadcast",
                        &flag_values->xla_eliminate_hlo_implicit_broadcast,
                        "Eliminate implicit broadcasts when lowering user "
                        "computations to HLO instructions; use explicit "
                        "broadcast instead."),
       tensorflow::Flag("xla_cpu_multi_thread_eigen",
                        &flag_values->xla_cpu_multi_thread_eigen,
                        "When generating calls to Eigen in the CPU backend, "
                        "use multi-threaded Eigen mode."),
       tensorflow::Flag("xla_gpu_cuda_data_dir",
                        &flag_values->xla_gpu_cuda_data_dir,
                        "If non-empty, speficies a local directory containing "
                        "ptxas and nvvm libdevice files; otherwise we use "
                        "those from runfile directories."),
       tensorflow::Flag("xla_gpu_ftz", &flag_values->xla_gpu_ftz,
                        "If true, flush-to-zero semantics are enabled in the "
                        "code generated for GPUs."),
       tensorflow::Flag(
           "xla_dump_debug_json_to", &flag_values->xla_dump_debug_json_to,
           "Dump compilation artifacts as JSON into this directory."),
       tensorflow::Flag("xla_backend_extra_options",
                        &flag_values->xla_backend_extra_options,
                        "Extra options to pass to a backend; "
                        "comma-separated list of 'key=val' strings (=val "
                        "may be omitted); no whitespace around commas.")});

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
  options.set_xla_hlo_graph_addresses(flag_values->xla_hlo_graph_addresses);
  options.set_xla_hlo_graph_layout(flag_values->xla_hlo_graph_layout);
  options.set_xla_hlo_graph_path(flag_values->xla_hlo_graph_path);
  options.set_xla_hlo_dump_as_graphdef(flag_values->xla_hlo_dump_as_graphdef);
  options.set_xla_log_hlo_text(flag_values->xla_log_hlo_text);
  options.set_xla_generate_hlo_text_to(flag_values->xla_generate_hlo_text_to);

  std::vector<string> disabled_passes =
      tensorflow::str_util::Split(flag_values->xla_disable_hlo_passes, ',');
  for (const auto& passname : disabled_passes) {
    options.add_xla_disable_hlo_passes(passname);
  }

  options.set_xla_enable_fast_math(flag_values->xla_enable_fast_math);
  options.set_xla_backend_optimization_level(
      flag_values->xla_backend_optimization_level);
  options.set_xla_embed_ir_in_executable(
      flag_values->xla_embed_ir_in_executable);
  options.set_xla_dump_ir_to(flag_values->xla_dump_ir_to);
  options.set_xla_eliminate_hlo_implicit_broadcast(
      flag_values->xla_eliminate_hlo_implicit_broadcast);
  options.set_xla_dump_debug_json_to(flag_values->xla_dump_debug_json_to);

  options.set_xla_cpu_multi_thread_eigen(
      flag_values->xla_cpu_multi_thread_eigen);
  options.set_xla_gpu_cuda_data_dir(flag_values->xla_gpu_cuda_data_dir);
  options.set_xla_gpu_ftz(flag_values->xla_gpu_ftz);
  options.set_xla_llvm_enable_alias_scope_metadata(
      flag_values->xla_llvm_enable_alias_scope_metadata);
  options.set_xla_llvm_enable_noalias_metadata(
      flag_values->xla_llvm_enable_noalias_metadata);
  options.set_xla_llvm_enable_invariant_load_metadata(
      flag_values->xla_llvm_enable_invariant_load_metadata);

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
