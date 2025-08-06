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

#include "tensorflow/compiler/aot/flags.h"

namespace tensorflow {
namespace tfcompile {

void AppendMainFlags(std::vector<Flag>* flag_list, MainFlags* flags) {
  const std::vector<Flag> tmp = {
      {"graph", &flags->graph,
       "Input GraphDef file.  If the file ends in '.pbtxt' it is expected to "
       "be in the human-readable proto text format, otherwise it is expected "
       "to be in the proto binary format."},
      {"debug_info", &flags->debug_info,
       "Graph debug info file.  If the file ends in '.pbtxt' it is expected to "
       "be in the human-readable proto text format, otherwise it is expected "
       "to be in the proto binary format."},
      {"debug_info_path_begin_marker", &flags->debug_info_path_begin_marker,
       "If not none, only keep the file path in the debug information after the"
       " marker. The default value is empty"},
      {"config", &flags->config,
       "Input file containing Config proto.  If the file ends in '.pbtxt' it "
       "is expected to be in the human-readable proto text format, otherwise "
       "it is expected to be in the proto binary format."},
      {"dump_fetch_nodes", &flags->dump_fetch_nodes,
       "If set, only flags related to fetches are processed, and the resulting "
       "fetch nodes will be dumped to stdout in a comma-separated list.  "
       "Typically used to format arguments for other tools, e.g. "
       "freeze_graph."},
      // Flags controlling the XLA ahead-of-time compilation, that correspond to
      // the fields of xla::cpu::CpuAotCompilationOptions.
      //
      // TODO(toddw): The following flags also need to be supported:
      //   --xla_cpu_llvm_opt_level
      //   --xla_cpu_llvm_cl_opts
      {"target_triple", &flags->target_triple,
       "Target platform, similar to the clang -target flag.  The general "
       "format is <arch><sub>-<vendor>-<sys>-<abi>.  "
       "http://clang.llvm.org/docs/CrossCompilation.html#target-triple."},
      {"target_cpu", &flags->target_cpu,
       "Target cpu, similar to the clang -mcpu flag.  "
       "http://clang.llvm.org/docs/CrossCompilation.html#cpu-fpu-abi"},
      {"target_features", &flags->target_features,
       "Target features, e.g. +avx2, +neon, etc."},
      {"entry_point", &flags->entry_point,
       "Name of the generated function.  If multiple generated object files "
       "will be linked into the same binary, each will need a unique entry "
       "point."},
      {"cpp_class", &flags->cpp_class,
       "Name of the generated C++ class, wrapping the generated function.  The "
       "syntax of this flag is [[<optional_namespace>::],...]<class_name>.  "
       "This mirrors the C++ syntax for referring to a class, where multiple "
       "namespaces may precede the class name, separated by double-colons.  "
       "The class will be generated in the given namespace(s), or if no "
       "namespaces are given, within the global namespace."},
      {"out_function_object", &flags->out_function_object,
       "Output object file containing the generated function for the "
       "TensorFlow model."},
      {"out_header", &flags->out_header, "Output header file name."},
      {"out_metadata_object", &flags->out_metadata_object,
       "Output object file name containing optional metadata for the generated "
       "function."},
      {"out_constant_buffers_object", &flags->out_constant_buffers_object,
       "Output object file name containing constant buffers for the runtime."},
      {"out_session_module", &flags->out_session_module,
       "Output session module proto."},
      {"mlir_components", &flags->mlir_components,
       "The MLIR components to enable. Currently only Bridge is supported."},
      {"experimental_quantize", &flags->experimental_quantize,
       "If set, quantization passes will run and dump the result before HLO "
       "code generation."},
      {"sanitize_dataflow", &flags->sanitize_dataflow,
       "Enable DataFlow Sanitizer pass."},
      {"sanitize_abilists_dataflow", &flags->sanitize_abilists_dataflow,
       "Comma separated list of ABIList file paths."},
      {"gen_name_to_index", &flags->gen_name_to_index,
       "Generate name-to-index data for Lookup{Arg,Result}Index methods."},
      {"gen_program_shape", &flags->gen_program_shape,
       "Generate program shape data for the ProgramShape method."},
      {"use_xla_nanort_runtime", &flags->use_xla_nanort_runtime,
       "Use xla cpu nanort runtime, otherwise each thunk execution gets "
       "serialized directly into the header."},
  };
  flag_list->insert(flag_list->end(), tmp.begin(), tmp.end());
}

}  // namespace tfcompile
}  // namespace tensorflow
