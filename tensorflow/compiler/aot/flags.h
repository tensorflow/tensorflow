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

#ifndef TENSORFLOW_COMPILER_AOT_FLAGS_H_
#define TENSORFLOW_COMPILER_AOT_FLAGS_H_

#include <string>
#include <vector>

#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace tfcompile {

// Flags for the tfcompile binary.  See *.cc file for descriptions.

struct MainFlags {
  std::string graph;
  std::string debug_info;
  std::string debug_info_path_begin_marker;
  std::string config;
  bool dump_fetch_nodes = false;
  std::string target_triple;
  std::string target_cpu;
  std::string target_features;
  std::string entry_point;
  std::string cpp_class;
  std::string out_function_object;
  std::string out_metadata_object;
  std::string out_header;
  std::string out_constant_buffers_object;
  std::string out_session_module;
  std::string mlir_components;
  bool experimental_quantize = false;

  // Sanitizer pass options
  bool sanitize_dataflow = false;
  std::string sanitize_abilists_dataflow;

  // C++ codegen options
  bool gen_name_to_index = false;
  bool gen_program_shape = false;
  bool use_xla_nanort_runtime = false;
};

// Appends to flag_list a tensorflow::Flag for each field in MainFlags.
void AppendMainFlags(std::vector<Flag>* flag_list, MainFlags* flags);

}  // namespace tfcompile
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_AOT_FLAGS_H_
