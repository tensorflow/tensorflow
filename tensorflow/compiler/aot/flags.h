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
  string graph;
  string config;
  bool dump_fetch_nodes = false;
  string target_triple;
  string target_cpu;
  string target_features;
  string entry_point;
  string cpp_class;
  string out_function_object;
  string out_metadata_object;
  string out_header;
  string out_session_module;
  string mlir_components;

  // C++ codegen options
  bool gen_name_to_index = false;
  bool gen_program_shape = false;
};

// Appends to flag_list a tensorflow::Flag for each field in MainFlags.
void AppendMainFlags(std::vector<Flag>* flag_list, MainFlags* flags);

}  // namespace tfcompile
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_AOT_FLAGS_H_
