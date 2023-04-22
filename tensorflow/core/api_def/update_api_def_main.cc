/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
// This program can be used to automatically create an api_def_*.pbtxt
// file based on op definition.
//
// To run, use the following script:
// tensorflow/core/api_def/update_api_def.sh
//
// There are 2 ways to use this script:
//   1. Define a REGISTER_OP call without a .Doc() call. Then, run
//      this script and add summaries and descriptions in the generated
//      api_def_*.pbtxt file manually.
//   2. Add .Doc() call to a REGISTER_OP call. Then run this script
//      to remove that .Doc() call and instead add corresponding summaries
//      and descriptions in api_def_*.pbtxt file automatically.
//      Note that .Doc() call must have the following format for this to work:
//      .Doc(R"doc(<doc goes here>)doc").
#include "tensorflow/core/api_def/update_api_def.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"

int main(int argc, char** argv) {
  tensorflow::string api_files_dir;
  tensorflow::string op_file_pattern;
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("api_def_dir", &api_files_dir,
                       "Base directory of api_def*.pbtxt files."),
      tensorflow::Flag("op_file_pattern", &op_file_pattern,
                       "Pattern that matches C++ files containing REGISTER_OP "
                       "calls. If specified, we will try to remove .Doc() "
                       "calls for new ops defined in these files.")};
  std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parsed_values_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parsed_values_ok) {
    std::cerr << usage << std::endl;
    return 2;
  }
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  tensorflow::OpList ops;
  tensorflow::OpRegistry::Global()->Export(false, &ops);
  tensorflow::CreateApiDefs(ops, api_files_dir, op_file_pattern);
}
