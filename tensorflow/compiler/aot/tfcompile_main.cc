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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/aot/compile.h"
#include "tensorflow/compiler/aot/flags.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "xla/debug_options_flags.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace tfcompile {

const char kUsageHeader[] =
    "tfcompile performs ahead-of-time compilation of a TensorFlow graph,\n"
    "resulting in an object file compiled for your target architecture, and a\n"
    "header file that gives access to the functionality in the object file.\n"
    "A typical invocation looks like this:\n"
    "\n"
    "   $ tfcompile --graph=mygraph.pb --config=myfile.pbtxt "
    "--cpp_class=\"mynamespace::MyComputation\"\n"
    "\n";

}  // end namespace tfcompile
}  // end namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::tfcompile::MainFlags flags;
#ifndef __s390x__
  flags.target_triple = "x86_64-pc-linux";
#endif
  flags.out_function_object = "out_model.o";
  flags.out_metadata_object = "out_helper.o";
  flags.out_header = "out.h";
  flags.entry_point = "entry";
  flags.debug_info_path_begin_marker = "";

  // Note that tfcompile.bzl's tf_library macro sets fast math flags as that is
  // generally the preferred case.
  std::vector<tensorflow::Flag> flag_list;
  AppendMainFlags(&flag_list, &flags);
  xla::AppendDebugOptionsFlags(&flag_list);

  tensorflow::string usage = tensorflow::tfcompile::kUsageHeader;
  usage += tensorflow::Flags::Usage(argv[0], flag_list);
  if (argc > 1 && absl::string_view(argv[1]) == "--help") {
    std::cerr << usage << "\n";
    return 0;
  }
  bool parsed_flags_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);
  QCHECK(parsed_flags_ok) << "\n" << usage;

  tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
  QCHECK(argc == 1) << "\nERROR: This command does not take any arguments "
                       "other than flags. See --help.\n\n";
  absl::Status status = tensorflow::tfcompile::Main(flags);
  if (status.code() == absl::StatusCode::kInvalidArgument) {
    std::cerr << "INVALID ARGUMENTS: " << status.message() << "\n\n";
    return 1;
  } else {
    TF_QCHECK_OK(status);
  }
  return 0;
}
