/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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

#include <stdio.h>

#include "tensorflow/core/framework/op_gen_lib.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace {

int Run(int argc, char** argv) {
  string FLAGS_in = "";
  string FLAGS_out = "";

  std::vector<Flag> flag_list = {
      Flag("in", &FLAGS_in, "Input multi-line proto text (.mlpbtxt) file name"),
      Flag("out", &FLAGS_out, "Output proto text (.pbtxt) file name")};

  // Parse the command-line.
  const string usage = Flags::Usage(argv[0], flag_list);
  const bool parse_ok = Flags::Parse(&argc, argv, flag_list);
  if (argc != 1 || !parse_ok) {
    printf("%s", usage.c_str());
    return 2;
  }

  port::InitMain(argv[0], &argc, &argv);

  // Read the input file --in.
  string in_contents;
  absl::Status s = ReadFileToString(Env::Default(), FLAGS_in, &in_contents);
  if (!s.ok()) {
    printf("Error reading file %s: %s\n", FLAGS_in.c_str(),
           s.ToString().c_str());
    return 1;
  }

  // Write the output file --out.
  const string out_contents = PBTxtFromMultiline(in_contents);
  s = WriteStringToFile(Env::Default(), FLAGS_out, out_contents);
  if (!s.ok()) {
    printf("Error writing file %s: %s\n", FLAGS_out.c_str(),
           s.ToString().c_str());
    return 1;
  }

  return 0;
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) { return tensorflow::Run(argc, argv); }
