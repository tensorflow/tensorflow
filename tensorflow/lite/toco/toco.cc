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
#include <cstdio>
#include <memory>
#include <string>

#include "tensorflow/lite/toco/model_cmdline_flags.h"
#include "tensorflow/lite/toco/toco_cmdline_flags.h"
#include "tensorflow/lite/toco/toco_convert.h"

int main(int argc, char** argv) {
  std::string msg;
  toco::ParsedTocoFlags parsed_toco_flags;
  toco::ParsedModelFlags parsed_model_flags;

  // If no args were specified, give a help string to be helpful.
  int* effective_argc = &argc;
  char** effective_argv = argv;
  if (argc == 1) {
    // No arguments, so manufacture help argv.
    static int dummy_argc = 2;
    static char* dummy_argv[] = {argv[0], const_cast<char*>("--help")};
    effective_argc = &dummy_argc;
    effective_argv = dummy_argv;
  }

  // Parse toco flags and command flags in sequence, each one strips off args,
  // giving InitGoogle a chance to handle all remaining arguments.
  bool toco_success = toco::ParseTocoFlagsFromCommandLineFlags(
      effective_argc, effective_argv, &msg, &parsed_toco_flags);
  bool model_success = toco::ParseModelFlagsFromCommandLineFlags(
      effective_argc, effective_argv, &msg, &parsed_model_flags);
  if (!toco_success || !model_success || !msg.empty()) {
    fprintf(stderr, "%s", msg.c_str());
    fflush(stderr);
    return 1;
  }
  toco::port::InitGoogle(argv[0], effective_argc, &effective_argv, true);
  auto status = toco::Convert(parsed_toco_flags, parsed_model_flags);
  if (!status.ok()) {
    fprintf(stderr, "%s\n", status.error_message().c_str());
    fflush(stderr);
    return 1;
  }
  return 0;
}
