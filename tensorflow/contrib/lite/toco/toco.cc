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

#include "absl/strings/string_view.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/model_cmdline_flags.h"
#include "tensorflow/contrib/lite/toco/model_flags.pb.h"
#include "tensorflow/contrib/lite/toco/toco_cmdline_flags.h"
#include "tensorflow/contrib/lite/toco/toco_flags.pb.h"
#include "tensorflow/contrib/lite/toco/toco_port.h"
#include "tensorflow/contrib/lite/toco/toco_tooling.h"
#include "tensorflow/contrib/lite/toco/toco_types.h"
#include "tensorflow/core/platform/logging.h"

#ifndef CHECK_OK
#define CHECK_OK(val) CHECK_EQ((val).ok(), true)
#define QCHECK_OK(val) QCHECK_EQ((val).ok(), true)
#endif

namespace toco {
namespace {

#define QCHECK_REQUIRE_TOCO_FLAG(arg) \
  QCHECK(parsed_toco_flags.arg.specified()) << "Missing required flag: " #arg;

void CheckFilePermissions(const ParsedTocoFlags& parsed_toco_flags,
                          const ParsedModelFlags& parsed_model_flags,
                          const TocoFlags& toco_flags) {
  port::CheckInitGoogleIsDone("InitGoogle is not done yet");

  QCHECK_REQUIRE_TOCO_FLAG(input_file)
  QCHECK_OK(port::file::Exists(parsed_toco_flags.input_file.value(),
                               port::file::Defaults()))
      << "Specified input_file does not exist: "
      << parsed_toco_flags.input_file.value();
  QCHECK_OK(port::file::Readable(parsed_toco_flags.input_file.value(),
                                 port::file::Defaults()))
      << "Specified input_file exists, but is not readable: "
      << parsed_toco_flags.input_file.value();

  QCHECK_REQUIRE_TOCO_FLAG(output_file);
  QCHECK_OK(port::file::Writable(parsed_toco_flags.output_file.value()))
      << "parsed_toco_flags.input_file.value() output_file is not writable: "
      << parsed_toco_flags.output_file.value();
}

void ToolMain(const ParsedTocoFlags& parsed_toco_flags,
              const ParsedModelFlags& parsed_model_flags) {
  ModelFlags model_flags;
  ReadModelFlagsFromCommandLineFlags(parsed_model_flags, &model_flags);

  TocoFlags toco_flags;
  ReadTocoFlagsFromCommandLineFlags(parsed_toco_flags, &toco_flags);

  CheckFilePermissions(parsed_toco_flags, parsed_model_flags, toco_flags);

  string input_file_contents;
  CHECK_OK(port::file::GetContents(parsed_toco_flags.input_file.value(),
                                   &input_file_contents,
                                   port::file::Defaults()));
  std::unique_ptr<Model> model =
      Import(toco_flags, model_flags, input_file_contents);
  Transform(toco_flags, model.get());
  string output_file_contents;
  Export(toco_flags, *model, toco_flags.allow_custom_ops(),
         &output_file_contents);
  CHECK_OK(port::file::SetContents(parsed_toco_flags.output_file.value(),
                                   output_file_contents,
                                   port::file::Defaults()));
}

}  // namespace
}  // namespace toco

int main(int argc, char** argv) {
  toco::string msg;
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
  toco::ToolMain(parsed_toco_flags, parsed_model_flags);
}
