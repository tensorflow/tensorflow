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
#include "tensorflow/contrib/lite/toco/toco_saved_model.h"
#include "tensorflow/contrib/lite/toco/toco_tooling.h"
#include "tensorflow/contrib/lite/toco/toco_types.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {
namespace {

// Checks the permissions of the output file to ensure it is writeable.
void CheckOutputFilePermissions(const Arg<string>& output_file) {
  QCHECK(output_file.specified()) << "Missing required flag --output_file.\n";
  QCHECK(port::file::Writable(output_file.value()).ok())
      << "Specified output_file is not writable: " << output_file.value()
      << ".\n";
}

// Checks the permissions of the frozen model file.
void CheckFrozenModelPermissions(const Arg<string>& input_file) {
  QCHECK(input_file.specified()) << "Missing required flag --input_file.\n";
  QCHECK(port::file::Exists(input_file.value(), port::file::Defaults()).ok())
      << "Specified input_file does not exist: " << input_file.value() << ".\n";
  QCHECK(port::file::Readable(input_file.value(), port::file::Defaults()).ok())
      << "Specified input_file exists, but is not readable: "
      << input_file.value() << ".\n";
}

// Checks the permissions of the SavedModel directory.
void CheckSavedModelPermissions(const Arg<string>& savedmodel_directory) {
  QCHECK(savedmodel_directory.specified())
      << "Missing required flag --savedmodel_directory.\n";
  QCHECK(
      port::file::Exists(savedmodel_directory.value(), port::file::Defaults())
          .ok())
      << "Specified savedmodel_directory does not exist: "
      << savedmodel_directory.value() << ".\n";
}

// Reads the contents of the GraphDef from either the frozen graph file or the
// SavedModel directory. If it reads the SavedModel directory, it updates the
// ModelFlags and TocoFlags accordingly.
void ReadInputData(const ParsedTocoFlags& parsed_toco_flags,
                   const ParsedModelFlags& parsed_model_flags,
                   TocoFlags* toco_flags, ModelFlags* model_flags,
                   string* graph_def_contents) {
  port::CheckInitGoogleIsDone("InitGoogle is not done yet.\n");

  bool has_input_file = parsed_toco_flags.input_file.specified();
  bool has_savedmodel_dir = parsed_toco_flags.savedmodel_directory.specified();

  // Ensure either input_file or savedmodel_directory flag has been set.
  QCHECK_NE(has_input_file, has_savedmodel_dir)
      << "Specify either input_file or savedmodel_directory flag.\n";

  // Checks the input file permissions and reads the contents.
  if (has_input_file) {
    CheckFrozenModelPermissions(parsed_toco_flags.input_file);
    CHECK(port::file::GetContents(parsed_toco_flags.input_file.value(),
                                  graph_def_contents, port::file::Defaults())
              .ok());
  } else {
    CheckSavedModelPermissions(parsed_toco_flags.savedmodel_directory);
    GetSavedModelContents(parsed_toco_flags, parsed_model_flags, toco_flags,
                          model_flags, graph_def_contents);
  }
}

void ToolMain(const ParsedTocoFlags& parsed_toco_flags,
              const ParsedModelFlags& parsed_model_flags) {
  ModelFlags model_flags;
  ReadModelFlagsFromCommandLineFlags(parsed_model_flags, &model_flags);

  TocoFlags toco_flags;
  ReadTocoFlagsFromCommandLineFlags(parsed_toco_flags, &toco_flags);

  string graph_def_contents;
  ReadInputData(parsed_toco_flags, parsed_model_flags, &toco_flags,
                &model_flags, &graph_def_contents);
  CheckOutputFilePermissions(parsed_toco_flags.output_file);

  std::unique_ptr<Model> model =
      Import(toco_flags, model_flags, graph_def_contents);
  Transform(toco_flags, model.get());
  string output_file_contents;
  Export(toco_flags, *model, toco_flags.allow_custom_ops(),
         &output_file_contents);
  CHECK(port::file::SetContents(parsed_toco_flags.output_file.value(),
                                output_file_contents, port::file::Defaults())
            .ok());
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
