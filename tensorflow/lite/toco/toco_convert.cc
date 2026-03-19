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

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/model_cmdline_flags.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/toco_cmdline_flags.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
#include "tensorflow/lite/toco/toco_port.h"
#include "tensorflow/lite/toco/toco_tooling.h"
#include "tensorflow/lite/toco/toco_types.h"

namespace toco {
namespace {

// Checks the permissions of the output file to ensure it is writeable.
void CheckOutputFilePermissions(const Arg<std::string>& output_file) {
  QCHECK(output_file.specified()) << "Missing required flag --output_file.\n";
  QCHECK(port::file::Writable(output_file.value()).ok())
      << "Specified output_file is not writable: " << output_file.value()
      << ".\n";
}

// Checks the permissions of the frozen model file.
void CheckFrozenModelPermissions(const Arg<std::string>& input_file) {
  QCHECK(input_file.specified()) << "Missing required flag --input_file.\n";
  QCHECK(port::file::Exists(input_file.value(), port::file::Defaults()).ok())
      << "Specified input_file does not exist: " << input_file.value() << ".\n";
  QCHECK(port::file::Readable(input_file.value(), port::file::Defaults()).ok())
      << "Specified input_file exists, but is not readable: "
      << input_file.value() << ".\n";
}

// Reads the contents of the GraphDef from either the frozen graph file or the
// SavedModel directory. If it reads the SavedModel directory, it updates the
// ModelFlags and TocoFlags accordingly.
void ReadInputData(const ParsedTocoFlags& parsed_toco_flags,
                   const ParsedModelFlags& parsed_model_flags,
                   std::string* graph_def_contents) {
  port::CheckInitGoogleIsDone("InitGoogle is not done yet.\n");

  // Ensure savedmodel_directory is not set.
  QCHECK(!parsed_toco_flags.savedmodel_directory.specified())
      << "Use `tensorflow/lite/python/tflite_convert` script with "
      << "SavedModel directories.\n";

  // Checks the input file permissions and reads the contents.
  CheckFrozenModelPermissions(parsed_toco_flags.input_file);
  CHECK_OK(port::file::GetContents(parsed_toco_flags.input_file.value(),
                                   graph_def_contents, port::file::Defaults()));
}
}  // namespace

absl::Status Convert(const std::string& graph_def_contents,
                     const TocoFlags& toco_flags, const ModelFlags& model_flags,
                     std::string* output_file_contents,
                     int64_t* arithmetic_ops_count = nullptr) {
  std::unique_ptr<Model> model =
      Import(toco_flags, model_flags, graph_def_contents);
  TF_RETURN_IF_ERROR(TransformWithStatus(toco_flags, model.get()));
  TF_RETURN_IF_ERROR(Export(toco_flags, *model, toco_flags.allow_custom_ops(),
                            output_file_contents));
  if (arithmetic_ops_count != nullptr) {
    *arithmetic_ops_count = model->ArithmeticOpsCount();
  }
  return absl::OkStatus();
}

absl::Status Convert(const ParsedTocoFlags& parsed_toco_flags,
                     const ParsedModelFlags& parsed_model_flags) {
  ModelFlags model_flags;
  ReadModelFlagsFromCommandLineFlags(parsed_model_flags, &model_flags);

  TocoFlags toco_flags;
  ReadTocoFlagsFromCommandLineFlags(parsed_toco_flags, &toco_flags);

  std::string graph_def_contents;
  ReadInputData(parsed_toco_flags, parsed_model_flags, &graph_def_contents);
  CheckOutputFilePermissions(parsed_toco_flags.output_file);

  std::string output_file_contents;
  TF_RETURN_IF_ERROR(Convert(graph_def_contents, toco_flags, model_flags,
                             &output_file_contents));

  TF_RETURN_IF_ERROR(
      port::file::SetContents(parsed_toco_flags.output_file.value(),
                              output_file_contents, port::file::Defaults()));
  return absl::Status();
}

}  // namespace toco
