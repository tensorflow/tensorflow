/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/saved_model/saved_model_aot_compile.h"

#include <string>

#include "absl/status/status.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/file_system_helper.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow::tfrt_stub {

Status AotCompileSavedModel(absl::string_view input_model_dir,
                            const AotOptions& aot_options,
                            absl::string_view output_model_dir) {
  Env* env = Env::Default();
  const std::string warmup_requests_path = io::JoinPath(
      input_model_dir, "assets.extra", "tf_serving_warmup_requests");
  TF_RETURN_IF_ERROR(env->FileExists(warmup_requests_path));

  const std::string saved_model_pb_path =
      io::JoinPath(input_model_dir, kSavedModelFilenamePb);
  const std::string saved_model_pbtxt_path =
      io::JoinPath(input_model_dir, kSavedModelFilenamePbTxt);
  bool pb_found = env->FileExists(saved_model_pb_path).ok();
  bool pbtxt_found = env->FileExists(saved_model_pbtxt_path).ok();
  if (!pb_found && !pbtxt_found) {
    return absl::NotFoundError(absl::StrCat(
        "saved_model not found in input directory: ", input_model_dir));
  }

  const bool new_directory = !output_model_dir.empty();
  std::string output_dir;
  if (!new_directory) {
    output_dir = std::string(input_model_dir);
  } else {
    // TODO(chrisminge) modify to copy everything in input directory
    output_dir = std::string(output_model_dir);
    TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(output_dir, {}));
  }
  const std::string aot_directory =
      io::JoinPath(std::string(output_model_dir), "aot_packages");
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(aot_directory, {}));
  if (pb_found) {
    const std::string output_file_directory =
        io::JoinPath(std::string(output_model_dir),
                     absl::StrCat("aot_", kSavedModelFilenamePb));
    return env->CopyFile(saved_model_pb_path, output_file_directory);
  } else {
    const std::string output_file_directory =
        io::JoinPath(std::string(output_model_dir),
                     absl::StrCat("aot_", kSavedModelFilenamePbTxt));
    return env->CopyFile(saved_model_pbtxt_path, output_file_directory);
  }
}

}  // namespace tensorflow::tfrt_stub
