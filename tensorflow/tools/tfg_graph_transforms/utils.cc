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

#include "tensorflow/tools/tfg_graph_transforms/utils.h"

#include <string>

#include "tensorflow/cc/saved_model/image_format/internal_api.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tsl/platform/stringpiece.h"

namespace mlir {
namespace tfg {
namespace graph_transforms {

namespace {

absl::string_view GetNameWithoutExtension(absl::string_view filename) {
  auto pos = filename.rfind('.');
  if (pos == absl::string_view::npos) return filename;
  return filename.substr(0, pos);
}

}  // namespace

bool IsTextProto(const std::string& input_file) {
  absl::string_view extension = tensorflow::io::Extension(input_file);
  return !extension.compare("pbtxt");
}

absl::Status ReadSavedModelImageFormat(const std::string& input_file,
                                       tensorflow::SavedModel& model_proto) {
  std::string saved_model_prefix(GetNameWithoutExtension(input_file));
  return tensorflow::image_format::ReadSavedModel(saved_model_prefix,
                                                  &model_proto);
}
absl::Status WriteSavedModelImageFormat(tensorflow::SavedModel* model_proto,
                                        const std::string& output_file,
                                        int debug_max_size) {
  std::string saved_model_prefix(GetNameWithoutExtension(output_file));
  if (debug_max_size > 0) {
    return tensorflow::image_format::WriteSavedModel(
        model_proto, saved_model_prefix, debug_max_size);
  } else {
    return tensorflow::image_format::WriteSavedModel(model_proto,
                                                     saved_model_prefix);
  }
}

}  // namespace graph_transforms
}  // namespace tfg
}  // namespace mlir
