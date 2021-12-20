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

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace mlir {
namespace tfg {
namespace graph_transforms {

static constexpr char kBinarySavedModelExtension[] = "pb";
static constexpr char kTextSavedModelExtension[] = "pbtxt";

tensorflow::Status ReadSavedModelProto(
    const std::string& input_file, tensorflow::SavedModel& saved_model_proto) {
  // Proto might be either in binary or text format.
  tensorflow::StringPiece extension = tensorflow::io::Extension(input_file);
  bool binary_extenstion = !extension.compare(kBinarySavedModelExtension);
  bool text_extension = !extension.compare(kTextSavedModelExtension);

  if (!binary_extenstion && !text_extension) {
    LOG(WARNING) << "Proto type cannot be identified based on the extension";
    // Try load binary first.
    auto status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                              input_file, &saved_model_proto);
    if (status.ok()) {
      return status;
    }

    // Binary proto loading failed, attempt to load text proto.
    return tensorflow::ReadTextProto(tensorflow::Env::Default(), input_file,
                                     &saved_model_proto);
  }

  if (binary_extenstion) {
    return tensorflow::ReadBinaryProto(tensorflow::Env::Default(), input_file,
                                       &saved_model_proto);
  }

  if (text_extension) {
    return tensorflow::ReadTextProto(tensorflow::Env::Default(), input_file,
                                     &saved_model_proto);
  }

  return tensorflow::errors::InvalidArgument(
      "Expected either binary or text saved model protobuf");
}

bool IsTextProto(const std::string& input_file) {
  tensorflow::StringPiece extension = tensorflow::io::Extension(input_file);
  return !extension.compare(kTextSavedModelExtension);
}

}  // namespace graph_transforms
}  // namespace tfg
}  // namespace mlir
