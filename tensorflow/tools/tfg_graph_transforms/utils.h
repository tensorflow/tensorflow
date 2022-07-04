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

#ifndef TENSORFLOW_TOOLS_TFG_GRAPH_TRANSFORMS_UTILS_H_
#define TENSORFLOW_TOOLS_TFG_GRAPH_TRANSFORMS_UTILS_H_

#include <string>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace mlir {
namespace tfg {
namespace graph_transforms {

// Reads the model proto from `input_file`.
// If the format of proto cannot be identified based on the file extension,
// attempts to load in a binary format first and then in a text format.
template <class T>
tensorflow::Status ReadModelProto(const std::string& input_file,
                                  T& model_proto) {
  // Proto might be either in binary or text format.
  tensorflow::StringPiece extension = tensorflow::io::Extension(input_file);
  bool binary_extenstion = !extension.compare("pb");
  bool text_extension = !extension.compare("pbtxt");

  if (!binary_extenstion && !text_extension) {
    LOG(WARNING) << "Proto type cannot be identified based on the extension";
    // Try load binary first.
    auto status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                              input_file, &model_proto);
    if (status.ok()) {
      return status;
    }

    // Binary proto loading failed, attempt to load text proto.
    return tensorflow::ReadTextProto(tensorflow::Env::Default(), input_file,
                                     &model_proto);
  }

  if (binary_extenstion) {
    return tensorflow::ReadBinaryProto(tensorflow::Env::Default(), input_file,
                                       &model_proto);
  }

  if (text_extension) {
    return tensorflow::ReadTextProto(tensorflow::Env::Default(), input_file,
                                     &model_proto);
  }

  return tensorflow::errors::InvalidArgument(
      "Expected either binary or text protobuf");
}

// Best effort to identify if the protobuf file `input_file` is
// in a text or binary format.
bool IsTextProto(const std::string& input_file);

template <class T>
tensorflow::Status SerializeProto(T model_proto,
                                  const std::string& output_file) {
  auto output_dir = tensorflow::io::Dirname(output_file);

  TF_RETURN_IF_ERROR(tensorflow::Env::Default()->RecursivelyCreateDir(
      {output_dir.data(), output_dir.length()}));
  if (IsTextProto(output_file)) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        tensorflow::WriteTextProto(tensorflow::Env::Default(), output_file,
                                   model_proto),
        "Error while writing the resulting model proto");
  } else {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        tensorflow::WriteBinaryProto(tensorflow::Env::Default(), output_file,
                                     model_proto),
        "Error while writing the resulting model proto");
  }
  return ::tensorflow::OkStatus();
}

}  // namespace graph_transforms
}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_TOOLS_TFG_GRAPH_TRANSFORMS_UTILS_H_
