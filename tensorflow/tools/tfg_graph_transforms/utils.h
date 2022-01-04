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

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace mlir {
namespace tfg {
namespace graph_transforms {

// Reads the SavedModel proto from `input_file`.
// If the format of proto cannot be identified based on the file extension,
// attempt to load in a binary format first and then in a text format.
tensorflow::Status ReadSavedModelProto(
    const std::string& input_file, tensorflow::SavedModel& saved_model_proto);

// Best effort to identify if the protobuf file `input_file` is
// in a text or binary format.
bool IsTextProto(const std::string& input_file);

}  // namespace graph_transforms
}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_TOOLS_TFG_GRAPH_TRANSFORMS_UTILS_H_
