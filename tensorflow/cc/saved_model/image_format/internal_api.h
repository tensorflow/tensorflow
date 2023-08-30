/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_SAVED_MODEL_IMAGE_FORMAT_INTERNAL_API_H_
#define TENSORFLOW_CC_SAVED_MODEL_IMAGE_FORMAT_INTERNAL_API_H_

#include <string>

#include "absl/status/status.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace tensorflow {
namespace image_format {

// Reads the SavedModel proto from {file_prefix}{.pb|.cpb}.
// Returns a failure status when the SavedModel file does not exist.
absl::Status ReadSavedModel(const std::string& file_prefix,
                            SavedModel* saved_model_proto);

// Writes the SavedModel proto to {file_prefix}{.pb|.cpb}.
// If the proto is < the protobuf maximum size, then it will be serialized
// as a `.pb` proto binary. When larger than the maximum size, the SavedModel
// proto is destructively separated into chunks and written to
// `.cpb` (chunked proto).
absl::Status WriteSavedModel(SavedModel* saved_model_proto,
                             const std::string& file_prefix);

// See above. The `debug_max_size` argument can be used to the maximum size to
// less than 2GB for testing purposes.
absl::Status WriteSavedModel(SavedModel* saved_model_proto,
                             const std::string& file_prefix,
                             int debug_max_size);

}  // namespace image_format
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_IMAGE_FORMAT_INTERNAL_API_H_
