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

#include "tensorflow/tools/proto_splitter_public/internal_api.h"

#include <string>

namespace tensorflow {
namespace image_format {

absl::Status ReadSavedModel(const std::string& file_prefix,
                            SavedModel* saved_model_proto) {
  return absl::UnimplementedError("Not yet available in OSS");
}

absl::Status WriteSavedModel(SavedModel* saved_model_proto,
                             const std::string& file_prefix) {
  return absl::UnimplementedError("Not yet available in OSS");
}

absl::Status WriteSavedModel(SavedModel* saved_model_proto,
                             const std::string& file_prefix,
                             int debug_max_size) {
  return absl::UnimplementedError("Not yet available in OSS");
}

}  // namespace image_format
}  // namespace tensorflow
