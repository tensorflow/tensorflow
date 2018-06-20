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

#include "tensorflow/tools/graph_transforms/file_utils.h"

#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace graph_transforms {

Status LoadTextOrBinaryGraphFile(const string& file_name, GraphDef* graph_def) {
  string file_data;
  Status load_file_status =
      ReadFileToString(Env::Default(), file_name, &file_data);
  if (!load_file_status.ok()) {
    errors::AppendToMessage(&load_file_status, " (for file ", file_name, ")");
    return load_file_status;
  }
  // Try to load in binary format first, and then try ascii if that fails.
  Status load_status = ReadBinaryProto(Env::Default(), file_name, graph_def);
  if (!load_status.ok()) {
    if (protobuf::TextFormat::ParseFromString(file_data, graph_def)) {
      load_status = Status::OK();
    } else {
      errors::AppendToMessage(&load_status,
                              " (both text and binary parsing failed for file ",
                              file_name, ")");
    }
  }
  return load_status;
}

}  // namespace graph_transforms
}  // namespace tensorflow
