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

#include "tensorflow/core/grappler/inputs/utils.h"

#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace grappler {

bool FilesExist(const std::vector<string>& files, std::vector<Status>* status) {
  return Env::Default()->FilesExist(files, status);
}

bool FilesExist(const std::set<string>& files) {
  return FilesExist(std::vector<string>(files.begin(), files.end()), nullptr);
}

bool FileExists(const string& file, Status* status) {
  *status = Env::Default()->FileExists(file);
  return status->ok();
}

Status ReadGraphDefFromFile(const string& graph_def_path, GraphDef* result) {
  Status status;
  if (!ReadBinaryProto(Env::Default(), graph_def_path, result).ok()) {
    return ReadTextProto(Env::Default(), graph_def_path, result);
  }
  return status;
}

Status ReadMetaGraphDefFromFile(const string& graph_def_path,
                                MetaGraphDef* result) {
  Status status;
  if (!ReadBinaryProto(Env::Default(), graph_def_path, result).ok()) {
    return ReadTextProto(Env::Default(), graph_def_path, result);
  }
  return status;
}

}  // End namespace grappler
}  // end namespace tensorflow
