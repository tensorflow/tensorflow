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

#ifndef TENSORFLOW_CORE_GRAPPLER_INPUTS_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_INPUTS_UTILS_H_

#include <set>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace grappler {

bool FilesExist(const std::vector<string>& files,
                std::vector<Status>* status = nullptr);
bool FilesExist(const std::set<string>& files);

bool FileExists(const string& file, Status* status);

// Reads GraphDef from file in either text or raw serialized format.
Status ReadGraphDefFromFile(const string& graph_def_path, GraphDef* result);

// Reads MetaGraphDef from file in either text or raw serialized format.
Status ReadMetaGraphDefFromFile(const string& meta_graph_def_path,
                                MetaGraphDef* result);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_INPUTS_UTILS_H_
