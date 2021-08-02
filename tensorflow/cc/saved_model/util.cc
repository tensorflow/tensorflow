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
#include "tensorflow/cc/saved_model/util.h"

#include <string>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace saved_model {

std::string GetWriteVersion(const SavedModel& saved_model) {
  if (saved_model.meta_graphs_size() == 1 &&
      saved_model.meta_graphs()[0].has_object_graph_def()) {
    return "2";
  }
  return "1";
}

}  // namespace saved_model
}  // namespace tensorflow
