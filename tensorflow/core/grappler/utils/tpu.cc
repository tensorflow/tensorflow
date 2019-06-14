/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/grappler/utils/tpu.h"

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace tensorflow {
namespace grappler {

bool IsTPUGraphDef(const GraphDef& def) {
  for (const auto& node : def.node()) {
    if (node.op() == "TPUCompile" || node.op() == "TPUPartitionedCall") {
      return true;
    }
  }
  if (!def.has_library()) return false;
  for (const auto& function_def : def.library().function()) {
    for (const auto& node : function_def.node_def()) {
      if (node.op() == "TPUCompile" || node.op() == "TPUPartitionedCall") {
        return true;
      }
    }
  }
  return false;
}

}  // end namespace grappler
}  // end namespace tensorflow
