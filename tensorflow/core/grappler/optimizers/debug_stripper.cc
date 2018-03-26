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

#include "tensorflow/core/grappler/optimizers/debug_stripper.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

Status DebugStripper::Optimize(Cluster* cluster, const GrapplerItem& item,
                               GraphDef* output) {
  // TODO(haoliang): Let's remove assertions here.
  *output = item.graph;
  return Status::OK();
}

void DebugStripper::Feedback(Cluster* cluster, const GrapplerItem& item,
                             const GraphDef& optimize_output, double result) {
  // Takes no feedback.
}

}  // end namespace grappler
}  // end namespace tensorflow
