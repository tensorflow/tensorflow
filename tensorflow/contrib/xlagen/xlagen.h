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

#ifndef TENSORFLOW_CONTRIB_XLAGEN_H_
#define TENSORFLOW_CONTRIB_XLAGEN_H_

#include <string>

#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace xlagen {

// Convert a tf graph to a xla session module
xla::StatusOr<std::unique_ptr<xla::SessionModule>>
GraphDefToXlaSessionModule(const std::vector<string> &output_tensor_names,
                           const GraphDef &graphdef);

}  // end namespace xlagen
}  // end namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_XLA_TF_GRAPH_XLAGEN_H_
