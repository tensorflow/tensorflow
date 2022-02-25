/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_GRAPH_UTILS_H_
#define TENSORFLOW_CORE_DATA_SERVICE_GRAPH_UTILS_H_

#include <string>
#include <utility>

#include "tensorflow/core/framework/graph.pb.h"

namespace tensorflow {
namespace data {

// Compares the structures of the GraphDefs. Returns true if they contain the
// same ops and inputs. Otherwise, returns false and an explanation. It only
// compares the structures. Most fields are ignored (e.g.: device, version).
std::pair<bool, std::string> HaveEquivalentStructures(const GraphDef& graph1,
                                                      const GraphDef& graph2);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_GRAPH_UTILS_H_
