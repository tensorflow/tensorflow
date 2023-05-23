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

#ifndef TENSORFLOW_CORE_GRAPH_REGULARIZATION_UTIL_H_
#define TENSORFLOW_CORE_GRAPH_REGULARIZATION_UTIL_H_

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow::graph_regularization {

// Computes the Fingerprint64 hash of the GraphDef.
uint64 ComputeHash(const GraphDef& graph_def);

// Returns the suffix UID of `function_name`, returns an error if there is none.
StatusOr<int> GetSuffixUID(absl::string_view function_name);

}  // namespace tensorflow::graph_regularization

#endif  // TENSORFLOW_CORE_GRAPH_REGULARIZATION_UTIL_H_
