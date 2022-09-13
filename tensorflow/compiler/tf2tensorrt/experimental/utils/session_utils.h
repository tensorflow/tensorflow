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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_EXPERIMENTAL_UTILS_SESSION_UTILS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_EXPERIMENTAL_UTILS_SESSION_UTILS_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

// Returns configuration used for TF-TRT conversion, building.
tensorflow::SessionOptions GetSessionConfig();

// Extend the session with the given GraphDef, prefixing node names with
// `prefix`.
Status ImportGraphDefToSession(Session* session, const GraphDef& graph_def,
                               const string& prefix,
                               bool ignore_existing = false);

// Run the session with the given inputs and node names.
Status RunSession(Session* session, const std::vector<std::string>& input_names,
                  const std::vector<std::string>& output_names,
                  const std::vector<Tensor>& input_tensors,
                  std::string prefix = "");

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_EXPERIMENTAL_UTILS_SESSION_UTILS_H_