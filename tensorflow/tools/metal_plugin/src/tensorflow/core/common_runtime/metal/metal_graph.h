/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_GRAPH_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_GRAPH_H_

#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace metal {

// Rewrites a serialized GraphDef, fusing what this backend has fused kernels
// for. Returns false and leaves `out` untouched if the graph could not be
// parsed, which the caller answers by passing the input through unchanged.
//
// Separate from the C API entry point so it can be driven by a test with no
// TensorFlow runtime in the process.
bool FuseGraph(const uint8_t* data, size_t size, std::vector<uint8_t>* out);

// Fills in the graph optimizer C API. Exported as TF_InitGraph.
void MetalInitGraph(TP_OptimizerRegistrationParams* params, TF_Status* status);

}  // namespace metal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_METAL_METAL_GRAPH_H_
