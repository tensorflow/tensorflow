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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_REWRITE_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_REWRITE_UTILS_H_

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {

// Rewrites the input dataset using the given config.
Status RewriteDataset(OpKernelContext* ctx, const DatasetBase* input,
                      std::function<RewriterConfig(void)> config_factory,
                      bool optimize_function_library,
                      DatasetBase** rewritten_input);

// Returns a stable hash of the portion of the graph `g` rooted at
// `node`, by creating a Merkle tree-like structure.
//
// Specifically, this function recursively walks the graph from `node` by
// following its inputs.
//
// The hash is computed by hashing its op name, device, attributes, and hashes
// of its inputs (if applicable).
//
// There is currently no guarantee that the hash of a subgraph will stay the
// same between TensorFlow builds.
uint64 HashSubgraph(const GraphDef& g, const NodeDef* node);

// Returns a stable hash of the function `f`.
//
// This function computes the hash by hashing the metadata of the
// function (disregarding the auto-generated names and descriptions) and also
// hashing the subgraph rooted at each of the output nodes.
//
// There is currently no guarantee that the hash of a function will stay the
// same between TensorFlow builds.
uint64 HashSubgraphFunction(const FunctionDefLibrary& library,
                            const FunctionDef* f);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_REWRITE_UTILS_H_
