/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// An optimization pass that performs node merging and rewrite on graph nodes

#ifndef TENSORFLOW_GRAPH_MKL_OPTIMIZER_MERGE_H_
#define TENSORFLOW_GRAPH_MKL_OPTIMIZER_MERGE_H_

#ifdef INTEL_MKL

#include <sys/types.h>
#include <vector>
#include <string>
#include <memory>
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// Interface to invoke the pass for unit test
//
// Returns true if and only if 'g' is mutated.
extern bool OptimizeNodeMerge(std::unique_ptr<Graph>* g);

}  // namespace tensorflow

#endif  // INTEL_MKL

#endif  // TENSORFLOW_GRAPH_MKL_OPTIMIZER_MERGE_H_
