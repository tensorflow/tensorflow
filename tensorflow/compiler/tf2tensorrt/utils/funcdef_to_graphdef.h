/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_FUNCDEF_TO_GRAPHDEF_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_UTILS_FUNCDEF_TO_GRAPHDEF_H_

#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {

namespace tensorrt {

string AppendIdToNodeName(const Node* n);

void ToGraphDefWithIOPrefix(const Graph* g, GraphDef* gdef);

Status FunctionDefToGraphDef(FunctionLibraryRuntime::Handle handle,
                             FunctionLibraryRuntime* flib_runtime,
                             GraphDef* graph_def,
                             std::vector<int>* input_node_ids,
                             std::vector<int>* output_node_ids);

}  // namespace tensorrt
}  // namespace tensorflow

#endif
#endif
#endif
