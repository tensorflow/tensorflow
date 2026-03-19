/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_LAYOUT_PASS_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_LAYOUT_PASS_UTIL_H_

#if defined(INTEL_MKL) || defined(AMD_ZENDNN)

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Temporarily wrapping these helper functions in the zendnn namespace
// to avoid crashing with similar functions in mkl_layout_pass.cc.
// TODO(penporn): Delete the functions in mkl_layout_pass and use the functions
// here after TF 2.12 branch cut.
namespace zendnn {

// Is OpDef::ArgDef a list type? It could be N * T or list(type).
// Refer to opdef.proto for details of list type.
inline bool ArgIsList(const OpDef::ArgDef &arg);

// Get length of a list in 'n' if 'arg' is of list type. Refer to
// description of ArgIsList for definition of list type.
inline int GetTensorListLength(const OpDef::ArgDef &arg, const Node *n);

// Can op represented by node 'n' run on DEVICE_CPU?
// Op can run on CPU with ZenDNN if the runtime assigned device or the
// user requested device contains device CPU, or both are empty.
bool CanOpRunOnCPUDevice(const Node *n);

// Get nodes that will feed a list of TF tensors to the new
// node that we are constructing.
//
// @input inputs - inputs to old node that we are using for constructing
//                 new inputs,
// @input input_idx - the index in the 'inputs' vector pointing to the
//                    current input that we have processed so far
// @output input_idx - index will be incremented by the number of nodes
//                     from 'inputs' that are processed
// @input list_length - The expected length of list of TF tensors
// @output output_nodes - the list of new nodes creating TF tensors
//
// @return None
void GetNodesProducingTFTensorList(
    const gtl::InlinedVector<std::pair<Node *, int>, 4> &inputs, int *input_idx,
    int list_length, std::vector<NodeBuilder::NodeOut> *output_nodes);

// Create new inputs by copying old inputs 'inputs' for the rewritten node
// in 'nb' in graph 'g'. Original node is input in 'orig_node'. This is mostly
// used in the context of rewrite for just operator name change in which
// inputs of old operator and new operator are same.
//
// Returns OkStatus() if setting up inputs is successful, otherwise
// returns appropriate status code.
Status CopyInputs(
    const Node *old_node,
    const gtl::InlinedVector<std::pair<Node *, int>, 4> &old_node_inputs,
    NodeBuilder *nb);

}  // namespace zendnn
}  // namespace tensorflow

#endif  // INTEL_MKL || AMD_ZENDNN
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_LAYOUT_PASS_UTIL_H_
