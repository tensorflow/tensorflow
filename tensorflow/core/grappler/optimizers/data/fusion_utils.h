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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_FUSION_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_FUSION_UTILS_H_

#include <functional>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace fusion_utils {

// These functions are invoked with first and second function signature,
// should set a signature of fused second_function.
using SetFunctionSignatureFn = std::function<void(
    const OpDef& first_function_signature,
    const OpDef& second_function_signature, OpDef* fused_function_signature)>;

using StringCollection = gtl::InlinedVector<string, 2>;

// These functions are invoked with nodes from second function that were
// previously taking arguments as input. The `arg_num` tells which
// function argument node was using as an input, e.g:
// node(arg_1, other_node, arg_4)
// would be called on the first and third input with arg_num equal 1 and 4.
// It should set up inputs based on first function inputs or outputs or
// second function inputs.
using SetInputFn =
    std::function<string(const StringCollection& first_function_inputs,
                         const StringCollection& second_function_inputs,
                         const StringCollection& parent_outputs, int arg_num)>;

// This function is invoked with first and second function ret. It is used to
// set up returns of fused function.
using SetOutputFn =
    std::function<void(const protobuf::Map<string, string>& parent_ret,
                       const protobuf::Map<string, string>& second_function_ret,
                       protobuf::Map<string, string>* fused_ret)>;

using SetNodesFn = std::function<void(
    const FunctionDef& first_function, const FunctionDef& second_function,
    FunctionDef* fused_function, FunctionDefLibrary* library)>;

void MergeNodes(const FunctionDef& first_function,
                const FunctionDef& second_function, FunctionDef* fused_function,
                FunctionDefLibrary* library);

// Returns true if functions can be composed.
bool CanCompose(const OpDef& first_signature, const OpDef& second_signature);

void ComposeSignature(const OpDef& first_signature,
                      const OpDef& second_signature, OpDef* fused_signature);

string ComposeInput(const StringCollection& first_inputs,
                    const StringCollection& second_inputs,
                    const StringCollection& first_outputs, int arg_num);

// Sets output to the composition of first and second function:
// second_function(first_function(args...)).
void ComposeOutput(const protobuf::Map<string, string>& first_ret,
                   const protobuf::Map<string, string>& second_ret,
                   protobuf::Map<string, string>* fused_ret);

// Set input signature to `first_function_signature` and output signature
// to `first_function_signature` + `second_function_signature`
void CombineSignature(const OpDef& first_signature,
                      const OpDef& second_signature, OpDef* fused_signature);

// Apart from first function returns, return values from second function as
// extra returns like:
// return *first_function(...), *second_function(...)
void CombineOutput(const protobuf::Map<string, string>& first_ret,
                   const protobuf::Map<string, string>& second_ret,
                   protobuf::Map<string, string>* fused_ret);

// Returns true if both signatures have the same number of input and output
// args.
bool HasSameSignature(const OpDef& first_signature,
                      const OpDef& second_signature);

// Check if both signatures are same and copy it from `first_signature`.
void SameSignature(const OpDef& first_signature, const OpDef& second_signature,
                   OpDef* fused_signature);

// Take the same input as first function.
string SameInput(const StringCollection& first_inputs,
                 const StringCollection& second_inputs,
                 const StringCollection& first_outputs, int arg_num);

// Create a fused function that computes the short-circuit logical AND of the
// result of the first function and the result of the second function.
void LazyConjunctionOutput(const protobuf::Map<string, string>& first_ret,
                           const protobuf::Map<string, string>& second_ret,
                           protobuf::Map<string, string>* fused_ret);

void LazyConjunctionNodes(const FunctionDef& first_function,
                          const FunctionDef& second_function,
                          FunctionDef* fused_function,
                          FunctionDefLibrary* library);

// Fuse `first_function` with `second_function`, setting `fused_name_prefix` as
// a name prefix.  The nodes from `first_function` are copied unmodified.  All
// of the setup functions are called with a copy of second function having names
// that are not conflicting with first function.  This means that copied nodes
// from  second function can end up having different names.  For explanation of
// set up functions see the documentation of the functions types.
FunctionDef* FuseFunctions(
    const FunctionDef& first_function, const FunctionDef& second_function,
    StringPiece fused_name_prefix, const SetFunctionSignatureFn& set_signature,
    const SetInputFn& set_input, const SetOutputFn& set_output,
    const SetNodesFn& set_nodes, FunctionDefLibrary* library);

}  // namespace fusion_utils
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_FUSION_UTILS_H_
