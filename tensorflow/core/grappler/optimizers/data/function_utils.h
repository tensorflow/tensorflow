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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_FUNCTION_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_FUNCTION_UTILS_H_

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {
namespace function_utils {
// This namespace contains utility functions for querying and modifying
// FunctionDefs.

// Describes a FunctionDef input tensor. In FunctionDefs, input tensor strings
// have the format node_name:node_output:position (if they derive from nodes),
// or input_name (if they derive from an argument).
struct FunctionDefTensorDesc {
  FunctionDefTensorDesc() = default;

  FunctionDefTensorDesc(const string& node_name, const string& output,
                        int position);

  // Parses node_name:node_output:position string into its components.
  explicit FunctionDefTensorDesc(const string& input);

  // TODO(rachelim): Add provisions to deal with special formats, like how
  // GrapplerFunctionItem expands node output range if position is not defined
  string full_str;
  string node_name;
  string node_output;
  int position = -1;
};

// Replaces all references to `from` tensor in func's nodes' inputs and retvals
// to `to` tensor. This is similar to `MutableGraphView::ReplaceInputs`.
void ReplaceReferences(const string& from, const string& to, FunctionDef* func);

// Adds a function output to the function def, ensuring that the output key
// is unique, and maps to output_tensor_name in the ret dict.
void AddFunctionOutputWithUniqueName(StringPiece prefix,
                                     StringPiece output_tensor_name,
                                     FunctionDef* fdef, DataType dtype);

// Adds an input to a FunctionDef.
OpDef_ArgDef* AddFunctionInput(const string& name, FunctionDef* fdef,
                               DataType dtype);

// Adds a node to a FunctionDef.
NodeDef* AddNode(StringPiece name, StringPiece op,
                 const std::vector<string>& inputs,
                 const std::vector<std::pair<string, AttrValue>>& attributes,
                 FunctionDef* fd);

// Checks whether the function contains a node with the given name.
bool ContainsFunctionNodeWithName(StringPiece name,
                                  const FunctionDef& function);

// Checks whether the function contains a node with the given op.
bool ContainsFunctionNodeWithOp(StringPiece op, const FunctionDef& function);

// Checks whether the function contains an output with the given name.
bool ContainsFunctionOutputWithName(StringPiece name,
                                    const FunctionDef& function);

// Returns the index of the function input with the given name or -1 if the
// function node does not exist.
int FindFunctionInputWithName(StringPiece name, const FunctionDef& function);

// Returns the index of the function output with the given name or -1 if the
// function node does not exist.
int FindFunctionOutputWithName(StringPiece name, const FunctionDef& function);

// Returns the index of the function node with the given name or -1 if the
// function node does not exist.
int FindFunctionNodeWithName(StringPiece name, const FunctionDef& function);

// Returns the index of the function node with the given op or -1 if the
// function node does not exist.
int FindFunctionNodeWithOp(StringPiece op, const FunctionDef& function);

// Sets the function node name using the `prefix` as a prefix while guaranteeing
// the name is unique across the functions nodes.
void SetUniqueFunctionNodeName(StringPiece prefix, FunctionDef* function,
                               NodeDef* node);

// Checks if the function is stateful by checking the function graph for
// stateful ops. Because the "If" and "While" ops are conservatively marked as
// stateful, the check recurses into their graph to determine whether they are
// actually stateful. The `skip_assert` argument determines whether the "Assert"
// op should be treated as stateful or not.
bool IsFunctionStateful(const FunctionLibraryDefinition& library,
                        const FunctionDef& function_def,
                        bool skip_assert = false);

// Checks if the node is stateful. Because the "If" or "While" ops are
// conservatively marked as stateful, the check recurses into their graph to
// determine whether they are actually stateful. The `skip_assert` argument
// determines whether the "Assert" op  should be treated as stateful or not.
bool IsNodeStateful(const FunctionLibraryDefinition& library,
                    const NodeDef& node, bool skip_assert = false);

}  // end namespace function_utils
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_FUNCTION_UTILS_H_
