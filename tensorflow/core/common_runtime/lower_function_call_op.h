/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_LOWER_FUNCTION_CALL_OP_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_LOWER_FUNCTION_CALL_OP_H_

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class FunctionLibraryDefinition;
class Graph;
class Node;

// Replaces function call node `n` with its function body. Uses
// InlineFunctionBody from `common_runtime/function.{h,cc}`. If function
// inlining is not possible or safe (see ValidateInlining), leaves the graph in
// unmodified state and returns OkStatus();
absl::Status RewriteFunctionCallNode(Node* n, Graph* g,
                                     const FunctionLibraryDefinition& flib_def,
                                     bool keep_caller_fetchable);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_LOWER_FUNCTION_CALL_OP_H_
