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

#ifndef TENSORFLOW_COMPILER_TF2XLA_REARRANGE_FUNCTION_ARGUMENT_H_
#define TENSORFLOW_COMPILER_TF2XLA_REARRANGE_FUNCTION_ARGUMENT_H_

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// For the given graph `g`:
// 1. Rewrite If/While node functions to rearrange arguments and return values,
//    so that all resource arguments/return values are placed in the end (as
//    required by XlaCompiler),
// 2. Inline StatefulPartitionedCall nodes so we do not need to rearrange
//    arguments and return values.
// `get_function_body_fn` is used to instantiate FunctionDef.
// `fld` is used to store rewritten functions.
// `global_fld` is used to potentially supply stack traces for functions when
// they are not found in `fld`.
Status RearrangeFunctionArguments(
    std::function<Status(const NameAttrList&, const FunctionBody**)>
        get_function_body_fn,
    Graph* g, FunctionLibraryDefinition* fld,
    const FunctionLibraryDefinition* global_fld = nullptr);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_REARRANGE_FUNCTION_ARGUMENT_H_
