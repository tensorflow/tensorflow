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

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_IF_WHILE_UTILS_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_IF_WHILE_UTILS_H_

#include <functional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

extern const char kPropagateCompileTimeConsts[];

// Convert arguments in `args` to constants provided they are compile-time
// constants and they satisfy the condition in `should_resolve_constant`. The
// argument `xla_expression_offset` determines what offset is needed to get the
// input expression from context given the argument index in `args`.
//
// Returns a list of indices which were converted to constants.
absl::InlinedVector<int, 5> ConvertCompileTimeConstArgumentsToConst(
    XlaOpKernelContext* ctx, std::vector<XlaCompiler::Argument>* args,
    int xla_expression_offset,
    std::function<bool(int arg_idx)> should_resolve_constant);

// Find and populate `must_be_const_nodes` and `body` of the function
// corresponding to the kernel with context `ctx` with name `func_name`.
absl::Status FindMustBeConstNodes(XlaOpKernelContext* ctx,
                                  const NameAttrList& func_name,
                                  std::vector<bool>* must_be_const_nodes,
                                  const FunctionBody** body);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_IF_WHILE_UTILS_H_
