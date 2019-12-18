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
#ifndef TENSORFLOW_COMPILER_JIT_XLA_KERNEL_CREATOR_UTIL_H_
#define TENSORFLOW_COMPILER_JIT_XLA_KERNEL_CREATOR_UTIL_H_

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class FunctionLibraryRuntime;
class OpKernel;

  // Given a NodeDef 'node_def' and the function library runtime 'flr', returns
  // true if 'node_def' is a call to a compilable function defined in 'flr',
  // with the kXlaCompileAttr set.
bool CanCreateXlaKernel(const FunctionLibraryRuntime& flr,
                        const NodeDef& node_def);

// Given a supported NodeDef, returns a XlaLaunchOp that computes the node.
Status CreateXlaKernel(FunctionLibraryRuntime* flr, const NodeDef& node_def,
                       std::unique_ptr<OpKernel>* kernel);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_KERNEL_CREATOR_UTIL_H_
