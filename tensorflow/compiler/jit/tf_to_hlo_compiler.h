/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_JIT_TF_TO_HLO_COMPILER_H_
#define TENSORFLOW_COMPILER_JIT_TF_TO_HLO_COMPILER_H_

#include <memory>
#include <vector>

#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class TfToHloCompiler {
 public:
  TfToHloCompiler() = default;
  virtual ~TfToHloCompiler() = default;

  // Compiles a Tensorflow `function` to an HloModuleProto stored in the
  // XlaCompilationResult pointed to by `result`.
  virtual Status Compile(const XlaCompiler::CompileOptions& options,
                         const NameAttrList& function,
                         absl::Span<const XlaArgument> args,
                         XlaCompilationResult* result) = 0;

  // Compiles a Tensorflow single op to an HloModuleProto stored in the
  // XlaCompilationResult pointed to by `result`.
  virtual Status CompileSingleOp(const XlaCompiler::CompileOptions& options,
                                 const OpKernelContext* ctx,
                                 absl::Span<const XlaArgument> args,
                                 XlaCompilationResult* result) = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TfToHloCompiler);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_TF_TO_HLO_COMPILER_H_
