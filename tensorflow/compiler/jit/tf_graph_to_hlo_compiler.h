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

#ifndef TENSORFLOW_COMPILER_JIT_TF_GRAPH_TO_HLO_COMPILER_H_
#define TENSORFLOW_COMPILER_JIT_TF_GRAPH_TO_HLO_COMPILER_H_

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/tf_to_hlo_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_argument.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class TfGraphToHloCompiler : public TfToHloCompiler {
 public:
  TfGraphToHloCompiler() = delete;

  explicit TfGraphToHloCompiler(const XlaCompiler::Options& options);

  // Compiles a Tensorflow `function` into an HloModuleProto stored in the
  // XlaCompilationResult pointed to by `result` by calling
  // XlaCompiler::CompileFunction.
  absl::Status Compile(const XlaCompiler::CompileOptions& options,
                       const NameAttrList& function,
                       absl::Span<const XlaArgument> args,
                       XlaCompilationResult* result) override;

  // Compiles a Tensorflow single op into an HloModuleProto stored in the
  // XlaCompilationResult pointed to by `result` by calling
  // XlaCompiler::CompileSingleOp.
  absl::Status CompileSingleOp(const XlaCompiler::CompileOptions& options,
                               const OpKernelContext* ctx,
                               absl::Span<const XlaArgument> args,
                               XlaCompilationResult* result) override;

 private:
  XlaCompiler xla_compiler_;
  std::string dump_dir_;

  TfGraphToHloCompiler(const TfGraphToHloCompiler&) = delete;
  void operator=(const TfGraphToHloCompiler&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_TF_GRAPH_TO_HLO_COMPILER_H_
