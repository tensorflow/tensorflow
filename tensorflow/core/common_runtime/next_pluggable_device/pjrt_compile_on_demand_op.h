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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PJRT_COMPILE_ON_DEMAND_OP_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PJRT_COMPILE_ON_DEMAND_OP_H_

#include <memory>
#include <variant>
#include <vector>

#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/core/lib/core/status.h"

// TODO(b/253052995): Move the logics to XlaCompileOnDemandOp and delete
// PjRtCompileOnDemandOp when it is ready.
namespace tensorflow {

// An OpKernel that compiles an op to an XLA computation and runs it. Unlike
// XlaLaunch this doesn't rely on any rewrites of the graphdef - it will run a
// vanilla TensorFlow op as long as the bridge supports it.
class PjRtCompileOnDemandOp : public OpKernel {
 public:
  explicit PjRtCompileOnDemandOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override;

 private:
  static Status Compile(OpKernelContext* ctx, xla::PjRtClient* pjrt_client,
                        const std::vector<XlaCompiler::Argument>& args,
                        XlaCompiler::CompilationResult* result,
                        std::unique_ptr<xla::PjRtLoadedExecutable>* executable);

  static Status Run(OpKernelContext* ctx, xla::PjRtClient* pjrt_client,
                    const std::vector<const Tensor*>& inputs,
                    const std::vector<VariableInfo>& variables,
                    const XlaCompiler::CompilationResult& result,
                    std::unique_ptr<xla::PjRtLoadedExecutable> executable);
};

void RegisterPjRtCompileOnDemand(const char* device, const char* jit_device);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PJRT_COMPILE_ON_DEMAND_OP_H_
