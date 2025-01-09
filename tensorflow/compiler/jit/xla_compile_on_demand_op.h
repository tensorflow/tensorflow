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

// The XlaCompileOnDemandOp is an OpKernel that, when its Compute method is
// called, will generate an xla::Computation and run it asynchronously.

#ifndef TENSORFLOW_COMPILER_JIT_XLA_COMPILE_ON_DEMAND_OP_H_
#define TENSORFLOW_COMPILER_JIT_XLA_COMPILE_ON_DEMAND_OP_H_

#include <vector>

#include "tensorflow/compiler/jit/device_compilation_profiler.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/variable_info_util.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/client/local_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// An OpKernel that compiles an op to an XLA computation and runs it. Unlike
// XlaLaunch this doesn't rely on any rewrites of the graphdef - it will run a
// vanilla TensorFlow op as long as the bridge supports it.
class XlaCompileOnDemandOp : public OpKernel {
 public:
  explicit XlaCompileOnDemandOp(OpKernelConstruction* ctx)
      : OpKernel(ctx),
        platform_info_(XlaPlatformInfoFromDevice(ctx->device())) {}
  void Compute(OpKernelContext* ctx) override;

 private:
  absl::Status Compile(const std::vector<XlaCompiler::Argument>& args,
                       OpKernelContext* ctx,
                       DeviceCompiler<xla::LocalExecutable, xla::LocalClient>**
                           xla_device_compiler,
                       DeviceCompilationProfiler** profiler,
                       const XlaCompiler::CompilationResult** result,
                       xla::LocalExecutable** executable);

  absl::Status Compile(const std::vector<XlaCompiler::Argument>& args,
                       OpKernelContext* ctx,
                       DeviceCompiler<xla::PjRtLoadedExecutable,
                                      xla::PjRtClient>** pjrt_device_compiler,
                       DeviceCompilationProfiler** profiler,
                       const XlaCompiler::CompilationResult** result,
                       xla::PjRtLoadedExecutable** executable);

  absl::Status Run(const ResourceVarsSnapshot& variable_args,
                   const XlaCompiler::CompilationResult* result,
                   const DeviceCompiler<xla::LocalExecutable, xla::LocalClient>*
                       xla_device_compiler,
                   xla::LocalExecutable* executable, OpKernelContext* ctx);

  const XlaPlatformInfo platform_info_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_COMPILE_ON_DEMAND_OP_H_
