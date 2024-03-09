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

#ifndef TENSORFLOW_COMPILER_JIT_XLA_COMPILE_UTIL_H_
#define TENSORFLOW_COMPILER_JIT_XLA_COMPILE_UTIL_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/tf2xla/xla_argument.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
// The number of compiler threads to use for asynchronous device compilation.
inline constexpr int64_t kNumAsyncDeviceCompilerThreads = 10;

enum class DeviceCompileMode {
  kLazy,
  kStrict,
  kAsync,
};

enum class DeviceCompileState {
  kUncompiled,
  kCompiling,
  kCompiled,
};

// Creates a single-node graph using the specified `node_def` as the only op
// apart from the arg and retval nodes corresponding to `args` and
// `result_types` respectively.
absl::StatusOr<std::unique_ptr<Graph>> CreateSingleOpGraph(
    const NodeDef& node_def, absl::Span<const XlaArgument> args,
    absl::Span<const DataType> result_types);

// Checks if single device compilation and execution with PJRT is enabled for
// `device_type` in either the XlaLaunch op or the XlaCompileOnDemand op.
bool UsePjRtForSingleDeviceCompilation(const DeviceType& device_type);

// Gets the resource name of the PjRt DeviceCompiler for `device_type`.
std::string GetPjRtDeviceCompilerResourceName(const DeviceType& device_type);

// Gets the resource name of the DeviceCompilationProfiler for `device_type`
// when PjRt is used for compilation and execution.
std::string GetPjRtDeviceCompilationProfilerResourceName(
    const DeviceType& device_type);

// Gets the ResourceMgr where the DeviceCompiler is/should be stored for the
// given `device_type`.
absl::StatusOr<ResourceMgr*> GetResourceMgrForDeviceCompiler(
    const OpKernelContext& ctx, const DeviceType& device_type);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_COMPILE_UTIL_H_
