/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_FUNCTION_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_FUNCTION_H_

#include <functional>
#include <memory>

#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// Get default customizable kernel creator if set
const CustomKernelCreator* GetDefaultCustomKernelCreator();

// Registers a default customizable kernel creator for a function call.
//
// If c->CanCreateKernel returns false, we still fall back to an executor-based
// interpreter op kernel to execute a function. Else c->CreateKernel() can be
// used to create a kernel that will compile the function with XLA and run the
// resulting program.
void RegisterDefaultCustomKernelCreator(CustomKernelCreator* c);

// Creates a FunctionLibraryRuntime, which instantiates functions
// defined in "lib_def" and executes functions on the "device".
// "device_mgr" must contain the "device".
//
// The returned object does not take ownerships of "device" or
// "lib_def".  The caller must ensure "device" and "lib_def" outlives
// the returned object.
//
// The "parent" is a pointer to the ProcessFunctionLibraryRuntime object that
// typically owns the created FunctionLibraryRuntime object. The parent pointer
// is not owned by the FunctionLibraryRuntime object.
std::unique_ptr<FunctionLibraryRuntime> NewFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, const ConfigProto* config,
    Device* device, int graph_def_version,
    const FunctionLibraryDefinition* lib_def, thread::ThreadPool* thread_pool,
    const OptimizerOptions& optimizer_options,
    const SessionMetadata* session_metadata,
    ProcessFunctionLibraryRuntime* parent);

// Given a numerical function "f", returns another numerical function
// "g", such that if "f" takes N inputs and produces M outputs, "g"
// takes N + M inputs and produces N outputs. I.e., if
//   (y1, y2, ..., y_M) = f(x1, x2, ..., x_N),
// g is a function which is
//   (dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N,
//                                     dL/dy1, dL/dy2, ..., dL/dy_M),
// where L is a scalar-value function of (...x_i...).
//
// TODO(zhifengc): Asks math expert to say the comment again.
std::unique_ptr<FunctionBody> SymbolicGradient(const FunctionBody& f);

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_FUNCTION_H_
