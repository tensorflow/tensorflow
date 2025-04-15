/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FUNCTION_RUNTIME_CLIENT_RUNTIME_CLIENT_H_
#define TENSORFLOW_CORE_FUNCTION_RUNTIME_CLIENT_RUNTIME_CLIENT_H_

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace tensorflow {
namespace core {
namespace function {

// TODO(mdan): Get rid of this once pybind can depend on MLIR headers.
// This empty struct serves to hide a pointer to an actual MLIR TFG dialect
// FuncOp object.
struct OpaqueTfgGraphFuncOp;

// TODO(xjun): Get rid of this once pybind can depend on MLIR headers.
// This empty struct serves to hide a pointer to an actual MLIR TF dialect
// FuncOp object.
struct OpaqueTfFuncOp;

// This is the current global context managed by the Python API. For historical
// reasons, the Python runtime controls this context and all other clients must
// use it. See tensorflow/python/eager/pywrap_tfe.h and
// tensorflow/python/eager/context.py.
//
// This must always be called after the Python eager context was initialized.
//
// If the Python runtime isn't involved, or when writing code that exclusively
// relies on functions defined in this namespace, users are encouraged to
// maintain their own EagerContext or use GlobalEagerContext.
EagerContext& GlobalPythonEagerContext();

// This global context is available for testing and to be shared among various
// APIs.
EagerContext& GlobalEagerContext();

using ReturnValues = std::vector<ImmediateTensorHandlePtr>;

// A public API for manipulating and executing functions in a TensorFlow
// runtime.
class Runtime {
 public:
  explicit Runtime(EagerContext& eager_ctx) : eager_ctx_(eager_ctx) {}

  enum class Dialect {
    TFG,
    TF,
  };

  absl::StatusOr<FunctionDef> GetFunctionProto(absl::string_view name);

  // TODO(mdan): Enforce creation or rename to SetFunction.
  absl::Status CreateFunction(const FunctionDef& fdef);
  // TODO(mdan): Change to mlir::tfg::GraphFuncOp once pybind can depend on it.
  absl::Status CreateFunction(OpaqueTfgGraphFuncOp* fop);
  // TODO(xjun): Change to mlir::func::FuncOp once pybind can depend on it.
  absl::Status CreateFunction(OpaqueTfFuncOp* fop);
  // Applies a MLIR pipeline to an existing function.
  // The pipeline may rename the function. If it does so, the old function
  // remains unchanged. If the new name specifies an existing function, it will
  // be overwritten.
  absl::Status TransformFunction(absl::string_view name,
                                 absl::string_view pipeline_name,
                                 Dialect dialect = Dialect::TFG);

  absl::StatusOr<ReturnValues> CallFunction(
      absl::string_view name, absl::Span<AbstractTensorHandle* const> args);

 private:
  EagerContext& eager_ctx_;
};

}  // namespace function
}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FUNCTION_RUNTIME_CLIENT_RUNTIME_CLIENT_H_
