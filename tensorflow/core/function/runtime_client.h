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

#ifndef TENSORFLOW_CORE_FUNCTION_RUNTIME_CLIENT_H_
#define TENSORFLOW_CORE_FUNCTION_RUNTIME_CLIENT_H_

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

// This global context is available for testing and to be shared with the
// various APIs, such as the default Python eager execution context.
EagerContext& GlobalEagerContext();

using ReturnValues = std::vector<ImmediateTensorHandlePtr>;

class Runtime {
 public:
  explicit Runtime(EagerContext& eager_ctx) : eager_ctx_(eager_ctx) {}

  StatusOr<FunctionDef> GetFunctionProto(StringPiece name);

  Status CreateFunctionProto(FunctionDef fdef);

  StatusOr<ReturnValues> CallFunction(
      StringPiece name, absl::Span<AbstractTensorHandle* const> args);

 private:
  EagerContext& eager_ctx_;
};

}  // namespace function
}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FUNCTION_RUNTIME_CLIENT_H_
