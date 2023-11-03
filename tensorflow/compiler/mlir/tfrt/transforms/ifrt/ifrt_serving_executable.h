/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_IFRT_SERVING_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_IFRT_SERVING_EXECUTABLE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace ifrt_serving {

class IfrtServingExecutable {
 public:
  static std::unique_ptr<IfrtServingExecutable> Create(
      absl::string_view model_name, absl::string_view signature_name,
      mlir::OwningOpRef<mlir::ModuleOp> module) {
    VLOG(1) << "Creating IfrtServingExecutable";
    std::string serialized_mlir_module =
        tensorflow::SerializeMlirModule(*module);

    return absl::WrapUnique(new IfrtServingExecutable(
        model_name, signature_name, std::move(serialized_mlir_module)));
  }

  // Movable but not copyable.
  IfrtServingExecutable(IfrtServingExecutable&& other) = default;
  IfrtServingExecutable& operator=(IfrtServingExecutable&& other) = default;
  IfrtServingExecutable(const IfrtServingExecutable& other) = delete;
  IfrtServingExecutable& operator=(const IfrtServingExecutable& other) = delete;

  absl::string_view model_name() const { return model_name_; }
  absl::string_view signature_name() const { return signature_name_; }

  // Executes the computation.
  absl::StatusOr<std::vector<tensorflow::Tensor>> Execute(
      absl::Span<const tensorflow::Tensor> inputs);

 private:
  std::string model_name_;
  std::string signature_name_;

  std::string serialized_mlir_module_;

  explicit IfrtServingExecutable(absl::string_view model_name,
                                 absl::string_view signature_name,
                                 std::string serialized_mlir_module)
      : model_name_(std::string(model_name)),
        signature_name_(std::string(signature_name)),
        serialized_mlir_module_(std::move(serialized_mlir_module)) {}
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_IFRT_IFRT_SERVING_EXECUTABLE_H_
