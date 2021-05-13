/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_VARIABLE_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_VARIABLE_H_

#include <memory>

#include "absl/types/optional.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tensorhandle_convertible.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"

namespace tensorflow {

class Variable : public TensorHandleConvertible {
 public:
  // Creates an uninitialized resource variable. Note that a caller must
  // call "assign" to associate a value with the variable.
  static Status CreateUninitialized(
      ImmediateExecutionContext* ctx, DataType dtype, TensorShape shape,
      absl::optional<std::string> name, const char* raw_device_name,
      const std::vector<std::string>& component_devices,
      std::unique_ptr<Variable>* output);

  // The dtype of the underlying variable.
  DataType dtype();

  // The shape of the underlying variable.
  TensorShape shape();

  // Updates the variable's contents with `handle`.
  Status Assign(ImmediateExecutionTensorHandle* handle);

  // Reads the value of the variable, and stores it in `out`
  Status ReadValue(ImmediateTensorHandlePtr* out);

  // Variable is movable, but not copyable.
  Variable(Variable&& other) = default;
  Variable& operator=(Variable&& other) = default;

  ~Variable() override;

 private:
  Variable(ImmediateExecutionContext* ctx, DataType dtype, TensorShape shape,
           absl::optional<std::string> name, ImmediateTensorHandlePtr handle);
  Variable(const Variable& variable) = delete;
  Variable& operator=(const Variable&) = delete;

  std::string name_;
  DataType dtype_;
  TensorShape shape_;

  // ctx_ must outlive Variable.
  ImmediateExecutionContext* ctx_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_VARIABLE_H_
