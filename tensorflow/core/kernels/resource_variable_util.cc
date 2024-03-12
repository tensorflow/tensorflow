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

#include "tensorflow/core/kernels/resource_variable_util.h"

#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

Status ValidateAssignUpdateVariableOpShapes(const TensorShape& variable_shape,
                                            const TensorShape& value_shape) {
  if (!variable_shape.IsSameSize(value_shape)) {
    return errors::InvalidArgument(
        "Cannot update variable with shape ", variable_shape.DebugString(),
        " using a Tensor with shape ", value_shape.DebugString(),
        ", shapes must be equal.");
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
