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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_OPS_VARIABLE_OPS_H
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_OPS_VARIABLE_OPS_H

#include "tensorflow/c/eager/context_interface.h"
#include "tensorflow/c/eager/tensor_handle_interface.h"
#include "tensorflow/c/experimental/saved_model/core/ops/owned_tensor_handle.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace internal {

// Executes a VarHandleOp using `ctx`, and fills `handle` with the DT_RESOURCE
// TensorHandle associated with the variable. This is equivalent to creating an
// unitialized TF2 tf.Variable.
// https://github.com/tensorflow/tensorflow/blob/516608035f85cec8b126712b0ff8407220206b22/tensorflow/python/ops/resource_variable_ops.py#L1867-L1872
Status CreateUninitializedResourceVariable(AbstractContextInterface* ctx,
                                           DataType dtype, TensorShape shape,
                                           AbstractTensorHandlePtr* handle);

// Executes DestroyResourceOp on `handle`, using `ctx`. This is equivalent to
// the cleanup that occurs in a tf.Variable's EagerResourceDeleter:
// https://github.com/tensorflow/tensorflow/blob/516608035f85cec8b126712b0ff8407220206b22/tensorflow/python/ops/resource_variable_ops.py#L289-L290
Status DestroyResource(AbstractContextInterface* ctx,
                       AbstractTensorHandleInterface* handle);

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_OPS_VARIABLE_OPS_H
