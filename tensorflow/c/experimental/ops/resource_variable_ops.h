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

// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_RESOURCE_VARIABLE_OPS_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_RESOURCE_VARIABLE_OPS_H_

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"

namespace tensorflow {
namespace ops {

// Creates a handle to a Variable resource.
Status VarHandleOp(AbstractContext* ctx, AbstractTensorHandle** resource,
                   DataType dtype, const PartialTensorShape shape,
                   const char* container = "", const char* shared_name = "",
                   absl::Span<string const> allowed_devices = {},
                   const char* name = nullptr,
                   const char* raw_device_name = nullptr);

// Reads the value of a variable.
Status ReadVariableOp(AbstractContext* ctx,
                      AbstractTensorHandle* const resource,
                      AbstractTensorHandle** value, DataType dtype,
                      const char* name = nullptr,
                      const char* raw_device_name = nullptr);

// Assigns a new value to a variable.
Status AssignVariableOp(AbstractContext* ctx,
                        AbstractTensorHandle* const resource,
                        AbstractTensorHandle* const value,
                        bool validate_shape = false, const char* name = nullptr,
                        const char* raw_device_name = nullptr);

// Deletes the resource specified by the handle.
Status DestroyResourceOp(AbstractContext* ctx,
                         AbstractTensorHandle* const resource,
                         bool ignore_lookup_error = true,
                         const char* name = nullptr,
                         const char* raw_device_name = nullptr);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_RESOURCE_VARIABLE_OPS_H_
