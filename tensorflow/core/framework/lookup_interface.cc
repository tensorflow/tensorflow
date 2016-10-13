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

#include "tensorflow/core/framework/lookup_interface.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace lookup {

Status LookupInterface::CheckKeyAndValueTensors(const Tensor& key,
                                                const Tensor& value) {
  if (key.dtype() != key_dtype()) {
    return errors::InvalidArgument("Key must be type ", key_dtype(),
                                   " but got ", key.dtype());
  }
  if (value.dtype() != value_dtype()) {
    return errors::InvalidArgument("Value must be type ", value_dtype(),
                                   " but got ", value.dtype());
  }
  TensorShape expected_value_shape = key.shape();
  expected_value_shape.AppendShape(value_shape());
  if (value.shape() != expected_value_shape) {
    return errors::InvalidArgument(
        "Expected shape ", expected_value_shape.DebugString(),
        " for value, got ", value.shape().DebugString());
  }
  return Status::OK();
}

Status LookupInterface::CheckFindArguments(const Tensor& key,
                                           const Tensor& default_value) {
  if (key.dtype() != key_dtype()) {
    return errors::InvalidArgument("Key must be type ", key_dtype(),
                                   " but got ", key.dtype());
  }
  if (default_value.dtype() != value_dtype()) {
    return errors::InvalidArgument("Default value must be type ", value_dtype(),
                                   " but got ", default_value.dtype());
  }
  if (default_value.shape() != value_shape()) {
    return errors::InvalidArgument(
        "Expected shape ", value_shape().DebugString(),
        " for default value, got ", default_value.shape().DebugString());
  }
  return Status::OK();
}

}  // namespace lookup
}  // namespace tensorflow
