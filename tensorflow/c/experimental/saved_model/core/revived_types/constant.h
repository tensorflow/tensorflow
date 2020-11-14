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

#ifndef TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_CONSTANT_H_
#define TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_CONSTANT_H_

#include <memory>

#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/saved_model/core/revived_types/tensorhandle_convertible.h"
#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow {

// This class corresponds to python's tf.constant, which is effectively a
// TensorHandle explicitly initialized to some value.
// For now this doesn't do much beyond wrap Context's CreateLocalHandle method,
// and offer a subclass of TensorHandleConvertible. Note that similar to
// the python's eager mode logic, we bypass calling the "Const" op:
// https://github.com/tensorflow/tensorflow/blob/1c064ab76064c58e54261b805027474885a1534d/tensorflow/python/framework/constant_op.py#L301
class Constant : public TensorHandleConvertible {
 public:
  static Status Create(ImmediateExecutionContext* ctx,
                       AbstractTensorInterface* tensor,
                       std::unique_ptr<Constant>* output);

  // RevivedConstant is movable, but not copyable.
  Constant(Constant&& other) = default;
  Constant& operator=(Constant&& other) = default;

  ~Constant() override = default;

 private:
  explicit Constant(ImmediateTensorHandlePtr handle);
  Constant(const Constant&) = delete;
  Constant& operator=(const Constant&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_SAVED_MODEL_CORE_REVIVED_CONSTANT_H_
