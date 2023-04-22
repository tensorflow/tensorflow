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
#ifndef TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_TENSOR_CONVERTER_H_
#define TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_TENSOR_CONVERTER_H_

#include <Python.h>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"

namespace tensorflow {

// Converts PyObject* values to Tensors.
//
// This converter attempts to convert values as efficiently as possible; but
// it has fallback paths to handle any PyObject* value for which tensor
// conversion is defined.
class PythonTensorConverter {
 public:
  // Constructs a new PythonTensorConverter.
  //
  // Note: the arguments to this constructor may change in the future, as
  // we move more of python tensor conversion from the Python layer to the
  // c++ layer.
  //
  // Args:
  //   py_eager_context: the value of context.context() from eager/context.py.
  //   ctx: The c++ eager context, or nullptr in graph mode.
  //   device_name: The current device name.
  //
  // All three argument values must remain alive until `this` is deleted.
  PythonTensorConverter(PyObject* py_eager_context, TFE_Context* ctx,
                        const char* device_name)
      : py_eager_context_(py_eager_context),
        ctx_(ctx),
        device_name_(device_name) {}

  // Converts `src` to a tensor (if it's not already one), and returns a new
  // reference to the converted value.
  //
  // Args:
  //   src: The object that should be converted to a Tensor.
  //   dtype: The requested dtype.  Use `DT_INVALID` if the dtype should be
  //     inferred from the `src` value (in which case `dtype` will be updated
  //     in-place to be the actual dtype of the converted value).
  //   used_fallback: Output parameter used to record whether the conversion
  //     was done by falling back to the Python `tf.convert_to_tensor()`
  //     function.  This is for testing/logging purposes only.  May be null.
  //
  // If `src` can't be converted to a tensor with the requested dtype, sets a
  // Python exception and returns nullptr.
  Safe_PyObjectPtr Convert(PyObject* src, DataType& dtype,
                           bool* used_fallback = nullptr) const;

 private:
  PyObject* py_eager_context_;
  TFE_Context* ctx_;
  const char* device_name_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_TENSOR_CONVERTER_H_
