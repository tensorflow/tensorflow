/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_PYTHON_EAGER_PYWRAP_TENSOR_H_
#define TENSORFLOW_PYTHON_EAGER_PYWRAP_TENSOR_H_

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/lib/core/numpy.h"

bool EagerTensor_CheckExact(const PyObject* o);
tensorflow::int64 PyEagerTensor_ID(const PyObject* tensor);
tensorflow::DataType PyEagerTensor_Dtype(const PyObject* tensor);
tensorflow::int64 PyEagerTensor_NumElements(const PyObject* tensor);

namespace tensorflow {
TFE_TensorHandle* ConvertToEagerTensor(PyObject* value, PyObject* dtype);

// TODO(nareshmodi): Move EagerCast and ReadVariableOp (which use the C API to
// execute TFE Ops) to a separate common library.
TFE_TensorHandle* EagerCast(TFE_Context* ctx, TFE_TensorHandle* handle,
                            TF_DataType src_type_enum,
                            TF_DataType dst_type_enum, TF_Status* out_status);
}

#endif  // TENSORFLOW_PYTHON_EAGER_PYWRAP_TENSOR_H_
