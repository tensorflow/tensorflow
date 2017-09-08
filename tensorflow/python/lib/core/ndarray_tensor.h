/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_PYTHON_LIB_CORE_NDARRAY_TENSOR_H_
#define THIRD_PARTY_TENSORFLOW_PYTHON_LIB_CORE_NDARRAY_TENSOR_H_

// Must be included first.
#include "tensorflow/python/lib/core/numpy.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

namespace tensorflow {

Status TF_TensorToPyArray(Safe_TF_TensorPtr tensor, PyObject** out_ndarray);

// Converts the given numpy ndarray to a (safe) TF_Tensor. The returned
// TF_Tensor in `out_tensor` may have its own Python reference to `ndarray`s
// data. After `out_tensor` is destroyed, this reference must (eventually) be
// decremented via ClearDecrefCache().
//
// `out_tensor` must be non-null. Caller retains ownership of `ndarray`.
Status PyArrayToTF_Tensor(PyObject* ndarray, Safe_TF_TensorPtr* out_tensor);

// Creates a tensor in 'ret' from the input Ndarray.
Status NdarrayToTensor(PyObject* obj, Tensor* ret);

// Creates a numpy array in 'ret' which either aliases the content of 't' or has
// a copy.
Status TensorToNdarray(const Tensor& t, PyObject** ret);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_PYTHON_LIB_CORE_NDARRAY_TENSOR_H_
