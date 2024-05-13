/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/tensor_list_util.h"

#include <functional>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/kernels/tensor_list.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

Status TensorListBinaryAdd(
    OpKernelContext* c, const TensorList& a, const TensorList& b,
    TensorList* out,
    std::function<Status(OpKernelContext* ctx, const Tensor& a, const Tensor& b,
                         Tensor* out)>
        binary_add_func) {
  if (a.element_dtype != b.element_dtype) {
    return errors::InvalidArgument(
        "Trying to add two lists of tensors of different dtypes. One is ",
        DataTypeString(a.element_dtype), " and the other is ",
        DataTypeString(b.element_dtype));
  }
  out->element_dtype = a.element_dtype;
  if (!a.element_shape.IsCompatibleWith(b.element_shape)) {
    return errors::InvalidArgument(
        "Trying to add two lists of tensors with incompatible element shapes. "
        "One is ",
        a.element_shape.DebugString(), " and the other is ",
        b.element_shape.DebugString());
  }

  TF_RETURN_IF_ERROR(
      a.element_shape.MergeWith(b.element_shape, &out->element_shape));
  if (a.tensors().size() != b.tensors().size()) {
    return errors::InvalidArgument(
        "Trying to add two lists of tensors with different lengths. One is ",
        a.tensors().size(), " and the other is ", b.tensors().size());
  }
  out->tensors().reserve(a.tensors().size());
  for (int i = 0; i < a.tensors().size(); ++i) {
    const Tensor& a_tensor = a.tensors()[i];
    const Tensor& b_tensor = b.tensors()[i];
    Tensor out_tensor;
    TF_RETURN_IF_ERROR(binary_add_func(c, a_tensor, b_tensor, &out_tensor));
    out->tensors().push_back(out_tensor);
  }
  return absl::OkStatus();
}

Status TensorListZerosLike(
    OpKernelContext* c, const TensorList& x, TensorList* y,
    std::function<Status(OpKernelContext* ctx, const Tensor& input,
                         Tensor* out)>
        zeros_like_func) {
  y->element_dtype = x.element_dtype;
  y->element_shape = x.element_shape;
  y->tensors().reserve(x.tensors().size());
  for (const Tensor& t : x.tensors()) {
    Tensor out_tensor;
    TF_RETURN_IF_ERROR(zeros_like_func(c, t, &out_tensor));
    y->tensors().emplace_back(out_tensor);
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
