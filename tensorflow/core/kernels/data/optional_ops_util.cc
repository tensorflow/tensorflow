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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/data/optional_ops_util.h"

#include <functional>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace data {

Status OptionalZerosLike(OpKernelContext* ctx, const OptionalVariant& x,
                         OptionalVariant* y,
                         std::function<Status(OpKernelContext* ctx,
                                              const Tensor& input, Tensor* out)>
                             zeros_like_func) {
  if (!x.has_value()) {
    return OkStatus();
  }
  std::vector<Tensor> zero_tensors;
  for (const Tensor& tensor : x.get_values()) {
    Tensor zero_t;
    TF_RETURN_IF_ERROR(zeros_like_func(ctx, tensor, &zero_t));
    zero_tensors.push_back(std::move(zero_t));
  }
  *y = OptionalVariant(zero_tensors);
  return OkStatus();
}

Status OptionalBinaryAdd(
    OpKernelContext* ctx, const OptionalVariant& a, const OptionalVariant& b,
    OptionalVariant* out,
    std::function<Status(OpKernelContext* ctx, const Tensor& a, const Tensor& b,
                         Tensor* out)>
        binary_add_func) {
  // TODO(skyewm): should adding a value to a non-value be a no-op instead?
  if (a.has_value() != b.has_value()) {
    return errors::InvalidArgument(
        "Cannot add optionals because one has a value and the other doesn't.");
  }
  if (!a.has_value()) {
    return OkStatus();
  }
  if (a.get_values().size() != b.get_values().size()) {
    return errors::InvalidArgument(
        "Cannot add optionals because they have different numbers of "
        "components (",
        a.get_values().size(), " vs. ", b.get_values().size(), ").");
  }
  std::vector<Tensor> out_tensors;
  for (int i = 0; i < a.get_values().size(); ++i) {
    const Tensor& a_tensor = a.get_values()[i];
    const Tensor& b_tensor = b.get_values()[i];
    Tensor out_tensor;
    TF_RETURN_IF_ERROR(binary_add_func(ctx, a_tensor, b_tensor, &out_tensor));
    out_tensors.push_back(std::move(out_tensor));
  }
  *out = OptionalVariant(out_tensors);
  return OkStatus();
}

}  // namespace data
}  // namespace tensorflow
