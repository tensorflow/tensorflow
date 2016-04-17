/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <string>
#include <unordered_set>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
template <typename T>
class ListDiffOp : public OpKernel {
 public:
  explicit ListDiffOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt, DT_INT32}));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& x = context->input(0);
    const Tensor& y = context->input(1);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(x.shape()),
                errors::InvalidArgument("x should be a 1D vector."));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(y.shape()),
                errors::InvalidArgument("y should be a 1D vector."));

    std::unordered_set<T> y_set;
    const auto Ty = y.vec<T>();
    const int y_size = Ty.size();
    y_set.reserve(y_size);
    for (int i = 0; i < y_size; ++i) {
      y_set.insert(Ty(i));
    }

    // Compute the size of the output.
    const auto Tx = x.vec<T>();
    const int x_size = Tx.size();

    int out_size = 0;
    for (int i = 0; i < x_size; ++i) {
      if (y_set.count(Tx(i)) == 0) {
        ++out_size;
      }
    }

    // Allocate and populate outputs.
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {out_size}, &out));
    auto Tout = out->vec<T>();

    Tensor* indices = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {out_size}, &indices));
    auto Tindices = indices->vec<int32>();

    for (int i = 0, p = 0; i < x_size; ++i) {
      if (y_set.count(Tx(i)) == 0) {
        OP_REQUIRES(context, p < out_size,
                    errors::InvalidArgument(
                        "Tried to set output index ", p,
                        " when output Tensor only had ", out_size,
                        " elements. Check that your "
                        "input tensors are not being concurrently mutated."));
        Tout(p) = Tx(i);
        Tindices(p) = i;
        p++;
      }
    }
  }
};

#define REGISTER_LISTDIFF(type)                                      \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ListDiff").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ListDiffOp<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_LISTDIFF);
REGISTER_LISTDIFF(string);
#undef REGISTER_LISTDIFF

}  // namespace tensorflow
