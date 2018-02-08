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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

namespace {
template <typename T>
struct mod_op {
  const T operator()(const T& a, const T& b) const { return a % b; }
};
}  // namespace

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Tidx>
class UnravelIndexOp : public OpKernel {
 public:
  explicit UnravelIndexOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& indices_tensor = ctx->input(0);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(indices_tensor.shape()) ||
                    TensorShapeUtils::IsScalar(indices_tensor.shape()),
                errors::InvalidArgument(
                    "The indices can only be scalar or vector, got \"",
                    indices_tensor.shape().DebugString(), "\""));

    const Tensor& dims_tensor = ctx->input(1);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(dims_tensor.shape()),
        errors::InvalidArgument("The indices can only be 1-D, got \"",
                                dims_tensor.shape().DebugString(), "\""));

    auto dims = dims_tensor.vec<Tidx>();

    Eigen::array<bool, 1> reverse({true});

    Tensor strides_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<Tidx>::value,
                                      TensorShape({dims_tensor.NumElements()}),
                                      &strides_tensor));

    auto strides = strides_tensor.vec<Tidx>();
    strides = dims.reverse(reverse)
                  .scan(0, Eigen::internal::ProdReducer<Tidx>(), false)
                  .reverse(reverse);

    Tensor strides_shifted_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<Tidx>::value,
                                      TensorShape({dims_tensor.NumElements()}),
                                      &strides_shifted_tensor));

    auto strides_shifted = strides_shifted_tensor.vec<Tidx>();
    strides_shifted = dims.reverse(reverse)
                          .scan(0, Eigen::internal::ProdReducer<Tidx>(), true)
                          .reverse(reverse);

    Tensor* output_tensor = nullptr;
    if (TensorShapeUtils::IsScalar(indices_tensor.shape())) {
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(0, TensorShape({dims_tensor.NumElements()}),
                                    &output_tensor));

      auto output = output_tensor->vec<Tidx>();

      output = output.constant(indices_tensor.scalar<Tidx>()());
      output = output.binaryExpr(strides, mod_op<Tidx>()) / strides_shifted;
    } else {
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(0,
                                    TensorShape({dims_tensor.NumElements(),
                                                 indices_tensor.NumElements()}),
                                    &output_tensor));

      auto output = output_tensor->matrix<Tidx>();

      Eigen::array<int64, 2> reshape{{dims_tensor.NumElements(), 1}};
      Eigen::array<int64, 2> bcast({1, indices_tensor.NumElements()});
      Eigen::array<int64, 2> indices_reshape{{1, indices_tensor.NumElements()}};
      Eigen::array<int64, 2> indices_bcast({dims_tensor.NumElements(), 1});

      output = indices_tensor.vec<Tidx>()
                   .reshape(indices_reshape)
                   .broadcast(indices_bcast);
      output = output.binaryExpr(strides.reshape(reshape).broadcast(bcast),
                                 mod_op<Tidx>()) /
               strides_shifted.reshape(reshape).broadcast(bcast);
    }
  }
};

#define REGISTER_KERNEL(type)                                               \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("UnravelIndex").Device(DEVICE_CPU).TypeConstraint<type>("Tidx"), \
      UnravelIndexOp<type>);
TF_CALL_int32(REGISTER_KERNEL) TF_CALL_int64(REGISTER_KERNEL)
#undef REGISTER_KERNEL

}  // namespace tensorflow
