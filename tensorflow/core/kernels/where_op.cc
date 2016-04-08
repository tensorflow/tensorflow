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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/where_op.h"

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
class WhereOp : public OpKernel {
 public:
  explicit WhereOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    const int input_dims = input.dims();
    Tensor num_true;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DT_INT64, TensorShape({}), &num_true));
    auto num_true_t = num_true.scalar<int64>();

    functor::NumTrue<Device>::Compute(context->eigen_device<Device>(),
                                      input.flat<bool>(), num_true_t);
    TensorShape output_shape({num_true_t(), input_dims});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

#define HANDLE_DIM(NDIM)                                                   \
  case NDIM:                                                               \
    functor::Where<Device, NDIM>::Compute(context->eigen_device<Device>(), \
                                          input.tensor<bool, NDIM>(),      \
                                          output->matrix<int64>());        \
    break;

    switch (input_dims) {
      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);

      default:
        OP_REQUIRES(context, false,
                    errors::InvalidArgument(
                        "WhereOp : Unhandled input dimensions: ", input_dims));
    }
#undef HANDLE_DIM
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(WhereOp);
};

#define REGISTER_WHERE() \
  REGISTER_KERNEL_BUILDER(Name("Where").Device(DEVICE_CPU), WhereOp<CPUDevice>);

REGISTER_WHERE();

}  // namespace tensorflow
