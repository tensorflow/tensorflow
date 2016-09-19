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

// See docs in ../ops/image_ops.cc.

#define EIGEN_USE_THREADS

#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/eigen_attention.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class ExtractGlimpseOp : public OpKernel {
 public:
  explicit ExtractGlimpseOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("normalized", &normalized_));
    OP_REQUIRES_OK(context, context->GetAttr("centered", &centered_));
    OP_REQUIRES_OK(context, context->GetAttr("uniform_noise", &uniform_noise_));
  }

  // Expect input tensor of rank 4 with dimensions (batch_size, height, width,
  // depth).
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const TensorShape& input_shape = input.shape();
    const int32 num_dims = input_shape.dims();
    OP_REQUIRES(
        context, num_dims == 4,
        errors::InvalidArgument(
            "input must be 4-dimensional (batch_size, height, width, depth)",
            input_shape.DebugString()));

    const int64 batch_size = input_shape.dim_size(0);

    const Tensor& window_size = context->input(1);
    OP_REQUIRES(context, (window_size.shape().dims() == 1) &&
                             window_size.shape().dim_size(0) == 2,
                errors::InvalidArgument(
                    "input must be a vector of size 2 (height, width)",
                    window_size.shape().DebugString()));

    const int64 output_height = window_size.tensor<int, 1>()(0);
    const int64 output_width = window_size.tensor<int, 1>()(1);
    TensorShape output_shape = input_shape;
    output_shape.set_dim(1, output_height);
    output_shape.set_dim(2, output_width);

    const Tensor& offsets = context->input(2);
    OP_REQUIRES(context, offsets.shape().dims() == 2,
                errors::InvalidArgument("input must be a matrix",
                                        offsets.shape().DebugString()));
    OP_REQUIRES(context, offsets.shape().dim_size(0) == batch_size,
                errors::InvalidArgument("first dimension should be batch",
                                        offsets.shape().DebugString()));
    OP_REQUIRES(
        context, offsets.shape().dim_size(1) == 2,
        errors::InvalidArgument("second dimension should be of size 2 (y,x)",
                                offsets.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) {
      // Nothing else to do.
      return;
    }

    std::vector<Eigen::IndexPair<float> > offset_vec;
    offset_vec.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      float offset_y = offsets.tensor<float, 2>()(i, 0);
      float offset_x = offsets.tensor<float, 2>()(i, 1);
      // Eigen::ExtractGlimpses expects offsets as (x,y), whereas the
      // calling TensorFlow operates with (y,x) as indices.
      offset_vec.push_back(Eigen::IndexPair<float>(offset_x, offset_y));
    }

    output->tensor<float, 4>().swap_layout().device(
        context->eigen_cpu_device()) =
        Eigen::ExtractGlimpses(input.tensor<float, 4>().swap_layout(),
                               output_width, output_height, offset_vec,
                               normalized_, centered_, uniform_noise_);
  }

 private:
  bool normalized_;
  bool centered_;
  bool uniform_noise_;
};

REGISTER_KERNEL_BUILDER(Name("ExtractGlimpse").Device(DEVICE_CPU),
                        ExtractGlimpseOp);

}  // end namespace tensorflow
