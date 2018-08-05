/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/correlation_cost/kernels/correlation_cost_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Dtype>
struct CorrelationCostFunctor<CPUDevice, Dtype> {
  Status operator()(OpKernelContext* context, const Tensor& input_a_t,
                    const Tensor& input_b_t, Tensor* output_t,
                    /* params */
                    int kernel_size, int max_displacement, int stride_1,
                    int stride_2, int pad, TensorFormat data_format) {
    const int32 oN = GetTensorDim(*output_t, FORMAT_NCHW, 'N');
    const int32 oH = GetTensorDim(*output_t, FORMAT_NCHW, 'H');
    const int32 oW = GetTensorDim(*output_t, FORMAT_NCHW, 'W');
    const int32 iH = GetTensorDim(input_a_t, data_format, 'H');
    const int32 iW = GetTensorDim(input_a_t, data_format, 'W');
    const int32 iC = GetTensorDim(input_a_t, data_format, 'C');

    const int K = kernel_size * kernel_size * iC;

    const auto input_a = input_a_t.tensor<Dtype, 4>();
    const auto input_b = input_b_t.tensor<Dtype, 4>();
    auto output = output_t->tensor<Dtype, 4>();
    output.setZero();

    const int kernel_rad = (kernel_size - 1) / 2;
    const int displacement_rad = max_displacement / stride_2;
    const int displacement_size = 2 * displacement_rad + 1;

    const bool is_NCHW = (data_format == FORMAT_NCHW);

    for (int n = 0; n < oN; ++n) {
      for (int h = 0; h < oH; ++h) {
        const int h1 = (h - pad) * stride_1 + max_displacement + kernel_rad;
        for (int w = 0; w < oW; ++w) {
          const int w1 = (w - pad) * stride_1 + max_displacement + kernel_rad;

          for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
            for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
              const int tc = (tj + displacement_rad) * displacement_size +
                             (ti + displacement_rad);

              const int w2 = w1 + ti * stride_2;
              const int h2 = h1 + tj * stride_2;

              for (int j = -kernel_rad; j <= kernel_rad; ++j) {
                // out-of-bound tests
                if ((h1 + j < 0) || (h1 + j >= iH)) continue;
                if ((h2 + j < 0) || (h2 + j >= iH)) continue;
                for (int i = -kernel_rad; i <= kernel_rad; ++i) {
                  if ((w1 + i < 0) || (w1 + i >= iW)) continue;
                  if ((w2 + i < 0) || (w2 + i >= iW)) continue;
                  for (int c = 0; c < iC; ++c) {
                    // eq. (1) in FlowNet: Learning Optical Flow with
                    // Convolutional Networks
                    if (is_NCHW) {
                      output(n, tc, h, w) += input_a(n, c, h1 + j, w1 + i) *
                                             input_b(n, c, h2 + j, w2 + i);
                    } else {
                      output(n, tc, h, w) += input_a(n, h1 + j, w1 + i, c) *
                                             input_b(n, h2 + j, w2 + i, c);
                    }
                  }
                }
              }
              output(n, tc, h, w) /= K;
            }
          }
        }
      }
    }
    return Status::OK();
  }
};

template <typename Dtype>
struct CorrelationCostGradFunctor<CPUDevice, Dtype> {
  Status operator()(OpKernelContext* context, const Tensor& input_a_t,
                    const Tensor& input_b_t, const Tensor& topdiff_t,
                    Tensor* output_a_gradient_t, Tensor* output_b_gradient_t,
                    /* params */
                    int kernel_size, int max_displacement, int stride_1,
                    int stride_2, int pad, TensorFormat data_format) {
    const int32 iN = GetTensorDim(input_a_t, data_format, 'N');
    const int32 iC = GetTensorDim(input_a_t, data_format, 'C');
    const int32 iH = GetTensorDim(input_a_t, data_format, 'H');
    const int32 iW = GetTensorDim(input_a_t, data_format, 'W');

    // topdiff is NCHW
    const int32 oH = GetTensorDim(topdiff_t, FORMAT_NCHW, 'H');
    const int32 oW = GetTensorDim(topdiff_t, FORMAT_NCHW, 'W');

    const auto topdiff = topdiff_t.tensor<Dtype, 4>();
    const auto input_a = input_a_t.tensor<Dtype, 4>();
    const auto input_b = input_b_t.tensor<Dtype, 4>();
    auto output_a_gradient = output_a_gradient_t->tensor<Dtype, 4>();
    auto output_b_gradient = output_b_gradient_t->tensor<Dtype, 4>();
    output_a_gradient.setZero();
    output_b_gradient.setZero();

    const int kernel_rad = (kernel_size - 1) / 2;
    const int displacement_rad = max_displacement / stride_2;
    const int displacement_size = 2 * displacement_rad + 1;
    const int K = kernel_size * kernel_size * iC;

    const bool is_NCHW = (data_format == FORMAT_NCHW);

    for (int n = 0; n < iN; ++n) {
      for (int h = 0; h < oH; ++h) {
        const int h1 = (h - pad) * stride_1 + max_displacement + kernel_rad;
        for (int w = 0; w < oW; ++w) {
          const int w1 = (w - pad) * stride_1 + max_displacement + kernel_rad;

          for (int tj = -displacement_rad; tj <= displacement_rad; ++tj) {
            for (int ti = -displacement_rad; ti <= displacement_rad; ++ti) {
              const int tc = (tj + displacement_rad) * displacement_size +
                             (ti + displacement_rad);

              const int w2 = w1 + ti * stride_2;
              const int h2 = h1 + tj * stride_2;

              for (int j = -kernel_rad; j <= kernel_rad; ++j) {
                // out-of-bound test
                if ((h1 + j < 0) || (h1 + j >= iH)) continue;
                if ((h2 + j < 0) || (h2 + j >= iH)) continue;
                for (int i = -kernel_rad; i <= kernel_rad; ++i) {
                  if ((w1 + i < 0) || (w1 + i >= iW)) continue;
                  if ((w2 + i < 0) || (w2 + i >= iW)) continue;
                  for (int c = 0; c < iC; ++c) {
                    // derivative of eq. (1) in FlowNet
                    if (is_NCHW) {
                      output_a_gradient(n, c, h1 + j, w1 + i) +=
                          topdiff(n, tc, h, w) * input_b(n, c, h2 + j, w2 + i) /
                          K;
                      output_b_gradient(n, c, h2 + j, w2 + i) +=
                          topdiff(n, tc, h, w) * input_a(n, c, h1 + j, w1 + i) /
                          K;
                    } else {
                      output_a_gradient(n, h1 + j, w1 + i, c) +=
                          topdiff(n, tc, h, w) * input_b(n, h2 + j, w2 + i, c) /
                          K;
                      output_b_gradient(n, h2 + j, w2 + i, c) +=
                          topdiff(n, tc, h, w) * input_a(n, h1 + j, w1 + i, c) /
                          K;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return Status::OK();
  }
};

}  // namespace functor

template <typename Device, typename T>
class CorrelationCostOp : public OpKernel {
 public:
  explicit CorrelationCostOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size));
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_displacement", &max_displacement));
    OP_REQUIRES_OK(context, context->GetAttr("stride_1", &stride_1));
    OP_REQUIRES_OK(context, context->GetAttr("stride_2", &stride_2));
    OP_REQUIRES_OK(context, context->GetAttr("pad", &pad));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, kernel_size % 2 != 0,
                errors::InvalidArgument("kernel_size must be odd"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_a_t = context->input(0);
    const Tensor& input_b_t = context->input(1);

    // we didn't check the batch-dimension during "SetShapeFn"
    OP_REQUIRES(context, input_a_t.shape() == input_b_t.shape(),
                errors::InvalidArgument("Input shapes have to be the same"));

    const int32 N = GetTensorDim(input_a_t, data_format_, 'N');
    const int32 H = GetTensorDim(input_a_t, data_format_, 'H');
    const int32 W = GetTensorDim(input_a_t, data_format_, 'W');

    // output channels are d**2 where, d = 2r + 1
    const int32 r = max_displacement / stride_2;
    const int32 d = 2 * r + 1;
    const int32 border = max_displacement + (kernel_size - 1) / 2;

    const int32 Cout = d * d;
    const int32 Hout =
        static_cast<int>(ceil(static_cast<float>(((H + 2 * pad) - border * 2)) /
                              static_cast<float>(stride_1)));
    const int32 Wout =
        static_cast<int>(ceil(static_cast<float>(((W + 2 * pad) - border * 2)) /
                              static_cast<float>(stride_1)));

    OP_REQUIRES(context, Hout >= 1,
                errors::InvalidArgument(
                    "Neighborhood and kernel don't fit in input height."));
    OP_REQUIRES(context, Wout >= 1,
                errors::InvalidArgument(
                    "Neighborhood and kernel don't fit in input width."));

    Tensor* output_t;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({N, Cout, Hout, Wout}),
                                          &output_t));


    functor::CorrelationCostFunctor<Device, T> correlationCostFunc;
    Status s = correlationCostFunc(
        context, input_a_t, input_b_t, output_t,
        /* params */
        kernel_size, max_displacement, stride_1, stride_2, pad, data_format_);

    OP_REQUIRES_OK(context, s);
  }

 private:
  int kernel_size;
  int max_displacement;
  int stride_1;
  int stride_2;
  int pad;
  TensorFormat data_format_;
};

template <typename Device, typename T>
class CorrelationCostGradOp : public OpKernel {
 public:
  explicit CorrelationCostGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size));
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_displacement", &max_displacement));
    OP_REQUIRES_OK(context, context->GetAttr("stride_1", &stride_1));
    OP_REQUIRES_OK(context, context->GetAttr("stride_2", &stride_2));
    OP_REQUIRES_OK(context, context->GetAttr("pad", &pad));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, kernel_size % 2 != 0,
                errors::InvalidArgument("kernel_size must be odd"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_a_t = context->input(0);
    const Tensor& input_b_t = context->input(1);
    const Tensor& topdiff_t = context->input(2);

    OP_REQUIRES(context, input_a_t.shape() == input_b_t.shape(),
                errors::InvalidArgument("Input shapes have to be the same"));

    // Allocate the memory for the bottom diffs
    Tensor* output_a_gradient_t;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_a_t.shape(),
                                                     &output_a_gradient_t));
    Tensor* output_b_gradient_t;
    OP_REQUIRES_OK(context, context->allocate_output(1, input_b_t.shape(),
                                                     &output_b_gradient_t));

    functor::CorrelationCostGradFunctor<Device, T> correlationCostGrad;
    Status s = correlationCostGrad(
        context, input_a_t, input_b_t, topdiff_t,
        output_a_gradient_t, output_b_gradient_t,
        /* params */
        kernel_size, max_displacement, stride_1, stride_2, pad, data_format_);

    OP_REQUIRES_OK(context, s);
  }

 private:
  int kernel_size;
  int max_displacement;
  int stride_1;
  int stride_2;
  int pad;
  TensorFormat data_format_;
};

// Register the CPU kernels.
#define REGISTER_CORRELATIONCOST_OP_CPU(T)                                   \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("CorrelationCost").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
      CorrelationCostOp<CPUDevice, T>)                                       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("CorrelationCostGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CorrelationCostGradOp<CPUDevice, T>)

TF_CALL_float(REGISTER_CORRELATIONCOST_OP_CPU);
TF_CALL_double(REGISTER_CORRELATIONCOST_OP_CPU);
#undef REGISTER_CORRELATIONCOST_OP_CPU

// Register the GPU kernels.
#ifdef GOOGLE_CUDA

#define REGISTER_CORRELATIONCOST_OP_GPU(T)                                   \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("CorrelationCost").Device(DEVICE_GPU).TypeConstraint<T>("T"),     \
      CorrelationCostOp<GPUDevice, T>)                                       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("CorrelationCostGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CorrelationCostGradOp<GPUDevice, T>)

TF_CALL_float(REGISTER_CORRELATIONCOST_OP_GPU);
TF_CALL_double(REGISTER_CORRELATIONCOST_OP_GPU);
#undef REGISTER_CORRELATIONCOST_OP_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
