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

// Implements quantized eight-bit versions of the deconvolution operations.
//
// Deconvolution operation defined in this file is similar to the
// 'conv2d_transpose' op in TensorFlow.
// Currently only a reference implementation is supported.
//
// TODO: use gemmlowp to optimise the performance

#include <algorithm>
#include <vector>

#define EIGEN_USE_THREADS

#define GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK
#include "public/gemmlowp.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/kernels/reference_gemm.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

// Similar to the ReferenceConvFunctor defined in quantized_conv_ops.cc,
// this ReferenceDeconvFunctor implements the deconvolution operation in
// the simplest way.
// As there is no backpropagation function designed for quantized convolution,
// this function implements the deconvolution operation by **definition**.
// Reference: https://youtu.be/Sb3b0ocD8mI?t=20m00s
// 
// TODO: add some detailed explanation of the algorithm.
template <class T1, class T2, class T3>
class ReferenceDeconvFunctor {
  public:
    void operator() (OpKernelContext* context,
                     const T1* input_data,
                     int input_batches,
                     int input_height,
                     int input_width,
                     int input_depth,
                     int input_offset,
                     const T2* filter_data,
                     int filter_height,
                     int filter_width,
                     int filter_depth,
                     int filter_offset,
                     int stride,
                     Padding padding,
                     T3* output_data,
                     int output_height,
                     int output_width,
                     int output_shift,
                     int output_offset,
                     int output_mult)
  {
    // TODO: add support for padding
    
    int output_depth = filter_depth;
    // Just copied from quantized_conv_ops
    int filter_left_offset =
      ((input_width - 1) * stride + filter_width - output_width + 1) / 2;
    int filter_top_offset =
      ((input_height - 1) * stride + filter_height - output_height + 1) / 2;
    // LOG(INFO) << filter_left_offset;
    // LOG(INFO) << filter_top_offset;

    for (int batch = 0; batch < input_batches; batch ++) {

      // for each channel in the output (which is the input of the conv2d)
      for (int c = 0; c < output_depth; c ++) {

        // we know that output_data is initialized as an array with zeros
        // h and w are the coordinate for an element in the gradient of output (input_data)
        for (int h = 0; h < input_height; h ++) {
          for (int w = 0; w < input_width; w ++) {
            // x and y are the coordinate of the center of the kernel that 
            // outputs the element at (h, w)
            int x = filter_height / 2 + h * stride - filter_top_offset;
            int y = filter_width / 2 + w * stride - filter_left_offset;

            for (int kx = 0; kx < filter_height; kx ++) {
              for (int ky = 0; ky < filter_width; ky ++) {
                int ox = x + kx - filter_height / 2;
                int oy = y + ky - filter_width / 2;

                T3 total = 0;
                for (int f = 0; f < input_depth; f++) {
                  const T1 input_value = input_data[
                    (batch * input_height * input_width * input_depth) +
                    (h * input_width * input_depth) +
                    (w * input_depth) +
                    (f)
                  ];

                  const T2 filter_value = filter_data[
                    (kx * filter_width * output_depth * input_depth) +
                    (ky * output_depth * input_depth) +
                    (c * input_depth) +
                    (f)
                  ];

                  total += input_value * filter_value;
                }
                
                output_data[
                  (batch * output_height * output_width * output_depth) +
                  (ox * output_width * output_depth) +
                  (oy * output_depth) +
                  (c)
                ] += total;

              }
            }


          }

        }
      }
    }

  }
};

template <class T1, class T2, class T3,
         template <class TF1, class TF2, class TF3> class ConvTransFunctor>
class QuantizedDeconv2DOp: public OpKernel {
  public:
    explicit QuantizedDeconv2DOp(OpKernelConstruction *context)
      : OpKernel(context) {
      //
      // Assertions for Attr

      // Assertions for strides
      OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
      OP_REQUIRES(context, strides_.size() == 4,
                  errors::InvalidArgument("Sliding window strides field must specify 4 dimensions"));
      OP_REQUIRES(context, strides_[1] == strides_[2],
                  errors::InvalidArgument("Current implementation only supports equal strides in the row and column dimensions"));
      //OP_REQUIRES(context, (strides_[1] == 1) & (strides_[2] == 1),
      //            errors::InvalidArgument("Current implementation doesn't support stride larger than 1"));
      OP_REQUIRES(context, (strides_[0] == 1) | (strides_[3] == 1),
                  errors::InvalidArgument("Current implementation doesn't support stride in batch or depth dimension"));
      OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
      OP_REQUIRES(context, padding_ == VALID,
                  errors::InvalidArgument("Only VALID padding is supported in the current implementation"));

      // These checkers are copied from the Conv2DFastBackpropInputOp
      // See: conv_grad_input_ops.cc
      // TODO: Assertions for data_format
      // Currently QuantizedConv2DTransposeOp doesn't support this parameter, could be implemented with
      // GetConvnetDataFormatAttrString()

      // The following checkers are set for the current implementation,
      // which are much stricter than the expected final release.
      // TODO: Remove these checkers when they are ready

      // Stride can only be 1 at the moment
      // OP_REQUIRES(context, strides_[1] == 1,
      //             errors::InvalidArgument("Current implementation only supports strides = {1,1,1,1}"));
    }

    void Compute(OpKernelContext* context) override {
      // fetch tensors from the context
      // input is the gradient w.r.t. to the output of the original convolution layer
      // shape [batch, in_rows, in_cols, in_depth]
      const Tensor& input = context->input(0); 

      // shape [filter_rows, filter_cols, out_depth, in_depth]
      const Tensor& filter = context->input(1);

      // 1-D Tensor
      const Tensor& output_sizes = context->input(2);

      // Check output_shape's dimension
      OP_REQUIRES(context,
                  TensorShapeUtils::IsVector(output_sizes.shape()),
                  errors::InvalidArgument("output_sizes should be a 1-D Tensor"));

      // Compute quantization related parameters
      const float min_input = context->input(3).flat<float>()(0);
      const float max_input = context->input(4).flat<float>()(0);
      const float min_filter = context->input(5).flat<float>()(0);
      const float max_filter = context->input(6).flat<float>()(0);

      // TODO: Use useful value here
      const int32 offset_input =
          FloatToQuantizedUnclamped<T1>(0.0f, min_input, max_input);
      const int32 offset_filter =
          FloatToQuantizedUnclamped<T2>(0.0f, min_filter, max_filter);
      const int32 offset_output = 0;
      const int32 mult_output = 1;
      const int32 shift_output = 0;

      // Get other constants
      const int64 in_depth = input.dim_size(3);
      OP_REQUIRES(context, in_depth == filter.dim_size(3),
          errors::InvalidArgument(
            "input(",
            in_depth,
            ") and filter(",
            filter.dim_size(3),
            ") doesn't match dimensions"));

      const int64 out_depth = filter.dim_size(2);

      const int64 input_rows = input.dim_size(1);
      const int64 filter_rows = filter.dim_size(0);
      const int64 out_rows = output_sizes.vec<int32>()(1);

      const int64 input_cols = input.dim_size(2);
      const int64 filter_cols = filter.dim_size(1);
      const int64 out_cols = output_sizes.vec<int32>()(2);

      const int64 batch = input.dim_size(0);
      CHECK_GT(batch, 0);

      const int stride = strides_[1];

      // Check output shape
      // The shape of the output should satisfy the following conditions:
      // 1. batch number should be the same
      // 2. the conditions for VALID padding should be matched

      OP_REQUIRES(context, batch == output_sizes.vec<int32>()(0),
          errors::InvalidArgument("batch should match output_sizes[0]"));

      // compute the expected input shape from the output shape
      // with the VALID padding condition
      const int expected_input_rows = ceil((float) (out_rows - filter_rows + 1) / stride);
      const int expected_input_cols = ceil((float) (out_cols - filter_cols + 1) / stride);
      // LOG(INFO) << expected_input_rows;
      // LOG(INFO) << expected_input_cols;
      CHECK_EQ(expected_input_rows, input_rows);
      CHECK_EQ(expected_input_cols, input_cols);

      // Create output_shape (TensorShape instance) from the input Tensor output_sizes
      TensorShape output_shape;
      OP_REQUIRES_OK(context,
                     TensorShapeUtils::MakeShape(output_sizes.vec<int32>(), &output_shape));

      Tensor *output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output)); 

      // Call the core computation
      ConvTransFunctor<T1, T2, T3> conv_trans_functor;
      conv_trans_functor(context,
                         input.flat<T1>().data(),
                         batch,
                         input_rows,
                         input_cols,
                         in_depth,
                         offset_input,
                         filter.flat<T2>().data(),
                         filter_rows,
                         filter_cols,
                         out_depth,
                         offset_filter,
                         stride,
                         padding_,
                         output->flat<T3>().data(),
                         out_rows,
                         out_cols,
                         shift_output,
                         offset_output,
                         mult_output);

      float min_output_value;
      float max_output_value;
      QuantizationRangeForMultiplication<T1, T2, T3>(
          min_input, max_input, min_filter, max_filter, &min_output_value,
          &max_output_value);

      Tensor *min_output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(1, {}, &min_output));
      min_output->flat<float>()(0) = min_output_value;

      Tensor *max_output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(2, {}, &max_output));
      max_output->flat<float>()(0) = max_output_value;
    }

  private:
    std::vector<int32> strides_;
    Padding padding_;
};

REGISTER_KERNEL_BUILDER(
    Name("QuantizedConv2DBackpropInput")
      .Device(DEVICE_CPU)
      .TypeConstraint<quint8>("Tinput")
      .TypeConstraint<quint8>("Tfilter")
      .TypeConstraint<qint32>("out_type"),
    QuantizedDeconv2DOp<quint8, quint8, qint32, ReferenceDeconvFunctor>);

} // namespace tensorflow
