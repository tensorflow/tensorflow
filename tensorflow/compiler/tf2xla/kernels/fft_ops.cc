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

// XLA-specific Ops for FFT.

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace {

using xla::FftType;

class GenericFftOp : public XlaOpKernel {
 public:
  explicit GenericFftOp(OpKernelConstruction* ctx, FftType fft_type,
                        int fft_rank)
      : XlaOpKernel(ctx), fft_type_(fft_type), fft_rank_(fft_rank) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVectorOrHigher(input_shape),
        errors::InvalidArgument("input must be at least 1 dimensional"));

    std::vector<int64> fft_length;
    if (fft_type_ == FftType::RFFT || fft_type_ == FftType::IRFFT) {
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &fft_length));
      OP_REQUIRES(ctx, fft_length.size() == fft_rank_,
                  errors::InvalidArgument("fft_length must be length ",
                                          fft_rank_, " vector"));
    } else {
      // Innermost axis provides the FFT length.
      for (int i = 0; i < fft_rank_; i++) {
        fft_length.push_back(
            input_shape.dim_size(input_shape.dims() - fft_rank_ + i));
      }
    }

    xla::XlaOp fft = xla::Fft(ctx->Input(0), fft_type_, fft_length);
    ctx->SetOutput(0, fft);
  }

 protected:
  const FftType fft_type_;
  const int fft_rank_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(GenericFftOp);
};

template <int FFTRank>
class FFTOp : public GenericFftOp {
 public:
  explicit FFTOp(OpKernelConstruction* ctx)
      : GenericFftOp(ctx, /*fft_type=*/FftType::FFT, /*fft_rank=*/FFTRank) {}
};
REGISTER_XLA_OP(Name("FFT").TypeConstraint("Tcomplex", DT_COMPLEX64), FFTOp<1>);
REGISTER_XLA_OP(Name("FFT2D").TypeConstraint("Tcomplex", DT_COMPLEX64),
                FFTOp<2>);
REGISTER_XLA_OP(Name("FFT3D").TypeConstraint("Tcomplex", DT_COMPLEX64),
                FFTOp<3>);

template <int FFTRank>
class IFFTOp : public GenericFftOp {
 public:
  explicit IFFTOp(OpKernelConstruction* ctx)
      : GenericFftOp(ctx, /*fft_type=*/FftType::IFFT, /*fft_rank=*/FFTRank) {}
};
REGISTER_XLA_OP(Name("IFFT").TypeConstraint("Tcomplex", DT_COMPLEX64),
                IFFTOp<1>);
REGISTER_XLA_OP(Name("IFFT2D").TypeConstraint("Tcomplex", DT_COMPLEX64),
                IFFTOp<2>);
REGISTER_XLA_OP(Name("IFFT3D").TypeConstraint("Tcomplex", DT_COMPLEX64),
                IFFTOp<3>);

template <int FFTRank>
class RFFTOp : public GenericFftOp {
 public:
  explicit RFFTOp(OpKernelConstruction* ctx)
      : GenericFftOp(ctx, /*fft_type=*/FftType::RFFT, /*fft_rank=*/FFTRank) {}
};
REGISTER_XLA_OP(Name("RFFT").CompileTimeConstInput("fft_length"), RFFTOp<1>);
REGISTER_XLA_OP(Name("RFFT2D").CompileTimeConstInput("fft_length"), RFFTOp<2>);
REGISTER_XLA_OP(Name("RFFT3D").CompileTimeConstInput("fft_length"), RFFTOp<3>);

template <int FFTRank>
class IRFFTOp : public GenericFftOp {
 public:
  explicit IRFFTOp(OpKernelConstruction* ctx)
      : GenericFftOp(ctx, /*fft_type=*/FftType::IRFFT, /*fft_rank=*/FFTRank) {}
};
REGISTER_XLA_OP(Name("IRFFT").CompileTimeConstInput("fft_length"), IRFFTOp<1>);
REGISTER_XLA_OP(Name("IRFFT2D").CompileTimeConstInput("fft_length"),
                IRFFTOp<2>);
REGISTER_XLA_OP(Name("IRFFT3D").CompileTimeConstInput("fft_length"),
                IRFFTOp<3>);

}  // namespace
}  // namespace tensorflow
