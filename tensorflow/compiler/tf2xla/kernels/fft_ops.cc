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

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.pb.h"
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

    std::vector<int64_t> fft_length;
    xla::XlaOp input = ctx->Input(0);
    if (fft_type_ == FftType::RFFT || fft_type_ == FftType::IRFFT) {
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &fft_length));
      OP_REQUIRES(ctx, fft_length.size() == fft_rank_,
                  errors::InvalidArgument("fft_length must be length ",
                                          fft_rank_, " vector"));

      // Zero pad or truncate the axes we're doing FFT on.
      absl::InlinedVector<int64_t, 4> slice_sizes = input_shape.dim_sizes();
      std::vector<std::pair<int64_t, int64_t>> padding_sizes(
          slice_sizes.size());
      std::vector<int64_t> expected_sizes = fft_length;
      // IRFFT wants the innermost axis to be n / 2 + 1.
      if (fft_type_ == FftType::IRFFT) {
        expected_sizes[fft_rank_ - 1] = fft_length[fft_rank_ - 1] / 2 + 1;
      }
      for (int i = 0; i < fft_rank_; i++) {
        int index = input_shape.dims() - fft_rank_ + i;
        OP_REQUIRES(
            ctx,
            input_shape.dim_size(index) == 0 ||
                input_shape.dim_size(index) >= expected_sizes[i],
            errors::InvalidArgument(
                "Input dimension ", index, " must have length of at least ",
                expected_sizes[i], " but got: ", input_shape.dim_size(index)));
        if (input_shape.dim_size(index) > expected_sizes[i]) {
          slice_sizes[index] = expected_sizes[i];
        } else {
          padding_sizes[index].second =
              expected_sizes[i] - input_shape.dim_size(index);
        }
      }

      std::vector<int64_t> start_indices(input_shape.dims(), 0);
      std::vector<int64_t> strides(input_shape.dims(), 1);
      input = xla::Pad(xla::Slice(input, start_indices, slice_sizes, strides),
                       XlaHelpers::Zero(ctx->builder(), ctx->input_type(0)),
                       xla::MakeEdgePaddingConfig(padding_sizes));
    } else {
      // Innermost axis provides the FFT length.
      for (int i = 0; i < fft_rank_; i++) {
        fft_length.push_back(
            input_shape.dim_size(input_shape.dims() - fft_rank_ + i));
      }
    }

    xla::XlaOp fft = xla::Fft(input, fft_type_, fft_length);
    ctx->SetOutput(0, fft);
  }

 protected:
  const FftType fft_type_;
  const int fft_rank_;

 private:
  GenericFftOp(const GenericFftOp&) = delete;
  void operator=(const GenericFftOp&) = delete;
};

template <int FFTRank>
class FFTOp : public GenericFftOp {
 public:
  explicit FFTOp(OpKernelConstruction* ctx)
      : GenericFftOp(ctx, /*fft_type=*/FftType::FFT, /*fft_rank=*/FFTRank) {}
};
REGISTER_XLA_OP(Name("FFT").TypeConstraint("Tcomplex",
                                           {DT_COMPLEX64, DT_COMPLEX128}),
                FFTOp<1>);
REGISTER_XLA_OP(Name("FFT2D").TypeConstraint("Tcomplex",
                                             {DT_COMPLEX64, DT_COMPLEX128}),
                FFTOp<2>);
REGISTER_XLA_OP(Name("FFT3D").TypeConstraint("Tcomplex",
                                             {DT_COMPLEX64, DT_COMPLEX128}),
                FFTOp<3>);

template <int FFTRank>
class IFFTOp : public GenericFftOp {
 public:
  explicit IFFTOp(OpKernelConstruction* ctx)
      : GenericFftOp(ctx, /*fft_type=*/FftType::IFFT, /*fft_rank=*/FFTRank) {}
};
REGISTER_XLA_OP(Name("IFFT").TypeConstraint("Tcomplex",
                                            {DT_COMPLEX64, DT_COMPLEX128}),
                MlirXlaOpKernel);
REGISTER_XLA_OP(Name("IFFT2D").TypeConstraint("Tcomplex",
                                              {DT_COMPLEX64, DT_COMPLEX128}),
                IFFTOp<2>);
REGISTER_XLA_OP(Name("IFFT3D").TypeConstraint("Tcomplex",
                                              {DT_COMPLEX64, DT_COMPLEX128}),
                IFFTOp<3>);

template <int FFTRank>
class RFFTOp : public GenericFftOp {
 public:
  explicit RFFTOp(OpKernelConstruction* ctx)
      : GenericFftOp(ctx, /*fft_type=*/FftType::RFFT, /*fft_rank=*/FFTRank) {}
};
REGISTER_XLA_OP(Name("RFFT")
                    .TypeConstraint("Treal", {DT_FLOAT, DT_DOUBLE})
                    .TypeConstraint("Tcomplex", {DT_COMPLEX64, DT_COMPLEX128})
                    .CompileTimeConstantInput("fft_length"),
                RFFTOp<1>);
REGISTER_XLA_OP(Name("RFFT2D")
                    .TypeConstraint("Treal", {DT_FLOAT, DT_DOUBLE})
                    .TypeConstraint("Tcomplex", {DT_COMPLEX64, DT_COMPLEX128})
                    .CompileTimeConstantInput("fft_length"),
                RFFTOp<2>);
REGISTER_XLA_OP(Name("RFFT3D")
                    .TypeConstraint("Treal", {DT_FLOAT, DT_DOUBLE})
                    .TypeConstraint("Tcomplex", {DT_COMPLEX64, DT_COMPLEX128})
                    .CompileTimeConstantInput("fft_length"),
                RFFTOp<3>);

template <int FFTRank>
class IRFFTOp : public GenericFftOp {
 public:
  explicit IRFFTOp(OpKernelConstruction* ctx)
      : GenericFftOp(ctx, /*fft_type=*/FftType::IRFFT, /*fft_rank=*/FFTRank) {}
};
REGISTER_XLA_OP(Name("IRFFT")
                    .TypeConstraint("Treal", {DT_FLOAT, DT_DOUBLE})
                    .TypeConstraint("Tcomplex", {DT_COMPLEX64, DT_COMPLEX128})
                    .CompileTimeConstantInput("fft_length"),
                IRFFTOp<1>);
REGISTER_XLA_OP(Name("IRFFT2D")
                    .TypeConstraint("Treal", {DT_FLOAT, DT_DOUBLE})
                    .TypeConstraint("Tcomplex", {DT_COMPLEX64, DT_COMPLEX128})
                    .CompileTimeConstantInput("fft_length"),
                IRFFTOp<2>);
REGISTER_XLA_OP(Name("IRFFT3D")
                    .TypeConstraint("Treal", {DT_FLOAT, DT_DOUBLE})
                    .TypeConstraint("Tcomplex", {DT_COMPLEX64, DT_COMPLEX128})
                    .CompileTimeConstantInput("fft_length"),
                IRFFTOp<3>);

}  // namespace
}  // namespace tensorflow
