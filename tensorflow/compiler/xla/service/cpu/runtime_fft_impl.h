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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FFT_IMPL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FFT_IMPL_H_

#include <array>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/types.h"

// 'tensorflow' namespace is used so that int64 and other types don't require
// qualification.
namespace tensorflow {
namespace xla {

namespace internal {

// Computes either a forward or reverse complex-to-complex FFT.
template <bool Forward, int FFTRank, typename EigenDevice>
void EigenFftC2C(const EigenDevice& device, complex64* out, complex64* operand,
                 int64 input_batch, int64 fft_length0, int64 fft_length1,
                 int64 fft_length2) {
  // Create the axes (which are always trailing).
  const auto axes = Eigen::ArrayXi::LinSpaced(FFTRank, 1, FFTRank);
  constexpr auto direction = Forward ? Eigen::FFT_FORWARD : Eigen::FFT_REVERSE;

  const std::array<int64, 3> fft_shape = {
      {fft_length0, fft_length1, fft_length2}};

  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> dims;
  dims[0] = input_batch;
  for (int i = 0; i < FFTRank; i++) {
    dims[i + 1] = fft_shape[i];
  }
  const Eigen::TensorMap<Eigen::Tensor<complex64, FFTRank + 1, Eigen::RowMajor>,
                         Eigen::Aligned>
      input(operand, dims);
  Eigen::TensorMap<Eigen::Tensor<complex64, FFTRank + 1, Eigen::RowMajor>,
                   Eigen::Aligned>
      output(out, dims);
  output.device(device) = input.template fft<Eigen::BothParts, direction>(axes);
}

// Computes a forward real->complex FFT, slicing out redundant negative
// frequencies from the innermost dimension.
template <int FFTRank, typename EigenDevice>
void EigenFftR2C(const EigenDevice& device, complex64* out, float* operand,
                 int64 input_batch, int64 fft_length0, int64 fft_length1,
                 int64 fft_length2) {
  const std::array<int64, 3> fft_shape = {
      {fft_length0, fft_length1, fft_length2}};

  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> in_dims;
  in_dims[0] = input_batch;
  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> out_dims;
  out_dims[0] = input_batch;
  TensorShape temp_shape{input_batch};
  for (int i = 0; i < FFTRank; i++) {
    in_dims[i + 1] = fft_shape[i];
    out_dims[i + 1] = i == FFTRank - 1 ? fft_shape[i] / 2 + 1 : fft_shape[i];
    temp_shape.AddDim(fft_shape[i]);
  }
  const Eigen::TensorMap<Eigen::Tensor<float, FFTRank + 1, Eigen::RowMajor>,
                         Eigen::Aligned>
      input(operand, in_dims);
  Eigen::TensorMap<Eigen::Tensor<complex64, FFTRank + 1, Eigen::RowMajor>,
                   Eigen::Aligned>
      output(out, out_dims);

  // Create the axes (which are always trailing).
  const auto axes = Eigen::ArrayXi::LinSpaced(FFTRank, 1, FFTRank);

  // Compute the full FFT using a temporary tensor.
  Tensor temp(DataTypeToEnum<complex64>::v(), temp_shape);
  auto full_fft = temp.flat_inner_dims<complex64, FFTRank + 1>();
  const Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> zero_start_indices;
  full_fft.device(device) =
      input.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(axes);

  // Slice away the negative frequency components.
  output.device(device) = full_fft.slice(zero_start_indices, out_dims);
}

// Computes a reverse complex->real FFT, reconstructing redundant negative
// frequencies using reverse conjugate on innermost dimension after doing IFFT
// on outer dimensions.
template <int FFTRank, typename EigenDevice>
void EigenFftC2R(const EigenDevice& device, float* out, complex64* operand,
                 int64 input_batch, int64 fft_length0, int64 fft_length1,
                 int64 fft_length2) {
  const std::array<int64, 3> fft_shape = {
      {fft_length0, fft_length1, fft_length2}};

  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> in_dims;
  in_dims[0] = input_batch;
  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> out_dims;
  out_dims[0] = input_batch;
  TensorShape temp_shape{input_batch};
  for (int i = 0; i < FFTRank; i++) {
    in_dims[i + 1] = i == FFTRank - 1 ? fft_shape[i] / 2 + 1 : fft_shape[i];
    out_dims[i + 1] = fft_shape[i];
    temp_shape.AddDim(fft_shape[i]);
  }
  const Eigen::TensorMap<Eigen::Tensor<complex64, FFTRank + 1, Eigen::RowMajor>,
                         Eigen::Aligned>
      input(operand, in_dims);
  Eigen::TensorMap<Eigen::Tensor<float, FFTRank + 1, Eigen::RowMajor>,
                   Eigen::Aligned>
      output(out, out_dims);

  // Calculate the shape of the temporary tensor for the full FFT and the
  // region we will slice from input given fft_shape. We slice input to
  // fft_shape on its inner-most dimensions, except the last (which we
  // slice to fft_shape[-1] / 2 + 1).
  Tensor temp(DataTypeToEnum<complex64>::v(), temp_shape);
  auto full_fft = temp.flat_inner_dims<complex64, FFTRank + 1>();

  // Calculate the starting point and range of the source of
  // negative frequency part.
  auto neg_sizes = in_dims;
  neg_sizes[FFTRank] = fft_shape[FFTRank - 1] - in_dims[FFTRank];
  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> neg_target_indices;
  neg_target_indices[FFTRank] = in_dims[FFTRank];

  const Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> zero_start_indices;
  Eigen::DSizes<Eigen::DenseIndex, FFTRank + 1> neg_start_indices;
  neg_start_indices[FFTRank] = 1;

  full_fft.slice(zero_start_indices, in_dims).device(device) = input;

  // First, conduct IFFTs on outer dimensions. We save computation (and
  // avoid touching uninitialized memory) by slicing full_fft to the
  // subregion we wrote input to.
  if (FFTRank > 1) {
    const auto outer_axes =
        Eigen::ArrayXi::LinSpaced(FFTRank - 1, 1, FFTRank - 1);
    full_fft.slice(zero_start_indices, in_dims).device(device) =
        full_fft.slice(zero_start_indices, in_dims)
            .template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(outer_axes);
  }

  // Reconstruct the full FFT by appending reversed and conjugated
  // spectrum as the negative frequency part.
  Eigen::array<bool, FFTRank + 1> reverse_last_axis;
  for (auto i = 0; i <= FFTRank; i++) {
    reverse_last_axis[i] = i == FFTRank;
  }

  if (neg_sizes[FFTRank] != 0) {
    full_fft.slice(neg_target_indices, neg_sizes).device(device) =
        full_fft.slice(neg_start_indices, neg_sizes)
            .reverse(reverse_last_axis)
            .conjugate();
  }

  auto inner_axis = Eigen::array<int, 1>{FFTRank};
  output.device(device) =
      full_fft.template fft<Eigen::RealPart, Eigen::FFT_REVERSE>(inner_axis);
}

template <int FFTRank, typename EigenDevice>
void EigenFftWithRank(const EigenDevice& device, void* out, void* operand,
                      int32 fft_type, int64 input_batch, int64 fft_length0,
                      int64 fft_length1, int64 fft_length2) {
  CHECK(::xla::FftType_IsValid(fft_type)) << fft_type;
  switch (fft_type) {
    case ::xla::FftType::FFT:
      EigenFftC2C<true, FFTRank, EigenDevice>(
          device, static_cast<complex64*>(out),
          static_cast<complex64*>(operand), input_batch, fft_length0,
          fft_length1, fft_length2);
      break;
    case ::xla::FftType::IFFT:
      EigenFftC2C<false, FFTRank, EigenDevice>(
          device, static_cast<complex64*>(out),
          static_cast<complex64*>(operand), input_batch, fft_length0,
          fft_length1, fft_length2);
      break;
    case ::xla::FftType::RFFT:
      EigenFftR2C<FFTRank, EigenDevice>(
          device, static_cast<complex64*>(out), static_cast<float*>(operand),
          input_batch, fft_length0, fft_length1, fft_length2);
      break;
    case ::xla::FftType::IRFFT:
      EigenFftC2R<FFTRank, EigenDevice>(
          device, static_cast<float*>(out), static_cast<complex64*>(operand),
          input_batch, fft_length0, fft_length1, fft_length2);
      break;
    default:
      LOG(FATAL) << "Unsupported FFT type: " << fft_type;
  }
}

}  // namespace internal

template <typename EigenDevice>
void EigenFftImpl(const EigenDevice& device, void* out, void* operand,
                  int32 fft_type, int32 fft_rank, int64 input_batch,
                  int64 fft_length0, int64 fft_length1, int64 fft_length2) {
  switch (fft_rank) {
    case 1:
      internal::EigenFftWithRank<1, EigenDevice>(
          device, out, operand, fft_type, input_batch, fft_length0, 0, 0);
      break;
    case 2:
      internal::EigenFftWithRank<2, EigenDevice>(device, out, operand, fft_type,
                                                 input_batch, fft_length0,
                                                 fft_length1, 0);
      break;
    case 3:
      internal::EigenFftWithRank<3, EigenDevice>(device, out, operand, fft_type,
                                                 input_batch, fft_length0,
                                                 fft_length1, fft_length2);
      break;
    default:
      LOG(FATAL) << "Unsupported FFT rank " << fft_rank;
  }
}

}  // namespace xla
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_FFT_IMPL_H_
