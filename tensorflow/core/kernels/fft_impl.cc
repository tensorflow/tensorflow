/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/fft_impl.h"  // NOLINT: declarations.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "ducc/google/threading.h"  // from @ducc
#include "ducc/src/ducc0/fft/fft.h"  // from @ducc
#include "ducc/src/ducc0/fft/fft1d_impl.h"  // from @ducc  // NOLINT: DUCC definitions.
#include "ducc/src/ducc0/fft/fftnd_impl.h"  // from @ducc  // NOLINT: DUCC definitions.
#include "ducc/src/ducc0/infra/mav.h"  // from @ducc
#include "ducc/src/ducc0/infra/threading.h"  // from @ducc
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/tsl/framework/numeric_types.h"

namespace tensorflow {
namespace internal {

using CPUDevice = Eigen::ThreadPoolDevice;

template <>
absl::Status FftImpl<CPUDevice>(const CPUDevice& device, const Tensor& in,
                                Tensor* out, const uint64_t* fft_shape,
                                const std::vector<size_t>& axes, bool forward) {
  const size_t fft_rank = axes.size();
  ducc0::fmav_info::shape_t in_shape(in.dims());
  ducc0::fmav_info::stride_t in_stride(in.dims());
  ducc0::fmav_info::shape_t out_shape(out->dims());
  ducc0::fmav_info::stride_t out_stride(out->dims());

  size_t next_stride = 1;
  for (int i = in.dims(); i-- > 0;) {
    in_shape[i] = in.dim_size(i);
    in_stride[i] = next_stride;
    next_stride *= in_shape[i];
  }
  next_stride = 1;
  for (int i = out->dims(); i-- > 0;) {
    out_shape[i] = out->dim_size(i);
    out_stride[i] = next_stride;
    next_stride *= out_shape[i];
  }

  // DUCC doesn't handle the case where fft_size[i] < input_size[i],
  // so manually adjust inputs if required.  If doing irfft, the limit
  // of the last axis is actually fft_size[i]/2 + 1.
  const bool is_iffrt = !(forward || out->dtype() == DT_COMPLEX128 ||
                          out->dtype() == DT_COMPLEX64);
  for (int i = 0; i < fft_rank; ++i) {
    int limit = (is_iffrt && (i == (fft_rank - 1))) ? fft_shape[i] / 2 + 1
                                                    : fft_shape[i];
    if (in_shape[axes[i]] > limit) {
      in_shape[axes[i]] = limit;
    }
  }

  double inv_scale = 1.0;
  for (int i = 0; i < fft_rank; ++i) {
    inv_scale *= out_shape[axes[i]];
  }
  double scale = forward ? 1.0 : 1.0 / inv_scale;

  // Set DUCC to use the current device threadpool.  Since this is a
  // thread-local setting, this is thread-safe.
  ducc0::google::EigenThreadPool thread_pool(*device.getPool());
  ducc0::detail_threading::ScopedUseThreadPool thread_pool_guard(thread_pool);
  size_t nthreads = thread_pool.nthreads();

  try {
    if (in.dtype() == DT_COMPLEX128 && out->dtype() == DT_COMPLEX128) {
      auto input = in.template flat<complex128>();
      auto output = out->template flat<complex128>();
      ducc0::cfmav<std::complex<double>> m_in(input.data(), in_shape,
                                              in_stride);
      ducc0::vfmav<std::complex<double>> m_out(output.data(), out_shape,
                                               out_stride);
      ducc0::c2c<double>(m_in, m_out, axes, forward, scale, nthreads);
    } else if (in.dtype() == DT_COMPLEX64 && out->dtype() == DT_COMPLEX64) {
      auto input = in.flat<complex64>();
      auto output = out->flat<complex64>();
      ducc0::cfmav<std::complex<float>> m_in(input.data(), in_shape, in_stride);
      ducc0::vfmav<std::complex<float>> m_out(output.data(), out_shape,
                                              out_stride);
      ducc0::c2c<float>(m_in, m_out, axes, forward, static_cast<float>(scale),
                        nthreads);
    } else if (in.dtype() == DT_DOUBLE && out->dtype() == DT_COMPLEX128 &&
               forward) {
      auto input = in.flat<double>();
      auto output = out->flat<complex128>();
      ducc0::cfmav<double> m_in(input.data(), in_shape, in_stride);
      ducc0::vfmav<std::complex<double>> m_out(output.data(), out_shape,
                                               out_stride);
      ducc0::r2c<double>(m_in, m_out, axes, forward, scale, nthreads);
    } else if (in.dtype() == DT_FLOAT && out->dtype() == DT_COMPLEX64 &&
               forward) {
      auto input = in.flat<float>();
      auto output = out->flat<complex64>();
      ducc0::cfmav<float> m_in(input.data(), in_shape, in_stride);
      ducc0::vfmav<std::complex<float>> m_out(output.data(), out_shape,
                                              out_stride);
      ducc0::r2c<float>(m_in, m_out, axes, forward, static_cast<float>(scale),
                        nthreads);
    } else if (in.dtype() == DT_COMPLEX128 && out->dtype() == DT_DOUBLE &&
               !forward) {
      auto input = in.flat<complex128>();
      auto output = out->flat<double>();
      ducc0::cfmav<std::complex<double>> m_in(input.data(), in_shape,
                                              in_stride);
      ducc0::vfmav<double> m_out(output.data(), out_shape, out_stride);
      ducc0::c2r<double>(m_in, m_out, axes, forward, scale, nthreads);
    } else if (in.dtype() == DT_COMPLEX64 && out->dtype() == DT_FLOAT &&
               !forward) {
      auto input = in.flat<complex64>();
      auto output = out->flat<float>();
      ducc0::cfmav<std::complex<float>> m_in(input.data(), in_shape, in_stride);
      ducc0::vfmav<float> m_out(output.data(), out_shape, out_stride);
      ducc0::c2r<float>(m_in, m_out, axes, forward, static_cast<float>(scale),
                        nthreads);
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid FFT parameters, in.dtype=", in.dtype(),
                       ", out->dtype=", out->dtype(), ", forward=", forward));
    }
  } catch (const std::runtime_error& ex) {
    return absl::InternalError(ex.what());
  } catch (const std::invalid_argument& ex) {
    return absl::InvalidArgumentError(ex.what());
  }
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace tensorflow
