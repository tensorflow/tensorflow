/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Make this file empty (or nearly empty) so that it can be compiled even when
// libxsmm is not available.

#ifndef TENSORFLOW_USE_LIBXSMM
void dummy_xsmm_conv2d_ensure_file_is_not_empty(void);
#else

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/xsmm_conv2d.h"

#include <stdlib.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"

#include "libxsmm/include/libxsmm_cpuid.h"

namespace tensorflow {

// Xsmm*Conv2D are wrappers for libxsmm direct convolutions.

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

static void chk_libxsmm_err(libxsmm_dnn_err_t status, string msg) {
  if (status != LIBXSMM_DNN_SUCCESS) {
    VLOG(0) << msg << " failed: " << libxsmm_dnn_get_error(status);
  }
}

template <typename InputPtr, typename FilterPtr, typename OutputPtr>
static bool CallLibxsmmConvGeneric(OpKernelContext* ctx,
                                   const libxsmm_dnn_conv_desc& desc,
                                   libxsmm_dnn_conv_kind kind, InputPtr input,
                                   FilterPtr filter, OutputPtr output) {
  libxsmm_dnn_err_t status;

  libxsmm_dnn_conv_handle* libxsmm_handle;
  libxsmm_handle = libxsmm_dnn_create_conv_handle_check(desc, &status);
  chk_libxsmm_err(status, "Create handle");

  status = libxsmm_dnn_get_codegen_success(libxsmm_handle, kind);
  if (status == LIBXSMM_DNN_WARN_FALLBACK) {
    chk_libxsmm_err(libxsmm_dnn_destroy_conv_handle(libxsmm_handle),
                    "Destroy handle");
    return false;  // Use non-libxsmm code
  }
  // libxsmm_dnn_get_codegen_success can return real errors as well
  chk_libxsmm_err(status, "Check codegen status");

  libxsmm_dnn_buffer* libxsmm_input;
  libxsmm_dnn_buffer* libxsmm_output;
  libxsmm_dnn_filter* libxsmm_filter;

  libxsmm_input = libxsmm_dnn_link_input_buffer_check(
      libxsmm_handle, input, LIBXSMM_DNN_CONV_FORMAT_NHWC_PTR, &status);
  chk_libxsmm_err(status, "Link input buffer");
  libxsmm_output = libxsmm_dnn_link_output_buffer_check(
      libxsmm_handle, output, LIBXSMM_DNN_CONV_FORMAT_NHWC_PTR, &status);
  chk_libxsmm_err(status, "Link output buffer");
  libxsmm_filter = libxsmm_dnn_link_filter_check(
      libxsmm_handle, filter, LIBXSMM_DNN_CONV_FORMAT_RSCK_PTR, &status);
  chk_libxsmm_err(status, "Link filter");

  chk_libxsmm_err(libxsmm_dnn_zero_buffer(libxsmm_output), "Zero output");

  chk_libxsmm_err(libxsmm_dnn_bind_input_buffer(libxsmm_handle, libxsmm_input),
                  "Bind input");
  chk_libxsmm_err(
      libxsmm_dnn_bind_output_buffer(libxsmm_handle, libxsmm_output),
      "Bind output");
  chk_libxsmm_err(libxsmm_dnn_bind_filter(libxsmm_handle, libxsmm_filter),
                  "Bind filter");

  if (kind == LIBXSMM_DNN_CONV_KIND_BWD) {
    libxsmm_dnn_transpose_filter(libxsmm_handle);
  }

  // TODO(maciejd) We would prefer raw threads instead of threadpool.
  auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
  int num_threads = worker_threads.num_threads;
  BlockingCounter counter(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    worker_threads.workers->Schedule([=, &counter]() {
      chk_libxsmm_err(libxsmm_dnn_convolve_st(libxsmm_handle, kind, 0, i),
                      "Worker");
      counter.DecrementCount();
    });
  }
  counter.Wait();

  chk_libxsmm_err(libxsmm_dnn_destroy_buffer(libxsmm_input), "Destroy input");
  chk_libxsmm_err(libxsmm_dnn_destroy_buffer(libxsmm_output), "Destroy output");
  chk_libxsmm_err(libxsmm_dnn_destroy_filter(libxsmm_filter), "Destroy filter");
  chk_libxsmm_err(libxsmm_dnn_destroy_conv_handle(libxsmm_handle),
                  "Destroy handle");

  return true;  // Succeeded
}

template <typename T>
struct XsmmFwdConv2D<CPUDevice, T> {
  bool operator()(OpKernelContext* ctx, const libxsmm_dnn_conv_desc& desc,
                  const T* input, const T* filter, T* output) {
    return CallLibxsmmConvGeneric(ctx, desc, LIBXSMM_DNN_CONV_KIND_FWD, input,
                                  filter, output);
  }
};

template <typename T>
struct XsmmBkwInputConv2D<CPUDevice, T> {
  bool operator()(OpKernelContext* ctx, const libxsmm_dnn_conv_desc& desc,
                  T* input, const T* filter, const T* output) {
    return CallLibxsmmConvGeneric(ctx, desc, LIBXSMM_DNN_CONV_KIND_BWD, input,
                                  filter, output);
  }
};

template <typename T>
struct XsmmBkwFilterConv2D<CPUDevice, T> {
  bool operator()(OpKernelContext* ctx, const libxsmm_dnn_conv_desc& desc,
                  const T* input, T* filter, const T* output) {
    return CallLibxsmmConvGeneric(ctx, desc, LIBXSMM_DNN_CONV_KIND_UPD, input,
                                  filter, output);
  }
};

}  // namespace functor

template struct functor::XsmmFwdConv2D<CPUDevice, float>;
template struct functor::XsmmBkwInputConv2D<CPUDevice, float>;
template struct functor::XsmmBkwFilterConv2D<CPUDevice, float>;

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_LIBXSMM
