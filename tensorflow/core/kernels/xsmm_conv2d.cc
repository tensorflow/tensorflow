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

// XsmmConv2D is a wrapper for libxsmm direct convolutions.

// Returns true if convolution can be computed efficiently by XsmmConv2D,
// returns false otherwise.
bool CanUseXsmmConv2D(const libxsmm_dnn_conv_desc& desc,
                      TensorFormat data_format) {
  int VECTOR_SIZE;
  int arch = libxsmm_cpuid_x86();

  if (arch == LIBXSMM_X86_AVX512_CORE) {
    VECTOR_SIZE = 16;
  } else if (arch == LIBXSMM_X86_AVX2) {
    VECTOR_SIZE = 8;
  } else {
    VLOG(1) << "Cannot use XSMM convolutions: unsupported architecture!";
    return false;
  }

  if (data_format != FORMAT_NHWC) {
    VLOG(1) << "Cannot use XSMM convolutions: unsupported format!";
    return false;
  }
  if (desc.pad_h_in != 0 || desc.pad_w_in != 0) {
    VLOG(1) << "Cannot use XSMM convolutions: unsupported padding!";
    return false;
  }
  if (desc.K % VECTOR_SIZE != 0) {
    VLOG(1) << "Cannot use XSMM convolutions: output features count not"
               " divisible by vector size!";
    return false;
  }
  VLOG(2) << "Can use XSMM convolutions.";
  return true;
}

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

template <typename T>
struct XsmmConv2D<CPUDevice, T> {
  static void chkerr(libxsmm_dnn_err_t status, string msg) {
    if (status != LIBXSMM_DNN_SUCCESS) {
      VLOG(0) << msg << " failed: " << libxsmm_dnn_get_error(status);
    }
  }

  void operator()(OpKernelContext* ctx, const libxsmm_dnn_conv_desc& desc,
                  const T* input, const T* filter, T* output) {
    libxsmm_dnn_err_t status;

    libxsmm_dnn_conv_handle* libxsmm_handle;
    libxsmm_handle = libxsmm_dnn_create_conv_handle_check(desc, &status);
    chkerr(status, "Create handle");

    libxsmm_dnn_buffer* libxsmm_input;
    libxsmm_dnn_buffer* libxsmm_output;
    libxsmm_dnn_filter* libxsmm_filter;

    libxsmm_input = libxsmm_dnn_link_input_buffer_check(
        libxsmm_handle, input, LIBXSMM_DNN_CONV_FORMAT_NHWC_PTR, &status);
    chkerr(status, "Link input buffer");
    libxsmm_output = libxsmm_dnn_link_output_buffer_check(
        libxsmm_handle, output, LIBXSMM_DNN_CONV_FORMAT_NHWC_PTR, &status);
    chkerr(status, "Link output buffer");
    libxsmm_filter = libxsmm_dnn_link_filter_check(
        libxsmm_handle, filter, LIBXSMM_DNN_CONV_FORMAT_RSCK_PTR, &status);
    chkerr(status, "Link filter");

    chkerr(libxsmm_dnn_zero_buffer(libxsmm_output), "Zero output");

    chkerr(libxsmm_dnn_bind_input_buffer(libxsmm_handle, libxsmm_input),
           "Bind input");
    chkerr(libxsmm_dnn_bind_output_buffer(libxsmm_handle, libxsmm_output),
           "Bind output");
    chkerr(libxsmm_dnn_bind_filter(libxsmm_handle, libxsmm_filter),
           "Bind filter");

    // TODO(maciejd) We would prefer raw threads instead of threadpool.
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    int num_threads = worker_threads.num_threads;
    BlockingCounter counter(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      worker_threads.workers->Schedule([=, &counter]() {
        chkerr(libxsmm_dnn_convolve_st(libxsmm_handle,
                                       LIBXSMM_DNN_CONV_KIND_FWD, 0, i),
               "Worker");
        counter.DecrementCount();
      });
    }
    counter.Wait();

    chkerr(libxsmm_dnn_destroy_buffer(libxsmm_input), "Destroy input");
    chkerr(libxsmm_dnn_destroy_buffer(libxsmm_output), "Destroy output");
    chkerr(libxsmm_dnn_destroy_filter(libxsmm_filter), "Destroy filter");
    chkerr(libxsmm_dnn_destroy_conv_handle(libxsmm_handle), "Destory handle");
  }
};

}  // namespace functor

template struct functor::XsmmConv2D<CPUDevice, float>;

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_LIBXSMM
