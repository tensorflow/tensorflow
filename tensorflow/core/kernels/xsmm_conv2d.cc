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
void dummy_xsmm_conv2d_ensure_file_is_not_empty();
#else

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/xsmm_conv2d.h"

#include <stdlib.h>
#include <cstring>
#if 0
#include <omp.h>
#endif

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"

#include "libxsmm_main.h"  // TODO(bsteiner): API to avoid incl. header from src/
#include "include/libxsmm_cpuid.h"
#include "include/libxsmm_malloc.h"

namespace tensorflow {

// Xsmm*Conv2D are wrappers for libxsmm direct convolutions.

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

static void chk_libxsmm_err(libxsmm_dnn_err_t status, string msg) {
  if (status != LIBXSMM_DNN_SUCCESS) {
    VLOG(0) << msg << " failed: " << libxsmm_dnn_get_error(status);
  }
}

LIBXSMM_INLINE void copy_RSCK_to_custom(const float* rsck, float* kcrs, int R,
                                        int S, int C, int K, int blocksifm,
                                        int blocksofm, int ifmblock,
                                        int ofmblock, int start, int end) {
  LIBXSMM_VLA_DECL(4, const float, input, rsck, S, C, K);
  LIBXSMM_VLA_DECL(6, float, output, kcrs, blocksifm, R, S, ifmblock, ofmblock);
  int r, s, k, c, v1, v2;

  for (k = start; k < end; k++) {
    for (c = 0; c < blocksifm; c++) {
      for (r = 0; r < R; r++) {
        for (s = 0; s < S; s++) {
          for (v1 = c * ifmblock; v1 < std::min(C, (c + 1) * ifmblock); v1++) {
            for (v2 = k * ofmblock; v2 < std::min(K, (k + 1) * ofmblock); v2++)
              LIBXSMM_VLA_ACCESS(6, output, k, c, r, s, v1 - c * ifmblock,
                                 v2 - k * ofmblock, blocksifm, R, S, ifmblock,
                                 ofmblock) =
                  LIBXSMM_VLA_ACCESS(4, input, r, s, v1, v2, S, C, K);
            for (v2 = K; v2 < (k + 1) * ofmblock; v2++)
              LIBXSMM_VLA_ACCESS(6, output, k, c, r, s, v1 - c * ifmblock,
                                 v2 - k * ofmblock, blocksifm, R, S, ifmblock,
                                 ofmblock) = 0.0f;
          }
          for (v1 = C; v1 < (c + 1) * ifmblock; v1++) {
            for (v2 = k * ofmblock; v2 < (k + 1) * ofmblock; v2++)
              LIBXSMM_VLA_ACCESS(6, output, k, c, r, s, v1 - c * ifmblock,
                                 v2 - k * ofmblock, blocksifm, R, S, ifmblock,
                                 ofmblock) = 0.0f;
          }
        }
      }
    }
  }
}

class libxsmm_dnn_conv_desc_wrap {
 public:
  const libxsmm_dnn_conv_desc d;

  libxsmm_dnn_conv_desc_wrap(const libxsmm_dnn_conv_desc& d_) : d(d_) {}
  bool operator==(const libxsmm_dnn_conv_desc_wrap& w) const {
    return (d.N == w.d.N && d.C == w.d.C && d.H == w.d.H && d.W == w.d.W &&
            d.K == w.d.K && d.R == w.d.R && d.S == w.d.S && d.u == w.d.u &&
            d.v == w.d.v && d.pad_h == w.d.pad_h && d.pad_w == w.d.pad_w);
  }
};

struct HashFunction {
  std::size_t operator()(const libxsmm_dnn_conv_desc_wrap& w) const {
    return libxsmm_hash(&w.d, sizeof(w.d), 25071975);
  }
};

class handles {
 public:
  libxsmm_dnn_layer* find(const libxsmm_dnn_conv_desc_wrap& w) {
    std::unordered_map<libxsmm_dnn_conv_desc_wrap, libxsmm_dnn_layer*,
                       HashFunction>::iterator i = libxsmm_handles.find(w);
    if (i == libxsmm_handles.end()) {
      libxsmm_dnn_err_t status;
      libxsmm_dnn_layer* libxsmm_handle =
          libxsmm_dnn_create_conv_layer(w.d, &status);
      chk_libxsmm_err(status, "Create handle");
      libxsmm_handles.insert(std::make_pair(w, libxsmm_handle));
      return libxsmm_handle;
    } else {
      return i->second;
    }
  }
  ~handles() {
    std::unordered_map<libxsmm_dnn_conv_desc_wrap, libxsmm_dnn_layer*,
                       HashFunction>::iterator i;
    for (i = libxsmm_handles.begin(); i != libxsmm_handles.end(); i++)
      chk_libxsmm_err(libxsmm_dnn_destroy_conv_layer(i->second),
                      "Destroy handle");
  }

 private:
  std::unordered_map<libxsmm_dnn_conv_desc_wrap, libxsmm_dnn_layer*,
                     HashFunction>
      libxsmm_handles;
};

static handles libxsmm_handles;

// #define LIBXSMM_DETAILED_TIMING

template <typename InputPtr, typename FilterPtr, typename OutputPtr>
static bool CallLibxsmmConvGeneric(OpKernelContext* ctx,
                                   const libxsmm_dnn_conv_desc& desc,
                                   libxsmm_dnn_compute_kind kind,
                                   InputPtr input, FilterPtr filter,
                                   OutputPtr output) {
#if defined(LIBXSMM_DETAILED_TIMING)
  unsigned long long l_tick1, l_tick2, l_tick3, l_tick4, l_tick5, l_tick6,
      l_tick7, l_tick8, l_tick9, l_tick10;
  l_tick1 = libxsmm_timer_tick();
#endif
  // setup scoped allocator, which adopts the allocator from the context
  const libxsmm_tf_allocator<libxsmm_scratch_allocator> tf_allocator(*ctx);
  libxsmm_dnn_err_t status;
  libxsmm_dnn_layer* libxsmm_handle;
  libxsmm_dnn_conv_desc_wrap w(desc);
  void* scratch;

  // if(kind == LIBXSMM_DNN_COMPUTE_KIND_FWD)
  libxsmm_handle = libxsmm_handles.find(w);
  // else{
  //  libxsmm_handle = libxsmm_dnn_create_conv_layer(desc, &status);
  //  chk_libxsmm_err(status, "Create handle");
  //}

  status = libxsmm_dnn_get_codegen_success(libxsmm_handle, kind);
  if (status == LIBXSMM_DNN_WARN_FALLBACK) {
    return false;  // Use non-libxsmm code
  }
  chk_libxsmm_err(status, "Check codegen status");

  libxsmm_dnn_buffer* libxsmm_input;
  libxsmm_dnn_buffer* libxsmm_output;
  libxsmm_dnn_filter* libxsmm_filter;

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick2 = libxsmm_timer_tick();
#endif

  int ifmblock = (libxsmm_handle->ifmblock);
  int ofmblock = (libxsmm_handle->ofmblock);

  int blocksifm =
      desc.C % ifmblock == 0 ? desc.C / ifmblock : desc.C / ifmblock + 1;
  int blocksofm =
      desc.K % ofmblock == 0 ? desc.K / ofmblock : desc.K / ofmblock + 1;
  float* native_filter =
      (float*)libxsmm_aligned_scratch(blocksofm * blocksifm * desc.R * desc.S *
                                          ifmblock * ofmblock * sizeof(float),
                                      2097152);

  const DeviceBase::CpuWorkerThreads* worker_threads =
      ctx->device()->tensorflow_cpu_worker_threads();

  int num_threads = worker_threads->num_threads;

#if 1
  if (kind == LIBXSMM_DNN_COMPUTE_KIND_FWD ||
      kind == LIBXSMM_DNN_COMPUTE_KIND_BWD) {
    if (blocksofm > num_threads) {
      int work = blocksofm;
      BlockingCounter count(num_threads);
      for (int i = 0; i < num_threads; ++i) {
        worker_threads->workers->Schedule([=, &count]() {
          int start = work / num_threads * i;
          int end = (start + work / num_threads) > work
                        ? work
                        : start + work / num_threads;
          copy_RSCK_to_custom(filter, native_filter, desc.R, desc.S, desc.C,
                              desc.K, blocksifm, blocksofm, ifmblock, ofmblock,
                              start, end);
          count.DecrementCount();
        });
      }
      count.Wait();
    } else {
      int work = blocksofm;
      int num_threads = work;

      BlockingCounter count(num_threads);
      for (int i = 0; i < num_threads; ++i) {
        worker_threads->workers->Schedule([=, &count]() {
          int start = i;
          int end = i + 1;
          copy_RSCK_to_custom(filter, native_filter, desc.R, desc.S, desc.C,
                              desc.K, blocksifm, blocksofm, ifmblock, ofmblock,
                              start, end);
          count.DecrementCount();
        });
      }
      count.Wait();
    }
  } else if (kind == LIBXSMM_DNN_COMPUTE_KIND_UPD) {
    // Added: for weight update
    libxsmm_filter =
        libxsmm_dnn_link_filter(libxsmm_handle, LIBXSMM_DNN_FILTER, filter,
                                LIBXSMM_DNN_TENSOR_FORMAT_RSCK_PTR, &status);
    chk_libxsmm_err(status,
                    "Link filter");  // weight update is in RSCK as
                                     // filter should be returned in RSCK
                                     // format
  }
#else
  memset(native_filter, 0,
         blocksofm * blocksifm * desc.R * desc.S * ifmblock * ofmblock *
             sizeof(float));
#endif

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick3 = libxsmm_timer_tick();
#endif

  libxsmm_input =
      libxsmm_dnn_link_buffer(libxsmm_handle, LIBXSMM_DNN_INPUT, input,
                              LIBXSMM_DNN_TENSOR_FORMAT_NHWC_PTR, &status);
  chk_libxsmm_err(status, "Link input buffer");
  libxsmm_output =
      libxsmm_dnn_link_buffer(libxsmm_handle, LIBXSMM_DNN_OUTPUT, output,
                              LIBXSMM_DNN_TENSOR_FORMAT_NHWC_PTR, &status);
  chk_libxsmm_err(status, "Link output buffer");
  if (kind == LIBXSMM_DNN_COMPUTE_KIND_FWD ||
      kind == LIBXSMM_DNN_COMPUTE_KIND_BWD) {
    libxsmm_filter = libxsmm_dnn_link_filter(
        libxsmm_handle, LIBXSMM_DNN_FILTER, native_filter,
        LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR, &status);
    chk_libxsmm_err(status, "Link filter");
  }
  if (kind == LIBXSMM_DNN_COMPUTE_KIND_FWD) {
    chk_libxsmm_err(libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_input,
                                            LIBXSMM_DNN_REGULAR_INPUT),
                    "Bind input forward");
    chk_libxsmm_err(libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_output,
                                            LIBXSMM_DNN_REGULAR_OUTPUT),
                    "Bind output forward");
    chk_libxsmm_err(libxsmm_dnn_bind_filter(libxsmm_handle, libxsmm_filter,
                                            LIBXSMM_DNN_REGULAR_FILTER),
                    "Bind filter forward");
  } else if (kind == LIBXSMM_DNN_COMPUTE_KIND_BWD) {
    chk_libxsmm_err(libxsmm_dnn_zero_buffer(libxsmm_input), "Zero input");

    chk_libxsmm_err(libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_input,
                                            LIBXSMM_DNN_GRADIENT_INPUT),
                    "Bind input backward");
    chk_libxsmm_err(libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_output,
                                            LIBXSMM_DNN_GRADIENT_OUTPUT),
                    "Bind output backward");
    chk_libxsmm_err(libxsmm_dnn_bind_filter(libxsmm_handle, libxsmm_filter,
                                            LIBXSMM_DNN_REGULAR_FILTER),
                    "Bind filter backward");
  } else if (kind == LIBXSMM_DNN_COMPUTE_KIND_UPD) {
    chk_libxsmm_err(libxsmm_dnn_zero_filter(libxsmm_filter), "Zero filter");

    chk_libxsmm_err(libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_input,
                                            LIBXSMM_DNN_REGULAR_INPUT),
                    "Bind input weight update");
    chk_libxsmm_err(libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_output,
                                            LIBXSMM_DNN_GRADIENT_OUTPUT),
                    "Bind output weight update");
    chk_libxsmm_err(libxsmm_dnn_bind_filter(libxsmm_handle, libxsmm_filter,
                                            LIBXSMM_DNN_GRADIENT_FILTER),
                    "Bind filter weight update");
  } else {
    /* shouldn't happen */
  }

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick4 = libxsmm_timer_tick();
#endif

  /* bind scratch */
  scratch = (void*)libxsmm_aligned_scratch(
      libxsmm_dnn_get_scratch_size(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL,
                                   &status),
      2097152);
  chk_libxsmm_err(status, "scratch allocation");
  chk_libxsmm_err(libxsmm_dnn_bind_scratch(
                      libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch),
                  "binding scratch");

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick5 = libxsmm_timer_tick();
#endif

  if (kind == LIBXSMM_DNN_COMPUTE_KIND_BWD) {
    libxsmm_dnn_transpose_filter(libxsmm_handle, LIBXSMM_DNN_FILTER);
  }

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick6 = libxsmm_timer_tick();
#endif

#if 1
  BlockingCounter counter(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    worker_threads->workers->Schedule([=, &counter]() {
      chk_libxsmm_err(libxsmm_dnn_execute_st(libxsmm_handle, kind, 0, i),
                      "Worker");
      counter.DecrementCount();
    });
  }
  counter.Wait();
#else
#pragma omp parallel
  {
    chk_libxsmm_err(
        libxsmm_dnn_execute_st(libxsmm_handle, kind, 0, omp_get_thread_num()),
        "Worker");
  }
#endif

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick7 = libxsmm_timer_tick();
#endif

  if (kind == LIBXSMM_DNN_COMPUTE_KIND_UPD) {
    libxsmm_dnn_reduce_wu_filters(libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER);
  }

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick8 = libxsmm_timer_tick();
#endif

  /* clean up */
  chk_libxsmm_err(
      libxsmm_dnn_release_scratch(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL),
      "release scratch");
  if (kind == LIBXSMM_DNN_COMPUTE_KIND_FWD) {
    chk_libxsmm_err(
        libxsmm_dnn_release_buffer(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT),
        "release input");
    chk_libxsmm_err(
        libxsmm_dnn_release_buffer(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT),
        "release output");
    chk_libxsmm_err(
        libxsmm_dnn_release_filter(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER),
        "release filter");
  } else if (kind == LIBXSMM_DNN_COMPUTE_KIND_BWD) {
    chk_libxsmm_err(
        libxsmm_dnn_release_buffer(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT),
        "release input");
    chk_libxsmm_err(
        libxsmm_dnn_release_buffer(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT),
        "release output");
    chk_libxsmm_err(
        libxsmm_dnn_release_filter(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER),
        "release filter");
  } else if (kind == LIBXSMM_DNN_COMPUTE_KIND_UPD) {
    chk_libxsmm_err(
        libxsmm_dnn_release_buffer(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT),
        "release input");
    chk_libxsmm_err(
        libxsmm_dnn_release_buffer(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT),
        "release output");
    chk_libxsmm_err(
        libxsmm_dnn_release_filter(libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER),
        "release filter");
  } else {
    /* shouldn't happen */
  }
  chk_libxsmm_err(libxsmm_dnn_destroy_buffer(libxsmm_input), "Destroy input");
  chk_libxsmm_err(libxsmm_dnn_destroy_buffer(libxsmm_output), "Destroy output");
  chk_libxsmm_err(libxsmm_dnn_destroy_filter(libxsmm_filter), "Destroy filter");

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick9 = libxsmm_timer_tick();
#endif

  // if(kind != LIBXSMM_DNN_COMPUTE_KIND_FWD)
  // chk_libxsmm_err(libxsmm_dnn_destroy_conv_layer(libxsmm_handle),
  //               "Destroy handle");

  libxsmm_free(native_filter);
  libxsmm_free(scratch);

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick10 = libxsmm_timer_tick();
  printf(
      "time for convolution (%i, %i, %i, %i, %i): %f, %f, %f, %f, %f, %f, %f, "
      "%f, %f, %f\n",
      desc.N, desc.C, desc.K, desc.R, desc.S,
      libxsmm_timer_duration(l_tick1, l_tick2),
      libxsmm_timer_duration(l_tick2, l_tick3),
      libxsmm_timer_duration(l_tick3, l_tick4),
      libxsmm_timer_duration(l_tick4, l_tick5),
      libxsmm_timer_duration(l_tick5, l_tick6),
      libxsmm_timer_duration(l_tick6, l_tick7),
      libxsmm_timer_duration(l_tick7, l_tick8),
      libxsmm_timer_duration(l_tick8, l_tick9),
      libxsmm_timer_duration(l_tick9, l_tick10),
      libxsmm_timer_duration(l_tick1, l_tick10));
#endif

  return true;  // Succeeded
}

template <typename T>
struct XsmmFwdConv2D<CPUDevice, T> {
  bool operator()(OpKernelContext* ctx, const libxsmm_dnn_conv_desc& desc,
                  const T* input, const T* filter, T* output) {
    return CallLibxsmmConvGeneric(ctx, desc, LIBXSMM_DNN_COMPUTE_KIND_FWD,
                                  input, filter, output);
  }
};

template <typename T>
struct XsmmBkwInputConv2D<CPUDevice, T> {
  bool operator()(OpKernelContext* ctx, const libxsmm_dnn_conv_desc& desc,
                  T* input, const T* filter, const T* output) {
    return CallLibxsmmConvGeneric(ctx, desc, LIBXSMM_DNN_COMPUTE_KIND_BWD,
                                  input, filter, output);
  }
};

template <typename T>
struct XsmmBkwFilterConv2D<CPUDevice, T> {
  bool operator()(OpKernelContext* ctx, const libxsmm_dnn_conv_desc& desc,
                  const T* input, T* filter, const T* output) {
    return CallLibxsmmConvGeneric(ctx, desc, LIBXSMM_DNN_COMPUTE_KIND_UPD,
                                  input, filter, output);
  }
};

}  // namespace functor

template struct functor::XsmmFwdConv2D<CPUDevice, float>;
template struct functor::XsmmBkwInputConv2D<CPUDevice, float>;
template struct functor::XsmmBkwFilterConv2D<CPUDevice, float>;

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_LIBXSMM
