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

#ifndef TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS
void dummy_xsmm_conv2d_ensure_file_is_not_empty();
#else

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/xsmm_conv2d.h"

#include <stdlib.h>
#include <cstring>
#if defined(_OPENMP) && defined(LIBXSMM_USE_OPENMP)
#include <omp.h>
#endif

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"

#include "include/libxsmm_cpuid.h"
#include "include/libxsmm_malloc.h"
#include "src/libxsmm_main.h"  // TODO(bsteiner): API to avoid incl. header from src/

#define CHECK_LIBXSMM(CONDITION_OK, MESSAGE) \
  if (!(CONDITION_OK)) VLOG(0) << (MESSAGE)
#define CHECK_LIBXSMM_DNN(STATUS, MESSAGE)                \
  CHECK_LIBXSMM(LIBXSMM_DNN_SUCCESS == (STATUS), MESSAGE) \
      << " failed: " << libxsmm_dnn_get_error(STATUS);

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

struct libxsmm_dnn_registry_key {
  const libxsmm_dnn_conv_desc descriptor;
  libxsmm_dnn_registry_key(const libxsmm_dnn_conv_desc& desc_)
      : descriptor(desc_) {}
  bool operator==(const libxsmm_dnn_registry_key& regkey) const {
    return 0 == memcmp(&descriptor, &regkey.descriptor, sizeof(descriptor));
  }
};

struct HashFunction {
  std::size_t operator()(const libxsmm_dnn_registry_key& regkey) const {
    return libxsmm_hash(&regkey.descriptor, sizeof(regkey.descriptor),
                        25071975);
  }
};

struct libxsmm_dnn_registry_value {
  libxsmm_dnn_tensor_datalayout* layout_input;
  libxsmm_dnn_tensor_datalayout* layout_filter;
  libxsmm_dnn_tensor_datalayout* layout_output;
  libxsmm_dnn_layer* handle;
};

typedef libxsmm_tf_allocator<libxsmm_scratch_allocator>
    libxsmm_tf_scratch_allocator;

static class libxsmm_dnn_registry_type {
 private:
  typedef std::unordered_map<libxsmm_dnn_registry_key,
                             libxsmm_dnn_registry_value, HashFunction>
      container_type;

 public:
  libxsmm_dnn_registry_type() {
    libxsmm_init(); /* must be first */
#if !defined(LIBXSMM_LOCAL_ALLOC)
    {
      libxsmm_malloc_function malloc_fn;
      libxsmm_free_function free_fn;
      malloc_fn.function = libxsmm_tf_scratch_allocator::malloc;
      free_fn.function = libxsmm_tf_scratch_allocator::free;
      libxsmm_set_scratch_allocator(0 /*context*/, malloc_fn, free_fn);
    }
#endif
    LIBXSMM_LOCK_ATTR_INIT(LIBXSMM_LOCK_RWLOCK, &attr);
    LIBXSMM_LOCK_INIT(LIBXSMM_LOCK_RWLOCK, &lock, &attr);
  }
  ~libxsmm_dnn_registry_type() {
    LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK_RWLOCK, &lock);
    const container_type::const_iterator end = container.end();
    for (container_type::const_iterator i = container.begin(); i != end; ++i) {
      CHECK_LIBXSMM_DNN(
          libxsmm_dnn_destroy_tensor_datalayout(i->second.layout_input),
          "destroy input layout");
      CHECK_LIBXSMM_DNN(
          libxsmm_dnn_destroy_tensor_datalayout(i->second.layout_output),
          "destroy output layout");
      CHECK_LIBXSMM_DNN(
          libxsmm_dnn_destroy_tensor_datalayout(i->second.layout_filter),
          "destroy filter layout");
      CHECK_LIBXSMM_DNN(libxsmm_dnn_destroy_conv_layer(i->second.handle),
                        "destroy handle");
    }
    LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK_RWLOCK, &lock);
    LIBXSMM_LOCK_DESTROY(LIBXSMM_LOCK_RWLOCK, &lock);
    LIBXSMM_LOCK_ATTR_DESTROY(LIBXSMM_LOCK_RWLOCK, &attr);
    libxsmm_finalize();
  }
  libxsmm_dnn_registry_value find(const libxsmm_dnn_registry_key& regkey) {
    container_type::iterator i;
    LIBXSMM_LOCK_ACQREAD(LIBXSMM_LOCK_RWLOCK, &lock);
    i = container.find(regkey);
    LIBXSMM_LOCK_RELREAD(LIBXSMM_LOCK_RWLOCK, &lock);
    if (i == container.end()) {
      libxsmm_dnn_err_t status;
      libxsmm_dnn_registry_value regentry;

      LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK_RWLOCK, &lock);
      i = container.find(regkey);
      if (i == container.end()) {  // re-check after lock acquisition
        regentry.handle =
            libxsmm_dnn_create_conv_layer(regkey.descriptor, &status);
        if (LIBXSMM_DNN_WARN_FALLBACK != status) {
          CHECK_LIBXSMM_DNN(status, "create handle");
        } else {  // warning
          VLOG(1) << libxsmm_dnn_get_error(status);
        }
        regentry.layout_input = libxsmm_dnn_create_tensor_datalayout(
            regentry.handle, LIBXSMM_DNN_INPUT, &status);
        CHECK_LIBXSMM_DNN(status, "create input layout");

        regentry.layout_output = libxsmm_dnn_create_tensor_datalayout(
            regentry.handle, LIBXSMM_DNN_OUTPUT, &status);
        CHECK_LIBXSMM_DNN(status, "create output layout");

        regentry.layout_filter = libxsmm_dnn_create_tensor_datalayout(
            regentry.handle, LIBXSMM_DNN_FILTER, &status);
        CHECK_LIBXSMM_DNN(status, "create filter layout");

        i = container.insert(std::make_pair(regkey, regentry)).first;
      }
      LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK_RWLOCK, &lock);
    }
    return i->second;
  }

 private:
  container_type container;
  LIBXSMM_LOCK_ATTR_TYPE(LIBXSMM_LOCK_RWLOCK) attr;
  LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK_RWLOCK) lock;
} libxsmm_dnn_registry;

// #define LIBXSMM_DETAILED_TIMING

template <typename InputPtr, typename FilterPtr, typename OutputPtr>
static bool CallLibxsmmConvGeneric(OpKernelContext* ctx,
                                   const libxsmm_dnn_conv_desc& desc,
                                   libxsmm_dnn_compute_kind kind,
                                   InputPtr input, FilterPtr filter,
                                   OutputPtr output) {
#if defined(LIBXSMM_DETAILED_TIMING)
  libxsmm_timer_tickint l_tick1;
  libxsmm_timer_tickint l_tick2;
  libxsmm_timer_tickint l_tick3;
  libxsmm_timer_tickint l_tick4;
  libxsmm_timer_tickint l_tick5;
  libxsmm_timer_tickint l_tick6;
  libxsmm_timer_tickint l_tick7;
  libxsmm_timer_tickint l_tick8;
  libxsmm_timer_tickint l_tick9;
  libxsmm_timer_tickint l_tick10;
  l_tick1 = libxsmm_timer_tick();
#endif
#if defined(LIBXSMM_LOCAL_ALLOC)
  // setup scoped allocator, which adopts the allocator of the current context
  const libxsmm_tf_scratch_allocator tf_allocator(*ctx);
#endif
  const libxsmm_dnn_registry_key regkey(desc);
  const libxsmm_dnn_registry_value regentry = libxsmm_dnn_registry.find(regkey);
  libxsmm_dnn_tensor *libxsmm_input, *libxsmm_output, *libxsmm_filter;
  libxsmm_dnn_err_t status;

  status = libxsmm_dnn_get_codegen_success(regentry.handle, kind);
  if (status == LIBXSMM_DNN_WARN_FALLBACK) {
    return false;  // Use non-libxsmm code
  }
  CHECK_LIBXSMM_DNN(status, "code generation");

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick2 = libxsmm_timer_tick();
#endif

  const int ifmblock = regentry.handle->ifmblock;
  const int ofmblock = regentry.handle->ofmblock;

  const int blocksifm =
      (desc.C % ifmblock == 0 ? desc.C / ifmblock : desc.C / ifmblock + 1);
  const int blocksofm =
      (desc.K % ofmblock == 0 ? desc.K / ofmblock : desc.K / ofmblock + 1);

  const size_t filter_size =
      blocksofm * blocksifm * desc.R * desc.S * ifmblock * ofmblock;
  float* const native_filter = (float*)libxsmm_aligned_scratch(
      filter_size * sizeof(float), 2097152 /*alignment*/);

  const DeviceBase::CpuWorkerThreads* const worker_threads =
      ctx->device()->tensorflow_cpu_worker_threads();
  const int num_threads = worker_threads->num_threads;

#if 1
  if (kind == LIBXSMM_DNN_COMPUTE_KIND_FWD ||
      kind == LIBXSMM_DNN_COMPUTE_KIND_BWD) {
    if (blocksofm > num_threads) {
      const int work = blocksofm;
      BlockingCounter count(num_threads);
      for (int i = 0; i < num_threads; ++i) {
        worker_threads->workers->Schedule([=, &count]() {
          const int start = work / num_threads * i;
          const int end = (start + work / num_threads) > work
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
      const int work = blocksofm;
      const int num_tasks = work;

      BlockingCounter count(num_tasks);
      for (int i = 0; i < num_tasks; ++i) {
        worker_threads->workers->Schedule([=, &count]() {
          const int start = i;
          const int end = i + 1;
          copy_RSCK_to_custom(filter, native_filter, desc.R, desc.S, desc.C,
                              desc.K, blocksifm, blocksofm, ifmblock, ofmblock,
                              start, end);
          count.DecrementCount();
        });
      }
      count.Wait();
    }
  } else if (kind == LIBXSMM_DNN_COMPUTE_KIND_UPD) {
    // weight update buffer must be in the right format
    // (LIBXSMM_DNN_TENSOR_FORMAT_RSCK_PTR)
    libxsmm_filter =
        libxsmm_dnn_link_tensor(regentry.layout_filter, filter, &status);
    CHECK_LIBXSMM_DNN(status, "link filter with layout");
  }
#else
  memset(native_filter, 0, filter_size * sizeof(float));
#endif

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick3 = libxsmm_timer_tick();
#endif

  // LIBXSMM_DNN_TENSOR_FORMAT_NHWC_PTR
  libxsmm_input =
      libxsmm_dnn_link_tensor(regentry.layout_input, input, &status);
  CHECK_LIBXSMM_DNN(status, "link input buffer with layout");

  // LIBXSMM_DNN_TENSOR_FORMAT_NHWC_PTR
  libxsmm_output =
      libxsmm_dnn_link_tensor(regentry.layout_output, output, &status);
  CHECK_LIBXSMM_DNN(status, "link output buffer with layout");

  if (kind == LIBXSMM_DNN_COMPUTE_KIND_FWD ||
      kind == LIBXSMM_DNN_COMPUTE_KIND_BWD) {
    // LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR
    libxsmm_filter =
        libxsmm_dnn_link_tensor(regentry.layout_filter, native_filter, &status);
    CHECK_LIBXSMM_DNN(status, "link filter with layout");
    CHECK_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(regentry.handle, libxsmm_filter,
                                              LIBXSMM_DNN_REGULAR_FILTER),
                      "bind filter to handle");
  }
  if (kind == LIBXSMM_DNN_COMPUTE_KIND_FWD) {
    CHECK_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(regentry.handle, libxsmm_input,
                                              LIBXSMM_DNN_REGULAR_INPUT),
                      "bind input forward");
    CHECK_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(regentry.handle, libxsmm_filter,
                                              LIBXSMM_DNN_REGULAR_FILTER),
                      "bind filter forward");
    CHECK_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(regentry.handle, libxsmm_output,
                                              LIBXSMM_DNN_REGULAR_OUTPUT),
                      "bind output forward");
  } else if (kind == LIBXSMM_DNN_COMPUTE_KIND_BWD) {
    CHECK_LIBXSMM_DNN(libxsmm_dnn_zero_tensor(libxsmm_input), "zeroing input");
    CHECK_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(regentry.handle, libxsmm_input,
                                              LIBXSMM_DNN_GRADIENT_INPUT),
                      "bind input backward");
    CHECK_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(regentry.handle, libxsmm_filter,
                                              LIBXSMM_DNN_REGULAR_FILTER),
                      "bind filter backward");
    CHECK_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(regentry.handle, libxsmm_output,
                                              LIBXSMM_DNN_GRADIENT_OUTPUT),
                      "bind output backward");
  } else if (kind == LIBXSMM_DNN_COMPUTE_KIND_UPD) {
    CHECK_LIBXSMM_DNN(libxsmm_dnn_zero_tensor(libxsmm_filter),
                      "zeroing filter");
    CHECK_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(regentry.handle, libxsmm_input,
                                              LIBXSMM_DNN_REGULAR_INPUT),
                      "bind input weight update");
    CHECK_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(regentry.handle, libxsmm_filter,
                                              LIBXSMM_DNN_GRADIENT_FILTER),
                      "bind filter weight update");
    CHECK_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(regentry.handle, libxsmm_output,
                                              LIBXSMM_DNN_GRADIENT_OUTPUT),
                      "bind output weight update");
  } else {
    assert(0 /*should not happen*/);
  }

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick4 = libxsmm_timer_tick();
#endif

  const size_t scratch_size = libxsmm_dnn_get_scratch_size(
      regentry.handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status);
  CHECK_LIBXSMM_DNN(status, "get scratch size");
  void* const scratch =
      libxsmm_aligned_scratch(scratch_size, 2097152 /*alignment*/);
  CHECK_LIBXSMM(0 != scratch, "scratch memory allocation");
  CHECK_LIBXSMM_DNN(libxsmm_dnn_bind_scratch(
                        regentry.handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch),
                    "binding scratch");

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick5 = libxsmm_timer_tick();
#endif

  if (kind == LIBXSMM_DNN_COMPUTE_KIND_BWD) {
    libxsmm_dnn_transpose_filter(regentry.handle, LIBXSMM_DNN_FILTER);
  }

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick6 = libxsmm_timer_tick();
#endif

#if !defined(_OPENMP) || !defined(LIBXSMM_USE_OPENMP)
  BlockingCounter counter(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    worker_threads->workers->Schedule([=, &counter]() {
      CHECK_LIBXSMM_DNN(libxsmm_dnn_execute_st(regentry.handle, kind, 0, i),
                        "worker");
      counter.DecrementCount();
    });
  }
  counter.Wait();
#else
#pragma omp parallel
  {
    CHECK_LIBXSMM_DNN(
        libxsmm_dnn_execute_st(regentry.handle, kind, 0, omp_get_thread_num()),
        "worker");
  }
#endif

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick7 = libxsmm_timer_tick();
#endif

  if (kind == LIBXSMM_DNN_COMPUTE_KIND_UPD) {
    libxsmm_dnn_reduce_wu_filters(regentry.handle, LIBXSMM_DNN_GRADIENT_FILTER);
  }

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick8 = libxsmm_timer_tick();
#endif

  /* clean up */
  CHECK_LIBXSMM_DNN(libxsmm_dnn_release_scratch(regentry.handle,
                                                LIBXSMM_DNN_COMPUTE_KIND_ALL),
                    "release scratch");
  if (kind == LIBXSMM_DNN_COMPUTE_KIND_FWD) {
    CHECK_LIBXSMM_DNN(
        libxsmm_dnn_release_tensor(regentry.handle, LIBXSMM_DNN_REGULAR_INPUT),
        "release input");
    CHECK_LIBXSMM_DNN(
        libxsmm_dnn_release_tensor(regentry.handle, LIBXSMM_DNN_REGULAR_OUTPUT),
        "release output");
    CHECK_LIBXSMM_DNN(
        libxsmm_dnn_release_tensor(regentry.handle, LIBXSMM_DNN_REGULAR_FILTER),
        "release filter");
  } else if (kind == LIBXSMM_DNN_COMPUTE_KIND_BWD) {
    CHECK_LIBXSMM_DNN(
        libxsmm_dnn_release_tensor(regentry.handle, LIBXSMM_DNN_GRADIENT_INPUT),
        "release input");
    CHECK_LIBXSMM_DNN(libxsmm_dnn_release_tensor(regentry.handle,
                                                 LIBXSMM_DNN_GRADIENT_OUTPUT),
                      "release output");
    CHECK_LIBXSMM_DNN(
        libxsmm_dnn_release_tensor(regentry.handle, LIBXSMM_DNN_REGULAR_FILTER),
        "release filter");
  } else if (kind == LIBXSMM_DNN_COMPUTE_KIND_UPD) {
    CHECK_LIBXSMM_DNN(
        libxsmm_dnn_release_tensor(regentry.handle, LIBXSMM_DNN_REGULAR_INPUT),
        "release input");
    CHECK_LIBXSMM_DNN(libxsmm_dnn_release_tensor(regentry.handle,
                                                 LIBXSMM_DNN_GRADIENT_OUTPUT),
                      "release output");
    CHECK_LIBXSMM_DNN(libxsmm_dnn_release_tensor(regentry.handle,
                                                 LIBXSMM_DNN_GRADIENT_FILTER),
                      "release filter");
  } else {
    /* shouldn't happen */
  }
  CHECK_LIBXSMM_DNN(libxsmm_dnn_destroy_tensor(libxsmm_input), "destroy input");
  CHECK_LIBXSMM_DNN(libxsmm_dnn_destroy_tensor(libxsmm_output),
                    "destroy output");
  CHECK_LIBXSMM_DNN(libxsmm_dnn_destroy_tensor(libxsmm_filter),
                    "destroy filter");

#if defined(LIBXSMM_DETAILED_TIMING)
  l_tick9 = libxsmm_timer_tick();
#endif

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

#ifdef TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS
template <typename T>
struct XsmmFwdConv2D<CPUDevice, T> {
  bool operator()(OpKernelContext* ctx, const libxsmm_dnn_conv_desc& desc,
                  const T* input, const T* filter, T* output) {
    return CallLibxsmmConvGeneric(ctx, desc, LIBXSMM_DNN_COMPUTE_KIND_FWD,
                                  input, filter, output);
  }
};
#endif

#ifdef TENSORFLOW_USE_LIBXSMM_BACKWARD_CONVOLUTIONS
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
#endif

}  // namespace functor

template struct functor::XsmmFwdConv2D<CPUDevice, float>;
template struct functor::XsmmBkwInputConv2D<CPUDevice, float>;
template struct functor::XsmmBkwFilterConv2D<CPUDevice, float>;

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_LIBXSMM_CONVOLUTIONS
