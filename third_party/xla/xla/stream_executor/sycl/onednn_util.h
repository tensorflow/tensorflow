/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_SYCL_ONEDNN_UTIL_H_
#define XLA_STREAM_EXECUTOR_SYCL_ONEDNN_UTIL_H_

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/str_util.h"

namespace stream_executor {
namespace sycl {

// Computes row-major strides where stride[i] = product(dims[i+1:]).
// Example: dims=[2, 3, 4] -> strides=[12, 4, 1]
dnnl::memory::dims ComputeRowMajorStrides(
    const dnnl::memory::dims& dims_tf_order);

// Thread-safe cache of oneDNN engines per SYCL queue.
dnnl::engine FindOrCreateEngine(::sycl::queue* stream);

// Reads XLA_FP32_MATH_MODE environment variable (fp32/tf32/bf32).
// Returns corresponding dnnl::fpmath_mode. Fatal error if unsupported.
dnnl::fpmath_mode GetFP32MathMode();

// Creates oneDNN memory for GPU or CPU. If data_handle is nullptr,
// allocates new memory; otherwise wraps existing memory.
dnnl::memory CreateDnnlMemory(const dnnl::memory::desc& md,
                              const dnnl::engine& engine,
                              void* data_handle = nullptr);
}  // namespace sycl
}  // namespace stream_executor
#endif  // XLA_STREAM_EXECUTOR_SYCL_ONEDNN_UTIL_H_
