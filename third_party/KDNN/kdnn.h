/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Minimal public C-API header for the KDNN (Kunpeng Deep Neural Network)
// library distributed by Huawei as part of openEuler's KAIL BoostKit.
//
// This header is INTENTIONALLY a thin wrapper. It exposes only the
// 7-8 entry points that the TensorFlow KDNN kernel templates need.
// The full KDNN API surface (matmul, softmax, sparse, RNN, etc.) is
// much larger and is *not* vendored here yet — each op is added as a
// separate PR with its own benchmark.

#ifndef TENSORFLOW_THIRD_PARTY_KDNN_KDNN_H_
#define TENSORFLOW_THIRD_PARTY_KDNN_KDNN_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a KDNN runtime context. Lifetime is managed by the
// caller; typically one per (process, device) pair.
typedef struct kdnn_context kdnn_context_t;

// Element-wise / activation function tags. The numeric values are
// stable; new tags are added at the end.
typedef enum {
  KDNN_ACT_SIGMOID = 0,
  KDNN_ACT_TANH = 1,
  KDNN_ACT_RELU = 2,
  KDNN_ACT_GELU = 3,
  KDNN_ACT_SWISH = 4,
} kdnn_activation_t;

// Data type tags. Mirrors the layout used by KAIL.
typedef enum {
  KDNN_DT_FLOAT = 0,
  KDNN_DT_BFLOAT16 = 1,
  KDNN_DT_HALF = 2,
} kdnn_data_type_t;

// Layout tag. KDNN is row-major; we always pass NHWC-equivalent row-major
// contiguous tensors and the library handles any internal blocking.
typedef enum {
  KDNN_LAYOUT_ROW_MAJOR = 0,
} kdnn_layout_t;

// Status codes returned by every KDNN entry point. KDNN_OK (== 0) is
// success; any non-zero value indicates failure and the caller MAY
// query kdnn_get_last_error() for a human-readable string.
typedef enum {
  KDNN_OK = 0,
  KDNN_ERR_INVALID_ARG = 1,
  KDNN_ERR_UNSUPPORTED = 2,
  KDNN_ERR_RUNTIME = 3,
  KDNN_ERR_OOM = 4,
} kdnn_status_t;

// Library lifecycle.
kdnn_status_t kdnn_create(kdnn_context_t** ctx);
void kdnn_destroy(kdnn_context_t* ctx);
const char* kdnn_get_last_error(const kdnn_context_t* ctx);
int kdnn_get_version(void);  // returns e.g. 0x00010000 for 1.0

// Element-wise activation: y = activation(x), element-wise,
// in-place (x == y) is allowed.
//
// Returns KDNN_OK on success. Returns KDNN_ERR_UNSUPPORTED if the
// requested activation / dtype combination is not implemented in the
// loaded library — the kernel MUST fall back to the native TF op in
// that case.
kdnn_status_t kdnn_apply_activation(
    kdnn_context_t* ctx,
    kdnn_activation_t activation,
    kdnn_data_type_t dtype,
    kdnn_layout_t layout,
    void* x,
    void* y,
    size_t n);

// Returns 1 if the given activation+dt combination is supported on the
// current CPU, 0 otherwise. Cheaper than kdnn_apply_activation; the
// kernel uses this in the shape-inference path to decide whether to
// emit the rewrite.
int kdnn_activation_supported(
    kdnn_activation_t activation,
    kdnn_data_type_t dtype);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_THIRD_PARTY_KDNN_KDNN_H_
