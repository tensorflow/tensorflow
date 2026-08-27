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

#include "tensorflow/core/common_runtime/metal/kernels/metal_shader_library.h"

#import <Metal/Metal.h>

#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// Shader source for the kernels this backend does not express in MPSGraph:
// fills, the random generators, and the optimizer updates. Arithmetic lives on
// MPSGraph instead, which broadcasts natively and brings Apple's tuned kernels.
//
// Every params struct here must stay layout-identical to its counterpart in
// the header, and every kernel bounds-checks against `count`, since callers
// use dispatchThreadgroups: and the grid rounds up to whole threadgroups.
constexpr char kShaderSource[] = R"METAL(
#include <metal_stdlib>
using namespace metal;

// ---- fills ----

struct FillParams {
  uint count;
  float value;
  uint pad0;
  uint pad1;
};

// Two variants, because the value's home differs by op. ZerosLike and OnesLike
// fill with a compile-time constant the host already knows, while Fill takes
// its value as a device scalar that must not be read back on the host.
#define TF_METAL_FILL_CONST(NAME, T)                                       \
  kernel void NAME(device T* out [[buffer(0)]],                            \
                   constant FillParams& params [[buffer(1)]],              \
                   uint gid [[thread_position_in_grid]]) {                 \
    if (gid >= params.count) return;                                       \
    out[gid] = static_cast<T>(params.value);                               \
  }

#define TF_METAL_FILL_BUFFER(NAME, T)                                      \
  kernel void NAME(device T* out [[buffer(0)]],                            \
                   device const T* value [[buffer(1)]],                    \
                   constant FillParams& params [[buffer(2)]],              \
                   uint gid [[thread_position_in_grid]]) {                 \
    if (gid >= params.count) return;                                       \
    out[gid] = value[0];                                                   \
  }

TF_METAL_FILL_CONST(tf_fill_const_float, float)
TF_METAL_FILL_CONST(tf_fill_const_half, half)
TF_METAL_FILL_BUFFER(tf_fill_float, float)
TF_METAL_FILL_BUFFER(tf_fill_half, half)

// ---- counter-based random ----

// Philox-style counter RNG.
//
// Counter-based rather than stateful because every element is generated
// independently by its own thread: the value depends only on the seed and the
// element index, so there is no sequence to carry between threads and no
// ordering to get wrong. Successive calls to the same op differ because the
// host feeds a fresh counter each time.
//
// This is the 4x32-10 Philox bijection, the same generator family TensorFlow's
// own stateless random ops use.
struct RandomParams {
  uint count;
  uint seed_lo;
  uint seed_hi;
  uint counter;
};

inline uint2 tf_mulhilo(uint a, uint b) {
  const uint lo = a * b;
  const uint hi = mulhi(a, b);
  return uint2(hi, lo);
}

inline uint4 tf_philox_round(uint4 ctr, uint2 key) {
  const uint2 p0 = tf_mulhilo(0xD2511F53u, ctr.x);
  const uint2 p1 = tf_mulhilo(0xCD9E8D57u, ctr.z);
  return uint4(p1.x ^ ctr.y ^ key.x, p1.y, p0.x ^ ctr.w ^ key.y, p0.y);
}

inline uint4 tf_philox(uint4 ctr, uint2 key) {
  for (uint i = 0; i < 10; ++i) {
    ctr = tf_philox_round(ctr, key);
    key.x += 0x9E3779B9u;
    key.y += 0xBB67AE85u;
  }
  return ctr;
}

// Uniform in [0, 1), from the top 24 bits so the spacing is exactly 2^-24.
inline float tf_to_unit_float(uint bits) {
  return static_cast<float>(bits >> 8) * (1.0f / 16777216.0f);
}

inline uint4 tf_random_bits(constant RandomParams& params, uint gid) {
  return tf_philox(uint4(gid, params.counter, 0u, 0u),
                   uint2(params.seed_lo, params.seed_hi));
}

kernel void tf_random_uniform_float(device float* out [[buffer(0)]],
                                    constant RandomParams& params [[buffer(1)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  out[gid] = tf_to_unit_float(tf_random_bits(params, gid).x);
}

// Box-Muller. Both normals come from one Philox call, and only the first is
// kept: generating pairs would make a thread responsible for two output
// elements and complicate the bounds check for no real gain.
inline float tf_box_muller(uint4 bits) {
  // Nudged off zero because log(0) is -inf.
  const float u1 = max(tf_to_unit_float(bits.x), 1.0e-7f);
  const float u2 = tf_to_unit_float(bits.y);
  return sqrt(-2.0f * log(u1)) * cos(6.28318530718f * u2);
}

kernel void tf_random_normal_float(device float* out [[buffer(0)]],
                                   constant RandomParams& params [[buffer(1)]],
                                   uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  out[gid] = tf_box_muller(tf_random_bits(params, gid));
}

// Truncated to two standard deviations, which is TensorFlow's definition.
// Resampled rather than clamped: clamping would pile mass onto the two
// boundary values and visibly distort weight initialisation.
kernel void tf_truncated_normal_float(device float* out [[buffer(0)]],
                                      constant RandomParams& params
                                          [[buffer(1)]],
                                      uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  float value = 0.0f;
  for (uint attempt = 0; attempt < 8; ++attempt) {
    uint4 bits = tf_philox(uint4(gid, params.counter, attempt, 0u),
                           uint2(params.seed_lo, params.seed_hi));
    value = tf_box_muller(bits);
    if (fabs(value) <= 2.0f) break;
    // After the last attempt the value is kept as is; the probability of
    // eight consecutive rejections is under 1e-13.
  }
  out[gid] = clamp(value, -2.0f, 2.0f);
}

// Uniform integers in [lo, hi). Derived from the same Philox stream as the
// float generator, but reduced by remainder rather than scaled, so the result
// is exact rather than a rounded float.
struct RandomIntParams {
  uint count;
  uint seed_lo;
  uint seed_hi;
  uint counter;
  int lo;
  uint span;
  uint pad0;
  uint pad1;
};

kernel void tf_random_uniform_int(device int* out [[buffer(0)]],
                                  constant RandomIntParams& params
                                      [[buffer(1)]],
                                  uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  if (params.span == 0u) { out[gid] = params.lo; return; }
  uint4 bits = tf_philox(uint4(gid, params.counter, 0u, 0u),
                         uint2(params.seed_lo, params.seed_hi));
  out[gid] = params.lo + int(bits.x % params.span);
}

// ---- optimizers ----

struct SgdParams {
  uint count;
  uint pad0;
  uint pad1;
  uint pad2;
};

kernel void tf_apply_gradient_descent_float(
    device float* var [[buffer(0)]],
    device const float* alpha [[buffer(1)]],
    device const float* delta [[buffer(2)]],
    constant SgdParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  var[gid] -= alpha[0] * delta[gid];
}

struct AdamParams {
  uint count;
  uint use_nesterov;
  uint pad0;
  uint pad1;
};

// TensorFlow's ResourceApplyAdam, with the bias correction folded into a
// single step size exactly as the reference kernel does:
//   alpha = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
//   m += (1 - beta1) * (g - m)
//   v += (1 - beta2) * (g*g - v)
//   var -= alpha * m / (sqrt(v) + epsilon)
// The scalars arrive as one-element device buffers because that is how
// TensorFlow places them.
// TensorFlow's ResourceApplyMomentum:
//   accum = accum * momentum + grad
//   var  -= use_nesterov ? (grad*lr + accum*momentum*lr) : accum*lr
kernel void tf_apply_momentum_float(device float* var [[buffer(0)]],
                                    device float* accum [[buffer(1)]],
                                    device const float* lr [[buffer(2)]],
                                    device const float* grad [[buffer(3)]],
                                    device const float* momentum [[buffer(4)]],
                                    constant AdamParams& params [[buffer(5)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  const float g = grad[gid];
  const float m = momentum[0];
  const float a = accum[gid] * m + g;
  accum[gid] = a;
  var[gid] -= params.use_nesterov ? (g * lr[0] + a * m * lr[0]) : (a * lr[0]);
}

// TensorFlow's ResourceApplyKerasMomentum, which is NOT the same update as
// ResourceApplyMomentum above. Keras folds the learning rate into the
// accumulator and adds it, where the classic op keeps the rate outside and
// subtracts:
//   accum = accum * momentum - grad * lr
//   var  += use_nesterov ? (accum*momentum - grad*lr) : accum
// Sharing one kernel between the two would train subtly differently rather
// than fail, which is why they are separate.
kernel void tf_apply_keras_momentum_float(
    device float* var [[buffer(0)]],
    device float* accum [[buffer(1)]],
    device const float* lr [[buffer(2)]],
    device const float* grad [[buffer(3)]],
    device const float* momentum [[buffer(4)]],
    constant AdamParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  const float g = grad[gid];
  const float m = momentum[0];
  const float a = accum[gid] * m - g * lr[0];
  accum[gid] = a;
  var[gid] += params.use_nesterov ? (a * m - g * lr[0]) : a;
}

// TensorFlow's ResourceApplyRMSProp:
//   ms  += (grad^2 - ms) * (1 - rho)
//   mom  = mom*momentum + grad*lr / sqrt(ms + epsilon)
//   var -= mom
// Note the epsilon sits inside the square root here, unlike Adam where it is
// added to the root. Moving it changes the update.
kernel void tf_apply_rms_prop_float(device float* var [[buffer(0)]],
                                    device float* ms [[buffer(1)]],
                                    device float* mom [[buffer(2)]],
                                    device const float* lr [[buffer(3)]],
                                    device const float* rho [[buffer(4)]],
                                    device const float* momentum [[buffer(5)]],
                                    device const float* epsilon [[buffer(6)]],
                                    device const float* grad [[buffer(7)]],
                                    constant AdamParams& params [[buffer(8)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  const float g = grad[gid];
  const float ms_new = ms[gid] + (g * g - ms[gid]) * (1.0f - rho[0]);
  ms[gid] = ms_new;
  const float mom_new =
      mom[gid] * momentum[0] + (g * lr[0]) / sqrt(ms_new + epsilon[0]);
  mom[gid] = mom_new;
  var[gid] -= mom_new;
}

kernel void tf_apply_adam_float(device float* var [[buffer(0)]],
                                device float* m [[buffer(1)]],
                                device float* v [[buffer(2)]],
                                device const float* beta1_power [[buffer(3)]],
                                device const float* beta2_power [[buffer(4)]],
                                device const float* lr [[buffer(5)]],
                                device const float* beta1 [[buffer(6)]],
                                device const float* beta2 [[buffer(7)]],
                                device const float* epsilon [[buffer(8)]],
                                device const float* grad [[buffer(9)]],
                                constant AdamParams& params [[buffer(10)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;

  const float b1 = beta1[0];
  const float b2 = beta2[0];
  const float alpha = lr[0] * sqrt(1.0f - beta2_power[0]) /
                      (1.0f - beta1_power[0]);

  const float g = grad[gid];
  const float m_new = m[gid] + (1.0f - b1) * (g - m[gid]);
  const float v_new = v[gid] + (1.0f - b2) * (g * g - v[gid]);
  m[gid] = m_new;
  v[gid] = v_new;

  const float numerator =
      params.use_nesterov ? (m_new * b1 + (1.0f - b1) * g) : m_new;
  var[gid] -= alpha * numerator / (sqrt(v_new) + epsilon[0]);
}

// ---- morphological dilation gradients ----

struct DilationParams {
  uint batch;
  uint in_h;
  uint in_w;
  uint channels;
  uint out_h;
  uint out_w;
  uint kh;
  uint kw;
  uint stride_h;
  uint stride_w;
  uint rate_h;
  uint rate_w;
  int pad_top;
  int pad_left;
  uint count;
  uint pad0;
};

// Both dilation gradients replay the forward pass's argmax over the filter
// window, then scatter one output gradient into the position that won. Many
// output positions can win the same input or filter element, so the
// accumulation has to be atomic.
//
// Ties go to the first position in scan order, and the comparison is strict
// for that reason: >= would pick the last instead. Either is a valid
// subgradient, but the CPU and CUDA kernels pick the first, and matching them
// is what makes a numerical comparison against them meaningful.
//
// A window entirely outside the input leaves the maximum unset. The fallback
// position is then the clamped window origin, which is what the CPU kernel
// uses, so a zero-sized overlap still deposits its gradient in the same place.
#define TF_METAL_DILATION_ARGMAX()                                             \
  const uint c = gid % params.channels;                                        \
  const uint ox = (gid / params.channels) % params.out_w;                      \
  const uint oy = (gid / (params.channels * params.out_w)) % params.out_h;     \
  const uint b = gid / (params.channels * params.out_w * params.out_h);        \
  const int h_beg = int(oy * params.stride_h) - params.pad_top;                \
  const int w_beg = int(ox * params.stride_w) - params.pad_left;               \
  int h_max = max(h_beg, 0);                                                   \
  int w_max = max(w_beg, 0);                                                   \
  uint k_h_max = 0;                                                            \
  uint k_w_max = 0;                                                            \
  float cur = -INFINITY;                                                       \
  for (uint ky = 0; ky < params.kh; ++ky) {                                    \
    const int iy = h_beg + int(ky * params.rate_h);                            \
    if (iy < 0 || iy >= int(params.in_h)) continue;                            \
    for (uint kx = 0; kx < params.kw; ++kx) {                                  \
      const int ix = w_beg + int(kx * params.rate_w);                          \
      if (ix < 0 || ix >= int(params.in_w)) continue;                          \
      const float v =                                                          \
          in[((b * params.in_h + uint(iy)) * params.in_w + uint(ix)) *         \
                 params.channels + c] +                                        \
          flt[(ky * params.kw + kx) * params.channels + c];                    \
      if (v > cur) {                                                           \
        cur = v;                                                               \
        h_max = iy;                                                            \
        w_max = ix;                                                            \
        k_h_max = ky;                                                          \
        k_w_max = kx;                                                          \
      }                                                                        \
    }                                                                          \
  }

kernel void tf_dilation_backprop_input_float(
    device const float* in [[buffer(0)]],
    device const float* flt [[buffer(1)]],
    device const float* grad [[buffer(2)]],
    device atomic_float* in_backprop [[buffer(3)]],
    constant DilationParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  TF_METAL_DILATION_ARGMAX()
  const uint index =
      ((b * params.in_h + uint(h_max)) * params.in_w + uint(w_max)) *
          params.channels + c;
  atomic_fetch_add_explicit(&in_backprop[index], grad[gid],
                            memory_order_relaxed);
}

kernel void tf_dilation_backprop_filter_float(
    device const float* in [[buffer(0)]],
    device const float* flt [[buffer(1)]],
    device const float* grad [[buffer(2)]],
    device atomic_float* filter_backprop [[buffer(3)]],
    constant DilationParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  TF_METAL_DILATION_ARGMAX()
  const uint index =
      (k_h_max * params.kw + k_w_max) * params.channels + c;
  atomic_fetch_add_explicit(&filter_backprop[index], grad[gid],
                            memory_order_relaxed);
}

// ---- max pooling with indices ----

struct PoolIndexParams {
  uint batch;
  uint in_h;
  uint in_w;
  uint channels;
  uint out_h;
  uint out_w;
  uint kh;
  uint kw;
  uint stride_h;
  uint stride_w;
  int pad_top;
  int pad_left;
  uint count;
  uint include_batch;
  uint pad0;
  uint pad1;
};

// The whole argmax family shares one window scan. It is written here rather
// than taken from MPSGraph because MPSGraph reports the winner's position
// inside the pooling window, while TensorFlow defines the index as a position
// in the flattened image; emitting the former under the latter's name would
// quietly corrupt every model that unpools with these indices.
//
// `offset` is the winner's linear position in the input including the batch;
// `flat` is the same in TensorFlow's convention, which drops the batch unless
// include_batch_in_index is set. Ties go to the earliest position in row-major
// scan order, matching the CPU and CUDA kernels.
#define TF_METAL_POOL_ARGMAX(SRC)                                             \
  const uint c = gid % params.channels;                                       \
  const uint ox = (gid / params.channels) % params.out_w;                     \
  const uint oy = (gid / (params.channels * params.out_w)) % params.out_h;    \
  const uint b = gid / (params.channels * params.out_w * params.out_h);       \
  const int h_beg = int(oy * params.stride_h) - params.pad_top;               \
  const int w_beg = int(ox * params.stride_w) - params.pad_left;              \
  float best = -INFINITY;                                                     \
  uint best_y = uint(max(h_beg, 0));                                          \
  uint best_x = uint(max(w_beg, 0));                                          \
  for (uint ky = 0; ky < params.kh; ++ky) {                                   \
    const int iy = h_beg + int(ky);                                           \
    if (iy < 0 || iy >= int(params.in_h)) continue;                           \
    for (uint kx = 0; kx < params.kw; ++kx) {                                 \
      const int ix = w_beg + int(kx);                                         \
      if (ix < 0 || ix >= int(params.in_w)) continue;                         \
      const float v =                                                         \
          SRC[((b * params.in_h + uint(iy)) * params.in_w + uint(ix)) *       \
                  params.channels + c];                                       \
      if (v > best) { best = v; best_y = uint(iy); best_x = uint(ix); }       \
    }                                                                         \
  }                                                                           \
  const uint offset =                                                         \
      ((b * params.in_h + best_y) * params.in_w + best_x) * params.channels + \
      c;                                                                      \
  const uint flat =                                                           \
      params.include_batch != 0                                               \
          ? offset                                                            \
          : ((best_y * params.in_w + best_x) * params.channels + c);

#define TF_METAL_POOL_ARGMAX_FWD(NAME, IDX)                                   \
  kernel void NAME(device const float* in [[buffer(0)]],                      \
                   device float* out [[buffer(1)]],                           \
                   device IDX* argmax [[buffer(2)]],                          \
                   constant PoolIndexParams& params [[buffer(3)]],            \
                   uint gid [[thread_position_in_grid]]) {                    \
    if (gid >= params.count) return;                                          \
    TF_METAL_POOL_ARGMAX(in)                                                  \
    out[gid] = best;                                                          \
    argmax[gid] = IDX(flat);                                                  \
  }

TF_METAL_POOL_ARGMAX_FWD(tf_maxpool_argmax_float_i32, int)
TF_METAL_POOL_ARGMAX_FWD(tf_maxpool_argmax_float_i64, long)

// Scatters one pooled gradient back to the input element that won its window.
// Overlapping windows share winners, so the accumulation is atomic. The index
// is turned back into a full offset here, since TensorFlow's default form
// drops the batch.
#define TF_METAL_POOL_GRAD_ARGMAX(NAME, IDX)                                  \
  kernel void NAME(device const float* grad [[buffer(0)]],                    \
                   device const IDX* argmax [[buffer(1)]],                    \
                   device atomic_float* out [[buffer(2)]],                    \
                   constant PoolIndexParams& params [[buffer(3)]],            \
                   uint gid [[thread_position_in_grid]]) {                    \
    if (gid >= params.count) return;                                          \
    const uint per_image = params.in_h * params.in_w * params.channels;       \
    const uint b = gid / (params.channels * params.out_w * params.out_h);     \
    uint index = uint(argmax[gid]);                                           \
    if (params.include_batch == 0) index += b * per_image;                    \
    if (index >= params.batch * per_image) return;                            \
    atomic_fetch_add_explicit(&out[index], grad[gid], memory_order_relaxed);  \
  }

TF_METAL_POOL_GRAD_ARGMAX(tf_maxpool_grad_with_argmax_float_i32, int)
TF_METAL_POOL_GRAD_ARGMAX(tf_maxpool_grad_with_argmax_float_i64, long)

// The second-order gradient is the gather that mirrors that scatter: each
// pooled position reads the input-shaped gradient at the element it selected.
// No atomics, since every pooled position writes its own slot.
#define TF_METAL_POOL_GRADGRAD_ARGMAX(NAME, IDX)                              \
  kernel void NAME(device const float* grad [[buffer(0)]],                    \
                   device const IDX* argmax [[buffer(1)]],                    \
                   device float* out [[buffer(2)]],                           \
                   constant PoolIndexParams& params [[buffer(3)]],            \
                   uint gid [[thread_position_in_grid]]) {                    \
    if (gid >= params.count) return;                                          \
    const uint per_image = params.in_h * params.in_w * params.channels;       \
    const uint b = gid / (params.channels * params.out_w * params.out_h);     \
    uint index = uint(argmax[gid]);                                           \
    if (params.include_batch == 0) index += b * per_image;                    \
    out[gid] = index < params.batch * per_image ? grad[index] : 0.0f;         \
  }

TF_METAL_POOL_GRADGRAD_ARGMAX(tf_maxpool_gradgrad_with_argmax_float_i32, int)
TF_METAL_POOL_GRADGRAD_ARGMAX(tf_maxpool_gradgrad_with_argmax_float_i64, long)

// MaxPoolGradGrad without stored indices: the winner is recomputed from the
// original input, then the input-shaped gradient is read there.
kernel void tf_maxpool_gradgrad_float(device const float* in [[buffer(0)]],
                                      device const float* grad [[buffer(1)]],
                                      device float* out [[buffer(2)]],
                                      constant PoolIndexParams& params
                                          [[buffer(3)]],
                                      uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  TF_METAL_POOL_ARGMAX(in)
  out[gid] = grad[offset];
}

// ---- bin counting ----

struct BincountParams {
  uint count;
  uint size;
  uint row_len;
  uint binary;
  uint has_weights;
  uint pad0;
  uint pad1;
  uint pad2;
};

// One thread per input value, accumulating into the bin it names. Many values
// land in the same bin, so the accumulation is atomic. Values outside
// [0, size) are dropped, which is what the CPU kernel does; they are not an
// error.
//
// The binary form stores one rather than accumulating, so repeated values
// still leave a one. The dense two-dimensional form gives each row its own
// stretch of bins, selected by `row_len`.
#define TF_METAL_BINCOUNT(NAME, IDX, T, ATOMIC, ONE)                          \
  kernel void NAME(device const IDX* values [[buffer(0)]],                    \
                   device const T* weights [[buffer(1)]],                     \
                   device ATOMIC* out [[buffer(2)]],                          \
                   constant BincountParams& params [[buffer(3)]],             \
                   uint gid [[thread_position_in_grid]]) {                    \
    if (gid >= params.count) return;                                          \
    const IDX v = values[gid];                                                \
    if (v < 0 || ulong(v) >= ulong(params.size)) return;                      \
    const uint row = params.row_len > 0 ? gid / params.row_len : 0;           \
    const uint index = row * params.size + uint(v);                           \
    if (params.binary != 0) {                                                 \
      atomic_store_explicit(&out[index], ONE, memory_order_relaxed);          \
      return;                                                                 \
    }                                                                         \
    const T w = params.has_weights != 0 ? weights[gid] : ONE;                 \
    atomic_fetch_add_explicit(&out[index], w, memory_order_relaxed);          \
  }

TF_METAL_BINCOUNT(tf_bincount_float_i32, int, float, atomic_float, 1.0f)
TF_METAL_BINCOUNT(tf_bincount_float_i64, long, float, atomic_float, 1.0f)
TF_METAL_BINCOUNT(tf_bincount_int_i32, int, int, atomic_int, 1)
TF_METAL_BINCOUNT(tf_bincount_int_i64, long, int, atomic_int, 1)

// ---- crop and resize ----

struct CropResizeParams {
  uint batch;
  uint in_h;
  uint in_w;
  uint depth;
  uint num_boxes;
  uint crop_h;
  uint crop_w;
  uint method_nearest;
  float extrapolation;
  uint count;
  uint pad0;
  uint pad1;
};

// All three shaders walk the same geometry: one thread per element of the
// crop, mapping it back to a fractional position in the source image. A box
// may reach outside the image, and the region outside is not clamped but
// filled with the extrapolation value, so the bounds test is part of the
// definition rather than a safety check.
#define TF_METAL_CROP_GEOMETRY()                                              \
  const uint d = gid % params.depth;                                          \
  const uint x = (gid / params.depth) % params.crop_w;                        \
  const uint y = (gid / (params.depth * params.crop_w)) % params.crop_h;      \
  const uint b = gid / (params.depth * params.crop_w * params.crop_h);        \
  const int b_in = box_index[b];                                              \
  const float y1 = boxes[b * 4 + 0];                                          \
  const float x1 = boxes[b * 4 + 1];                                          \
  const float y2 = boxes[b * 4 + 2];                                          \
  const float x2 = boxes[b * 4 + 3];                                          \
  const float height_ratio =                                                  \
      params.crop_h > 1 ? float(params.in_h - 1) / float(params.crop_h - 1)   \
                        : 0.0f;                                               \
  const float width_ratio =                                                   \
      params.crop_w > 1 ? float(params.in_w - 1) / float(params.crop_w - 1)   \
                        : 0.0f;                                               \
  const float in_y =                                                          \
      params.crop_h > 1                                                       \
          ? y1 * float(params.in_h - 1) + float(y) * ((y2 - y1) * height_ratio) \
          : 0.5f * (y1 + y2) * float(params.in_h - 1);                        \
  const float in_x =                                                          \
      params.crop_w > 1                                                       \
          ? x1 * float(params.in_w - 1) + float(x) * ((x2 - x1) * width_ratio) \
          : 0.5f * (x1 + x2) * float(params.in_w - 1);                        \
  const bool outside = b_in < 0 || b_in >= int(params.batch) || in_y < 0.0f || \
                       in_y > float(params.in_h - 1) || in_x < 0.0f ||        \
                       in_x > float(params.in_w - 1);

#define TF_METAL_CROP_IMAGE_INDEX(YI, XI)                                     \
  ((uint(b_in) * params.in_h + uint(YI)) * params.in_w + uint(XI)) *          \
      params.depth + d

kernel void tf_crop_and_resize_float(device const float* image [[buffer(0)]],
                                     device const float* boxes [[buffer(1)]],
                                     device const int* box_index [[buffer(2)]],
                                     device float* out [[buffer(3)]],
                                     constant CropResizeParams& params
                                         [[buffer(4)]],
                                     uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  TF_METAL_CROP_GEOMETRY()
  if (outside) {
    out[gid] = params.extrapolation;
    return;
  }
  if (params.method_nearest != 0) {
    // round() ties away from zero, which is what roundf does in the CPU
    // kernel.
    out[gid] = image[TF_METAL_CROP_IMAGE_INDEX(uint(round(in_y)),
                                               uint(round(in_x)))];
    return;
  }
  const uint top = uint(floor(in_y));
  const uint bottom = uint(ceil(in_y));
  const uint left = uint(floor(in_x));
  const uint right = uint(ceil(in_x));
  const float y_lerp = in_y - floor(in_y);
  const float x_lerp = in_x - floor(in_x);
  const float tl = image[TF_METAL_CROP_IMAGE_INDEX(top, left)];
  const float tr = image[TF_METAL_CROP_IMAGE_INDEX(top, right)];
  const float bl = image[TF_METAL_CROP_IMAGE_INDEX(bottom, left)];
  const float br = image[TF_METAL_CROP_IMAGE_INDEX(bottom, right)];
  const float t = tl + (tr - tl) * x_lerp;
  const float bo = bl + (br - bl) * x_lerp;
  out[gid] = t + (bo - t) * y_lerp;
}

// The image gradient is the transpose of that sampling: each crop element
// pushes its gradient back into the four pixels it interpolated, weighted the
// same way. Neighbouring crop elements share pixels, so the accumulation is
// atomic.
kernel void tf_crop_and_resize_grad_image_float(
    device const float* grads [[buffer(0)]],
    device const float* boxes [[buffer(1)]],
    device const int* box_index [[buffer(2)]],
    device atomic_float* out [[buffer(3)]],
    constant CropResizeParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  TF_METAL_CROP_GEOMETRY()
  if (outside) return;
  const float g = grads[gid];
  if (params.method_nearest != 0) {
    atomic_fetch_add_explicit(
        &out[TF_METAL_CROP_IMAGE_INDEX(uint(round(in_y)), uint(round(in_x)))],
        g, memory_order_relaxed);
    return;
  }
  const uint top = uint(floor(in_y));
  const uint bottom = uint(ceil(in_y));
  const uint left = uint(floor(in_x));
  const uint right = uint(ceil(in_x));
  const float y_lerp = in_y - floor(in_y);
  const float x_lerp = in_x - floor(in_x);
  atomic_fetch_add_explicit(&out[TF_METAL_CROP_IMAGE_INDEX(top, left)],
                            (1.0f - y_lerp) * (1.0f - x_lerp) * g,
                            memory_order_relaxed);
  atomic_fetch_add_explicit(&out[TF_METAL_CROP_IMAGE_INDEX(top, right)],
                            (1.0f - y_lerp) * x_lerp * g,
                            memory_order_relaxed);
  atomic_fetch_add_explicit(&out[TF_METAL_CROP_IMAGE_INDEX(bottom, left)],
                            y_lerp * (1.0f - x_lerp) * g,
                            memory_order_relaxed);
  atomic_fetch_add_explicit(&out[TF_METAL_CROP_IMAGE_INDEX(bottom, right)],
                            y_lerp * x_lerp * g, memory_order_relaxed);
}

// The box gradient differentiates the sampling position rather than the
// sampled value, so it needs the image itself. Bilinear only, which is what
// TensorFlow defines: the nearest-neighbour position is piecewise constant and
// its derivative is zero almost everywhere.
kernel void tf_crop_and_resize_grad_boxes_float(
    device const float* grads [[buffer(0)]],
    device const float* image [[buffer(1)]],
    device const float* boxes [[buffer(2)]],
    device const int* box_index [[buffer(3)]],
    device atomic_float* out [[buffer(4)]],
    constant CropResizeParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  TF_METAL_CROP_GEOMETRY()
  if (outside) return;
  const uint top = uint(floor(in_y));
  const uint bottom = uint(ceil(in_y));
  const uint left = uint(floor(in_x));
  const uint right = uint(ceil(in_x));
  const float y_lerp = in_y - floor(in_y);
  const float x_lerp = in_x - floor(in_x);
  const float tl = image[TF_METAL_CROP_IMAGE_INDEX(top, left)];
  const float tr = image[TF_METAL_CROP_IMAGE_INDEX(top, right)];
  const float bl = image[TF_METAL_CROP_IMAGE_INDEX(bottom, left)];
  const float br = image[TF_METAL_CROP_IMAGE_INDEX(bottom, right)];
  const float g = grads[gid];
  const float grad_y =
      ((1.0f - x_lerp) * (bl - tl) + x_lerp * (br - tr)) * g;
  const float grad_x =
      ((1.0f - y_lerp) * (tr - tl) + y_lerp * (br - bl)) * g;
  float dy1, dy2, dx1, dx2;
  if (params.crop_h > 1) {
    dy1 = grad_y * (float(params.in_h - 1) - float(y) * height_ratio);
    dy2 = grad_y * (float(y) * height_ratio);
  } else {
    dy1 = grad_y * 0.5f * float(params.in_h - 1);
    dy2 = dy1;
  }
  if (params.crop_w > 1) {
    dx1 = grad_x * (float(params.in_w - 1) - float(x) * width_ratio);
    dx2 = grad_x * (float(x) * width_ratio);
  } else {
    dx1 = grad_x * 0.5f * float(params.in_w - 1);
    dx2 = dx1;
  }
  atomic_fetch_add_explicit(&out[b * 4 + 0], dy1, memory_order_relaxed);
  atomic_fetch_add_explicit(&out[b * 4 + 1], dx1, memory_order_relaxed);
  atomic_fetch_add_explicit(&out[b * 4 + 2], dy2, memory_order_relaxed);
  atomic_fetch_add_explicit(&out[b * 4 + 3], dx2, memory_order_relaxed);
}

// ---- projective transform ----

struct TransformParams {
  uint batch;
  uint in_h;
  uint in_w;
  uint depth;
  uint out_h;
  uint out_w;
  uint count;
  uint nearest;
  uint fill_mode;
  uint num_transforms;
  float fill_value;
  uint pad0;
};

// TensorFlow's MapCoordinate, one function per fill mode. The clamp at the end
// of the reflect and wrap modes is not redundant: a coordinate of 3.5 in a
// length of 4 is in range, but nearest interpolation would round it to 4.
static inline float tf_map_coordinate(float coord, int len, uint mode) {
  if (mode == 0u) return coord;                       // CONSTANT
  if (mode == 3u) return clamp(coord, 0.0f, float(len - 1));  // NEAREST
  if (len <= 1) return 0.0f;
  if (mode == 1u) {                                   // REFLECT
    if (coord < 0.0f) {
      const int sz2 = 2 * len;
      if (coord < float(sz2)) coord = float(sz2) * float(int(-coord / float(sz2))) + coord;
      coord = (coord < float(-len)) ? coord + float(sz2) : -coord - 1.0f;
    } else if (coord > float(len - 1)) {
      const int sz2 = 2 * len;
      coord -= float(sz2) * float(int(coord / float(sz2)));
      if (coord >= float(len)) coord = float(sz2) - coord - 1.0f;
    }
    return clamp(coord, 0.0f, float(len - 1));
  }
  // WRAP
  if (coord < 0.0f) {
    const int sz = len - 1;
    coord += float(len) * float(int(-coord / float(sz)) + 1);
  } else if (coord > float(len - 1)) {
    const int sz = len - 1;
    coord -= float(len) * float(int(coord / float(sz)));
  }
  return clamp(coord, 0.0f, float(len - 1));
}

#define TF_METAL_TRANSFORM_READ(YI, XI)                                       \
  ((YI) >= 0 && (YI) < int(params.in_h) && (XI) >= 0 &&                       \
   (XI) < int(params.in_w))                                                   \
      ? image[((b * params.in_h + uint(YI)) * params.in_w + uint(XI)) *       \
                  params.depth + c]                                           \
      : params.fill_value

kernel void tf_image_projective_transform_float(
    device const float* image [[buffer(0)]],
    device const float* transforms [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant TransformParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  const uint c = gid % params.depth;
  const uint ox = (gid / params.depth) % params.out_w;
  const uint oy = (gid / (params.depth * params.out_w)) % params.out_h;
  const uint b = gid / (params.depth * params.out_w * params.out_h);

  // One transform for the whole batch, or one per image.
  const uint t = params.num_transforms == 1u ? 0u : b;
  device const float* m = transforms + t * 8u;

  const float projection = m[6] * float(ox) + m[7] * float(oy) + 1.0f;
  if (projection == 0.0f || !isfinite(projection)) {
    out[gid] = params.fill_value;
    return;
  }
  const float raw_x =
      (m[0] * float(ox) + m[1] * float(oy) + m[2]) / projection;
  const float raw_y =
      (m[3] * float(ox) + m[4] * float(oy) + m[5]) / projection;
  const float x = tf_map_coordinate(raw_x, int(params.in_w), params.fill_mode);
  const float y = tf_map_coordinate(raw_y, int(params.in_h), params.fill_mode);

  if (params.nearest != 0u) {
    const int yi = int(round(y));
    const int xi = int(round(x));
    out[gid] = TF_METAL_TRANSFORM_READ(yi, xi);
    return;
  }
  // The corners are floor and floor+1, not floor and ceil: on an exact
  // integer the two coincide under ceil and the interpolation degenerates,
  // where TensorFlow keeps a unit-wide cell whose upper weight is zero.
  const float yf = floor(y);
  const float xf = floor(x);
  const int y0 = int(yf), x0 = int(xf), y1 = int(yf + 1.0f), x1 = int(xf + 1.0f);
  const float v00 = TF_METAL_TRANSFORM_READ(y0, x0);
  const float v01 = TF_METAL_TRANSFORM_READ(y0, x1);
  const float v10 = TF_METAL_TRANSFORM_READ(y1, x0);
  const float v11 = TF_METAL_TRANSFORM_READ(y1, x1);
  const float top = (xf + 1.0f - x) * v00 + (x - xf) * v01;
  const float bottom = (xf + 1.0f - x) * v10 + (x - xf) * v11;
  out[gid] = (yf + 1.0f - y) * top + (y - yf) * bottom;
}

// ---- resize gradients ----

struct ResizeGradParams {
  uint batch;
  uint in_h;
  uint in_w;
  uint channels;
  uint out_h;
  uint out_w;
  float height_scale;
  float width_scale;
  uint half_pixel;
  uint align_corners;
  uint count;
  uint pad0;
};

// TensorFlow's two scalers: the half-pixel one places sample centres between
// pixels, the legacy one on them.
static inline float tf_resize_scale(uint index, float scale, uint half_pixel) {
  return half_pixel != 0u ? (float(index) + 0.5f) * scale - 0.5f
                          : float(index) * scale;
}

// Both gradients are the transpose of a resize: every resized pixel pushes its
// gradient back to the source pixels it read, with the same weights. Source
// pixels are shared between neighbours, so the accumulation is atomic.
kernel void tf_resize_bilinear_grad_float(
    device const float* grads [[buffer(0)]],
    device atomic_float* out [[buffer(1)]],
    constant ResizeGradParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  const uint c = gid % params.channels;
  const uint x = (gid / params.channels) % params.in_w;
  const uint y = (gid / (params.channels * params.in_w)) % params.in_h;
  const uint b = gid / (params.channels * params.in_w * params.in_h);

  const float in_y = tf_resize_scale(y, params.height_scale, params.half_pixel);
  const float in_x = tf_resize_scale(x, params.width_scale, params.half_pixel);
  // The corners are clamped independently, so a sample that falls off an edge
  // pushes its whole weight onto the edge pixel rather than being dropped.
  const int top = max(int(floor(in_y)), 0);
  const int bottom = min(int(ceil(in_y)), int(params.out_h) - 1);
  const int left = max(int(floor(in_x)), 0);
  const int right = min(int(ceil(in_x)), int(params.out_w) - 1);
  const float y_lerp = in_y - floor(in_y);
  const float x_lerp = in_x - floor(in_x);
  const float g = grads[gid];

  const uint base = b * params.out_h;
  atomic_fetch_add_explicit(
      &out[((base + uint(top)) * params.out_w + uint(left)) * params.channels + c],
      (1.0f - y_lerp) * (1.0f - x_lerp) * g, memory_order_relaxed);
  atomic_fetch_add_explicit(
      &out[((base + uint(top)) * params.out_w + uint(right)) * params.channels + c],
      (1.0f - y_lerp) * x_lerp * g, memory_order_relaxed);
  atomic_fetch_add_explicit(
      &out[((base + uint(bottom)) * params.out_w + uint(left)) * params.channels + c],
      y_lerp * (1.0f - x_lerp) * g, memory_order_relaxed);
  atomic_fetch_add_explicit(
      &out[((base + uint(bottom)) * params.out_w + uint(right)) * params.channels + c],
      y_lerp * x_lerp * g, memory_order_relaxed);
}

kernel void tf_resize_nearest_grad_float(
    device const float* grads [[buffer(0)]],
    device atomic_float* out [[buffer(1)]],
    constant ResizeGradParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  const uint c = gid % params.channels;
  const uint x = (gid / params.channels) % params.in_w;
  const uint y = (gid / (params.channels * params.in_w)) % params.in_h;
  const uint b = gid / (params.channels * params.in_w * params.in_h);

  const float fy = tf_resize_scale(y, params.height_scale, params.half_pixel);
  const float fx = tf_resize_scale(x, params.width_scale, params.half_pixel);
  // Aligned corners round to the nearest source pixel; otherwise the source
  // pixel is the one the sample falls inside, which is the floor.
  int oy = params.align_corners != 0u ? int(round(fy)) : int(floor(fy));
  int ox = params.align_corners != 0u ? int(round(fx)) : int(floor(fx));
  oy = clamp(oy, 0, int(params.out_h) - 1);
  ox = clamp(ox, 0, int(params.out_w) - 1);
  atomic_fetch_add_explicit(
      &out[((b * params.out_h + uint(oy)) * params.out_w + uint(ox)) *
               params.channels + c],
      grads[gid], memory_order_relaxed);
}

// ---- volume patches ----

struct VolumePatchParams {
  uint batch;
  uint in_d;
  uint in_h;
  uint in_w;
  uint channels;
  uint out_d;
  uint out_h;
  uint out_w;
  uint kd;
  uint kh;
  uint kw;
  uint stride_d;
  uint stride_h;
  uint stride_w;
  int pad_d;
  int pad_h;
  int pad_w;
  uint count;
  uint pad0;
  uint pad1;
};

// One thread per output element. The last axis packs the whole window, with
// the channel varying fastest, then the width, the height and the depth, which
// is the order TensorFlow lays a patch out in. Positions outside the volume
// contribute zero, which is what padding means here.
kernel void tf_extract_volume_patches_float(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant VolumePatchParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  const uint patch = params.kd * params.kh * params.kw * params.channels;
  const uint k = gid % patch;
  const uint ox = (gid / patch) % params.out_w;
  const uint oy = (gid / (patch * params.out_w)) % params.out_h;
  const uint oz = (gid / (patch * params.out_w * params.out_h)) % params.out_d;
  const uint b = gid / (patch * params.out_w * params.out_h * params.out_d);

  const uint c = k % params.channels;
  const uint kx = (k / params.channels) % params.kw;
  const uint ky = (k / (params.channels * params.kw)) % params.kh;
  const uint kz = k / (params.channels * params.kw * params.kh);

  const int iz = int(oz * params.stride_d) - params.pad_d + int(kz);
  const int iy = int(oy * params.stride_h) - params.pad_h + int(ky);
  const int ix = int(ox * params.stride_w) - params.pad_w + int(kx);
  if (iz < 0 || iz >= int(params.in_d) || iy < 0 || iy >= int(params.in_h) ||
      ix < 0 || ix >= int(params.in_w)) {
    out[gid] = 0.0f;
    return;
  }
  const uint index =
      (((b * params.in_d + uint(iz)) * params.in_h + uint(iy)) * params.in_w +
       uint(ix)) * params.channels + c;
  out[gid] = in[index];
}

// ---- parameterised random distributions ----

struct ParamTruncatedParams {
  uint count;
  uint seed_lo;
  uint seed_hi;
  uint counter;
  uint samples_per_batch;
  uint num_params;
  uint pad0;
  uint pad1;
};

struct MultinomialParams {
  uint count;
  uint seed_lo;
  uint seed_hi;
  uint counter;
  uint batch;
  uint classes;
  uint samples;
  uint pad0;
};

struct GammaParams {
  uint count;
  uint seed_lo;
  uint seed_hi;
  uint counter;
  uint num_alphas;
  uint pad0;
  uint pad1;
  uint pad2;
};

// A normal truncated to an arbitrary interval, per element of a batch.
//
// Rejection sampling rather than a clamp: clamping piles the rejected mass
// onto the two endpoints, which is a different distribution and a visible one.
// A bounded number of attempts keeps the thread from spinning on a very
// narrow interval; the fallback is the clamp, which is wrong in the same way
// but only for intervals so narrow that acceptance is hopeless.
kernel void tf_parameterized_truncated_normal_float(
    device float* out [[buffer(0)]],
    device const float* means [[buffer(1)]],
    device const float* stdevs [[buffer(2)]],
    device const float* minvals [[buffer(3)]],
    device const float* maxvals [[buffer(4)]],
    constant ParamTruncatedParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  const uint batch =
      params.samples_per_batch > 0u ? gid / params.samples_per_batch : 0u;
  const uint p = params.num_params == 1u ? 0u : batch;
  const float mean = means[p];
  const float stdev = stdevs[p];
  const float lo = minvals[p];
  const float hi = maxvals[p];

  float value = mean;
  for (uint attempt = 0u; attempt < 32u; ++attempt) {
    const uint4 bits = tf_philox(uint4(gid, params.counter, attempt, 0u),
                                 uint2(params.seed_lo, params.seed_hi));
    value = mean + stdev * tf_box_muller(bits);
    if (value >= lo && value <= hi) {
      out[gid] = value;
      return;
    }
  }
  out[gid] = clamp(value, lo, hi);
}

// Categorical sampling by the Gumbel-max trick: adding a Gumbel variate to
// each logit and taking the argmax draws exactly from the softmax of the
// logits. It needs one pass over the classes and no normalisation, no
// cumulative sum and no search, which is what a per-thread sampler wants.
#define TF_METAL_MULTINOMIAL(NAME, IDX)                                       \
  kernel void NAME(device const float* logits [[buffer(0)]],                  \
                   device IDX* out [[buffer(1)]],                             \
                   constant MultinomialParams& params [[buffer(2)]],          \
                   uint gid [[thread_position_in_grid]]) {                    \
    if (gid >= params.count) return;                                          \
    const uint b = params.samples > 0u ? gid / params.samples : 0u;           \
    device const float* row = logits + b * params.classes;                    \
    float best = -INFINITY;                                                   \
    uint best_class = 0u;                                                     \
    for (uint c = 0u; c < params.classes; ++c) {                              \
      /* One Philox call per class, indexed by the class, so the draw depends \
         only on the seed and the position, not on any iteration order. */    \
      const uint4 bits = tf_philox(uint4(gid, params.counter, c, 1u),         \
                                   uint2(params.seed_lo, params.seed_hi));    \
      const float u = max(tf_to_unit_float(bits.x), 1.0e-7f);                 \
      const float score = row[c] + (-log(-log(u)));                           \
      if (score > best) {                                                     \
        best = score;                                                         \
        best_class = c;                                                       \
      }                                                                       \
    }                                                                         \
    out[gid] = IDX(best_class);                                               \
  }

TF_METAL_MULTINOMIAL(tf_multinomial_float, int)
TF_METAL_MULTINOMIAL(tf_multinomial_float_i64, long)

// Marsaglia and Tsang's gamma sampler, which is what TensorFlow uses.
//
// It is defined for a shape parameter of at least one; below that the
// identity gamma(a) = gamma(a + 1) * u^(1/a) moves the draw into range, which
// is the standard boost and is what the CPU kernel does as well.
kernel void tf_random_gamma_float(device float* out [[buffer(0)]],
                                  device const float* alphas [[buffer(1)]],
                                  constant GammaParams& params [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  const uint a_index = params.num_alphas > 0u ? gid % params.num_alphas : 0u;
  const float alpha_in = alphas[a_index];
  if (!(alpha_in > 0.0f)) {
    out[gid] = 0.0f;
    return;
  }

  const bool boost = alpha_in < 1.0f;
  const float alpha = boost ? alpha_in + 1.0f : alpha_in;
  const float d = alpha - 1.0f / 3.0f;
  const float c = 1.0f / sqrt(9.0f * d);

  float result = 0.0f;
  for (uint attempt = 0u; attempt < 64u; ++attempt) {
    const uint4 bits = tf_philox(uint4(gid, params.counter, attempt, 2u),
                                 uint2(params.seed_lo, params.seed_hi));
    const float x = tf_box_muller(bits);
    const float v = 1.0f + c * x;
    if (v <= 0.0f) continue;
    const float v3 = v * v * v;
    const float u = max(tf_to_unit_float(bits.z), 1.0e-7f);
    const float x2 = x * x;
    // The squeeze first, then the exact test, exactly as in the paper.
    if (u < 1.0f - 0.0331f * x2 * x2) {
      result = d * v3;
      break;
    }
    if (log(u) < 0.5f * x2 + d * (1.0f - v3 + log(v3))) {
      result = d * v3;
      break;
    }
  }
  if (boost) {
    const uint4 bits = tf_philox(uint4(gid, params.counter, 99u, 3u),
                                 uint2(params.seed_lo, params.seed_hi));
    const float u = max(tf_to_unit_float(bits.x), 1.0e-7f);
    result *= pow(u, 1.0f / alpha_in);
  }
  out[gid] = result;
}

// ---- row gather and scatter ----

struct RowMoveParams {
  uint count;
  uint slice;
  uint limit;
  uint pad0;
};

// The two halves of every index-driven data movement: gather takes rows named
// by an index vector, scatter puts rows where an index vector says.
//
// They copy 32-bit words rather than a typed element, so one pair covers every
// dtype: the caller expresses a row's width in words and doubles it for the
// eight-byte types. Copying bits also avoids a float round trip turning a
// signalling NaN into a quiet one, which a copy has no business doing.
kernel void tf_gather_rows_u32(device const uint* data [[buffer(0)]],
                               device const int* indices [[buffer(1)]],
                               device uint* out [[buffer(2)]],
                               constant RowMoveParams& params [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  const uint row = gid / params.slice;
  const uint word = gid % params.slice;
  const int index = indices[row];
  // An out-of-range index yields zero rather than reading out of bounds; the
  // callers here never produce one, and a shader must not trust that.
  out[gid] = (index < 0 || uint(index) >= params.limit)
                 ? 0u
                 : data[uint(index) * params.slice + word];
}

kernel void tf_scatter_rows_u32(device const uint* data [[buffer(0)]],
                                device const int* indices [[buffer(1)]],
                                device uint* out [[buffer(2)]],
                                constant RowMoveParams& params [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  const uint row = gid / params.slice;
  const uint word = gid % params.slice;
  const int index = indices[row];
  if (index < 0 || uint(index) >= params.limit) return;
  out[uint(index) * params.slice + word] = data[gid];
}

// ---- pivots to permutation ----

struct PivotParams {
  uint batch;
  uint order;
  uint pad0;
  uint pad1;
};

// A factorisation returns its row interchanges the way LAPACK does: entry i
// names the row that row i was swapped with, in order. TensorFlow wants the
// permutation itself, so the swaps have to be replayed.
//
// One thread per matrix, looping over the order. That is deliberately serial:
// the swaps are only meaningful in sequence, and a matrix small enough to
// factorise is small enough for one thread to walk. Batches still run in
// parallel, which is where the width is.
#define TF_METAL_PIVOTS(NAME, IDX)                                            \
  kernel void NAME(device const uint* pivots [[buffer(0)]],                   \
                   device IDX* out [[buffer(1)]],                             \
                   constant PivotParams& params [[buffer(2)]],                \
                   uint gid [[thread_position_in_grid]]) {                    \
    if (gid >= params.batch) return;                                          \
    device const uint* p = pivots + gid * params.order;                       \
    device IDX* result = out + gid * params.order;                            \
    for (uint i = 0; i < params.order; ++i) result[i] = IDX(i);               \
    for (uint i = 0; i < params.order; ++i) {                                 \
      const uint j = p[i];                                                    \
      if (j >= params.order) continue;                                        \
      const IDX tmp = result[i];                                              \
      result[i] = result[j];                                                  \
      result[j] = tmp;                                                        \
    }                                                                         \
  }

TF_METAL_PIVOTS(tf_pivots_to_permutation_i32, int)
TF_METAL_PIVOTS(tf_pivots_to_permutation_i64, long)

// ---- dense factorisations ----

struct FactorParams {
  uint batch;
  uint rows;
  uint cols;
  uint k;
  uint full_matrices;
  uint compute_vectors;
  uint pad0;
  uint pad1;
};

// QR by Householder reflections, and the symmetric eigenproblem by Jacobi
// rotations. Metal Performance Shaders has neither.
//
// One thread per matrix, working serially through its own scratch. That is not
// a compromise for lack of a better shape: both algorithms are sweeps whose
// every step depends on the one before, so the parallelism that exists is
// across matrices, which is exactly what a thread per matrix uses. A single
// large matrix will not go fast here, and a batch of small ones will.
//
// The scratch is passed in rather than declared: a thread's working copy is
// larger than any thread-local allocation Metal offers, and it has to be
// device memory for that reason.
kernel void tf_qr_float(device const float* input [[buffer(0)]],
                        device float* q_out [[buffer(1)]],
                        device float* r_out [[buffer(2)]],
                        device float* scratch [[buffer(3)]],
                        constant FactorParams& params [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= params.batch) return;
  const uint m = params.rows;
  const uint n = params.cols;
  const uint k = params.k;
  const uint stride = m * n + m * m + m;
  device float* work = scratch + gid * stride;
  device float* q = work + m * n;
  device float* v = q + m * m;

  device const float* a = input + gid * m * n;
  for (uint i = 0; i < m * n; ++i) work[i] = a[i];
  for (uint i = 0; i < m; ++i)
    for (uint j = 0; j < m; ++j) q[i * m + j] = (i == j) ? 1.0f : 0.0f;

  for (uint step = 0; step < k; ++step) {
    // The reflector that zeroes column `step` below the diagonal.
    float norm = 0.0f;
    for (uint i = step; i < m; ++i) {
      const float x = work[i * n + step];
      norm += x * x;
    }
    norm = sqrt(norm);
    if (norm == 0.0f) continue;
    const float head = work[step * n + step];
    // Choosing the sign away from the head avoids cancellation, which is the
    // whole reason Householder is preferred to Gram-Schmidt.
    const float alpha = head >= 0.0f ? -norm : norm;
    for (uint i = step; i < m; ++i) v[i] = work[i * n + step];
    v[step] -= alpha;
    float vnorm = 0.0f;
    for (uint i = step; i < m; ++i) vnorm += v[i] * v[i];
    if (vnorm == 0.0f) continue;
    const float scale = 2.0f / vnorm;

    for (uint j = step; j < n; ++j) {
      float dot = 0.0f;
      for (uint i = step; i < m; ++i) dot += v[i] * work[i * n + j];
      dot *= scale;
      for (uint i = step; i < m; ++i) work[i * n + j] -= dot * v[i];
    }
    // Q accumulates the reflectors from the right, so it ends up as the
    // product that takes the factored form back to the input.
    for (uint i = 0; i < m; ++i) {
      float dot = 0.0f;
      for (uint j = step; j < m; ++j) dot += q[i * m + j] * v[j];
      dot *= scale;
      for (uint j = step; j < m; ++j) q[i * m + j] -= dot * v[j];
    }
  }

  const uint q_cols = params.full_matrices != 0u ? m : k;
  const uint r_rows = params.full_matrices != 0u ? m : k;
  device float* qd = q_out + gid * m * q_cols;
  device float* rd = r_out + gid * r_rows * n;
  for (uint i = 0; i < m; ++i)
    for (uint j = 0; j < q_cols; ++j) qd[i * q_cols + j] = q[i * m + j];
  for (uint i = 0; i < r_rows; ++i)
    for (uint j = 0; j < n; ++j)
      rd[i * n + j] = (j >= i) ? work[i * n + j] : 0.0f;
}

kernel void tf_selfadjoint_eig_float(device const float* input [[buffer(0)]],
                                     device float* e_out [[buffer(1)]],
                                     device float* v_out [[buffer(2)]],
                                     device float* scratch [[buffer(3)]],
                                     constant FactorParams& params
                                         [[buffer(4)]],
                                     uint gid [[thread_position_in_grid]]) {
  if (gid >= params.batch) return;
  const uint n = params.rows;
  const uint stride = 2u * n * n;
  device float* work = scratch + gid * stride;
  device float* vec = work + n * n;

  device const float* a = input + gid * n * n;
  // Symmetrised on the way in: the op promises to read one triangle, and
  // averaging costs nothing and removes any question of which.
  for (uint i = 0; i < n; ++i) {
    for (uint j = 0; j < n; ++j) {
      work[i * n + j] = 0.5f * (a[i * n + j] + a[j * n + i]);
      vec[i * n + j] = (i == j) ? 1.0f : 0.0f;
    }
  }

  for (uint sweep = 0; sweep < 60u; ++sweep) {
    float off = 0.0f;
    for (uint i = 0; i < n; ++i)
      for (uint j = i + 1; j < n; ++j) off += work[i * n + j] * work[i * n + j];
    if (off <= 1.0e-16f) break;
    for (uint p = 0; p + 1 < n; ++p) {
      for (uint q = p + 1; q < n; ++q) {
        const float apq = work[p * n + q];
        if (fabs(apq) < 1.0e-12f) continue;
        // The rotation that zeroes this off-diagonal entry.
        const float theta = (work[q * n + q] - work[p * n + p]) / (2.0f * apq);
        const float t = (theta >= 0.0f ? 1.0f : -1.0f) /
                        (fabs(theta) + sqrt(theta * theta + 1.0f));
        const float c = 1.0f / sqrt(t * t + 1.0f);
        const float s = t * c;
        for (uint i = 0; i < n; ++i) {
          const float aip = work[i * n + p];
          const float aiq = work[i * n + q];
          work[i * n + p] = c * aip - s * aiq;
          work[i * n + q] = s * aip + c * aiq;
        }
        for (uint i = 0; i < n; ++i) {
          const float api = work[p * n + i];
          const float aqi = work[q * n + i];
          work[p * n + i] = c * api - s * aqi;
          work[q * n + i] = s * api + c * aqi;
        }
        for (uint i = 0; i < n; ++i) {
          const float vip = vec[i * n + p];
          const float viq = vec[i * n + q];
          vec[i * n + p] = c * vip - s * viq;
          vec[i * n + q] = s * vip + c * viq;
        }
      }
    }
  }

  // TensorFlow returns the eigenvalues in ascending order, so the diagonal is
  // sorted and the eigenvectors follow their own values.
  device float* e = e_out + gid * n;
  for (uint i = 0; i < n; ++i) e[i] = work[i * n + i];
  for (uint i = 1; i < n; ++i) {
    for (uint j = i; j > 0 && e[j - 1] > e[j]; --j) {
      const float tmp = e[j - 1];
      e[j - 1] = e[j];
      e[j] = tmp;
      for (uint r = 0; r < n; ++r) {
        const float v0 = vec[r * n + j - 1];
        vec[r * n + j - 1] = vec[r * n + j];
        vec[r * n + j] = v0;
      }
    }
  }
  if (params.compute_vectors != 0u) {
    device float* vd = v_out + gid * n * n;
    for (uint i = 0; i < n * n; ++i) vd[i] = vec[i];
  }
}
)METAL";

class ShaderLibrary {
 public:
  static ShaderLibrary& Global() {
    static ShaderLibrary* library = new ShaderLibrary();
    return *library;
  }

  id<MTLComputePipelineState> PipelineFor(id<MTLDevice> device,
                                          const char* function_name,
                                          TF_Status* status) {
    ScopedAutoreleasePool pool;
    absl::MutexLock lock(&mu_);
    if (!EnsureLibraryLocked(device, status)) return nil;

    const std::string name(function_name);
    auto it = pipelines_.find(name);
    if (it != pipelines_.end()) return it->second;

    id<MTLFunction> function = [library_
        newFunctionWithName:[NSString stringWithUTF8String:function_name]];
    if (function == nil) {
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   ("Metal: no shader function named '" + name + "'.").c_str());
      return nil;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (pipeline == nil) {
      const char* reason = error.localizedDescription.UTF8String;
      TF_SetStatus(status, TF_INTERNAL,
                   ("Metal: could not create a compute pipeline for '" + name +
                    "': " + (reason != nullptr ? reason : "unknown error"))
                       .c_str());
      return nil;
    }

    pipelines_.emplace(name, pipeline);  // Retained for the process lifetime.
    return pipeline;
  }

 private:
  ShaderLibrary() = default;

  bool EnsureLibraryLocked(id<MTLDevice> device, TF_Status* status)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (library_ != nil) return true;
    // A previous attempt already failed; recompiling would fail identically
    // and flood the log, so report the same outcome without retrying.
    if (compile_failed_) {
      TF_SetStatus(status, TF_INTERNAL,
                   "Metal: shader library failed to compile earlier.");
      return false;
    }

    NSError* error = nil;
    // Compiled without fast math, which Metal enables by default. Fast math
    // lets the compiler contract a multiply and an add into a fused multiply
    // and add, so a sum that is exact on the CPU comes back a fraction of a
    // unit in the last place away on the GPU. Samplers decide which pixel to
    // read by taking the floor of such a sum, so a value that should land
    // exactly on an integer instead lands just below it, and the kernel reads
    // the wrong pixel with a weight of nearly one. Correct results in the
    // ordinary case are worth more than the optimisation.
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    if ([options respondsToSelector:@selector(setMathMode:)]) {
      options.mathMode = MTLMathModeSafe;
    } else {
      options.fastMathEnabled = NO;
    }
    library_ = [[device
        newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource]
                     options:options
                       error:&error] retain];
    [options release];
    if (library_ == nil) {
      compile_failed_ = true;
      const char* reason = error.localizedDescription.UTF8String;
      LOG(ERROR) << "Metal: shader library compilation failed: "
                 << (reason != nullptr ? reason : "unknown error");
      TF_SetStatus(status, TF_INTERNAL,
                   "Metal: shader library failed to compile.");
      return false;
    }
    return true;
  }

  absl::Mutex mu_;
  id<MTLLibrary> library_ ABSL_GUARDED_BY(mu_) = nil;
  bool compile_failed_ ABSL_GUARDED_BY(mu_) = false;
  absl::flat_hash_map<std::string, id<MTLComputePipelineState>> pipelines_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace

id<MTLComputePipelineState> PipelineFor(id<MTLDevice> device,
                                        const char* function_name,
                                        TF_Status* status) {
  return ShaderLibrary::Global().PipelineFor(device, function_name, status);
}

}  // namespace metal
}  // namespace tensorflow
