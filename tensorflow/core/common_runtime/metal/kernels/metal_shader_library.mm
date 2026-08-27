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
    library_ = [[device
        newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource]
                     options:nil
                       error:&error] retain];
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
