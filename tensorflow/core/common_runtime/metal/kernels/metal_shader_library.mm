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

// Shader source for the backend's elementwise kernels.
//
// ElementwiseParams here must stay layout-identical to the C++ struct in the
// header. Broadcasting is deliberately limited to the scalar case: full
// NumPy-style broadcasting needs stride arithmetic per operand, which belongs
// with the wider op coverage rather than in this first set. Kernels that
// receive shapes they cannot handle reject them on the host side with a clear
// message instead of computing something wrong here.
//
// Every kernel bounds-checks against `count` so that a caller using
// dispatchThreadgroups:, which rounds the grid up to whole threadgroups,
// cannot write past the end of the output.
constexpr char kShaderSource[] = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct ElementwiseParams {
  uint count;
  uint lhs_is_scalar;
  uint rhs_is_scalar;
  uint padding;
};

#define TF_METAL_BINARY(NAME, T, EXPR)                                     \
  kernel void NAME(device const T* lhs [[buffer(0)]],                      \
                   device const T* rhs [[buffer(1)]],                      \
                   device T* out [[buffer(2)]],                            \
                   constant ElementwiseParams& params [[buffer(3)]],       \
                   uint gid [[thread_position_in_grid]]) {                 \
    if (gid >= params.count) return;                                       \
    const T a = lhs[params.lhs_is_scalar ? 0 : gid];                       \
    const T b = rhs[params.rhs_is_scalar ? 0 : gid];                       \
    out[gid] = (EXPR);                                                     \
  }

TF_METAL_BINARY(tf_add_float, float, a + b)
TF_METAL_BINARY(tf_add_half, half, a + b)
TF_METAL_BINARY(tf_sub_float, float, a - b)
TF_METAL_BINARY(tf_sub_half, half, a - b)
TF_METAL_BINARY(tf_mul_float, float, a * b)
TF_METAL_BINARY(tf_mul_half, half, a * b)

#define TF_METAL_CAST(NAME, IN_T, OUT_T)                                   \
  kernel void NAME(device const IN_T* in [[buffer(0)]],                    \
                   device OUT_T* out [[buffer(1)]],                        \
                   constant ElementwiseParams& params [[buffer(2)]],       \
                   uint gid [[thread_position_in_grid]]) {                 \
    if (gid >= params.count) return;                                       \
    out[gid] = static_cast<OUT_T>(in[gid]);                                \
  }

TF_METAL_CAST(tf_cast_float_to_half, float, half)
TF_METAL_CAST(tf_cast_half_to_float, half, float)
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
