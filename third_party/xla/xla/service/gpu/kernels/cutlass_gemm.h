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

#ifndef XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_H_
#define XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_H_

//===-------------------------------------------------------------------------//
//                 ! ! ! ! !      WARNING      ! ! ! ! !                      //
//===-------------------------------------------------------------------------//
//                                                                            //
//   Do not add external dependencies to this header. Use only std library.   //
//                                                                            //
//===-------------------------------------------------------------------------//
//                 ! ! ! ! !      WARNING      ! ! ! ! !                      //
//===-------------------------------------------------------------------------//

#include <cstdint>
#include <optional>

namespace xla::gpu::kernel::gemm_universal {

//===----------------------------------------------------------------------===//
// Tag based GEMM dispatching
//===----------------------------------------------------------------------===//

// We use tag-based template specializations to carefully avoid including
// CUTLASS headers into regular libraries, and specialize templates in separate
// CUDA build targets that have no dependencies on other parts of XLA or ABSL to
// enable parallel compilation and minimize recompilations on code changes.
//
// Here we re-define some of the enums and types defined in CUTLASS and CUTE to
// break a dependency on them from XLA.

enum class Arch { kDefault, kSm80 };

template <Arch arch>
struct Bf16xBf16ToBf16 {};

template <Arch arch>
struct F32xF32ToF32 {};

//===----------------------------------------------------------------------===//
// CUTLASS gemm arguments
//===----------------------------------------------------------------------===//

struct Arguments {
  int32_t m;
  int32_t n;
  int32_t k;

  void* a;
  void* b;
  void* c;
};

// Indices of a custom fusion parameters corresponding to Gemm kernel arguments.
//
// Example:
//   se::KernelArgsDeviceMemoryArray args = ...
//   void* lhs = args->device_memory_ptr(indices.lhs);
//
// Custom fusion instruction can have parameters in arbitrary order, and we need
// a mapping from a custom kernel argument to the fusion instruction parameter.
struct ArgsIndices {
  int64_t lhs;
  int64_t rhs;
  int64_t out;
};

// Following structs encode how a custom kernel arguments packing and a custom
// CUTLASS kernel itself can find dynamic-slice offsets at run time.
//
// Example: CUTLASS gemm with a dynamic-update-slice
//
//   cutlass_gemm {
//     p0 = f32[2,2]{1,0} parameter(0)
//     p1 = f32[2,2,2]{2,1,0} parameter(1)
//     p2 = s32[] parameter(2)               <--- major dim offset
//     p3 = s32[] parameter(3)               <--- minor dims offset
//     dot = f32[2,2]{1,0} dot(p0, p0)
//     ...
//     ROOT r = f32[2,2,2]{2,1,0} dynamic-update-slice(p1, ..., p2, p3, p3)
//   }
//
// In this example `p2` parameter defines a dynamic slice offset along the
// major dimension (0-th dimension for a row major layout). In practice
// parameters can be passed to fusions in arbitrary order, and when we pack
// custom kernel arguments into device kernel parameters we need to know
// how to find correct device pointers in the list of fusion arguments.
//
// For this example:
//
//   DynamicSliceIndices::out = 2
//   DynamicSliceParams::out = <pointer to p2 buffer>
//
// `DynamicSliceIndices` used in the host-code to fetch device memory pointers
// from arguments and pass it as `DynamicSliceParams` to a device kernel.
//
// Example:
//   se::KernelArgsDeviceMemoryArray args = ...
//   void* out_ptr = args->device_memory_ptr(*slice_indices.out);
//
//   DynamicSliceParams params { // this struct passed to a kernel
//     out_ptr,                  // kernel loads offset value from this pointer
//     ...
//   };
//

// TODO(ezhulenev): Support dynamic slices along all dimensions, today we assume
// that we can slice only along the leading dimension (batch).

// Indices of a custom fusion parameters corresponding to dynamic slice offsets.
struct DynamicSliceIndices {
  // Index of a dynamic slice offset along the major dimension.
  std::optional<int64_t> out;
};

// Pointers to buffers (s32[] buffers in HLO) holding dynamic slice offsets.
struct DynamicSliceParams {
  // Dynamic slice offset along the major dimension.
  std::optional<int32_t*> out;
};

//===----------------------------------------------------------------------===//
// CUTLASS Host Side Adaptor
//===----------------------------------------------------------------------===//

template <typename Tag>
struct Traits;

struct Dim3 {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

// This is a type-erased adaptor that has all details required for launching
// CUTLASS kernel on a device. At run time device kernel parameters is really
// just a bag of bytes that driver sends to a kernel, so we rely on it to hide
// CUTLASS templates inside individual build targets and don't leak them into
// XLA, as they contain device code and can't be parsed by regular clang.
//
// TODO(ezhulenev): For simplicity adaptor has all functions defined as static,
// however we should support adaptors for loading kernels from disk, and all
// functions should become member functions. Figure out how to do it!
template <typename Tag>
struct Adaptor {
  static int32_t shared_memory_bytes();

  static std::optional<Dim3> ClusterDim();
  static Dim3 BlockDim(int32_t m, int32_t n, int32_t k);
  static Dim3 ThreadDim();

  static bool CanImplement(const Arguments& args);
  static void Initialize(void* params, const Arguments& args,
                         int32_t device_sms, int32_t sm_occupancy);
};

//===----------------------------------------------------------------------===//
// CUTLASS Device Side Adaptor
//===----------------------------------------------------------------------===//

// We keep device side adaptor separate from host side adaptor so that we could
// easily split host and device code compilation if needed.

template <typename Tag>
struct DeviceKernel {
  static void* symbol();
};

}  // namespace xla::gpu::kernel::gemm_universal

#endif  // XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_H_
