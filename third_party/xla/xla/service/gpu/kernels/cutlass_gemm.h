/* Copyright 2023 The OpenXLA Authors.

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
#include <string>

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

enum class Arch { kDefault, kSm80, kSm90 };

// Keep in sync with cutlass::gemm::GemmUniversalMode.
enum class GemmMode { kGemm, kGemmSplitKParallel, kBatched, kArray, kInvalid };

template <Arch arch>
struct Bf16xBf16ToBf16 {};

template <Arch arch>
struct F32xF32ToF32 {};

template <Arch arch>
struct Bf16xBf16ToF32 {};

template <Arch arch>
struct F32xBf16ToF32 {};

template <Arch arch>
struct Bf16xS8ToF32 {};

// A tag to specialize CUTLASS kernel adaptors for loading kernels from shared
// libraries using dlopen.
struct DlOpenedKernel {};

//===----------------------------------------------------------------------===//
// CUTLASS gemm arguments
//===----------------------------------------------------------------------===//

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

  // Workspace parameter is a special case, as it's always passed as a last
  // parameter at run time (only if requested).
  bool has_workspace;
};

// Custom CUTLASS gemm kernels support on-device address arithmetics for input
// and output buffers, so that we can fuse dynamic-slice/dynamic-update-slice
// operations into the GEMM kernel.
//
// Base pointers and memory layout known on the host before kernel launch, but
// offsets are computed on device and available only in device memory. We can't
// load offsets to the host as it would require stream synchronization.
//
// Following structs encode how dynamic offsets passed to custom kernels.
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
//   DynamicSliceArguments::out = <pointer to p2 buffer>
//
// `DynamicSliceIndices` used in the host-code to fetch device memory pointers
// from arguments and pass it as `DynamicSliceArguments` to a device kernel.
//
// Kernel arguments packing function can pass dynamic slices as a part of
// CUTLASS kernel parameters, or as a separate argument to a device kernel entry
// function (CUTLASS 3x vs 2x).

// Indices of a custom fusion parameters corresponding to dynamic slice offsets.
struct DynamicSliceIndices {
  // Index of a dynamic slice offset along the major dimension.
  std::optional<int64_t> out;
};

// Pointers to buffers (s32[] buffers in HLO) holding dynamic slice offsets.
struct DynamicSliceArguments {
  int32_t* out = nullptr;
};

// Type-erased CUTLASS gemm arguments structure that has all of the details
// required for packing CUTLASS kernel parameters.
struct Arguments {
  GemmMode mode;

  // Number of batches when mode is `kBatched`.
  // Number of k-slices when mode is `kGemmSplitKParallel`.
  int32_t batch_count;

  int32_t m;
  int32_t n;
  int32_t k;

  void* lhs;
  void* rhs;
  void* out;
  void* workspace;

  DynamicSliceArguments slices;
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
template <typename Tag>
class Adaptor {
 public:
  std::optional<Dim3> ClusterDim() const;
  Dim3 BlockDim(int32_t m, int32_t n, int32_t k) const;
  Dim3 ThreadDim() const;

  int32_t SharedMemoryBytes() const;

  bool CanImplement(const Arguments& args) const;
  int64_t WorkspaceSize(const Arguments& args) const;

  void Initialize(void* params, const Arguments& args, int32_t device_sms,
                  int32_t sm_occupancy) const;
};

// This is a specialization of adaptor that can load CUTLASS kernels from
// pre-compiled shared libraries on disk. Libraries can be compiled ahead of
// time using external toolchain, e.g. NVCC, as long as they export required
// symbols with a plain C calling convention.
template <>
class Adaptor<DlOpenedKernel> {
 public:
  static std::optional<Adaptor> Load(const std::string& path);

  std::optional<Dim3> ClusterDim() const;
  Dim3 BlockDim(int32_t m, int32_t n, int32_t k) const;
  Dim3 ThreadDim() const;

  int32_t SharedMemoryBytes() const;

  bool CanImplement(const Arguments& args) const;
  int64_t WorkspaceSize(const Arguments& args) const;

  void Initialize(void* params, const Arguments& args, int32_t device_sms,
                  int32_t sm_occupancy) const;

 private:
  Adaptor(void* handle, void* block_dim_fn, void* thread_dim_fn,
          void* shared_memory_bytes_fn, void* can_implement_fn,
          void* workspace_size_fn, void* initialize_fn);

  void* handle_;
  void* block_dim_fn_;
  void* thread_dim_fn_;
  void* shared_memory_bytes_fn_;
  void* can_implement_fn_;
  void* workspace_size_fn_;
  void* initialize_fn_;
};

//===----------------------------------------------------------------------===//
// CUTLASS Device Side Adaptor
//===----------------------------------------------------------------------===//

// We keep device side adaptor separate from host side adaptor so that we could
// easily split host and device code compilation if needed.

template <typename Tag>
class DeviceKernel {
 public:
  void* symbol() const;
};

// This is a specialization of device kernel for loading CUTLASS kernels from
// shared libraries on disk (see Adaptor specialization above).
template <>
class DeviceKernel<DlOpenedKernel> {
 public:
  static std::optional<DeviceKernel> Load(const std::string& path);

  void* symbol() const;

 private:
  DeviceKernel(void* handle, void* symbol_fn);

  void* handle_;
  void* symbol_fn_;
};

}  // namespace xla::gpu::kernel::gemm_universal

#endif  // XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_H_
