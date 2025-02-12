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

#ifndef XLA_SERVICE_GPU_KERNELS_CUSTOM_KERNEL_FUSION_H_
#define XLA_SERVICE_GPU_KERNELS_CUSTOM_KERNEL_FUSION_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/logging.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// CustomKernelFusion
//===----------------------------------------------------------------------===//

// Custom kernel fusion is a mechanism for registering custom kernels
// corresponding to HLO fusions.
//
// Example: row-major mixed dtype gemm with fused bitcast
//
//   %gemm (parameter_0: s8[19,17], parameter_1: f16[15,19]) -> f16[15,17] {
//     %parameter_1 = f16[15,19]{1,0} parameter(1)
//     %parameter_0 = s8[19,17]{1,0} parameter(0)
//     %cp1.1 = f16[19,17]{1,0} convert(%parameter_0)
//     ROOT %r.1 = f16[15,17]{1,0} dot(%parameter_1, %cp1.1),
//                                   lhs_contracting_dims={1},
//                                   rhs_contracting_dims={0}
//  }
//
//  ENTRY %e (p0: f16[15,19], p1: s8[19,17]) -> f16[15,17] {
//    %p1 = s8[19,17]{1,0} parameter(1)
//    %p0 = f16[15,19]{1,0} parameter(0)
//    ROOT %gemm = f16[15,17]{1,0} fusion(%p1, %p0), kind=kCustom,
//                                 <implementation detail backend config>
//  }
//
// XLA:GPU has multiple strategies for executing this fusion on device:
//
// (1) cuBLAS library call: a lot of simple gemm operations are supported by
//     cuBLAS out of the box. However some combinations of parameters casting
//     and epilogue fusion are not supported, which means that XLA has to form
//     smaller fusions or use code generation to compiled a device kernel.
//
// (2) Triton: XLA:GPU uses Triton to codegen gemm fusion into device kernels
//     (PTX and CUBIN for NVIDIA gpus).
//
// (3) Custom kernel fusion is another mechanism to execute fusion on device,
// which
//     relies on pre-compiled libraries of custom kernels authored by CUDA C++
//     experts. Custom kernel fusion implements one particular fusion pattern
//     (e.g. type casting plus a dot operation like in the example above) with
//     custom kernels that XLA has to choose from at run time based on auto
//     tuning.
//
//     In practice custom kernel fusion almost always implemented with multiple
//     kernels, because input shapes are not known at compile time, and custom
//     fusion has multiple kernels with different tiling schemes.
//
// What differentiates custom kernel fusions from custom calls, is that custom
// kernel fusion should be implemented with a device kernel, and this allows
// XLA:GPU to treat custom kernel fusion just like any other device kernel: it's
// launched as a regular KernelThunk and automatically captured into command
// buffers.
//
// Custom calls (registered with XLA:FFI) on the other hand gives much more
// flexibility, and can be implemented as a combination of a non-trivial host
// side code plus multiple kernel launches or library calls.
//
// Also XLA:FFI offers a stable C API that allows registering external functions
// loaded from dynamic libraries compiled with a different toolchain of XLA
// version. Custom kernel fusions integration relies on C++ ABI and static
// linking.
//
// TODO(ezhulenev): It should be possible to lower `stablehlo.custom_call`
// operations to custom kernel fusions, albeit with a static linking
// restriction.
class CustomKernelFusion {
 public:
  virtual ~CustomKernelFusion() = default;

  // Loads kernels implementing `hlo_computation` optimized for a given device.
  virtual absl::StatusOr<std::vector<CustomKernel>> LoadKernels(
      const se::DeviceDescription& device,
      const HloComputation* computation) const = 0;
};

//===----------------------------------------------------------------------===//
// CustomKernelFusionRegistry
//===----------------------------------------------------------------------===//

// Custom fusion registry is a mapping from a custom kernel fusion name to the
// custom fusion implementation, and XLA compiler uses this registry to lower
// fusion operations to kernels when emitting thunks.
class CustomKernelFusionRegistry {
 public:
  // Returns a pointer to a default custom fusion registry, which is a global
  // static registry.
  static CustomKernelFusionRegistry* Default();

  // Registers custom kernel fusion in the registry. Returns error if fusion
  // with the given name already registered.
  absl::Status Register(std::string name,
                        std::unique_ptr<CustomKernelFusion> fusion);

  // Looks up custom kernel fusion by name. Return nullptr if it's not found.
  CustomKernelFusion* Lookup(absl::string_view name) const;

 private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, std::unique_ptr<CustomKernelFusion>>
      registry_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla::gpu

#define XLA_REGISTER_CUSTOM_FUSION(NAME, FUSION) \
  XLA_REGISTER_CUSTOM_FUSION_(NAME, FUSION, __COUNTER__)

#define XLA_REGISTER_CUSTOM_FUSION_(NAME, FUSION, N) \
  XLA_REGISTER_CUSTOM_FUSION__(NAME, FUSION, N)

#define XLA_REGISTER_CUSTOM_FUSION__(NAME, FUSION, N)                      \
  [[maybe_unused]] static const bool xla_custom_fusion_##N##_registered_ = \
      [] {                                                                 \
        absl::Status status =                                              \
            ::xla::gpu::CustomKernelFusionRegistry::Default()->Register(   \
                NAME, std::make_unique<FUSION>());                         \
        if (!status.ok()) LOG(ERROR) << status;                            \
        return status.ok();                                                \
      }()

#endif  // XLA_SERVICE_GPU_KERNELS_CUSTOM_KERNEL_FUSION_H_
