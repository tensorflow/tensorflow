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

#include <cstdint>

#include "xla/service/gpu/kernels/cutlass_gemm.h"

namespace xla::gpu::kernel::gemm_universal {

using CutlassGemm = F32xF32ToF32<Arch::kDefault>;

extern template class Adaptor<CutlassGemm>;
extern template class DeviceKernel<CutlassGemm>;

extern "C" void xla_cutlass_kernel_block_dim(int32_t m, int32_t n, int32_t k,
                                             uint32_t* x, uint32_t* y,
                                             uint32_t* z) {
  Adaptor<CutlassGemm> adaptor;
  auto dim = adaptor.BlockDim(m, n, k);
  *x = dim.x;
  *y = dim.y;
  *z = dim.z;
}

extern "C" void xla_cutlass_kernel_thread_dim(uint32_t* x, uint32_t* y,
                                              uint32_t* z) {
  Adaptor<CutlassGemm> adaptor;
  auto dim = adaptor.ThreadDim();
  *x = dim.x;
  *y = dim.y;
  *z = dim.z;
}

extern "C" int32_t xla_cutlass_kernel_shared_memory_bytes() {
  Adaptor<CutlassGemm> adaptor;
  return adaptor.SharedMemoryBytes();
}

extern "C" bool xla_cutlass_kernel_can_implement(int32_t m, int32_t n,
                                                 int32_t k) {
  Adaptor<CutlassGemm> adaptor;
  Arguments arguments = {GemmMode::kGemm, /*batch_count=*/1, m, n, k};
  return adaptor.CanImplement(arguments);
}

extern "C" int64_t xla_cutlass_kernel_workspace_size(int32_t m, int32_t n,
                                                     int32_t k) {
  Adaptor<CutlassGemm> adaptor;
  Arguments arguments = {GemmMode::kGemm, /*batch_count=*/1, m, n, k};
  return adaptor.WorkspaceSize(arguments);
}

extern "C" void xla_cutlass_kernel_initialize(
    void* params, int32_t m, int32_t n, int32_t k, void* lhs, void* rhs,
    void* out, void* workspace, int32_t* out_offset, int32_t device_sms,
    int32_t sm_occupancy) {
  Adaptor<CutlassGemm> adaptor;
  Arguments arguments = {
      GemmMode::kGemm, /*batch_count=*/1, m, n, k, lhs, rhs, out,
      workspace,       {out_offset}};
  adaptor.Initialize(params, arguments, device_sms, sm_occupancy);
}

extern "C" void* xla_cutlass_kernel_symbol() {
  DeviceKernel<CutlassGemm> kernel;
  return kernel.symbol();
}

}  // namespace xla::gpu::kernel::gemm_universal
