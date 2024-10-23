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

#ifndef XLA_SERVICE_CPU_ONEDNN_UTIL_H_
#define XLA_SERVICE_CPU_ONEDNN_UTIL_H_

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"
#include "dnnl.hpp"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_config.pb.h"
#include "xla/tsl/util/onednn_threadpool.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"

namespace xla {
namespace cpu {

inline bool IsSupportedType(xla::PrimitiveType dtype) {
  using tsl::port::CPUFeature;
  // TODO(intel-tf): Enable more types.
  switch (dtype) {
    case F32:
      return true;
    case BF16:
      return TestCPUFeature(CPUFeature::AVX512F) ||
             TestCPUFeature(CPUFeature::AVX_NE_CONVERT) ||
             TestCPUFeature(CPUFeature::AMX_BF16);
    case F16:
      return (TestCPUFeature(CPUFeature::AVX512BW) &&
              (TestCPUFeature(CPUFeature::AVX512_FP16) ||
               TestCPUFeature(CPUFeature::AMX_FP16))) ||
             TestCPUFeature(CPUFeature::AVX_NE_CONVERT);
    default:
      return false;
  }
  return false;
}

struct FusedOperandsRef {
  const std::vector<void*>& bufs;
  std::vector<std::pair<int, dnnl::memory>>& postop_args;
};

std::unique_ptr<tsl::OneDnnThreadPool> CreateOneDnnThreadPool(
    const Eigen::ThreadPoolDevice* threadpool_device);

dnnl::stream MakeOneDnnStream(
    const dnnl::engine& cpu_engine,
    dnnl::threadpool_interop::threadpool_iface* thread_pool);

typedef BackendConfig::BackendConfigOneofCase BackendConfigOneofCase;

// These template functions must have explicit specialization at the definition
// site.
template <typename PrimDesc>
std::unique_ptr<PrimDesc> CreateOneDnnPrimDesc(HloInstruction*);

template <BackendConfigOneofCase config>
struct PrimitiveTrait;

template <BackendConfigOneofCase config>
typename PrimitiveTrait<config>::pointer_type GetKernelConfig(
    absl::StatusOr<BackendConfig>*);

dnnl::post_ops PopulateOneDnnPostOps(
    const dnnl::engine& cpu_engine,
    const std::vector<dnnl::memory::desc>& fused_mds,
    const OneDnnFusionConfig* fusion_config, const int output_ndims,
    FusedOperandsRef* fused_operands_ref = nullptr,
    dnnl::memory::desc* bias_md = nullptr);

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
#endif  // XLA_SERVICE_CPU_ONEDNN_UTIL_H_
