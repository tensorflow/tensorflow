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

#ifndef XLA_SERVICE_CPU_ONEDNN_MATMUL_H_
#define XLA_SERVICE_CPU_ONEDNN_MATMUL_H_
#if defined(INTEL_MKL)

#include "dnnl.hpp"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/shape.h"

namespace xla {
namespace cpu {

constexpr auto kOnednnMatmulConfig =
    BackendConfigOneofCase::kOnednnMatmulConfig;

Shape OneDnnMatMulOptWeightsShape(const Shape& input_shape,
                                  const Shape& weights_shape,
                                  const Shape& bias_shape,
                                  const Shape& output_shape,
                                  const OneDnnMatMulConfig* matmul_config);

extern "C" {
extern void __xla_cpu_runtime_OneDnnMatMul(void* result, void* scratch,
                                           void** args);
extern void __xla_cpu_runtime_OneDnnMatMulReorder(void* result, void** args);
}  // extern "C"

template <>
struct PrimitiveTrait<kOnednnMatmulConfig> {
  using pointer_type = xla::cpu::OneDnnMatMulConfig*;
  static const BackendConfigOneofCase kConfigVal = kOnednnMatmulConfig;
};

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL
#endif  // XLA_SERVICE_CPU_ONEDNN_MATMUL_H_
