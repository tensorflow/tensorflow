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

#ifndef XLA_SERVICE_CPU_ONEDNN_LAYER_NORM_H_
#define XLA_SERVICE_CPU_ONEDNN_LAYER_NORM_H_
#if defined(INTEL_MKL)

namespace xla {
namespace cpu {

extern "C" {
extern void __xla_cpu_runtime_OneDnnLayerNorm(void* result, void** args);
}  // extern "C"

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL
#endif  // XLA_SERVICE_CPU_ONEDNN_LAYER_NORM_H_
