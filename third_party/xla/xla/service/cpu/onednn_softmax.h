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

#ifndef XLA_SERVICE_CPU_ONEDNN_SOFTMAX_H_
#define XLA_SERVICE_CPU_ONEDNN_SOFTMAX_H_
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

namespace xla {
namespace cpu {

extern "C" {
extern void __xla_cpu_runtime_OneDnnSoftmax(const void* run_options_ptr,
                                            void* input, void* result,
                                            void* softmax_config_ptr);
}  // extern "C"

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && && ENABLE_ONEDNN_V3
#endif  // XLA_SERVICE_CPU_ONEDNN_SOFTMAX_H_
