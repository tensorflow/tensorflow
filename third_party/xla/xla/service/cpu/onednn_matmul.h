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

#ifndef XLA_SERVICE_CPU_ONEDNN_MATMUL_H_
#define XLA_SERVICE_CPU_ONEDNN_MATMUL_H_
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

namespace xla {
namespace cpu {

extern "C" {
// TODO(intel-tf): Change the function signature as
//    void onednn_matmul(void* result, void** args)
// where
//        args[0]: num_args (>=3, including itself)
//        args[1]: ExecutableRunOption
//        args[2]: OneDnnMatMulConfig
//        args[3...]: Actual Operands
// so that it can take variable number of arguments.
//
// For now, we are using a fixed number of arguments.
extern void __xla_cpu_runtime_OneDnnMatMul(const void* run_options_ptr,
                                           void* lhs, void* rhs, void* result,
                                           void* config);
}  // extern "C"

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
#endif  // XLA_SERVICE_CPU_ONEDNN_MATMUL_H_
