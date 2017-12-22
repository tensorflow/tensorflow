/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/cpu_runtime_sse4_1.h"

#define EIGEN_USE_THREADS

#include "third_party/eigen3/Eigen/Core"

#ifdef TF_XLA_HAS_SSE4_1

xla::cpu::runtime::V4F32SSE __xla_cpu_runtime_ExpV4F32SSE(
    xla::cpu::runtime::V4F32SSE x) {
  Eigen::internal::Packet4f p = x;
  return Eigen::internal::pexp(p);
}

xla::cpu::runtime::V4F32SSE __xla_cpu_runtime_LogV4F32SSE(
    xla::cpu::runtime::V4F32SSE x) {
  Eigen::internal::Packet4f p = x;
  return Eigen::internal::plog(p);
}

#endif  // TF_XLA_HAS_SSE4_1

namespace xla {
namespace cpu {
namespace runtime {

const char *const kExpV4F32SSESymbolName = "__xla_cpu_runtime_ExpV4F32SSE";
const char *const kLogV4F32SSESymbolName = "__xla_cpu_runtime_LogV4F32SSE";

}  // namespace runtime
}  // namespace cpu
}  // namespace xla
