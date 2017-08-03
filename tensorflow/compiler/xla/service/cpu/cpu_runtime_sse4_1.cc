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

#ifdef __SSE4_1__

xla::cpu::runtime::V4F32 __xla_cpu_runtime_ExpV4F32(
    xla::cpu::runtime::V4F32 x) {
  Eigen::internal::Packet4f p = x;
  return Eigen::internal::pexp(p);
}

xla::cpu::runtime::V4F32 __xla_cpu_runtime_LogV4F32(
    xla::cpu::runtime::V4F32 x) {
  Eigen::internal::Packet4f p = x;
  return Eigen::internal::plog(p);
}

#endif  // __SSE4_1__

namespace xla {
namespace cpu {
namespace runtime {

const char *const kExpV4F32SymbolName = "__xla_cpu_runtime_ExpV4F32";
const char *const kLogV4F32SymbolName = "__xla_cpu_runtime_LogV4F32";

}  // namespace runtime
}  // namespace cpu
}  // namespace xla
