/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_GEMM_H_
#define XLA_SERVICE_GPU_RUNTIME_GEMM_H_

#include "absl/container/node_hash_map.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/runtime/state.h"
#include "xla/service/gpu/matmul_utils.h"

namespace xla {
namespace gpu {

// Registers XLA Gpu runtime Gemm# custom calls.
void RegisterGemmCustomCalls(runtime::DirectCustomCallRegistry& registry);

// Keep GemmConfigs for all gemm/matmul instances in the executable.
class GemmConfigs : public runtime::StateVector<GemmConfig> {};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_GEMM_H_
