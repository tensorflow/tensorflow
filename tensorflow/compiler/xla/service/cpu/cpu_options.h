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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_OPTIONS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_OPTIONS_H_

#include "tensorflow/compiler/xla/service/hlo_module_config.h"

// Helper functions for querying options that are specific to the CPU backend.

namespace xla {
namespace cpu {
namespace options {

bool OptimizeForSizeRequested(const HloModuleConfig& config);
bool VectorizedReduceDisabled(const HloModuleConfig& config);
bool SlpVectorizerDisabled(const HloModuleConfig& config);
bool ForceEnableExperimentalLlvmIrGemm(const HloModuleConfig& config);
std::optional<int64_t> LlvmIrGemvTilingFactor(const HloModuleConfig& config);
std::optional<std::tuple<int64_t, int64_t, int64_t>> LlvmIrGemmTileSize(
    const HloModuleConfig& config);

}  // namespace options
}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_OPTIONS_H_
