/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_AUTOTUNING_TRITON_CONFIGS_H_
#define XLA_SERVICE_GPU_AUTOTUNING_TRITON_CONFIGS_H_

#include <vector>

#include "xla/service/gpu/matmul_utils.h"

namespace xla {
namespace gpu {

using Config = TritonGemmConfig;

static const std::vector<TritonGemmConfig>* const kBlackwellConfigs =
    new std::vector<TritonGemmConfig>(
        {Config(128, 128, 32, 1, 4, 4), Config(128, 128, 64, 1, 1, 8),
         Config(128, 128, 64, 8, 3, 4), Config(128, 16, 16, 512, 4, 2),
         Config(128, 16, 32, 16, 3, 2), Config(128, 16, 64, 1, 5, 4),
         Config(128, 16, 64, 16, 3, 4), Config(128, 16, 64, 64, 1, 2),
         Config(128, 256, 64, 1, 4, 8), Config(128, 256, 64, 2, 4, 8),
         Config(128, 256, 64, 4, 3, 8), Config(128, 64, 64, 1, 3, 4),
         Config(128, 64, 64, 16, 4, 8), Config(128, 64, 64, 8, 4, 4),
         Config(16, 16, 128, 1, 3, 2),  Config(16, 16, 16, 1, 1, 2),
         Config(16, 16, 64, 8, 3, 2),   Config(16, 32, 64, 1, 3, 2),
         Config(256, 128, 64, 1, 3, 8), Config(256, 16, 16, 1, 1, 2),
         Config(256, 32, 32, 16, 3, 4), Config(32, 16, 32, 1, 4, 2),
         Config(32, 16, 512, 1, 1, 4),  Config(32, 16, 64, 1, 1, 2),
         Config(32, 16, 64, 1, 4, 2),   Config(64, 128, 16, 1, 1, 16),
         Config(64, 128, 16, 1, 3, 2),  Config(64, 128, 64, 1, 4, 4),
         Config(64, 16, 64, 1, 2, 2),   Config(64, 32, 128, 1, 3, 2),
         Config(64, 32, 32, 1, 4, 2),   Config(64, 32, 64, 64, 3, 2),
         Config(64, 64, 128, 8, 1, 8),  Config(64, 64, 16, 1, 1, 2),
         Config(64, 64, 16, 1, 3, 2)});

static const std::vector<TritonGemmConfig>* const kHopperAmpereConfigs =
    new std::vector<TritonGemmConfig>(
        {Config(16, 16, 64, 1, 4, 2),    Config(16, 16, 128, 1, 4, 4),
         Config(16, 16, 128, 128, 4, 2), Config(16, 16, 128, 16, 1, 2),
         Config(16, 256, 16, 1, 1, 2),   Config(32, 32, 128, 16, 1, 4),
         Config(32, 256, 32, 1, 3, 4),   Config(32, 256, 32, 16, 3, 8),
         Config(64, 16, 32, 1, 4, 2),    Config(64, 16, 32, 16, 4, 2),
         Config(64, 16, 64, 1, 1, 4),    Config(64, 16, 64, 4, 3, 2),
         Config(64, 16, 64, 16, 4, 4),   Config(64, 16, 128, 1, 4, 2),
         Config(64, 16, 128, 16, 4, 4),  Config(64, 32, 32, 1, 4, 4),
         Config(64, 32, 64, 16, 3, 4),   Config(64, 32, 128, 1, 3, 2),
         Config(64, 32, 128, 128, 2, 4), Config(64, 64, 32, 1, 4, 4),
         Config(64, 64, 64, 1, 4, 4),    Config(64, 64, 64, 4, 4, 4),
         Config(64, 64, 128, 16, 3, 4),  Config(64, 64, 256, 16, 4, 8),
         Config(64, 128, 16, 1, 4, 2),   Config(64, 128, 64, 1, 3, 4),
         Config(64, 128, 128, 8, 1, 4),  Config(64, 256, 32, 1, 4, 4),
         Config(128, 16, 32, 8, 4, 2),   Config(128, 16, 64, 16, 3, 2),
         Config(128, 16, 64, 16, 1, 4),  Config(128, 32, 32, 8, 4, 2),
         Config(128, 128, 32, 8, 4, 8),  Config(128, 256, 32, 1, 4, 8),
         Config(128, 256, 64, 1, 4, 8)});

static const std::vector<TritonGemmConfig>* const kDefaultCudaConfigs =
    new std::vector<TritonGemmConfig>(
        {Config(32, 32, 256, 1, 1, 4),   Config(64, 32, 32, 16, 1, 4),
         Config(32, 64, 64, 4, 1, 4),    Config(128, 128, 64, 4, 1, 4),
         Config(16, 16, 256, 1, 1, 4),   Config(16, 128, 32, 16, 1, 4),
         Config(16, 64, 128, 1, 1, 4),   Config(16, 128, 32, 8, 1, 4),
         Config(16, 16, 512, 1, 1, 4),   Config(32, 16, 512, 1, 1, 4),
         Config(64, 32, 64, 1, 2, 8),    Config(128, 256, 32, 1, 3, 8),
         Config(256, 128, 32, 1, 3, 8),  Config(256, 64, 32, 1, 4, 4),
         Config(64, 256, 32, 1, 4, 4),   Config(128, 64, 32, 1, 4, 4),
         Config(64, 128, 32, 1, 4, 4),   Config(256, 128, 128, 1, 3, 8),
         Config(256, 64, 128, 1, 4, 4),  Config(64, 256, 128, 1, 4, 4),
         Config(128, 128, 128, 1, 4, 4), Config(128, 64, 64, 1, 4, 4),
         Config(64, 128, 64, 1, 4, 4),   Config(128, 32, 64, 1, 4, 4),
         Config(64, 32, 64, 1, 4, 4),    Config(32, 128, 32, 1, 4, 4),
         Config(128, 128, 32, 1, 4, 4),  Config(16, 16, 256, 1, 3, 4),
         Config(128, 128, 64, 2, 1, 8),  Config(64, 64, 64, 1, 2, 4),
         Config(16, 64, 256, 8, 1, 4),   Config(256, 256, 128, 1, 3, 8)});

static const std::vector<TritonGemmConfig>* const kDefaultRocmConfigs =
    new std::vector<TritonGemmConfig>(
        {Config(32, 32, 256, 1, 1, 4), Config(64, 32, 32, 16, 1, 4),
         Config(32, 64, 64, 4, 1, 4), Config(128, 128, 64, 4, 1, 4),
         Config(16, 16, 256, 1, 1, 4), Config(16, 128, 32, 16, 1, 4)});

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_AUTOTUNING_TRITON_CONFIGS_H_
