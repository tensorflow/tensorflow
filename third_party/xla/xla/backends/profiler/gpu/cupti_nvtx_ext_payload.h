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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_NVTX_EXT_PAYLOAD_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_NVTX_EXT_PAYLOAD_H_

#include <cstdint>
#include <string>

#include "third_party/gpus/cuda/include/nvtx3/nvToolsExtPayload.h"

namespace xla {
namespace profiler {

void CuptiParseNvtxPayload(uint32_t cupti_domain_id,
                           nvtxPayloadData_t* payload_data,
                           std::string& result_str);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_NVTX_EXT_PAYLOAD_H_
