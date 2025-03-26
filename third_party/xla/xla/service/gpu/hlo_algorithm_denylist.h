/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_HLO_ALGORITHM_DENYLIST_H_
#define XLA_SERVICE_GPU_HLO_ALGORITHM_DENYLIST_H_

#include <string>
#include <vector>

#include "xla/autotuning.pb.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/dnn.h"

namespace xla {
namespace gpu {

// Get the list of convolution algorithms which are disabled for the given 'hlo'
// when using compute capability 'cc', cudnn version 'cudnn_version' and blas
// version 'blas_version'. In addition to the hardcoded denylist used in this
// function, extra entries for the denylist can be added via a file pointed to
// by the --xla_gpu_algorithm_denylist_path flag.
std::vector<stream_executor::dnn::AlgorithmDesc> GetDisabledConvAlgorithms(
    ComputeCapability cc, CudnnVersion cudnn_version,
    const std::string& blas_version, const std::string& hlo);

// Attaches a serialized backend config to the given HLO string.
std::string HloStringWithGpuBackendConfig(const std::string& hlo,
                                          GpuBackendConfig config);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_HLO_ALGORITHM_DENYLIST_H_
