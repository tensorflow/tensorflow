/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "xla/autotuning.pb.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

absl::Span<const stream_executor::dnn::AlgorithmDesc> GetDisabledConvAlgorithms(
    ComputeCapability cc, CudnnVersion cudnn_version,
    const std::string& blas_version, const std::string& hlo);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_HLO_ALGORITHM_DENYLIST_H_
