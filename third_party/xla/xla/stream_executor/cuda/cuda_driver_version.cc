/* Copyright 2015 The OpenXLA Authors.

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

#include <stdint.h>
#include <stdlib.h>

#include <cstdint>

#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "tsl/platform/errors.h"

namespace stream_executor::gpu {

absl::StatusOr<int32_t> CudaDriverVersion() {
  int32_t version;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuDriverGetVersion(&version),
                                    "Could not get driver version"));
  return version;
}

}  // namespace stream_executor::gpu
