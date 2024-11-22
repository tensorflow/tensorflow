/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/collectives/nccl_communicator.h"

#include <string>

#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "tsl/platform/logging.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200
#else
#include "third_party/nccl/nccl.h"
#endif  // TENSORFLOW_USE_ROCM

namespace xla::gpu {

NcclCommunicator::NcclCommunicator(ncclComm_t comm) : comm_(comm) {
  VLOG(1) << "Created " << *this;
}

NcclCommunicator::~NcclCommunicator() {
  VLOG(1) << "Destroy " << *this;
  XLA_NCCL_LOG_IF_ERROR(ncclCommDestroy(comm_));
}

std::string NcclCommunicator::ToString() const {
  return absl::StrFormat("NccCommunicator(ncclComm_t=%p)", comm_);
}

}  // namespace xla::gpu
