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

#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace gpu {

struct NcclCollectiveConfig::AuxData {};

NcclCollectiveConfig::NcclCollectiveConfig() = default;
NcclCollectiveConfig::NcclCollectiveConfig(NcclCollectiveConfig &&) = default;
NcclCollectiveConfig::~NcclCollectiveConfig() = default;
NcclCollectiveConfig &NcclCollectiveConfig::operator=(NcclCollectiveConfig &&) =
    default;

NcclCollectiveConfig GetNcclCollectiveConfig(const HloInstruction *hlo,
                                             int64 replica_count) {
  return NcclCollectiveConfig();
}

/* static */ bool NcclCollectiveThunk::NcclIsEnabled() {
  return false;  // Skylark selects this source file if NCCL is disabled.
}

Status NcclCollectiveThunk::ExecuteOnStream(const ExecuteParams &) {
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
}

}  // namespace gpu
}  // namespace xla
