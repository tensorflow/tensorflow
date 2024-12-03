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

#ifndef XLA_SERVICE_GPU_GPU_P2P_PIPELINER_H_
#define XLA_SERVICE_GPU_GPU_P2P_PIPELINER_H_

#include "xla/hlo/pass/hlo_pass_pipeline.h"

namespace xla {
namespace gpu {
// Adds a collective-pipeliner pass for pipelining P2P Send-Recv chains.
void AddP2PPipeliner(HloPassPipeline& pipeline);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_P2P_PIPELINER_H_
