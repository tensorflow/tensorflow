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

#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"

#include <memory>

#include "absl/status/statusor.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_client.h"

namespace xla {

absl::StatusOr<std::unique_ptr<PjRtClient>> GetXlaPjrtCpuClient(
    CpuClientOptions options) {
  // TODO(masonchang): Wrap the PjRtCPU Client inside the PJRT Sandwich.
  return xla::GetPjRtCpuClient(options);
}

}  // namespace xla
