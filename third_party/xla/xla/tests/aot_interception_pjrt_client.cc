/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/tests/aot_interception_pjrt_client.h"

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"

namespace xla {

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
AOTInterceptionPjrtClient::Compile(const XlaComputation& computation,
                                   CompileOptions options) {
  // Skeleton implementation: directly delegate to the underlying client.
  return inner_client_->Compile(computation, std::move(options));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
AOTInterceptionPjrtClient::CompileAndLoad(const XlaComputation& computation,
                                          CompileOptions options) {
  // Skeleton implementation: directly delegate to the underlying client.
  return inner_client_->CompileAndLoad(computation, std::move(options));
}

}  // namespace xla
