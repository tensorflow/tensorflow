/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/builder/xla_computation.h"

#include <memory>

#include "absl/status/statusor.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/util.h"

namespace xla {

absl::StatusOr<ProgramShape> XlaComputation::GetProgramShape() const {
  TF_RET_CHECK(proto_.has_host_program_shape());
  return ProgramShape::FromProto(proto_.host_program_shape());
}

absl::StatusOr<std::unique_ptr<HloSnapshot>> XlaComputation::Snapshot() const {
  if (IsNull()) {
    return InvalidArgument("Computation is invalid.");
  }
  auto session = std::make_unique<HloSnapshot>();
  *session->mutable_hlo()->mutable_hlo_module() = proto_;
  return session;
}

}  // namespace xla
