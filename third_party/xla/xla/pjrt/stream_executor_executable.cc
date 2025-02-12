/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/stream_executor_executable.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/stream_executor_executable.pb.h"
#include "xla/service/compiler.h"
#include "tsl/platform/statusor.h"

namespace xla {
absl::StatusOr<std::string> StreamExecutorExecutable::SerializeExecutable()
    const {
  if (aot_executables_.empty()) {
    return absl::InternalError("No local executable");
  }
  if (aot_executables_.size() != 1) {
    return absl::UnimplementedError(
        "PjRtStreamExecutorClient::SerializeExecutable unimplemented for MPMD "
        "executables");
  }

  TF_ASSIGN_OR_RETURN(std::string serialized,
                      aot_executables_[0]->SerializeAsString());
  if (serialized.empty()) {
    return absl::InternalError(
        "PjRtStreamExecutorClient::SerializeExecutable proto serialization "
        "failed");
  }
  ExecutableAndOptionsProto proto;
  *proto.mutable_serialized_executable() = std::move(serialized);
  TF_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                      compile_options_.ToProto());
  return proto.SerializeAsString();
}
}  // namespace xla
