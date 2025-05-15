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
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "xla/client/local_client.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/stream_executor_executable.pb.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
absl::StatusOr<std::string> StreamExecutorExecutable::SerializeExecutable()
    const {
  std::string serialized;
  if (std::holds_alternative<
          std::vector<std::unique_ptr<xla::AotCompilationResult>>>(
          executables_)) {
    const auto& aot_executables =
        std::get<std::vector<std::unique_ptr<xla::AotCompilationResult>>>(
            executables_);
    if (aot_executables.empty()) {
      return absl::InternalError("No local executable");
    }
    if (aot_executables.size() != 1) {
      return absl::UnimplementedError(
          "PjRtStreamExecutorClient::SerializeExecutable unimplemented for "
          "MPMD executables");
    }
    TF_ASSIGN_OR_RETURN(serialized, aot_executables[0]->SerializeAsString());
  } else {
    const auto& local_executables =
        std::get<std::vector<std::unique_ptr<LocalExecutable>>>(executables_);
    Executable* built_executable = local_executables[0]->executable();
    CHECK(local_client_ != nullptr);
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<AotCompilationResult> aot_result,
        local_client_->backend().compiler()->Export(built_executable));

    TF_ASSIGN_OR_RETURN(serialized, aot_result->SerializeAsString());
  }

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

namespace {

absl::StatusOr<absl::string_view> MemoryKindFromSimpleShape(
    const Shape& shape, absl::string_view default_memory_kind) {
  if (!shape.has_layout()) {
    return default_memory_kind;
  }
  switch (shape.layout().memory_space()) {
    case Layout::kHostMemorySpace:
      return PinnedHostMemorySpace::kKind;
    case Layout::kGenericFastMemorySpace:
    case Layout::kDefaultMemorySpace:
      return default_memory_kind;
    default:
      return InvalidArgument("Unexpected memory space %d in output layout",
                             shape.layout().memory_space());
  }
}

absl::StatusOr<std::vector<absl::string_view>> MemoryKindsFromShape(
    const Shape& shape, absl::string_view default_memory_kind) {
  if (!shape.IsTuple()) {
    TF_ASSIGN_OR_RETURN(absl::string_view memory_kind,
                        MemoryKindFromSimpleShape(shape, default_memory_kind));
    return {{memory_kind}};
  }
  std::vector<absl::string_view> result;
  result.reserve(shape.tuple_shapes().size());
  for (const auto& element_shape : shape.tuple_shapes()) {
    TF_ASSIGN_OR_RETURN(
        absl::string_view element_memory_kind,
        MemoryKindFromSimpleShape(element_shape, default_memory_kind));
    result.push_back(element_memory_kind);
  }
  return result;
}

}  // namespace

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
StreamExecutorExecutable::GetOutputMemoryKinds() const {
  TF_ASSIGN_OR_RETURN(auto shapes, GetOutputShapes());
  std::vector<std::vector<absl::string_view>> out;
  out.reserve(shapes.size());
  for (const auto& shape : shapes) {
    TF_ASSIGN_OR_RETURN(std::vector<absl::string_view> memory_kind,
                        MemoryKindsFromShape(shape, default_memory_kind_));
    out.push_back(memory_kind);
  }
  return out;
}

absl::StatusOr<std::vector<std::unique_ptr<LocalExecutable>>>
StreamExecutorExecutable::ConsumeExecutable(
    LocalClient* client, const CompileOptions& compile_options) {
  if (std::holds_alternative<std::vector<std::unique_ptr<LocalExecutable>>>(
          executables_)) {
    return std::get<std::vector<std::unique_ptr<LocalExecutable>>>(
        std::move(executables_));
  } else if (std::holds_alternative<
                 std::vector<std::unique_ptr<xla::AotCompilationResult>>>(
                 executables_)) {
    auto aot_executables =
        std::get<std::vector<std::unique_ptr<xla::AotCompilationResult>>>(
            std::move(executables_));
    std::vector<std::unique_ptr<LocalExecutable>> local_executables;
    local_executables.reserve(aot_executables.size());
    for (int i = 0; i < aot_executables.size(); ++i) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<LocalExecutable> local_executable,
          client->Load(std::move(aot_executables[i]),
                       compile_options.executable_build_options));
      local_executables.push_back(std::move(local_executable));
    }
    return local_executables;
  }
  return absl::UnimplementedError("Unsupported executable type.");
}

}  // namespace xla
