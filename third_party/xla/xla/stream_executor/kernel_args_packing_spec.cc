/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/kernel_args_packing_spec.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/status/status_macros.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_args_packed_vector.h"
#include "xla/stream_executor/kernel_args_packing_spec.pb.h"

namespace stream_executor {
namespace {

KernelArgPackingRelocationProto::Kind ToProtoKind(
    KernelArgPackingRelocation::Kind kind) {
  switch (kind) {
    case KernelArgPackingRelocation::Kind::kBits64Absolute:
      return KernelArgPackingRelocationProto::KIND_BITS64_ABSOLUTE;
  }
  LOG(FATAL) << "Unsupported relocation kind: " << static_cast<int>(kind);
  return KernelArgPackingRelocationProto::KIND_BITS64_ABSOLUTE;
}

absl::StatusOr<KernelArgPackingRelocation::Kind> FromProtoKind(
    KernelArgPackingRelocationProto::Kind kind) {
  switch (kind) {
    case KernelArgPackingRelocationProto::KIND_BITS64_ABSOLUTE:
      return KernelArgPackingRelocation::Kind::kBits64Absolute;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported relocation kind: %d", static_cast<int>(kind)));
  }
}
}  // namespace

absl::StatusOr<std::vector<char>> KernelArgPackingSpec::BuildArgument(
    absl::Span<const std::unique_ptr<PackedArgBase>> args) const {
  if (relocation_.has_value()) {
    CHECK(constant_.empty());
    switch (relocation_->kind()) {
      case KernelArgPackingRelocation::Kind::kBits64Absolute: {
        if (args.size() <= relocation_->argument_index()) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Not enough arguments for relocation (expected "
                              "at least %d, but got %zu)",
                              relocation_->argument_index() + 1, args.size()));
        }
        const auto& arg = args[relocation_->argument_index()];
        const char* arg_addr =
            static_cast<const char*>(arg->argument_address());
        return std::vector<char>(arg_addr, arg_addr + arg->size());
      }
    }
    CHECK(false) << "Unsupported relocation kind: "
                 << static_cast<int>(relocation_->kind());
  }
  return constant_;
}

KernelArgPackingSpec KernelArgPackingSpec::BuildArgRelocation(
    int argument_index) {
  CHECK_GE(argument_index, 0);
  return KernelArgPackingSpec(
      {},
      {KernelArgPackingRelocation(
          KernelArgPackingRelocation::Kind::kBits64Absolute, argument_index)});
}

absl::StatusOr<std::unique_ptr<KernelArgsPackedVector>>
KernelArgsPackingSpec::BuildArguments(
    absl::Span<const std::unique_ptr<PackedArgBase>> args,
    size_t shared_memory_bytes) const {
  std::vector<std::vector<char>> result;
  result.reserve(kernel_arguments_.size());
  for (const KernelArgPackingSpec& kernel_argument : kernel_arguments_) {
    ASSIGN_OR_RETURN(std::vector<char> arg,
                     kernel_argument.BuildArgument(args));
    result.push_back(std::move(arg));
  }
  return std::make_unique<KernelArgsPackedVector>(std::move(result),
                                                  shared_memory_bytes);
}
absl::StatusOr<KernelArgPackingSpecProto> KernelArgPackingSpec::ToProto()
    const {
  KernelArgPackingSpecProto proto;
  if (relocation_.has_value()) {
    ASSIGN_OR_RETURN(*proto.add_relocations(), relocation_->ToProto());
  } else {
    proto.set_data(constant_.data(), constant_.size());
  }
  return proto;
}

absl::StatusOr<KernelArgPackingSpec> KernelArgPackingSpec::FromProto(
    const KernelArgPackingSpecProto& proto) {
  if (proto.relocations().size() > 1) {
    return absl::InvalidArgumentError(
        "Multiple relocations not supported for single arg.");
  }
  if (proto.relocations().size() == 1) {
    if (!proto.data().empty()) {
      return absl::InvalidArgumentError(
          "Both relocation and constant data cannot be provided "
          "simultaneously.");
    }
    ASSIGN_OR_RETURN(
        KernelArgPackingRelocation relocation,
        KernelArgPackingRelocation::FromProto(proto.relocations()[0]));
    return KernelArgPackingSpec({}, std::move(relocation));
  }

  if (proto.data().empty()) {
    return absl::InvalidArgumentError(
        "Either relocation or constant has to be provided.");
  }

  std::vector<char> storage(proto.data().begin(), proto.data().end());
  return KernelArgPackingSpec(std::move(storage), std::nullopt);
}

absl::StatusOr<KernelArgPackingRelocationProto>
KernelArgPackingRelocation::ToProto() const {
  KernelArgPackingRelocationProto proto;
  proto.set_kind(ToProtoKind(kind_));
  proto.set_argument_index(argument_index_);
  return proto;
}

absl::StatusOr<KernelArgPackingRelocation>
KernelArgPackingRelocation::FromProto(
    const KernelArgPackingRelocationProto& proto) {
  ASSIGN_OR_RETURN(KernelArgPackingRelocation::Kind kind,
                   FromProtoKind(proto.kind()));
  if (proto.argument_index() < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Argument index cannot be negative: %d", proto.argument_index()));
  }
  return KernelArgPackingRelocation(kind, proto.argument_index());
}

absl::StatusOr<KernelArgsPackingSpecProto> KernelArgsPackingSpec::ToProto()
    const {
  KernelArgsPackingSpecProto proto;
  for (const KernelArgPackingSpec& kernel_argument : kernel_arguments_) {
    ASSIGN_OR_RETURN(*proto.add_kernel_arguments(), kernel_argument.ToProto());
  }
  return proto;
}

absl::StatusOr<KernelArgsPackingSpec> KernelArgsPackingSpec::FromProto(
    const KernelArgsPackingSpecProto& proto) {
  std::vector<KernelArgPackingSpec> kernel_arguments;
  kernel_arguments.reserve(proto.kernel_arguments().size());
  for (const KernelArgPackingSpecProto& kernel_argument_proto :
       proto.kernel_arguments()) {
    ASSIGN_OR_RETURN(KernelArgPackingSpec kernel_argument,
                     KernelArgPackingSpec::FromProto(kernel_argument_proto));
    kernel_arguments.push_back(std::move(kernel_argument));
  }
  return KernelArgsPackingSpec(std::move(kernel_arguments));
}

}  // namespace stream_executor
