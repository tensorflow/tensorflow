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

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_args_packed_vector.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"

namespace stream_executor {
namespace {

KernelArgPackingRelocationProto::Kind ToProtoKind(
    KernelArgPackingRelocation::Kind kind) {
  switch (kind) {
    case KernelArgPackingRelocation::Kind::kBits64Absolute:
      return KernelArgPackingRelocationProto::KIND_BITS64_ABSOLUTE;
  }
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
    absl::Span<const DeviceAddressBase> args) const {
  auto argument = storage_;

  for (const KernelArgPackingRelocation& relocation : relocations_) {
    switch (relocation.kind()) {
      case KernelArgPackingRelocation::Kind::kBits64Absolute: {
        if (args.size() <= relocation.argument_index()) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Not enough arguments for relocation (expected "
                              "at least %d, but got %d)",
                              relocation.argument_index(), args.size()));
        }
        if (relocation.offset() + sizeof(uint64_t) > argument.size()) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Not enough storage for relocation (expected "
                              "at least %d, but got %d)",
                              sizeof(void*), argument.size()));
        }
        uint64_t ptr =
            static_cast<uint64_t>(tsl::safe_reinterpret_cast<uintptr_t>(
                args.at(relocation.argument_index()).opaque()));
        std::memcpy(argument.data() + relocation.offset(), &ptr, sizeof(ptr));
        break;
      }
      default:
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported relocation kind: %d",
                            static_cast<int>(relocation.kind())));
    }
  }
  return argument;
}

void KernelArgPackingSpec::WriteArgumentAddress(int argument_index) {
  relocations_.push_back(KernelArgPackingRelocation(
      KernelArgPackingRelocation::Kind::kBits64Absolute, argument_index,
      /*offset=*/storage_.size()));
  storage_.insert(storage_.end(), sizeof(uint64_t), 0);
}

absl::StatusOr<std::unique_ptr<KernelArgsPackedVector>>
KernelArgsPackingSpec::BuildArguments(absl::Span<const DeviceAddressBase> args,
                                      size_t shared_memory_bytes) const {
  std::vector<std::vector<char>> result;
  result.reserve(kernel_arguments_.size());
  for (const KernelArgPackingSpec& kernel_argument : kernel_arguments_) {
    TF_ASSIGN_OR_RETURN(result.emplace_back(),
                        kernel_argument.BuildArgument(args));
  }
  return std::make_unique<KernelArgsPackedVector>(std::move(result),
                                                  shared_memory_bytes);
}
absl::StatusOr<KernelArgPackingSpecProto> KernelArgPackingSpec::ToProto()
    const {
  KernelArgPackingSpecProto proto;
  for (const KernelArgPackingRelocation& relocation : relocations_) {
    TF_ASSIGN_OR_RETURN(*proto.add_relocations(), relocation.ToProto());
  }
  proto.set_data(storage_.data(), storage_.size());
  return proto;
}

absl::StatusOr<KernelArgPackingSpec> KernelArgPackingSpec::FromProto(
    const KernelArgPackingSpecProto& proto) {
  std::vector<char> storage(proto.data().begin(), proto.data().end());
  std::vector<KernelArgPackingRelocation> relocations;
  for (const KernelArgPackingRelocationProto& relocation_proto :
       proto.relocations()) {
    TF_ASSIGN_OR_RETURN(
        KernelArgPackingRelocation relocation,
        KernelArgPackingRelocation::FromProto(relocation_proto));
    relocations.push_back(std::move(relocation));
  }
  return KernelArgPackingSpec(std::move(storage), std::move(relocations));
}

absl::StatusOr<KernelArgPackingRelocationProto>
KernelArgPackingRelocation::ToProto() const {
  KernelArgPackingRelocationProto proto;
  proto.set_kind(ToProtoKind(kind_));
  proto.set_argument_index(argument_index_);
  proto.set_offset(offset_);
  return proto;
}

absl::StatusOr<KernelArgPackingRelocation>
KernelArgPackingRelocation::FromProto(
    const KernelArgPackingRelocationProto& proto) {
  TF_ASSIGN_OR_RETURN(KernelArgPackingRelocation::Kind kind,
                      FromProtoKind(proto.kind()));
  return KernelArgPackingRelocation(kind, proto.argument_index(),
                                    proto.offset());
}

absl::StatusOr<KernelArgsPackingSpecProto> KernelArgsPackingSpec::ToProto()
    const {
  KernelArgsPackingSpecProto proto;
  for (const KernelArgPackingSpec& kernel_argument : kernel_arguments_) {
    TF_ASSIGN_OR_RETURN(*proto.add_kernel_arguments(),
                        kernel_argument.ToProto());
  }
  return proto;
}

absl::StatusOr<KernelArgsPackingSpec> KernelArgsPackingSpec::FromProto(
    const KernelArgsPackingSpecProto& proto) {
  std::vector<KernelArgPackingSpec> kernel_arguments;
  for (const KernelArgPackingSpecProto& kernel_argument_proto :
       proto.kernel_arguments()) {
    TF_ASSIGN_OR_RETURN(KernelArgPackingSpec kernel_argument,
                        KernelArgPackingSpec::FromProto(kernel_argument_proto));
    kernel_arguments.push_back(std::move(kernel_argument));
  }
  return KernelArgsPackingSpec(std::move(kernel_arguments));
}

}  // namespace stream_executor
