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

#include "xla/stream_executor/kernel_argument_packing_spec.h"

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
ArgumentPackingRelocationProto::Type ToProtoType(
    ArgumentPackingRelocation::Type type) {
  switch (type) {
    case ArgumentPackingRelocation::Type::kBits64Absolute:
      return ArgumentPackingRelocationProto::TYPE_BITS64_ABSOLUTE;
  }
}

absl::StatusOr<ArgumentPackingRelocation::Type> FromProtoType(
    ArgumentPackingRelocationProto::Type type) {
  switch (type) {
    case ArgumentPackingRelocationProto::TYPE_BITS64_ABSOLUTE:
      return ArgumentPackingRelocation::Type::kBits64Absolute;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported relocation type: %d", static_cast<int>(type)));
  }
}
}  // namespace

absl::StatusOr<std::vector<char>> SingleArgumentPackingSpec::BuildArgument(
    absl::Span<const DeviceAddressBase> args) const {
  auto argument = storage_;

  for (const ArgumentPackingRelocation& relocation : relocations_) {
    switch (relocation.type()) {
      case ArgumentPackingRelocation::Type::kBits64Absolute: {
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
            absl::StrFormat("Unsupported relocation type: %d",
                            static_cast<int>(relocation.type())));
    }
  }
  return argument;
}

void SingleArgumentPackingSpec::WriteArgumentAddress(int argument_index) {
  relocations_.push_back(ArgumentPackingRelocation(
      ArgumentPackingRelocation::Type::kBits64Absolute, argument_index,
      /*offset=*/storage_.size()));
  storage_.insert(storage_.end(), sizeof(uint64_t), 0);
}

absl::StatusOr<std::unique_ptr<KernelArgsPackedVector>>
KernelArgumentsPackingSpec::BuildArguments(
    absl::Span<const DeviceAddressBase> thunk_arguments,
    size_t shared_memory_bytes) const {
  std::vector<std::vector<char>> result;
  result.reserve(kernel_arguments_.size());
  for (const SingleArgumentPackingSpec& kernel_argument : kernel_arguments_) {
    TF_ASSIGN_OR_RETURN(result.emplace_back(),
                        kernel_argument.BuildArgument(thunk_arguments));
  }
  return std::make_unique<KernelArgsPackedVector>(std::move(result),
                                                  shared_memory_bytes);
}
absl::StatusOr<SingleArgumentPackingSpecProto>
SingleArgumentPackingSpec::ToProto() const {
  SingleArgumentPackingSpecProto proto;
  for (const ArgumentPackingRelocation& relocation : relocations_) {
    TF_ASSIGN_OR_RETURN(*proto.add_relocations(), relocation.ToProto());
  }
  proto.set_data(storage_.data(), storage_.size());
  return proto;
}

absl::StatusOr<SingleArgumentPackingSpec> SingleArgumentPackingSpec::FromProto(
    const SingleArgumentPackingSpecProto& proto) {
  std::vector<char> storage(proto.data().begin(), proto.data().end());
  std::vector<ArgumentPackingRelocation> relocations;
  for (const ArgumentPackingRelocationProto& relocation_proto :
       proto.relocations()) {
    TF_ASSIGN_OR_RETURN(ArgumentPackingRelocation relocation,
                        ArgumentPackingRelocation::FromProto(relocation_proto));
    relocations.push_back(std::move(relocation));
  }
  return SingleArgumentPackingSpec(std::move(storage), std::move(relocations));
}

absl::StatusOr<ArgumentPackingRelocationProto>
ArgumentPackingRelocation::ToProto() const {
  ArgumentPackingRelocationProto proto;
  proto.set_type(ToProtoType(type_));
  proto.set_argument_index(argument_index_);
  proto.set_offset(offset_);
  return proto;
}

absl::StatusOr<ArgumentPackingRelocation> ArgumentPackingRelocation::FromProto(
    const ArgumentPackingRelocationProto& proto) {
  TF_ASSIGN_OR_RETURN(ArgumentPackingRelocation::Type type,
                      FromProtoType(proto.type()));
  return ArgumentPackingRelocation(type, proto.argument_index(),
                                   proto.offset());
}

absl::StatusOr<KernelArgumentsPackingSpecProto>
KernelArgumentsPackingSpec::ToProto() const {
  KernelArgumentsPackingSpecProto proto;
  for (const SingleArgumentPackingSpec& kernel_argument : kernel_arguments_) {
    TF_ASSIGN_OR_RETURN(*proto.add_kernel_arguments(),
                        kernel_argument.ToProto());
  }
  return proto;
}

absl::StatusOr<KernelArgumentsPackingSpec>
KernelArgumentsPackingSpec::FromProto(
    const KernelArgumentsPackingSpecProto& proto) {
  std::vector<SingleArgumentPackingSpec> kernel_arguments;
  for (const SingleArgumentPackingSpecProto& kernel_argument_proto :
       proto.kernel_arguments()) {
    TF_ASSIGN_OR_RETURN(
        SingleArgumentPackingSpec kernel_argument,
        SingleArgumentPackingSpec::FromProto(kernel_argument_proto));
    kernel_arguments.push_back(std::move(kernel_argument));
  }
  return KernelArgumentsPackingSpec(std::move(kernel_arguments));
}

}  // namespace stream_executor
