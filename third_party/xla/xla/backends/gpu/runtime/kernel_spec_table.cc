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
#include "xla/backends/gpu/runtime/kernel_spec_table.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.pb.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/stream_executor/kernel_spec.pb.h"
#include "xla/tsl/lib/strings/proto_serialization.h"

namespace xla::gpu {
namespace {

using ::google::protobuf::Descriptor;
using ::google::protobuf::FieldDescriptor;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;

// Applies `fn` to every `CustomKernelProto` reachable from `message`.
//
// The walk is driven by proto reflection instead of an explicit list of the
// nesting thunk kinds, so newly added nested thunks are handled automatically.
// `Reflection::ListFields` returns fields ordered by field number, which makes
// the traversal - and therefore the index assignment it drives - deterministic.
absl::Status ForEachCustomKernel(
    Message& message, absl::FunctionRef<absl::Status(CustomKernelProto&)> fn) {
  const Descriptor* descriptor = message.GetDescriptor();
  if (descriptor == CustomKernelProto::descriptor()) {
    return fn(static_cast<CustomKernelProto&>(message));
  }

  const Reflection* reflection = message.GetReflection();
  std::vector<const FieldDescriptor*> fields;
  reflection->ListFields(message, &fields);
  for (const FieldDescriptor* field : fields) {
    if (field->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE) {
      continue;
    }
    if (field->is_repeated()) {
      const int size = reflection->FieldSize(message, field);
      for (int i = 0; i < size; ++i) {
        ABSL_RETURN_IF_ERROR(ForEachCustomKernel(
            *reflection->MutableRepeatedMessage(&message, field, i), fn));
      }
    } else {
      ABSL_RETURN_IF_ERROR(ForEachCustomKernel(
          *reflection->MutableMessage(&message, field), fn));
    }
  }
  return absl::OkStatus();
}

}  // namespace

int32_t KernelSpecTable::Append(stream_executor::KernelLoaderSpecProto spec) {
  const uint64_t hash = tsl::DeterministicProtoHash64(spec);
  const int32_t index = static_cast<int32_t>(specs_.size());
  specs_.push_back(std::move(spec));
  index_[hash].push_back(index);
  return index;
}

int32_t KernelSpecTable::Intern(stream_executor::KernelLoaderSpecProto spec) {
  const uint64_t hash = tsl::DeterministicProtoHash64(spec);
  const absl::InlinedVector<int32_t, 1>* bucket = nullptr;
  if (auto it = index_.find(hash); it != index_.end()) {
    bucket = &it->second;
  }
  if (bucket != nullptr) {
    for (const int32_t candidate : *bucket) {
      if (tsl::AreSerializedProtosEqual(specs_[candidate], spec)) {
        return candidate;
      }
    }
  }
  return Append(std::move(spec));
}

absl::StatusOr<const stream_executor::KernelLoaderSpecProto*>
KernelSpecTable::Get(int32_t index) const {
  if (index < 0 || index >= static_cast<int64_t>(specs_.size())) {
    return absl::OutOfRangeError(absl::StrFormat(
        "Kernel spec index %d is out of range; the table holds %d specs.",
        index, specs_.size()));
  }
  return &specs_[index];
}

absl::Status InternKernelSpecs(ThunkProto& thunk, KernelSpecTable& table) {
  return ForEachCustomKernel(
      thunk, [&table](CustomKernelProto& custom_kernel) -> absl::Status {
        if (!custom_kernel.has_kernel_spec()) {
          return absl::OkStatus();
        }
        const int32_t index =
            table.Intern(std::move(*custom_kernel.mutable_kernel_spec()));
        // Setting the index clears the (moved-from) inline spec, since both
        // live in the same oneof.
        custom_kernel.set_kernel_spec_index(index);
        return absl::OkStatus();
      });
}

absl::Status InlineKernelSpecs(ThunkProto& thunk,
                               const KernelSpecTable& table) {
  return ForEachCustomKernel(
      thunk, [&table](CustomKernelProto& custom_kernel) -> absl::Status {
        if (!custom_kernel.has_kernel_spec_index()) {
          return absl::OkStatus();
        }
        ABSL_ASSIGN_OR_RETURN(const stream_executor::KernelLoaderSpecProto* spec,
                         table.Get(custom_kernel.kernel_spec_index()));
        // Setting the spec clears the index, since both live in the same
        // oneof.
        *custom_kernel.mutable_kernel_spec() = *spec;
        return absl::OkStatus();
      });
}

}  // namespace xla::gpu
