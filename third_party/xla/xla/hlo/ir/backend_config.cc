/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/ir/backend_config.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "google/protobuf/message.h"
#include "re2/re2.h"
#include "xla/util.h"
#include "tsl/platform/human_readable_json.h"
#include "tsl/platform/protobuf.h"

// TODO(dasenov): Remove this after 2026-07-15.
namespace {
std::string RemoveWaitOnOperationQueues(std::string&& s) {
  static constexpr LazyRE2 kReWaitOnOperationQueues = {
      R"("wait_on_operation_queues"\s*:\s*\[\s*\]\s*,)"};
  RE2::GlobalReplace(&s, *kReWaitOnOperationQueues, "");
  return std::move(s);
}
}  // namespace

namespace xla {

std::unique_ptr<tsl::protobuf::Message> CloneBackendConfigProto(
    const tsl::protobuf::Message* proto) {
  if (proto == nullptr) {
    return nullptr;
  }
  std::unique_ptr<tsl::protobuf::Message> result(proto->New());
  result->CopyFrom(*proto);
  return result;
}

absl::StatusOr<std::string> BackendConfigToRawString(
    const tsl::protobuf::Message& proto) {
  return tsl::ProtoToHumanReadableJson(proto);
}

BackendConfigWrapper::BackendConfigWrapper(std::string raw_string)
    : raw_string_(RemoveWaitOnOperationQueues(std::move(raw_string))) {}

const std::string& BackendConfigWrapper::GetRawStringWithoutMutex() const {
  if (proto_ && raw_string_.empty()) {
    // Cache the raw string.
    raw_string_ = BackendConfigToRawString(*proto_).value();
  }
  static const std::string* const kEmptyString = new std::string();
  return raw_string_.empty() ? *kEmptyString : raw_string_;
}

absl::Status BackendConfigWrapper::GetProto(
    tsl::protobuf::Message* output_proto) const {
  output_proto->Clear();

  auto copy_from_cache =
      [&]() ABSL_SHARED_LOCKS_REQUIRED(mutex_) -> absl::Status {
    if (proto_->GetDescriptor() != output_proto->GetDescriptor()) {
      return Internal("Mismatched backend config descriptors.");
    }
    output_proto->CopyFrom(*proto_);
    return absl::OkStatus();
  };

  // Fast path: check with reader lock if proto is already cached.
  {
    absl::ReaderMutexLock lock{mutex_};
    if (proto_ != nullptr) {
      return copy_from_cache();
    }
    // Empty string does not parse as valid JSON, but it's a valid backend
    // config, corresponding to the empty proto.
    if (raw_string_.empty()) {
      return absl::OkStatus();
    }
  }

  absl::WriterMutexLock lock{mutex_};
  // Check again if another thread parsed and cached the proto while we were
  // waiting for the writer lock.
  if (proto_ != nullptr) {
    return copy_from_cache();
  }

  RETURN_IF_ERROR(tsl::HumanReadableJsonToProto(raw_string_, output_proto));
  // Cache the proto into the empty proto_.
  proto_ = CloneBackendConfigProto(output_proto);
  return absl::OkStatus();
}

BackendConfigWrapper& BackendConfigWrapper::operator=(
    BackendConfigWrapper&& other) {
  std::unique_ptr<tsl::protobuf::Message> temp_proto;
  std::string temp_string;

  // Do not hold two mutexes at the same time to avoid deadlocks.
  {
    absl::MutexLock other_lock{other.mutex_};
    temp_proto = std::move(other.proto_);
    temp_string = std::move(other.raw_string_);
  }

  absl::MutexLock this_lock{mutex_};

  proto_ = std::move(temp_proto);
  raw_string_ = std::move(temp_string);
  return *this;
}

bool BackendConfigWrapper::operator==(const BackendConfigWrapper& other) const {
  const std::string* other_raw_string = nullptr;
  {
    // Make sure to drop the lock on this mutex before calling GetRawString()
    // to avoid deadlock.
    absl::MutexLock other_lock{other.mutex_};
    other_raw_string = &other.GetRawStringWithoutMutex();
  }

  return GetRawString() == *other_raw_string;
}

namespace {
CoreAssignmentHandler* g_core_assignment_handler = nullptr;
}  // namespace

void RegisterCoreAssignmentHandler(CoreAssignmentHandler* handler) {
  g_core_assignment_handler = handler;
}

absl::Status SetCoreAssignment(HloInstruction* inst,
                               absl::Span<const int64_t> core_ids) {
  if (g_core_assignment_handler != nullptr) {
    return g_core_assignment_handler->SetCoreAssignment(inst, core_ids);
  }
  return absl::UnimplementedError(
      "Core assignment is not implemented for this target.");
}

absl::StatusOr<std::vector<int64_t>> GetCoreAssignment(
    const HloInstruction* inst) {
  if (g_core_assignment_handler != nullptr) {
    return g_core_assignment_handler->GetCoreAssignment(inst);
  }
  return absl::UnimplementedError(
      "Core assignment is not implemented for this target.");
}

}  // namespace xla
