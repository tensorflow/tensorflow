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

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "tsl/platform/human_readable_json.h"
#include "tsl/platform/protobuf.h"

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
  // Pass ignore_accuracy_loss = true because estimated_cycles field can be
  // INT64_MAX. If ignore_accuracy_loss = false and estimated_cycles =
  // INT64_MAX, JsonFormat will return an error status, although there is no
  // accuracy loss for int64_t.
  return tsl::ProtoToHumanReadableJson(proto, /*ignore_accuracy_loss=*/true);
}

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

  absl::WriterMutexLock lock{&mutex_};
  if (proto_ != nullptr) {
    if (proto_->GetDescriptor() != output_proto->GetDescriptor()) {
      return Internal("Mismatched backend config descriptors.");
    }
    output_proto->CopyFrom(*proto_);
    return absl::OkStatus();
  }

  // Empty string does not parse as valid JSON, but it's a valid backend config,
  // corresponding to the empty proto.
  if (raw_string_.empty()) {
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(tsl::HumanReadableJsonToProto(raw_string_, output_proto));
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
    absl::MutexLock other_lock{&other.mutex_};
    temp_proto = std::move(other.proto_);
    temp_string = std::move(other.raw_string_);
  }

  absl::MutexLock this_lock{&mutex_};

  proto_ = std::move(temp_proto);
  raw_string_ = std::move(temp_string);
  return *this;
}

bool BackendConfigWrapper::operator==(const BackendConfigWrapper& other) const {
  tsl::protobuf::Message* this_proto = nullptr;

  // Do not hold two mutexes at the same time to avoid deadlocks.
  {
    absl::MutexLock this_lock{&mutex_};
    this_proto = proto_.get();
  }

  const std::string* other_raw_string = nullptr;
  {
    absl::MutexLock other_lock{&other.mutex_};
    if (this_proto != nullptr && other.proto_ != nullptr) {
      using ::tsl::protobuf::util::MessageDifferencer;
      return MessageDifferencer::Equals(*this_proto, *other.proto_);
    }
    other_raw_string = &other.GetRawStringWithoutMutex();
  }

  return GetRawString() == *other_raw_string;
}

}  // namespace xla
