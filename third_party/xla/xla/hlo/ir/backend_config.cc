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

#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/human_readable_json.h"
#include "tsl/platform/protobuf.h"

namespace xla {

absl::StatusOr<std::string> BackendConfigToRawString(
    const tsl::protobuf::Message& proto) {
  std::string ret;
  // Pass ignore_accuracy_loss = true because estimated_cycles field can be
  // INT64_MAX. If ignore_accuracy_loss = false and estimated_cycles =
  // INT64_MAX, JsonFormat will return an error status, although there is no
  // accuracy loss for int64_t.
  TF_RETURN_IF_ERROR(tsl::ProtoToHumanReadableJson(
      proto, &ret, /*ignore_accuracy_loss=*/true));
  return ret;
}

const std::string& BackendConfigWrapper::GetRawString() const {
  absl::WriterMutexLock lock{&mutex_};
  if (proto_ && raw_string_.empty()) {
    raw_string_ = BackendConfigToRawString(*proto_).value();
  }
  return raw_string_;
}

BackendConfigWrapper BackendConfigWrapper::Clone() const {
  // Prefer cloning protobuf, raw_string_ will be lazily generated if accessed.
  BackendConfigWrapper cloned;
  if (auto* proto = GetProtoPtr()) {
    cloned.SetProto(*proto);
  } else {
    absl::MutexLock source_lock{&mutex_};
    absl::MutexLock target_lock{&cloned.mutex_};
    cloned.raw_string_ = raw_string_;
  }
  return cloned;
}

BackendConfigWrapper& BackendConfigWrapper::operator=(std::string raw_string) {
  absl::MutexLock lock{&mutex_};
  raw_string_ = std::move(raw_string);
  proto_.reset();
  return *this;
}

BackendConfigWrapper& BackendConfigWrapper::operator=(
    const tsl::protobuf::Message& proto) {
  SetProto(proto);
  absl::MutexLock lock{&mutex_};
  raw_string_.clear();
  return *this;
}

void BackendConfigWrapper::SetProto(const tsl::protobuf::Message& proto) {
  proto_.reset(proto.New());
  proto_->CopyFrom(proto);
}

bool BackendConfigWrapper::operator==(const BackendConfigWrapper& other) const {
  auto* proto_a = GetProtoPtr();
  auto* proto_b = other.GetProtoPtr();
  if (proto_a != nullptr && proto_b != nullptr) {
    using ::tsl::protobuf::util::MessageDifferencer;
    return MessageDifferencer::Equals(*proto_a, *proto_b);
  }
  // TODO(b/225956414): Consider canonicalizing raw string form.
  return GetRawString() == other.GetRawString();
}

}  // namespace xla
