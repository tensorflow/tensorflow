/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_HLO_IR_BACKEND_CONFIG_H_
#define XLA_HLO_IR_BACKEND_CONFIG_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "tsl/platform/protobuf.h"

namespace xla {

// Returns a string representation of a proto in the format used by
// HloInstruction::raw_backend_config_string.
//
// This is morally equivalent to:
//
//   HloInstruction instr;
//   TF_RETURN_IF_ERROR(instr.set_backend_config(proto));
//   return instr.raw_backend_config_string();
//
absl::StatusOr<std::string> BackendConfigToRawString(
    const tsl::protobuf::Message& proto);

// A wrapper around the BackendConfig proto. The wrapper holds either a proto
// object or a proto encoded as a JSON string. If the wrapper holds a proto and
// the string is requested, it is lazily computed and stored. If the wrapper
// holds only a string, a nullptr proto is always returned.
class BackendConfigWrapper {
 public:
  const tsl::protobuf::Message* GetProtoPtr() const { return proto_.get(); }

  const std::string& GetRawString() const;

  BackendConfigWrapper Clone() const;

  bool operator==(const BackendConfigWrapper& other) const;
  bool operator!=(const BackendConfigWrapper& other) const {
    return !(*this == other);
  }

  bool empty() const {
    absl::MutexLock lock{&mutex_};
    return proto_ == nullptr && raw_string_.empty();
  }

  void clear() {
    proto_.reset();
    absl::MutexLock lock{&mutex_};
    raw_string_.clear();
  }

  BackendConfigWrapper() = default;
  BackendConfigWrapper(BackendConfigWrapper&& other)
      : proto_(std::move(other.proto_)), raw_string_([&] {
          absl::MutexLock lock{&other.mutex_};
          return std::move(other.raw_string_);
        }()) {}

  BackendConfigWrapper& operator=(std::string raw_string);
  BackendConfigWrapper& operator=(const tsl::protobuf::Message& proto);
  BackendConfigWrapper& operator=(BackendConfigWrapper&& other) {
    proto_ = std::move(other.proto_);
    absl::MutexLock destination_lock{&mutex_};
    absl::MutexLock source_lock{&other.mutex_};
    raw_string_ = std::move(other.raw_string_);
    return *this;
  }

  void SetProto(const tsl::protobuf::Message& proto);

 private:
  std::unique_ptr<tsl::protobuf::Message> proto_;
  // If proto_ is not null, raw_string_ is a lazy cache of its string format.
  mutable absl::Mutex mutex_;
  mutable std::string raw_string_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla

#endif  // XLA_HLO_IR_BACKEND_CONFIG_H_
