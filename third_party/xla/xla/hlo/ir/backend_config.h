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
#include "absl/status/status.h"
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

// Clones the provided proto. If the input is nullptr, the result is also
// nullptr.
std::unique_ptr<tsl::protobuf::Message> CloneBackendConfigProto(
    const tsl::protobuf::Message* proto);

// A wrapper around the BackendConfig proto. It can be initialized either with
// a proto object or a string representing the JSON encoding of a proto. Once
// the wrapper is initialized (either during construction or via an assignment)
// it becomes immutable and any further assignment attempts will fail.
//
// When the wrapper is initialized only the provided format is stored. If the
// other format is requested from the wrapper later, it is lazily computed and
// cached internally, before it is returned. Subsequent accesses will directly
// return the cached value.
//
// All accesses are protected via a mutex because instances of this class are
// accessed concurrently during auto tuning.
class BackendConfigWrapper {
 public:
  BackendConfigWrapper() = default;
  explicit BackendConfigWrapper(std::string raw_string)
      : raw_string_(std::move(raw_string)) {}
  explicit BackendConfigWrapper(const tsl::protobuf::Message& proto)
      : proto_(CloneBackendConfigProto(&proto)) {}
  BackendConfigWrapper(const BackendConfigWrapper& other) {
    absl::MutexLock other_lock{&other.mutex_};
    proto_ = CloneBackendConfigProto(other.proto_.get());
    raw_string_ = other.raw_string_;
  }

  BackendConfigWrapper& operator=(BackendConfigWrapper&& other);
  bool operator==(const BackendConfigWrapper& other) const;
  bool operator!=(const BackendConfigWrapper& other) const {
    return !(*this == other);
  }

  // Returns a reference to the raw string that corresponds to this backend
  // config.
  //
  // WARNING: This function returns a reference which is valid at the time the
  //          call terminates. If the BackendConfig is reassigned the reference
  //          becomes invalid, which could lead to subtle and hard to detect
  //          bugs, especially in multi-threaded code. The caller is responsible
  //          for ensuring the lifetime of the referenced string.
  //
  //          Prefer to use the safer (but potentially slower) GetProto().
  const std::string& GetRawString() const {
    absl::WriterMutexLock lock{&mutex_};
    return GetRawStringWithoutMutex();
  }
  absl::Status GetProto(tsl::protobuf::Message* output_proto) const;

  bool empty() const {
    absl::MutexLock lock{&mutex_};
    return proto_ == nullptr && raw_string_.empty();
  }

 private:
  const std::string& GetRawStringWithoutMutex() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // proto_ and raw_string_ must be consistent. If one is set, the other
  // will be lazily initialized when requested. Because this class is accessed
  // concurrently, a mutex is used to protect all access.
  //
  // Unfortunately, all members have to be mutable, since either of them can be
  // the cached one.
  mutable absl::Mutex mutex_;
  mutable std::unique_ptr<tsl::protobuf::Message> proto_
      ABSL_GUARDED_BY(mutex_);
  mutable std::string raw_string_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla

#endif  // XLA_HLO_IR_BACKEND_CONFIG_H_
