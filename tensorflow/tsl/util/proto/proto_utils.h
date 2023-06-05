/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_UTIL_PROTO_PROTO_UTILS_H_
#define TENSORFLOW_TSL_UTIL_PROTO_PROTO_UTILS_H_

#include "google/protobuf/duration.pb.h"
#include "absl/time/time.h"

namespace tsl {
namespace proto_utils {

// Converts an absl::Duration to a google::protobuf::Duration.
inline google::protobuf::Duration ToDurationProto(absl::Duration duration) {
  google::protobuf::Duration proto;
  proto.set_seconds(absl::IDivDuration(duration, absl::Seconds(1), &duration));
  proto.set_nanos(
      absl::IDivDuration(duration, absl::Nanoseconds(1), &duration));
  return proto;
}

// Converts a google::protobuf::Duration to an absl::Duration.
inline absl::Duration FromDurationProto(google::protobuf::Duration proto) {
  return absl::Seconds(proto.seconds()) + absl::Nanoseconds(proto.nanos());
}

}  // namespace proto_utils
}  // namespace tsl

#endif  // TENSORFLOW_TSL_UTIL_PROTO_PROTO_UTILS_H_
