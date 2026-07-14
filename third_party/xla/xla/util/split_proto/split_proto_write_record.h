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

#ifndef XLA_UTIL_SPLIT_PROTO_SPLIT_PROTO_WRITE_RECORD_H_
#define XLA_UTIL_SPLIT_PROTO_SPLIT_PROTO_WRITE_RECORD_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message_lite.h"
#include "riegeli/messages/serialize_message.h"
#include "riegeli/records/record_writer.h"

namespace xla {

template <typename Src>
absl::Status WriteRecord(riegeli::RecordWriter<Src>& record_writer,
                         const google::protobuf::MessageLite& record) {
  // Proto serialization isn't deterministic by default (e.g. proto maps).
  // Setting this flag can make things a bit slower, but its important so that
  // if we compile the same model twice we get the same bit-identical binary.
  if (!record_writer.WriteRecord(
          record, riegeli::SerializeMessageOptions().set_deterministic(true))) {
    return record_writer.status().ok()
               ? absl::InternalError("Failed to write proto record")
               : record_writer.status();
  }
  return absl::OkStatus();
}

template <typename Src>
absl::Status WriteRecord(riegeli::RecordWriter<Src>& record_writer,
                         absl::string_view record) {
  if (!record_writer.WriteRecord(record)) {
    return record_writer.status().ok()
               ? absl::InternalError("Failed to write bytes record")
               : record_writer.status();
  }
  return absl::OkStatus();
}

}  // namespace xla

#endif  // XLA_UTIL_SPLIT_PROTO_SPLIT_PROTO_WRITE_RECORD_H_
