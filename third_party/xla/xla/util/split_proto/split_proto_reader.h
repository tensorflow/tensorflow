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

#ifndef XLA_UTIL_SPLIT_PROTO_SPLIT_PROTO_READER_H_
#define XLA_UTIL_SPLIT_PROTO_SPLIT_PROTO_READER_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/message.h"
#include "riegeli/bytes/reader.h"
#include "xla/service/gpu/gpu_executable.pb.h"

namespace xla {

// Reads a split proto into writing it into a the regular `proto` messages.
// See proto_splitter.proto for more details on the split proto format.
absl::Status ReadSplitProto(std::unique_ptr<riegeli::Reader> reader,
                            google::protobuf::Message& proto);

// Return true if the data being read by the reader is a split proto.
absl::StatusOr<bool> IsSplitProto(riegeli::Reader& reader);

}  // namespace xla

#endif  // XLA_UTIL_SPLIT_PROTO_SPLIT_PROTO_READER_H_
