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

#ifndef XLA_RUNTIME_LARGE_HLO_SNAPSHOT_SERIALIZATION_SERIALIZATION_H_
#define XLA_RUNTIME_LARGE_HLO_SNAPSHOT_SERIALIZATION_SERIALIZATION_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/service/hlo.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {

// Serialize HloUnoptimizedSnapshot to a zero copy output stream.
// The snapshot is serialized in the following format order:
//
// * Metadata: `HloUnoptimizedSnapshot` with HloModuleProto and a descriptor for
//    each argument. The descriptor specifies the size of the argument in bytes.
// * Raw arguments: serialized in the same format as `Literal::Serialize`.
//
// This serialization format is designed to bypass the 2GiB proto size
// limitation.
absl::Status SerializeHloUnoptimizedSnapshot(
    const HloUnoptimizedSnapshot& snapshot,
    tsl::protobuf::io::ZeroCopyOutputStream* zero_copy_output_stream);

// Deserialization of the HLO unoptimized snapshot. The snapshot is expected to
// be in the format produced by `SerializeHloUnoptimizedSnapshot`.
absl::StatusOr<HloUnoptimizedSnapshot> DeserializeHloUnoptimizedSnapshot(
    tsl::protobuf::io::ZeroCopyInputStream* zero_copy_input_stream);
}  // namespace xla

#endif  // XLA_RUNTIME_LARGE_HLO_SNAPSHOT_SERIALIZATION_SERIALIZATION_H_
