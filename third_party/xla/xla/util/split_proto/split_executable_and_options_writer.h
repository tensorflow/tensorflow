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

#ifndef XLA_UTIL_SPLIT_PROTO_SPLIT_EXECUTABLE_AND_OPTIONS_WRITER_H_
#define XLA_UTIL_SPLIT_PROTO_SPLIT_EXECUTABLE_AND_OPTIONS_WRITER_H_

#include <memory>

#include "absl/status/status.h"
#include "riegeli/bytes/writer.h"
#include "xla/pjrt/proto/compile_options.pb.h"

namespace xla {

// Serialized the `executable_and_options` into the `writer` using the split
// proto format. This supports serializing protos bigger than 2GiB, unlike
// regular proto serialization.
absl::Status WriteSplitExecutableAndOptions(
    const ExecutableAndOptionsProto& executable_and_options,
    std::unique_ptr<riegeli::Writer> writer);

}  // namespace xla

#endif  // XLA_UTIL_SPLIT_PROTO_SPLIT_EXECUTABLE_AND_OPTIONS_WRITER_H_
