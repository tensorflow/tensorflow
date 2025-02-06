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

#ifndef XLA_BACKENDS_CPU_RUNTIME_THUNK_PROTO_SERDES_H_
#define XLA_BACKENDS_CPU_RUNTIME_THUNK_PROTO_SERDES_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/serdes_base.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

class ThunkSequenceSerDesProtobuf : public SerDesBase<ThunkSequence> {
 public:
  explicit ThunkSequenceSerDesProtobuf(
      const std::vector<BufferAllocation>* buffer_allocations =
          nullptr);  // NOTE buffer allocations aren't
                     // needed for serialization.

  absl::StatusOr<std::string> Serialize(
      const ThunkSequence& thunk_sequence) override;
  absl::StatusOr<std::unique_ptr<ThunkSequence>> Deserialize(
      const std::string& serialized) override;

  absl::StatusOr<ThunkSequenceProto> ToProto(
      const ThunkSequence& thunk_sequence) const;
  absl::StatusOr<std::unique_ptr<ThunkSequence>> FromProto(
      const ThunkSequenceProto& proto) const;

 private:
  const std::vector<BufferAllocation>* buffer_allocations_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_THUNK_PROTO_SERDES_H_
