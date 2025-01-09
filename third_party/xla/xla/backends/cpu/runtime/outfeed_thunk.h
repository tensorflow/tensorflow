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

#ifndef XLA_BACKENDS_CPU_RUNTIME_OUTFEED_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_OUTFEED_THUNK_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/resource_use.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

class OutfeedThunk final : public Thunk {
 public:
  struct OutfeedBuffer {
    BufferAllocation::Slice slice;
    Shape shape;
  };

  struct OutfeedResources {
    std::shared_ptr<Resource> consume_token;
    std::shared_ptr<Resource> produce_token;
  };

  static absl::StatusOr<std::unique_ptr<OutfeedThunk>> Create(
      Info info, absl::Span<const OutfeedBuffer> outfeed_buffers,
      OutfeedResources outfeed_resources);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;
  ResourceUses resource_uses() const final;

 private:
  OutfeedThunk(Info info, absl::Span<const OutfeedBuffer> outfeed_buffers,
               OutfeedResources outfeed_resources);

  std::vector<OutfeedBuffer> outfeed_buffers_;
  OutfeedResources outfeed_resources_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_OUTFEED_THUNK_H_
