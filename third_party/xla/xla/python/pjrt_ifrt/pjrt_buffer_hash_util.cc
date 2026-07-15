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

#include "xla/python/pjrt_ifrt/pjrt_buffer_hash_util.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/primitive_util.h"
#include "xla/python/ifrt/buffer_hash_util.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/shape.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/concurrency/future.h"

namespace xla {
namespace ifrt {

namespace {

// Shared state for all `InflightHashing` objects.
struct State {
  absl::Mutex mu;
  // Total size of all buffers that are currently being hashed.
  int64_t current_in_flight_byte_size ABSL_GUARDED_BY(mu) = 0;
};

// RAII object that updates the state when an inflight hashing completes.
class InflightHashing {
 public:
  InflightHashing(std::shared_ptr<State> state, int64_t byte_size)
      : state_(std::move(state)), byte_size_(byte_size) {}

  ~InflightHashing() {
    absl::MutexLock lock(state_->mu);
    state_->current_in_flight_byte_size -= byte_size_;
  }

 private:
  std::shared_ptr<State> state_;
  const int64_t byte_size_;
};

}  // namespace

tsl::Future<std::vector<uint64_t>> HashPjRtBuffers(
    tsl::Executor& executor, absl::Span<PjRtBuffer* const> pjrt_buffers,
    absl::Span<const IndexDomain> index_domains, Client::HashMode mode,
    int64_t max_inflight_memory) {
  if (pjrt_buffers.size() != index_domains.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "buffers and index_domains must have the same size, but have ",
        pjrt_buffers.size(), " vs. ", index_domains.size()));
  }

  if (pjrt_buffers.empty()) {
    return std::vector<uint64_t>();
  }

  auto state = std::make_shared<State>();

  std::vector<tsl::Future<uint64_t>> futures;
  futures.reserve(pjrt_buffers.size());

  for (int i = 0; i < pjrt_buffers.size(); ++i) {
    // Use an unpacked element type for sub-byte types.
    const int64_t element_byte_size = xla::primitive_util::ByteWidth(
        pjrt_buffers[i]->on_device_shape().element_type());
    const int64_t buffer_byte_size =
        element_byte_size * index_domains[i].shape().num_elements();

    {
      absl::MutexLock lock(state->mu);
      auto condition = [&]() ABSL_SHARED_LOCKS_REQUIRED(state->mu) {
        // If a single buffer is larger than the limit, we allow it to exceed
        // the limit, but only if there is no other inflight hashing.
        if (buffer_byte_size > max_inflight_memory) {
          return state->current_in_flight_byte_size == 0;
        }
        return state->current_in_flight_byte_size + buffer_byte_size <=
               max_inflight_memory;
      };
      state->mu.Await(absl::Condition(&condition));
      state->current_in_flight_byte_size += buffer_byte_size;
    }

    auto inflight_hashing =
        std::make_unique<InflightHashing>(state, buffer_byte_size);

    const xla::Shape& device_shape = pjrt_buffers[i]->on_device_shape();

    // For simplicity, use a descending layout without paddings for the output
    // literal even for `kPhysical` hashing. This ensures robust hashing
    // regardless of whether raw D2H without delinearization will have
    // deterministic padding data in the output literal.
    //
    // If we can ensure that the paddings are deterministic, we should use the
    // original on-device layout and avoid the delinearization cost.
    xla::Shape descending_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
        device_shape.element_type(), device_shape.dimensions());
    ASSIGN_OR_RETURN(std::unique_ptr<xla::Literal> literal,
                     xla::Literal::MakeUnique(descending_shape));

    xla::Literal* literal_ptr = literal.get();
    futures.push_back(
        pjrt_buffers[i]
            ->ToLiteral(literal_ptr)
            .Map(executor,
                 [pjrt_buffer = pjrt_buffers[i],
                  index_domain = index_domains[i], mode, element_byte_size,
                  literal = std::move(literal),
                  inflight_hashing = std::move(
                      inflight_hashing)]() -> absl::StatusOr<uint64_t> {
                   absl::Span<const char> literal_span(
                       static_cast<const char*>(literal->untyped_data()),
                       literal->size_bytes());

                   std::shared_ptr<const PjRtLayout> pjrt_layout =
                       pjrt_buffer->layout();
                   const xla::Layout& xla_layout = pjrt_layout->xla_layout();

                   switch (mode) {
                     case Client::HashMode::kPhysical:
                       return HashBufferPhysical(literal_span, xla_layout);
                     case Client::HashMode::kLogical:
                       return HashBufferLogical(literal_span, element_byte_size,
                                                index_domain);
                   }
                 }));
  }

  return tsl::JoinFutures(absl::MakeSpan(futures));
}

}  // namespace ifrt
}  // namespace xla
