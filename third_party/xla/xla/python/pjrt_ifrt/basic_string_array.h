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

#ifndef XLA_PYTHON_PJRT_IFRT_BASIC_STRING_ARRAY_H_
#define XLA_PYTHON_PJRT_IFRT_BASIC_STRING_ARRAY_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// `BasicStringArray` implements an `ifrt::Array` by wrapping a local (aka host)
// string buffer. This object is expected to live exclusively in the IFRT layer,
// and thus is not specific to any particular backend. However, it is currently
// located in the pjrt_ifrt directory because we expect the main use of this
// class is to implement IO Callable support in pjrt_ifrt.
class BasicStringArray final
    : public llvm::RTTIExtends<BasicStringArray, Array> {
 public:
  // Must be in dense major to minor order.
  using Buffer = absl::Span<const absl::Cord>;

  // One Buffer per shard.
  static constexpr int kBuffersInlineSize = 1;
  using Buffers = absl::InlinedVector<Buffer, kBuffersInlineSize>;

  // Called when this object is done with the string buffer provided at the
  // construction time.
  using OnDoneWithBuffer = std::function<void()>;

  // General array construction. The `buffers` and their elements
  // (absl::Cords) must live until the `on_done_with_buffer` is called.
  // The number and order of buffers must match the number and order of devices
  // in `sharding`.
  static absl::StatusOr<tsl::RCReference<BasicStringArray>> Create(
      Client* client, Shape shape, ShardingRef sharding,
      Future<Buffers> buffers, OnDoneWithBuffer on_done_with_buffer);

  ~BasicStringArray() override;

  absl::StatusOr<ArrayRef> FullyReplicatedShard(
      ArrayCopySemantics semantics) override;

  // ifrt::Array API

  Client* client() const override {
    DCHECK(this);
    return client_;
  }

  DType dtype() const override {
    DCHECK(this);
    return DType(DType::kString);
  }

  const Shape& shape() const override {
    DCHECK(this);
    return shape_;
  }

  const Sharding& sharding() const override {
    DCHECK(this);
    return *sharding_;
  }

  ShardingRef shared_ptr_sharding() const override {
    DCHECK(this);
    return sharding_;
  }

  absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>> pjrt_layout()
      const override;

  absl::StatusOr<std::vector<ArrayRef>> DisassembleIntoSingleDeviceArrays(
      ArrayCopySemantics array_copy_semantics,
      SingleDeviceShardSemantics single_device_shard_semantics) override;

  ABSL_MUST_USE_RESULT
  Future<> CopyToHostBuffer(
      void* data, std::optional<absl::Span<const int64_t>> byte_strides,
      ArrayCopySemantics semantics) override;

  absl::StatusOr<ArrayRef> Copy(
      std::optional<xla::ifrt::DeviceListRef> devices,
      std::optional<xla::ifrt::MemoryKind> memory_kind,
      ArrayCopySemantics semantics);

  Future<> GetReadyFuture() const override;

  Future<> Delete() override;
  bool IsDeleted() const override;

  std::string DebugString() const override;

  // Methods specific to this Array variant (i.e., not from `ifrt::Array`).

  // Returns a future holding the string buffers underlying this array. Valid
  // only while this Array object is alive.
  Future<Buffers> buffers() const {
    return buffers_;  // Future copying is not considered expensive.
  }

  static char ID;  // NOLINT

 private:
  template <typename T, typename... Args>
  friend tsl::RCReference<T> tsl::MakeRef(Args&&... args);

  BasicStringArray(Client* client, Shape shape, ShardingRef sharding,
                   Future<Buffers> buffers, Future<> ready_future,
                   OnDoneWithBuffer on_done_with_buffer);

  // Internal implementation of delete.
  void DeleteInternal() ABSL_LOCKS_EXCLUDED(mu_);

  Client* client_;
  Shape shape_;
  ShardingRef sharding_;
  Future<Buffers> buffers_;
  Future<> ready_future_;

  mutable absl::Mutex mu_;
  OnDoneWithBuffer on_done_with_buffer_ ABSL_GUARDED_BY(mu_);
  bool is_deleted_ ABSL_GUARDED_BY(mu_) = false;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_BASIC_STRING_ARRAY_H_
