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
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
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
  using Buffer = absl::Span<const absl::string_view>;

  // One Buffer per shard.
  static constexpr int kBuffersInlineSize = 1;
  using Buffers = absl::InlinedVector<Buffer, kBuffersInlineSize>;

  // Called when this object is done with the string buffer provided at the
  // construction time.
  using OnDoneWithBuffer = absl::AnyInvocable<void() &&>;

  // General array construction (with static shape). The `buffers` and their
  // elements (absl::string_views) must live until the `on_done_with_buffer` is
  // called. The number and order of buffers must match the number and order
  // of devices in `sharding`.
  static absl::StatusOr<tsl::RCReference<BasicStringArray>> Create(
      Client* client, Shape shape, std::shared_ptr<const Sharding> sharding,
      Future<Buffers> buffers, OnDoneWithBuffer on_done_with_buffer);

  absl::StatusOr<tsl::RCReference<Array>> FullyReplicatedShard(
      ArrayCopySemantics semantics) override;

  // ifrt::Array API

  ~BasicStringArray() override = default;

  Client* client() const override {
    DCHECK(this);
    return client_;
  }

  DType dtype() const override {
    DCHECK(this);
    return DType(DType::kString);
  }

  const Shape& shape() const override { return shape_; }

  const Sharding& sharding() const override {
    DCHECK(this);
    return *sharding_;
  }

  std::shared_ptr<const Sharding> shared_ptr_sharding() const override {
    DCHECK(this);
    return sharding_;
  }

  absl::StatusOr<std::unique_ptr<PjRtLayout>> layout() const override;

  absl::StatusOr<std::vector<tsl::RCReference<Array>>>
  DisassembleIntoSingleDeviceArrays(ArrayCopySemantics semantics) override;

  ABSL_MUST_USE_RESULT
  Future<> CopyToHostBuffer(
      void* data, std::optional<absl::Span<const int64_t>> byte_strides,
      ArrayCopySemantics semantics) override;

  absl::StatusOr<tsl::RCReference<Array>> Reshard(
      std::shared_ptr<const Sharding> new_sharding,
      ArrayCopySemantics semantics) override;

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

  BasicStringArray(Client* client, Shape shape,
                   std::shared_ptr<const Sharding> sharding,
                   Future<Buffers> buffers,
                   OnDoneWithBuffer on_done_with_buffer);

  Client* client_;
  Shape shape_;
  std::shared_ptr<const Sharding> sharding_;

  Future<Buffers> buffers_;
  OnDoneWithBuffer on_done_with_buffer_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_BASIC_STRING_ARRAY_H_
