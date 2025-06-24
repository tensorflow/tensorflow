/*
 * Copyright 2023 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_ARRAY_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_ARRAY_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {
namespace proxy {

// Implementation of the xla::ifrt::Array interface.
class Array final : public llvm::RTTIExtends<Array, xla::ifrt::Array> {
 public:
  // `Array::MakeArrayFromHostBuffer()` implements
  // `Client::MakeArrayFromHostBuffer()`.
  // TODO(b/261226026): Implement logic directly in client.cc.
  static absl::StatusOr<xla::ifrt::ArrayRef> MakeArrayFromHostBuffer(
      xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
      const void* data, DType dtype, Shape shape,
      std::optional<absl::Span<const int64_t>> byte_strides,
      ShardingRef sharding, xla::ifrt::Client::HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer);

  // `Array::MakeArraysFromHostBufferShards()` implements
  // `Client::MakeArraysFromHostBufferShards()`.
  // TODO(b/261226026): Implement logic directly in client.cc.
  static absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
  MakeArraysFromHostBufferShards(
      xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
      absl::Span<xla::ifrt::Client::MakeArraysFromHostBufferShardsSpec> specs,
      xla::ifrt::Client::HostBufferSemantics semantics,
      tsl::RCReference<xla::ifrt::UserContext> user_context);

  static absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> MakeErrorArrays(
      xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
      const absl::Status& error, absl::Span<const ArraySpec> array_specs,
      tsl::RCReference<UserContext> user_context);

  // `Array::AssembleArrayFromSingleDeviceArrays()` implements
  // `Client::AssembleArrayFromSingleDeviceArrays()`.
  // TODO(b/261226026): Implement logic directly in client.cc.
  static absl::StatusOr<xla::ifrt::ArrayRef>
  AssembleArrayFromSingleDeviceArrays(
      xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
      DType dtype, Shape shape, ShardingRef sharding,
      absl::Span<xla::ifrt::ArrayRef> arrays,
      ArrayCopySemantics array_copy_semantics,
      SingleDeviceShardSemantics single_device_shard_semantics);

  // `Array::RemapArrays()` implements `Client::RemapArrays()`.
  // TODO(b/261226026): Implement logic directly in client.cc.
  static absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> RemapArrays(
      xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
      const RemapPlan& plan, absl::Span<xla::ifrt::ArrayRef> arrays,
      ArrayCopySemantics semantics);

  // Destructs the array associated with the given handle. The corresponding
  // array becomes unusable afterwards.
  static void Destruct(RpcHelper* rpc_helper, ArrayHandle handle);

  Array(xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
        DType dtype, Shape shape, ShardingRef sharding, ArrayHandle arr_handle,
        std::shared_ptr<const xla::PjRtLayout> layout)
      : client_(client),
        rpc_helper_(std::move(rpc_helper)),
        dtype_(dtype),
        shape_(std::move(shape)),
        sharding_(std::move(sharding)),
        custom_layout_(std::move(layout)),
        handle_(arr_handle) {}

  ~Array() override { Destruct(rpc_helper_.get(), handle_); }

  absl::StatusOr<ArrayHandle> GetHandle(ArrayCopySemantics semantics) {
    absl::MutexLock l(&mu_);
    if (deleted_ == DeletionState::kDeleted) {
      return absl::InvalidArgumentError("Array already deleted.");
    }
    if (semantics == ArrayCopySemantics::kDonateInput) {
      deleted_ = DeletionState::kDeleted;
    }
    return handle_;
  }

  // Fetches the ArrayHandle when the ArrayCopySemantics (i.e., whether the
  // array is meant to be donated or copied) is not known.
  //
  // Calling this function may cause `IsDelete()` calls to result in a
  // synchronous RPC to the proxy-server. To avoid such performance overhead,
  // prefer using `GetHandle(semantics)` whenever the semantics are known.
  absl::StatusOr<ArrayHandle> GetHandleUnknownIfBeingDonated() {
    absl::MutexLock l(&mu_);
    if (deleted_ == DeletionState::kDeleted) {
      return absl::InvalidArgumentError("Array already deleted.");
    }
    deleted_ = DeletionState::kUnknown;
    return handle_;
  }

  std::shared_ptr<const xla::PjRtLayout> custom_layout() const {
    return custom_layout_;
  }

  xla::ifrt::Client* client() const override;
  Future<> GetReadyFuture() const override;
  Future<> Delete() override;
  bool IsDeleted() const override;
  std::string DebugString() const override;

  DType dtype() const override { return dtype_; }
  const Shape& shape() const override { return shape_; }
  const Sharding& sharding() const override { return *sharding_; }
  ShardingRef shared_ptr_sharding() const override { return sharding_; }
  absl::StatusOr<std::shared_ptr<const PjRtLayout>> layout() const override;

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
  DisassembleIntoSingleDeviceArrays(
      ArrayCopySemantics array_copy_semantics,
      SingleDeviceShardSemantics single_device_shard_semantics) override;

  absl::StatusOr<xla::ifrt::ArrayRef> FullyReplicatedShard(
      xla::ifrt::ArrayCopySemantics semantics) override;

  ABSL_MUST_USE_RESULT
  Future<> CopyToHostBuffer(
      void* data, std::optional<absl::Span<const int64_t>> byte_strides,
      ArrayCopySemantics semantics) override;

  static char ID;  // NOLINT

 private:
  template <typename T, typename... Args>
  friend tsl::RCReference<T> tsl::MakeRef(Args&&... args);

  Future<> CopyToStringHostBuffer(
      void* data, std::optional<absl::Span<const int64_t>> byte_strides,
      ArrayCopySemantics semantics);

  // Not owned. Used only for implementing `client()` interface method. Note
  // that `client()` will still return the pointer even if the pointed-to memory
  // is freed; this unfortunate behavior currently exists in all IFRT
  // implementations.
  xla::ifrt::Client* const client_;

  const std::shared_ptr<RpcHelper> rpc_helper_;
  const DType dtype_;
  const Shape shape_;
  const ShardingRef sharding_;

  // This is layout explicitly supplied at creation time. we explicitly
  // distinguish it from default layouts since some functions
  // behaves differently depending on where the layout came from.
  const std::shared_ptr<const xla::PjRtLayout> custom_layout_;

  const ArrayHandle handle_
      ABSL_DEPRECATED("Use GetHandle() function instead.");

  mutable absl::Mutex mu_;
  enum class DeletionState {
    kUnknown,  // Need to ask the proxy-server whether the array is deleted.
    kDeleted,  // IsDeleted() will return true.
    kAlive     // IsDeleted() will return false.
  };
  mutable DeletionState deleted_ ABSL_GUARDED_BY(mu_) = DeletionState::kAlive;

  mutable Future<> ready_future_ ABSL_GUARDED_BY(mu_);
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_ARRAY_H_
