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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/tuple.h"
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
  static absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
  MakeArrayFromHostBuffer(xla::ifrt::Client* client,
                          std::shared_ptr<RpcHelper> rpc_helper,
                          const void* data, DType dtype, Shape shape,
                          std::optional<absl::Span<const int64_t>> byte_strides,
                          std::shared_ptr<const Sharding> sharding,
                          xla::ifrt::Client::HostBufferSemantics semantics,
                          std::function<void()> on_done_with_host_buffer);

  // `Array::MakeArraysFromHostBufferShards()` implements
  // `Client::MakeArraysFromHostBufferShards()`.
  // TODO(b/261226026): Implement logic directly in client.cc.
  static absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
  MakeArraysFromHostBufferShards(
      xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
      absl::Span<xla::ifrt::Client::MakeArraysFromHostBufferShardsSpec> specs,
      xla::ifrt::Client::HostBufferSemantics semantics);

  // `Array::AssembleArrayFromSingleDeviceArrays()` implements
  // `Client::AssembleArrayFromSingleDeviceArrays()`.
  // TODO(b/261226026): Implement logic directly in client.cc.
  static absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
  AssembleArrayFromSingleDeviceArrays(
      xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
      DType dtype, Shape shape, std::shared_ptr<const Sharding> sharding,
      absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
      ArrayCopySemantics array_copy_semantics,
      SingleDeviceShardSemantics single_device_shard_semantics);

  // `Array::RemapArrays()` implements `Client::RemapArrays()`.
  // TODO(b/261226026): Implement logic directly in client.cc.
  static absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
  RemapArrays(xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
              const RemapPlan& plan,
              absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
              ArrayCopySemantics semantics);

  // Destructs the array associated with the given handle. The corresponding
  // array becomes unusable afterwards.
  static void Destruct(RpcHelper* rpc_helper, ArrayHandle handle);

  Array(xla::ifrt::Client* const client, std::shared_ptr<RpcHelper> rpc_helper,
        DType dtype, Shape shape, std::shared_ptr<const Sharding> sharding,
        ArrayHandle handle)
      : client_(client),
        rpc_helper_(std::move(rpc_helper)),
        dtype_(dtype),
        shape_(std::move(shape)),
        sharding_(std::move(sharding)),
        handle_(handle) {}

  ~Array() override { Destruct(rpc_helper_.get(), handle_); }

  ArrayHandle handle() const { return handle_; }

  xla::ifrt::Client* client() const override;
  Future<> GetReadyFuture() const override;
  Future<> Delete() override;
  bool IsDeleted() const override;
  std::string DebugString() const override;

  DType dtype() const override { return dtype_; }
  const Shape& shape() const override { return shape_; }
  const Sharding& sharding() const override { return *sharding_; }
  std::shared_ptr<const Sharding> shared_ptr_sharding() const override {
    return sharding_;
  }
  absl::StatusOr<std::shared_ptr<const PjRtLayout>> layout() const override {
    return absl::UnimplementedError(
        "Array::layout() not implemented for IFRT proxy");
  };

  absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
  DisassembleIntoSingleDeviceArrays(
      ArrayCopySemantics array_copy_semantics,
      SingleDeviceShardSemantics single_device_shard_semantics) override;

  absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> FullyReplicatedShard(
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
  const std::shared_ptr<const Sharding> sharding_;
  const ArrayHandle handle_;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_ARRAY_H_
