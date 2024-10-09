// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/client/array.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/common/array_util.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace proxy {

char Array::ID = 0;

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
Array::MakeArrayFromHostBuffer(
    xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
    const void* data, DType dtype, Shape shape,
    std::optional<absl::Span<const int64_t>> byte_strides,
    std::shared_ptr<const Sharding> sharding,
    xla::ifrt::Client::HostBufferSemantics semantics,
    std::function<void()> on_done_with_host_buffer) {
  TF_ASSIGN_OR_RETURN(const auto array_mem_region,
                      ArrayMemRegion::FromZerothElementPointer(
                          /*zeroth_element=*/data, dtype, shape, byte_strides));

  const uint64_t host_buffer_handle =
      rpc_helper->host_buffer_store()->NextHandle();
  TF_RETURN_IF_ERROR(
      rpc_helper->host_buffer_store()
          ->Store(host_buffer_handle, array_mem_region.mem_region())
          .Await());

  auto req = std::make_unique<MakeArrayFromHostBufferRequest>();
  req->set_host_buffer_handle(host_buffer_handle);
  *req->mutable_dtype() = dtype.ToProto();
  *req->mutable_shape() = shape.ToProto();
  TF_ASSIGN_OR_RETURN(*req->mutable_sharding(), sharding->ToProto());
  if (byte_strides.has_value()) {
    *req->mutable_byte_strides() = ToByteStridesProto(*byte_strides);
  }

  TF_ASSIGN_OR_RETURN(
      auto response,
      rpc_helper->MakeArrayFromHostBuffer(std::move(req)).Await());
  const ArrayHandle handle{response->array_handle()};

  if (on_done_with_host_buffer != nullptr) {
    std::move(on_done_with_host_buffer)();
  }

  return tsl::RCReference<xla::ifrt::Array>(
      tsl::MakeRef<Array>(client, std::move(rpc_helper), dtype,
                          std::move(shape), std::move(sharding), handle));
}

void Array::Destruct(RpcHelper* rpc_helper, ArrayHandle handle) {
  if (rpc_helper->version().protocol_version() >= 5) {
    rpc_helper->Batch(RpcHelper::kDestructArray, handle);
    return;
  }

  auto req = std::make_unique<DestructArrayRequest>();
  req->set_array_handle_deprecated(handle.handle);
  rpc_helper->DestructArray(std::move(req))
      .OnReady(
          [](absl::StatusOr<std::shared_ptr<DestructArrayResponse>> response) {
            if (!response.ok()) {
              LOG(WARNING)
                  << "Server returned an error when asked to destruct array: "
                  << response.status();
            }
          });
}

Future<> Array::GetReadyFuture() const {
  auto req = std::make_unique<CheckValueReadyRequest>();
  req->add_value_handles(handle_.handle);

  auto promise = Future<>::CreatePromise();
  rpc_helper_->CheckValueReady(std::move(req))
      .OnReady(
          [promise](absl::StatusOr<std::shared_ptr<CheckValueReadyResponse>>
                        resp) mutable { promise.Set(resp.status()); });
  return Future<>(std::move(promise));
}

Future<> Array::Delete() {
  if (rpc_helper_->version().protocol_version() >= 5) {
    rpc_helper_->Batch(RpcHelper::kDeleteArray, handle_);
    return Future<>(absl::OkStatus());
  }

  auto req = std::make_unique<DeleteArrayRequest>();
  req->set_array_handle_deprecated(handle_.handle);

  absl::StatusOr<std::shared_ptr<DeleteArrayResponse>> response =
      rpc_helper_->DeleteArray(std::move(req)).Await();
  if (!response.ok()) {
    return Future<>(response.status());
  }

  // TODO(b/266635130): So that the caller is not blocked until the server
  // replies with the deletion's response, from within
  // `Future(status_handle_promise).OnReady()`, schedule `CheckFuture()` on a
  // separate thread.
  return rpc_helper_->CheckFuture((*response)->deletion_future_handle());
}

bool Array::IsDeleted() const {
  auto req = std::make_unique<IsArrayDeletedRequest>();
  req->set_array_handle(handle_.handle);

  absl::StatusOr<std::shared_ptr<IsArrayDeletedResponse>> response =
      rpc_helper_->IsArrayDeleted(std::move(req)).Await();
  if (response.ok()) {
    return (*response)->deleted();
  } else {
    LOG(ERROR) << "Internal error from proxy server during Array::IsDeleted(): "
               << response.status();
    // Return false so that the user likely queries the array with some
    // method that returns an absl::Status, and ends up with the real
    // error being returned to them by that method.
    return false;
  }
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
Array::AssembleArrayFromSingleDeviceArrays(
    xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
    Shape shape, std::shared_ptr<const Sharding> sharding,
    absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
    ArrayCopySemantics array_copy_semantics,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  auto req = std::make_unique<AssembleArrayFromSingleDeviceArraysRequest>();
  TF_RET_CHECK(!arrays.empty());
  *req->mutable_shape() = shape.ToProto();
  TF_ASSIGN_OR_RETURN(*req->mutable_sharding(), sharding->ToProto());
  req->set_copy_semantics(ToArrayCopySemanticsProto(array_copy_semantics));
  req->set_single_device_shard_semantics(
      ToSingleDeviceShardSemanticsProto(single_device_shard_semantics));
  for (const tsl::RCReference<xla::ifrt::Array>& rcref : arrays) {
    Array* array = llvm::dyn_cast<Array>(rcref.get());
    if (array == nullptr) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Array at $0 supplied to AssembleArrayFromSingleDeviceArrays() is "
          "not a xla::ifrt::proxy::Array.",
          rcref.get()));
    }
    req->add_single_device_array_handles(array->handle_.handle);
  }

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<AssembleArrayFromSingleDeviceArraysResponse> response,
      rpc_helper->AssembleArrayFromSingleDeviceArrays(std::move(req)).Await());
  ArrayHandle handle{response->array_handle()};

  return tsl::RCReference<xla::ifrt::Array>(
      tsl::MakeRef<Array>(client, std::move(rpc_helper), arrays[0]->dtype(),
                          std::move(shape), std::move(sharding), handle));
}

absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
Array::RemapArrays(xla::ifrt::Client* client,
                   std::shared_ptr<RpcHelper> rpc_helper, const RemapPlan& plan,
                   absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
                   ArrayCopySemantics semantics) {
  auto req = std::make_unique<RemapArraysRequest>();
  TF_RET_CHECK(!arrays.empty());
  TF_ASSIGN_OR_RETURN(*req->mutable_plan(), plan.ToProto());
  req->set_copy_semantics(ToArrayCopySemanticsProto(semantics));
  for (const tsl::RCReference<xla::ifrt::Array>& rcref : arrays) {
    Array* array = llvm::dyn_cast<Array>(rcref.get());
    if (array == nullptr) {
      return absl::InvalidArgumentError(
          absl::Substitute("Array at $0 supplied to RemapArrays() is "
                           "not a xla::ifrt::proxy::Array.",
                           rcref.get()));
    }
    req->add_array_handles(array->handle_.handle);
  }

  TF_ASSIGN_OR_RETURN(std::shared_ptr<RemapArraysResponse> response,
                      rpc_helper->RemapArrays(std::move(req)).Await());

  std::vector<ArrayHandle> handles;
  for (auto& handle : response->array_handles()) {
    handles.push_back(ArrayHandle{handle});
  }
  TF_RET_CHECK(handles.size() == plan.output_specs.size());

  std::vector<tsl::RCReference<xla::ifrt::Array>> result;
  result.reserve(handles.size());
  for (int i = 0; i < handles.size(); ++i) {
    result.push_back(tsl::RCReference<xla::ifrt::Array>(
        tsl::MakeRef<Array>(client, rpc_helper, plan.output_specs[i].dtype,
                            plan.output_specs[i].shape,
                            plan.output_specs[i].sharding, handles[i])));
  }
  return result;
}

absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
Array::DisassembleIntoSingleDeviceArrays(ArrayCopySemantics semantics) {
  return DisassembleIntoSingleDeviceArrays(
      semantics, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
Array::DisassembleIntoSingleDeviceArrays(
    ArrayCopySemantics array_copy_semantics,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  auto req = std::make_unique<DisassembleIntoSingleDeviceArraysRequest>();
  req->set_array_handle(handle_.handle);
  req->set_copy_semantics(ToArrayCopySemanticsProto(array_copy_semantics));
  req->set_single_device_shard_semantics(
      ToSingleDeviceShardSemanticsProto(single_device_shard_semantics));

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<DisassembleIntoSingleDeviceArraysResponse> response,
      rpc_helper_->DisassembleIntoSingleDeviceArrays(std::move(req)).Await());
  std::vector<ArrayHandle> handles;
  for (auto& handle : response->single_device_array_handles()) {
    handles.push_back(ArrayHandle{handle});
  }

  TF_ASSIGN_OR_RETURN(auto shape_and_shardings, sharding_->Disassemble(shape_));
  CHECK_EQ(handles.size(), shape_and_shardings.size())
      << " " << absl::StrJoin(handles, ",") << " " << shape_ << " "
      << *sharding_ << " ";

  std::vector<tsl::RCReference<xla::ifrt::Array>> result;
  result.reserve(handles.size());
  for (int i = 0; i < handles.size(); ++i) {
    result.push_back(tsl::RCReference<xla::ifrt::Array>(tsl::MakeRef<Array>(
        client_, rpc_helper_, dtype_, std::move(shape_and_shardings[i].first),
        std::move(shape_and_shardings[i].second), handles[i])));
  }

  return result;
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> Array::FullyReplicatedShard(
    ArrayCopySemantics semantics) {
  auto req = std::make_unique<FullyReplicatedShardRequest>();
  req->set_array_handle(handle_.handle);
  req->set_copy_semantics(ToArrayCopySemanticsProto(semantics));

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<FullyReplicatedShardResponse> response,
      rpc_helper_->FullyReplicatedShard(std::move(req)).Await());

  ArrayHandle handle{response->array_handle()};

  // We are making the assumption the Array returned by the server corresponds
  // to the first device. Revisit this when IFRT supports: (1) an inexpensive
  // way to derive a SingleDeviceSharding from a fully replicated Array's
  // sharding and (2) A generalized `Reshard` API that allows the user to
  // request an Array to be made out of a specific single shard.
  std::unique_ptr<xla::ifrt::SingleDeviceSharding> single_device_sharding =
      xla::ifrt::SingleDeviceSharding::Create(
          sharding_->devices()->devices().front(), sharding_->memory_kind());

  return tsl::RCReference<xla::ifrt::Array>(
      tsl::MakeRef<Array>(client_, rpc_helper_, dtype_, shape_,
                          std::move(single_device_sharding), handle));
}

Future<> Array::CopyToHostBuffer(
    void* data, std::optional<absl::Span<const int64_t>> byte_strides,
    ArrayCopySemantics semantics) {
  const auto mem_region = ArrayMemRegion::FromZerothElementPointer(
      /*zeroth_element=*/data, dtype_, shape_, byte_strides);
  if (!mem_region.ok()) {
    return Future<>(mem_region.status());
  }

  auto req = std::make_unique<CopyToHostBufferRequest>();
  req->set_array_handle(handle_.handle);
  if (byte_strides.has_value()) {
    *req->mutable_byte_strides() = ToByteStridesProto(*byte_strides);
  }
  const uint64_t host_buffer_handle =
      rpc_helper_->host_buffer_store()->NextHandle();
  req->set_host_buffer_handle(host_buffer_handle);

  auto promise = Future<>::CreatePromise();
  auto on_ready = [host_buffer_store = rpc_helper_->host_buffer_store(),
                   promise, host_buffer_handle,
                   mem_region = mem_region->mem_region()](
                      absl::StatusOr<std::shared_ptr<CopyToHostBufferResponse>>
                          resp) mutable {
    if (!resp.ok()) {
      promise.Set(resp.status());
      return;
    }

    auto host_buffer = host_buffer_store->Lookup(host_buffer_handle);
    host_buffer.OnReady(
        [promise, mem_region, host_buffer_store,
         host_buffer_handle](absl::StatusOr<absl::Cord> data) mutable {
          absl::Cleanup cleanup = [&]() {
            host_buffer_store->Delete(host_buffer_handle)
                .OnReady([buffer_status = data.status()](absl::Status status) {
                  if (!status.ok()) {
                    LOG(WARNING) << "Failed to delete host buffer: " << status
                                 << " (buffer status: " << buffer_status << ")";
                  }
                });
          };

          if (!data.ok()) {
            promise.Set(data.status());
            return;
          }
          if (data->size() != mem_region.size()) {
            auto status = absl::InternalError(
                absl::StrCat("During CopyToHostBuffer, size mismatch in "
                             "response from proxy: ",
                             mem_region.size(), " vs ", data->size()));
            LOG(ERROR) << status;
            promise.Set(status);
            return;
          }
#if defined(PLATFORM_GOOGLE)
          data->CopyToArray(const_cast<char*>(mem_region.data()));
#else
          std::memcpy(const_cast<char*>(mem_region.data()),
                      data->Flatten().data(), data->size());
#endif
          promise.Set();
        });
  };
  rpc_helper_->CopyToHostBuffer(std::move(req)).OnReady(std::move(on_ready));
  return Future<>(std::move(promise));
}

xla::ifrt::Client* Array::client() const { return client_; }

std::string Array::DebugString() const {
  return absl::Substitute("proxy::Array, this=$0, handle=$1", this,
                          handle_.handle);
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
