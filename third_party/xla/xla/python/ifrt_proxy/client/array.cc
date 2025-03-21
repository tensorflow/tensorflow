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
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/client_impl_util.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt_proxy/client/global_flags.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/common/array_util.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"
#include "xla/python/ifrt_proxy/common/versions.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace ifrt {
namespace proxy {

namespace {

template <typename T>
void CheckResponseAfterAsyncCall(const Future<std::shared_ptr<T>>& f,
                                 ArrayHandle handle) {
  f.OnReady([handle](absl::StatusOr<std::shared_ptr<T>> r) {
    if (r.ok()) {
      CHECK_EQ(r.value()->array_handle(), handle.handle);
    } else {
      LOG(ERROR) << "Received error response for background call of type "
                 << T::GetDescriptor()->full_name() << " relating to handle "
                 << handle.handle << ": " << r.status();
    }
  });
}

template <typename T>
void CheckResponseAfterAsyncCall(const Future<std::shared_ptr<T>>& f,
                                 const std::vector<ArrayHandle>& handles) {
  f.OnReady([handles = handles](absl::StatusOr<std::shared_ptr<T>> r) {
    if (r.ok()) {
      for (int i = 0; i < handles.size(); ++i) {
        CHECK_EQ(r.value()->array_handles(i), handles[i].handle);
      }
    } else {
      LOG(ERROR) << "Received error response for background call of type "
                 << T::GetDescriptor()->full_name() << "relating to handles "
                 << absl::StrJoin(handles, ",") << ": " << r.status();
    }
  });
}

using HostBufferSemantics = ::xla::ifrt::Client::HostBufferSemantics;

// Makes a host buffer on the server.
absl::StatusOr<uint64_t> MakeHostBuffer(
    xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
    const void* data, DType dtype, Shape shape,
    std::optional<absl::Span<const int64_t>> byte_strides,
    HostBufferSemantics semantics,
    std::function<void()> on_done_with_host_buffer) {
  absl::string_view mem_region;
  if (dtype.kind() != DType::kString) {
    TF_ASSIGN_OR_RETURN(
        auto array_mem_region,
        ArrayMemRegion::FromZerothElementPointer(
            /*zeroth_element=*/data, dtype, shape, byte_strides));
    mem_region = array_mem_region.mem_region();
  } else {
    // DType::kString
    if (rpc_helper->version().protocol_version() < 9) {
      return absl::UnimplementedError(
          "String arrays are not supported in ifrt-proxy version < 9");
    }
    tsl::profiler::TraceMe traceme("IfrtProxySerializeStringHostBuffer");
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<std::string> owned_data,
        SerializeStringHostBuffer(absl::MakeConstSpan(
            static_cast<const absl::Cord*>(data), shape.num_elements())));
    mem_region = *owned_data;
    semantics = HostBufferSemantics::kImmutableUntilTransferCompletes;
    std::function<void()> on_done(std::move(on_done_with_host_buffer));
    on_done_with_host_buffer = [owned_data = std::move(owned_data),
                                on_done = std::move(on_done)]() {
      if (on_done) {
        std::move(on_done)();
      }
    };
  }
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      [s = mem_region.size(), semantics]() {
        return tsl::profiler::TraceMeEncode(
            "IfrtProxyEntrypointMakeArrayFromHostBuffer",
            {{"size", s}, {"semantics", static_cast<int>(semantics)}});
      });

  const uint64_t host_buffer_handle = rpc_helper->NextHandle();

  if (GetGlobalClientFlags()->synchronous_host_buffer_store ||
      rpc_helper->version().protocol_version() < 10) {
    // Synchronously send data and await.
    TF_RETURN_IF_ERROR(rpc_helper->host_buffer_store()
                           ->Store(host_buffer_handle, mem_region)
                           .Await());
    if (on_done_with_host_buffer != nullptr) {
      std::move(on_done_with_host_buffer)();
    }
  } else {
    // Asynchronously send data.

    if (semantics == HostBufferSemantics::kImmutableOnlyDuringCall) {
      char* alloc = static_cast<char*>(malloc(mem_region.size()));
      memcpy(alloc, mem_region.data(), mem_region.size());
      mem_region = absl::string_view(alloc, mem_region.size());
      if (on_done_with_host_buffer != nullptr) {
        std::move(on_done_with_host_buffer)();
      }
      on_done_with_host_buffer = [alloc]() { free(alloc); };
    }

    // If the async-send results in an error, ignoring it may mean that the
    // control-path hangs forever. Instead, we explicitly ensure the
    // control-path gets disconnected (and so the entire session ends).
    //
    // While there are more fine-grained approaches to handle errors, we do not
    // expect an error except for one that indicates being already disconnected
    // from the server.
    rpc_helper->host_buffer_store()
        ->Store(host_buffer_handle, mem_region)
        .OnReady([on_done = std::move(on_done_with_host_buffer),
                  rpc_helper = std::weak_ptr<RpcHelper>(rpc_helper)](
                     absl::Status s) mutable {
          if (!s.ok()) {
            LOG(WARNING) << "Handling error in background data-transfer by "
                         << "disconnecting from server (if not already "
                         << "disconnected), error: " << s;
            if (auto locked = rpc_helper.lock()) {
              locked->Disconnect();
            }
          };
          if (on_done != nullptr) {
            std::move(on_done)();
          }
        });
  }
  return host_buffer_handle;
}

}  // namespace

char Array::ID = 0;

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
Array::MakeArrayFromHostBuffer(
    xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
    const void* data, DType dtype, Shape shape,
    std::optional<absl::Span<const int64_t>> byte_strides,
    std::shared_ptr<const Sharding> sharding, HostBufferSemantics semantics,
    std::function<void()> on_done_with_host_buffer) {
  TF_ASSIGN_OR_RETURN(
      const uint64_t host_buffer_handle,
      MakeHostBuffer(client, rpc_helper, data, dtype, shape, byte_strides,
                     semantics, std::move(on_done_with_host_buffer)));
  auto cleanup = absl::MakeCleanup([&]() {
    rpc_helper->host_buffer_store()
        ->Delete(host_buffer_handle)
        .OnReady([](absl::Status status) {
          if (!status.ok()) {
            LOG(WARNING) << "Failed to delete host buffer: " << status;
          }
        });
  });

  auto req = std::make_unique<MakeArrayFromHostBufferRequest>();
  req->set_host_buffer_handle(host_buffer_handle);
  // Reuse the host_buffer_handle as also the client-generated
  // array_handle.
  req->set_array_handle(host_buffer_handle);
  *req->mutable_dtype() = dtype.ToProto();
  *req->mutable_shape() = shape.ToProto();
  TF_ASSIGN_OR_RETURN(*req->mutable_sharding(), sharding->ToProto());
  if (byte_strides.has_value()) {
    *req->mutable_byte_strides() = ToByteStridesProto(*byte_strides);
  }

  ArrayHandle arr_handle;
  if (GetGlobalClientFlags()->synchronous_host_buffer_store ||
      rpc_helper->version().protocol_version() < 10) {
    TF_ASSIGN_OR_RETURN(
        auto resp, rpc_helper->MakeArrayFromHostBuffer(std::move(req)).Await());
    arr_handle.handle = resp->array_handle();
  } else {
    arr_handle.handle = host_buffer_handle;
    CheckResponseAfterAsyncCall(
        rpc_helper->MakeArrayFromHostBuffer(std::move(req)), arr_handle);
  }

  std::move(cleanup).Cancel();

  return tsl::RCReference<xla::ifrt::Array>(
      tsl::MakeRef<Array>(client, std::move(rpc_helper), dtype,
                          std::move(shape), std::move(sharding), arr_handle));
}

absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
Array::MakeArraysFromHostBufferShards(
    xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
    absl::Span<xla::ifrt::Client::MakeArraysFromHostBufferShardsSpec> specs,
    xla::ifrt::Client::HostBufferSemantics semantics) {
  if (rpc_helper->version().protocol_version() <
      protocol_version::kMakeArraysFromHostBufferShards) {
    return xla::ifrt::ClientMakeArraysFromHostBufferShards(client, specs,
                                                           semantics);
  }

  absl::InlinedVector<absl::InlinedVector<uint64_t, 1>, 1>
      host_buffer_handles_for_specs;
  auto cleanup = absl::MakeCleanup([&]() {
    for (const auto& host_buffer_handles : host_buffer_handles_for_specs) {
      for (const uint64_t host_buffer_handle : host_buffer_handles) {
        rpc_helper->host_buffer_store()
            ->Delete(host_buffer_handle)
            .OnReady([](absl::Status status) {
              if (!status.ok()) {
                LOG(WARNING) << "Failed to delete host buffer: " << status;
              }
            });
      }
    }
  });
  host_buffer_handles_for_specs.reserve(specs.size());
  for (const auto& spec : specs) {
    auto& host_buffer_handles = host_buffer_handles_for_specs.emplace_back();
    host_buffer_handles.reserve(spec.buffers.size());
    for (const auto& [_, host_buffer] : spec.buffers) {
      TF_ASSIGN_OR_RETURN(
          const uint64_t host_buffer_handle,
          MakeHostBuffer(client, rpc_helper, host_buffer.data,
                         host_buffer.dtype, host_buffer.shape,
                         host_buffer.byte_strides, semantics,
                         /*on_done_with_host_buffer=*/host_buffer.on_done));
      host_buffer_handles.push_back(host_buffer_handle);
    }
  }

  std::vector<ArrayHandle> arr_handles;
  arr_handles.reserve(specs.size());

  auto req = std::make_unique<MakeArraysFromHostBufferShardsRequest>();
  req->mutable_specs()->Reserve(specs.size());
  if (!GetGlobalClientFlags()->synchronous_host_buffer_store) {
    req->mutable_array_handles()->Reserve(specs.size());
  }
  for (int spec_idx = 0; spec_idx < specs.size(); ++spec_idx) {
    const xla::ifrt::Client::MakeArraysFromHostBufferShardsSpec& spec =
        specs[spec_idx];
    MakeArraysFromHostBufferShardsRequest::MakeArraysFromHostBufferShardsSpec*
        spec_proto = req->add_specs();
    spec_proto->mutable_host_buffers()->Reserve(spec.buffers.size());
    for (int buffer_idx = 0; buffer_idx < spec.buffers.size(); ++buffer_idx) {
      const auto& [addressable_shard_indices, host_buffer] =
          spec.buffers[buffer_idx];
      MakeArraysFromHostBufferShardsRequest::ShardIndices*
          addressable_shard_indices_proto =
              spec_proto->add_addressable_shard_indices();
      addressable_shard_indices_proto->mutable_indices()->Reserve(
          addressable_shard_indices.size());
      for (const int shard_index : addressable_shard_indices) {
        addressable_shard_indices_proto->add_indices(shard_index);
      }

      MakeArraysFromHostBufferShardsRequest::HostBuffer* host_buffer_proto =
          spec_proto->add_host_buffers();
      *host_buffer_proto->mutable_dtype() = host_buffer.dtype.ToProto();
      *host_buffer_proto->mutable_shape() = host_buffer.shape.ToProto();
      host_buffer_proto->set_host_buffer_handle(
          host_buffer_handles_for_specs[spec_idx][buffer_idx]);
      if (host_buffer.byte_strides.has_value()) {
        *host_buffer_proto->mutable_byte_strides() =
            ToByteStridesProto(*host_buffer.byte_strides);
      }
    }
    TF_ASSIGN_OR_RETURN(*spec_proto->mutable_array_spec(),
                        spec.array_spec.ToProto());

    if (!GetGlobalClientFlags()->synchronous_host_buffer_store) {
      uint64_t arr_handle;
      if (spec.buffers.empty()) {
        arr_handle = rpc_helper->NextHandle();
      } else {
        // Reuse the host_buffer_handle as also the client-generated arr_handle.
        arr_handle = spec_proto->host_buffers(0).host_buffer_handle();
      }

      req->add_array_handles(arr_handle);
      arr_handles.push_back(ArrayHandle{arr_handle});
    }
  }

  if (GetGlobalClientFlags()->synchronous_host_buffer_store) {
    TF_ASSIGN_OR_RETURN(
        auto resp,
        rpc_helper->MakeArraysFromHostBufferShards(std::move(req)).Await());
    for (const uint64_t array_handle : resp->array_handles()) {
      arr_handles.push_back(ArrayHandle{array_handle});
    }
  } else {
    CheckResponseAfterAsyncCall(
        rpc_helper->MakeArraysFromHostBufferShards(std::move(req)),
        arr_handles);
  }

  std::move(cleanup).Cancel();

  std::vector<tsl::RCReference<xla::ifrt::Array>> arrays;
  arrays.reserve(specs.size());
  for (int spec_idx = 0; spec_idx < specs.size(); ++spec_idx) {
    xla::ifrt::Client::MakeArraysFromHostBufferShardsSpec& spec =
        specs[spec_idx];
    arrays.push_back(tsl::MakeRef<Array>(
        client, rpc_helper, spec.array_spec.dtype,
        std::move(spec.array_spec.shape), std::move(spec.array_spec.sharding),
        arr_handles[spec_idx]));
  }
  return arrays;
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
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointArrayGetReadyFuture");

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
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointIsDeleted");
  if (GetGlobalClientFlags()->array_is_deleted_hack) {
    return false;
  }
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
    DType dtype, Shape shape, std::shared_ptr<const Sharding> sharding,
    absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
    ArrayCopySemantics array_copy_semantics,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      [n_arrays = arrays.size(), single_device_shard_semantics]() {
        return tsl::profiler::TraceMeEncode(
            "IfrtProxyEntrypointAssembleArrayFromSingleDeviceArrays",
            {{"n_arrays", n_arrays},
             {"sds_semantics",
              static_cast<int>(single_device_shard_semantics)}});
      });
  if (single_device_shard_semantics ==
          SingleDeviceShardSemantics::kAddressableShards &&
      rpc_helper->version().protocol_version() < 8) {
    return absl::UnimplementedError(
        "SingleDeviceShardSemantics::kAdressableShards is not supported in "
        "ifrt-proxy version < 8");
  }
  auto req = std::make_unique<AssembleArrayFromSingleDeviceArraysRequest>();
  *req->mutable_shape() = shape.ToProto();
  TF_ASSIGN_OR_RETURN(*req->mutable_sharding(), sharding->ToProto());
  req->set_copy_semantics(ToArrayCopySemanticsProto(array_copy_semantics));
  req->set_single_device_shard_semantics(
      ToSingleDeviceShardSemanticsProto(single_device_shard_semantics));
  *req->mutable_dtype() = dtype.ToProto();
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

  ArrayHandle result_handle;
  if (rpc_helper->version().protocol_version() <
      protocol_version::kClientHandlesOptimization2) {
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<AssembleArrayFromSingleDeviceArraysResponse> response,
        rpc_helper->AssembleArrayFromSingleDeviceArrays(std::move(req))
            .Await());

  } else {
    result_handle.handle = rpc_helper->NextHandle();
    req->set_result_handle(result_handle.handle);
    CheckResponseAfterAsyncCall(
        rpc_helper->AssembleArrayFromSingleDeviceArrays(std::move(req)),
        result_handle);
  }

  return tsl::RCReference<xla::ifrt::Array>(tsl::MakeRef<Array>(
      client, std::move(rpc_helper), dtype, std::move(shape),
      std::move(sharding), result_handle));
}

absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
Array::RemapArrays(xla::ifrt::Client* client,
                   std::shared_ptr<RpcHelper> rpc_helper, const RemapPlan& plan,
                   absl::Span<tsl::RCReference<xla::ifrt::Array>> arrays,
                   ArrayCopySemantics semantics) {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint([n_arrays = arrays.size()]() {
    return tsl::profiler::TraceMeEncode("IfrtProxyEntrypointRemapArrays",
                                        {{"n_arrays", n_arrays}});
  });

  TF_RETURN_IF_ERROR(plan.CheckArrayCopySemantics(semantics));
  const int num_inputs = plan.input_specs.size();
  const int num_actual_inputs = arrays.size();
  if (num_inputs != num_actual_inputs) {
    return absl::InvalidArgumentError(
        absl::StrFormat("RemapArrays expects %d input arrays, but got %d",
                        num_inputs, num_actual_inputs));
  }

  auto req = std::make_unique<RemapArraysRequest>();
  TF_RET_CHECK(!arrays.empty());
  TF_ASSIGN_OR_RETURN(*req->mutable_plan(), plan.ToProto());
  req->set_copy_semantics(ToArrayCopySemanticsProto(semantics));
  for (int i = 0; i < num_inputs; ++i) {
    const tsl::RCReference<xla::ifrt::Array>& rcref = arrays[i];
    Array* array = llvm::dyn_cast<Array>(rcref.get());
    if (array == nullptr) {
      return absl::InvalidArgumentError(
          absl::Substitute("Array at $0 supplied to RemapArrays() is "
                           "not a xla::ifrt::proxy::Array.",
                           rcref.get()));
    }

    if (plan.input_specs[i].dtype != arrays[i]->dtype()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "RemapArrays expects input #%d to have dtype %v, but got %v", i,
          plan.input_specs[i].dtype, arrays[i]->dtype()));
    }
    if (plan.input_specs[i].shape != arrays[i]->shape()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "RemapArrays expects input #%d to have shape %v, but got %v", i,
          plan.input_specs[i].shape, arrays[i]->shape().DebugString()));
    }
    // Skip xla::ifrt::Sharding::HasSamePartitioning() check because RemapArrays
    // is currently called with input arrays with implicit sharding
    // reinterpretation. Such patterns should be fixed before enabling stricter
    // checking to avoid false positives.
    if (*plan.input_specs[i].sharding->devices() !=
            *arrays[i]->sharding().devices() ||
        plan.input_specs[i].sharding->memory_kind() !=
            arrays[i]->sharding().memory_kind()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("RemapArrays expects input #%d to be on %v with "
                          "%v, but is on %v with %v",
                          i, *plan.input_specs[i].sharding->devices(),
                          plan.input_specs[i].sharding->memory_kind(),
                          *arrays[i]->sharding().devices(),
                          arrays[i]->sharding().memory_kind()));
    }

    req->add_array_handles(array->handle_.handle);
  }

  std::vector<ArrayHandle> result_handles;
  if (rpc_helper->version().protocol_version() <
      protocol_version::kClientHandlesOptimization2) {
    TF_ASSIGN_OR_RETURN(std::shared_ptr<RemapArraysResponse> response,
                        rpc_helper->RemapArrays(std::move(req)).Await());
    TF_RET_CHECK(result_handles.size() == plan.output_specs.size());
    for (auto& handle : response->array_handles()) {
      result_handles.push_back(ArrayHandle{handle});
    }
  } else {
    for (int i = 0; i < plan.output_specs.size(); ++i) {
      uint64_t h = rpc_helper->NextHandle();
      result_handles.push_back(ArrayHandle{h});
      req->add_result_handles(h);
    }
    CheckResponseAfterAsyncCall(rpc_helper->RemapArrays(std::move(req)),
                                result_handles);
  }

  std::vector<tsl::RCReference<xla::ifrt::Array>> result;
  result.reserve(result_handles.size());
  for (int i = 0; i < result_handles.size(); ++i) {
    result.push_back(tsl::RCReference<xla::ifrt::Array>(
        tsl::MakeRef<Array>(client, rpc_helper, plan.output_specs[i].dtype,
                            plan.output_specs[i].shape,
                            plan.output_specs[i].sharding, result_handles[i])));
  }
  return result;
}

absl::StatusOr<std::vector<tsl::RCReference<xla::ifrt::Array>>>
Array::DisassembleIntoSingleDeviceArrays(
    ArrayCopySemantics array_copy_semantics,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointDisassembleIntoSingleDeviceArrays");
  if (single_device_shard_semantics ==
          SingleDeviceShardSemantics::kAddressableShards &&
      rpc_helper_->version().protocol_version() < 8) {
    return absl::UnimplementedError(
        "SingleDeviceShardSemantics::kAdressableShards is not supported in "
        "version < 8");
  }
  auto req = std::make_unique<DisassembleIntoSingleDeviceArraysRequest>();
  req->set_array_handle(handle_.handle);
  req->set_copy_semantics(ToArrayCopySemanticsProto(array_copy_semantics));
  req->set_single_device_shard_semantics(
      ToSingleDeviceShardSemanticsProto(single_device_shard_semantics));

  std::vector<ArrayHandle> result_handles;
  TF_ASSIGN_OR_RETURN(auto shape_and_shardings, sharding_->Disassemble(shape_));
  result_handles.reserve(shape_and_shardings.size());

  if (rpc_helper_->version().protocol_version() <
      protocol_version::kClientHandlesOptimization2) {
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<DisassembleIntoSingleDeviceArraysResponse> response,
        rpc_helper_->DisassembleIntoSingleDeviceArrays(std::move(req)).Await());
    for (auto& handle : response->array_handles()) {
      result_handles.push_back(ArrayHandle{handle});
    }
  } else {
    for (int i = 0; i < shape_and_shardings.size(); ++i) {
      uint64_t h = rpc_helper_->NextHandle();
      result_handles.push_back(ArrayHandle{h});
      req->add_result_handles(h);
    }
    CheckResponseAfterAsyncCall(
        rpc_helper_->DisassembleIntoSingleDeviceArrays(std::move(req)),
        result_handles);
  }

  CHECK_EQ(result_handles.size(), shape_and_shardings.size())
      << " " << absl::StrJoin(result_handles, ",") << " " << shape_ << " "
      << *sharding_ << " ";

  std::vector<tsl::RCReference<xla::ifrt::Array>> result;
  result.reserve(result_handles.size());
  for (int i = 0; i < result_handles.size(); ++i) {
    result.push_back(tsl::RCReference<xla::ifrt::Array>(tsl::MakeRef<Array>(
        client_, rpc_helper_, dtype_, std::move(shape_and_shardings[i].first),
        std::move(shape_and_shardings[i].second), result_handles[i])));
  }

  return result;
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> Array::FullyReplicatedShard(
    ArrayCopySemantics semantics) {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointFullyReplicatedShard");
  auto req = std::make_unique<FullyReplicatedShardRequest>();
  req->set_array_handle(handle_.handle);
  req->set_copy_semantics(ToArrayCopySemanticsProto(semantics));

  ArrayHandle result_handle;
  if (rpc_helper_->version().protocol_version() <
      protocol_version::kClientHandlesOptimization2) {
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<FullyReplicatedShardResponse> response,
        rpc_helper_->FullyReplicatedShard(std::move(req)).Await());
    result_handle.handle = response->array_handle();
  } else {
    result_handle.handle = rpc_helper_->NextHandle();
    req->set_result_handle(result_handle.handle);
    CheckResponseAfterAsyncCall(
        rpc_helper_->FullyReplicatedShard(std::move(req)), result_handle);
  }

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
                          std::move(single_device_sharding), result_handle));
}

Future<> Array::CopyToStringHostBuffer(
    void* data, std::optional<absl::Span<const int64_t>> byte_strides,
    ArrayCopySemantics semantics) {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointCopyToStringHostBuffer");
  if (rpc_helper_->version().protocol_version() < 9) {
    return Future<>(absl::UnimplementedError(
        "String arrays are not supported in ifrt-proxy version < 9"));
  }
  auto req = std::make_unique<CopyToHostBufferRequest>();
  req->set_array_handle(handle_.handle);
  if (byte_strides.has_value()) {
    return Future<>(absl::InvalidArgumentError(
        "Byte strides are not supported for string arrays."));
  }

  const uint64_t host_buffer_handle = rpc_helper_->NextHandle();
  req->set_host_buffer_handle(host_buffer_handle);
  auto promise = Future<>::CreatePromise();
  auto on_ready = [promise,
                   host_buffer_store = rpc_helper_->host_buffer_store(),
                   host_buffer_handle,
                   dst_buffer = static_cast<absl::Cord*>(data)](
                      absl::StatusOr<std::shared_ptr<CopyToHostBufferResponse>>
                          resp) mutable {
    if (!resp.ok()) {
      promise.Set(resp.status());
      return;
    }
    host_buffer_store->Lookup(host_buffer_handle)
        .OnReady([promise, dst_buffer, host_buffer_store, host_buffer_handle](
                     absl::StatusOr<absl::Cord> array_contents) mutable {
          absl::Cleanup cleanup = [&]() {
            host_buffer_store->Delete(host_buffer_handle)
                .OnReady([buffer_deletion_status =
                              array_contents.status()](absl::Status status) {
                  if (!status.ok()) {
                    LOG(WARNING)
                        << "Failed to delete host buffer: " << status
                        << " (buffer status: " << buffer_deletion_status << ")";
                  }
                });
          };

          if (!array_contents.ok()) {
            promise.Set(array_contents.status());
            return;
          }
          auto deserialization_status =
              DeserializeFromCordIntoPreallocatedStringHostBuffer(
                  *array_contents, dst_buffer);
          promise.Set(deserialization_status);
        });
  };
  rpc_helper_->CopyToHostBuffer(std::move(req)).OnReady(std::move(on_ready));
  return Future<>(std::move(promise));
}

Future<> Array::CopyToHostBuffer(
    void* data, std::optional<absl::Span<const int64_t>> byte_strides,
    ArrayCopySemantics semantics) {
  if (dtype_.kind() == DType::kString) {
    return CopyToStringHostBuffer(data, byte_strides, semantics);
  }
  tsl::profiler::TraceMe traceme("IfrtProxyEntrypointCopyToHostBuffer");
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
  const uint64_t host_buffer_handle = rpc_helper_->NextHandle();
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
