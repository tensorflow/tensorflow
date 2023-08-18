/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_client.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_array.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_tuple.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tfrt/concurrency/ref_count.h"  // from @tf_runtime

namespace xla {
namespace ifrt {

char PjRtCompatibleClient::ID = 0;
char PjRtClient::ID = 0;

std::unique_ptr<PjRtClient> PjRtClient::Create(
    std::shared_ptr<xla::PjRtClient> pjrt_client) {
  return absl::WrapUnique(new PjRtClient(std::move(pjrt_client)));
}

StatusOr<tsl::RCReference<PjRtCompatibleArray>> PjRtClient::CreatePjRtArray(
    std::shared_ptr<PjRtBuffer> pjrt_buffer) {
  TF_ASSIGN_OR_RETURN(auto array,
                      PjRtArray::Create(this, std::move(pjrt_buffer)));
  return tsl::RCReference<PjRtCompatibleArray>(std::move(array));
}

StatusOr<tsl::RCReference<PjRtCompatibleArray>> PjRtClient::CreatePjRtArray(
    Shape shape, PjRtBuffers pjrt_buffers) {
  TF_ASSIGN_OR_RETURN(auto array, PjRtArray::Create(this, std::move(shape),
                                                    std::move(pjrt_buffers)));
  return tsl::RCReference<PjRtCompatibleArray>(std::move(array));
}

StatusOr<tsl::RCReference<Array>> PjRtClient::MakeArrayFromHostBuffer(
    const void* data, DType dtype, Shape shape,
    std::optional<absl::Span<const int64_t>> byte_strides,
    std::shared_ptr<const Sharding> sharding,
    Client::HostBufferSemantics semantics,
    std::function<void()> on_done_with_host_buffer) {
  DCHECK(this);
  if (!llvm::isa<const SingleDeviceSharding>(sharding.get())) {
    return InvalidArgument(
        "Only SingleDeviceSharding is supported: sharding=%s",
        sharding->DebugString());
  }
  TF_ASSIGN_OR_RETURN(auto primitive_type, ToPrimitiveType(dtype));

  std::unique_ptr<PjRtBuffer> buffer;
  // If the sharding has memory_kind specified, use a version of
  // `PjRtClient::BufferFromHostBuffer` that accepts `PjRtMemorySpace`.
  // Otherwise, use a non-`PjRtMemorySpace` version that is compatible with PjRt
  // implementations without memories support.
  if (sharding->memory_kind().memory_kind().has_value()) {
    // Find `PjRtMemorySpace` that is associated with the sharding's device and
    // matches the sharding's memory_kind.
    PjRtMemorySpace* memory_space = nullptr;
    for (PjRtMemorySpace* ms : sharding->devices().front()->memory_spaces()) {
      if (ms->memory_space_kind() == *sharding->memory_kind().memory_kind()) {
        memory_space = ms;
        break;
      }
    }
    if (memory_space == nullptr) {
      return InvalidArgument(
          "Invalid memory kind: %s; available memory kinds: %s",
          *sharding->memory_kind().memory_kind(),
          absl::StrJoin(sharding->devices().front()->memory_spaces(), ", ",
                        [](std::string* out, PjRtMemorySpace* ms) {
                          absl::StrAppend(out, ms->memory_space_kind());
                        }));
    }
    TF_ASSIGN_OR_RETURN(
        buffer, pjrt_client_->BufferFromHostBuffer(
                    data, primitive_type, shape.dims(), byte_strides, semantics,
                    std::move(on_done_with_host_buffer), memory_space,
                    /*device_layout=*/nullptr));
  } else {
    TF_ASSIGN_OR_RETURN(
        buffer,
        pjrt_client_->BufferFromHostBuffer(
            data, primitive_type, shape.dims(), byte_strides, semantics,
            std::move(on_done_with_host_buffer), sharding->devices().front()));
  }
  return PjRtArray::Create(
      this, dtype, std::move(shape), std::move(sharding),
      PjRtArray::PjRtBuffers({std::shared_ptr<PjRtBuffer>(buffer.release())}));
}

StatusOr<tsl::RCReference<Array>>
PjRtClient::AssembleArrayFromSingleDeviceArrays(
    Shape shape, std::shared_ptr<const Sharding> sharding,
    absl::Span<tsl::RCReference<Array>> arrays, ArrayCopySemantics semantics) {
  DCHECK(this);
  if (llvm::isa<const SingleDeviceSharding>(sharding.get())) {
    // Assemble with SingleDeviceSharding is No-op.
    if (arrays.size() != 1) {
      return InvalidArgument(
          "When the sharding is SingleDeviceSharding, the input arrays size "
          "must be one, but the actual size is %d",
          arrays.size());
    }
    return arrays[0];
  } else if (!llvm::isa<const OpaqueSharding, const ConcreteSharding,
                        const ConcreteEvenSharding,
                        const ShardingParamSharding>(sharding.get())) {
    return InvalidArgument(
        "Only SingleDeviceSharding, OpaqueSharding, ConcreteSharding, "
        "ConcreteEvenSharding, and ShardingParamSharding are supported: "
        "sharding=%s",
        sharding->DebugString());
  }
  if (sharding->devices().size() != arrays.size()) {
    return InvalidArgument(
        "Number of output shards must match the number of single-shard arrays: "
        "%d vs. %d",
        sharding->devices().size(), arrays.size());
  }
  PjRtArray::PjRtBuffers buffers;
  buffers.reserve(arrays.size());
  DType dtype = arrays[0]->dtype();
  for (int i = 0; i < arrays.size(); ++i) {
    if (!llvm::isa<PjRtCompatibleArray>(arrays[i].get())) {
      return InvalidArgument(
          "Only PjRtCompatibleArray is supported: arrays[%d]=%s", i,
          arrays[i]->DebugString());
    }
    auto* array = static_cast<PjRtCompatibleArray*>(arrays[i].get());
    if (array->dtype() != dtype) {
      return InvalidArgument(
          "Every input must have the same dtype: %s (shard 0) vs. %s (shard "
          "%d)",
          dtype.DebugString(), array->dtype().DebugString(), i);
    }
    if (array->sharding().devices().size() != 1) {
      return InvalidArgument(
          "Every input must use a single device sharding, but input %d has "
          "sharding=%s",
          i, array->sharding().DebugString());
    }
    switch (semantics) {
      case ArrayCopySemantics::kAlwaysCopy:
        // TODO(hyeontaek): kAlwaysCopy should clone the buffer, but the PjRt
        // API does not have efficient buffer cloning on the same device.
        buffers.push_back(array->pjrt_buffers().front());
        break;
      case ArrayCopySemantics::kReuseInput:
        buffers.push_back(array->pjrt_buffers().front());
        break;
      case ArrayCopySemantics::kDonateInput:
        buffers.push_back(std::move(array->pjrt_buffers().front()));
        break;
    }
  }
  return PjRtArray::Create(this, dtype, std::move(shape), std::move(sharding),
                           std::move(buffers));
}

StatusOr<tsl::RCReference<Tuple>> PjRtClient::MakeTuple(
    absl::Span<tsl::RCReference<Value>> values) {
  return PjRtTuple::Create(this, values);
}

StatusOr<std::shared_ptr<const xla::PjRtTopologyDescription>>
PjRtClient::GetTopologyForDevices(absl::Span<Device* const> devices) const {
  // TODO(parkers): Consider constructing a sub-slice topology based on the
  // provided devices.
  TF_ASSIGN_OR_RETURN(auto topology, pjrt_client_->GetTopologyDescription());
  return std::shared_ptr<const xla::PjRtTopologyDescription>(pjrt_client_,
                                                             topology);
}

}  // namespace ifrt
}  // namespace xla
