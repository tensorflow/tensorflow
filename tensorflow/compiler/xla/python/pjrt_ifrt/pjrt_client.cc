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
#include <utility>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_array.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

char PjRtClient::ID = 0;

std::unique_ptr<ifrt::Client> PjRtClient::Create(
    std::shared_ptr<xla::PjRtClient> pjrt_client) {
  return std::unique_ptr<ifrt::Client>(new PjRtClient(std::move(pjrt_client)));
}

std::unique_ptr<ifrt::Client> PjRtClient::Create(
    std::unique_ptr<xla::PjRtClient> pjrt_client) {
  return Create(std::shared_ptr<xla::PjRtClient>(pjrt_client.release()));
}

StatusOr<std::unique_ptr<Array>> PjRtClient::MakeArrayFromHostBuffer(
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
  TF_ASSIGN_OR_RETURN(
      auto buffer,
      pjrt_client_->BufferFromHostBuffer(
          data, primitive_type, shape.dims(), byte_strides, semantics,
          std::move(on_done_with_host_buffer), sharding->devices().front()));
  return PjRtArray::Create(
      this, dtype, std::move(shape), std::move(sharding),
      PjRtArray::PjRtBuffers({std::shared_ptr<PjRtBuffer>(buffer.release())}));
}

StatusOr<std::unique_ptr<Array>>
PjRtClient::AssembleArrayFromSingleDeviceArrays(
    Shape shape, std::shared_ptr<const Sharding> sharding,
    absl::Span<Array* const> arrays, ArrayCopySemantics semantics) {
  DCHECK(this);
  if (!llvm::isa<const OpaqueSharding>(sharding.get())) {
    return InvalidArgument("Only OpaqueSharding is supported: sharding=%s",
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
    if (!llvm::isa<PjRtArray>(arrays[i])) {
      return InvalidArgument("Only PjRtArray is supported: arrays[%d]=%s", i,
                             arrays[i]->DebugString());
    }
    const auto* array = static_cast<const PjRtArray*>(arrays[i]);
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

}  // namespace ifrt
}  // namespace xla
